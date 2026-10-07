# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md (adapt a little code)

"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def masked_l1_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean L1 over valid action cells, preserving legacy behavior without a mask."""
    error = torch.abs(prediction - target)
    if mask is None:
        return error.mean()
    valid = mask.to(device=error.device, dtype=error.dtype)
    if valid.shape != error.shape:
        raise ValueError(f"action_mask shape {tuple(valid.shape)} != action shape {tuple(error.shape)}")
    denominator = valid.sum()
    if not torch.any(mask):
        raise ValueError("action_mask contains no valid targets")
    return (error * valid).sum() / denominator

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import add_discretized_state_to_instruction, merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.motion.head import MotionHead
from starVLA.model.modules.motion.loss import motion_credit, token_motion_loss
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.vlm.QWen3_5 import ACTION_TOKEN, qwen_apply_processor, qwen_build_messages
from starVLA.training.trainer_utils.trainer_tools import resize_images


def assemble_prompts(examples: List[dict], chunk_len: int, action_token: str) -> List[str]:
    """Assemble per-sample prompts (state tokens + action-token suffix); shared by forward/eval and worker collates."""
    instructions = [example["lang"] for example in examples]
    state = [example["state"] for example in examples] if "state" in examples[0] else None
    instructions = add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
    action_tokens = action_token * chunk_len
    prompt_suffix = f" Please predict the next {chunk_len} robot actions: <action>{action_tokens}<action>."
    return [instruction + prompt_suffix for instruction in instructions]


def make_qwen_preprocess_collate(cfg, processor):
    """Opt-in collate factory: run prompt assembly + Qwen processor in DataLoader workers."""
    chunk_len = int(cfg.framework.action_model.action_horizon)
    cot_prompt = cfg.datasets.vla_data.get("CoT_prompt", None)

    def collate(batch):
        instructions = assemble_prompts(batch, chunk_len, ACTION_TOKEN)
        messages = qwen_build_messages([example["image"] for example in batch], instructions, cot_prompt)
        qwen_inputs = qwen_apply_processor(processor, messages)
        batch[0]["qwen_inputs"] = qwen_inputs
        return batch

    return collate


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenOFT
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenOFTDefaultConfig:
    """QwenOFT framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenOFT"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (MLP regression over action special tokens) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "MLP",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # Hidden dim for the action MLP (auto-set from VLM hidden_size at runtime)
            "action_hidden_dim": 2560,
            # How many future steps to predict
            "future_action_window_size": 8,
            # How many past steps included in action chunk (usually 0)
            "past_action_window_size": 0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    Multimodal vision-language-action model (OFT variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Action special token injected into the VLM sequence
      - MLP regression head over action token hidden states (L1 loss)

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenOFTDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align action_hidden_dim to VLM hidden_size at runtime
        self.config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon
        # self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_token = ACTION_TOKEN  # TODO also can add spacail token to Qwen, but too complex
        self.action_token_id = self.qwen_vl_interface.action_token_id

        motion_cfg = self.config.framework.get("motion", None)
        self.motion_mode = motion_cfg.get("mode", "uniform") if motion_cfg else None
        if motion_cfg is not None:
            if self.motion_mode not in ("uniform", "compat", "random", "shuffled"):
                raise ValueError(f"unsupported framework.motion.mode {self.motion_mode}")
            self.motion_loss_weight = float(motion_cfg.get("loss_weight", 1.0))
            self.motion_head = MotionHead(self.qwen_vl_interface.model.config.hidden_size)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly regress future actions (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        if "qwen_inputs" in examples[0]:
            qwen_inputs = examples[0]["qwen_inputs"]
        else:
            instructions = assemble_prompts(examples, self.chunk_len, self.action_token)
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=[example["image"] for example in examples],
                instructions=instructions,
            )
        actions = [example["action"] for example in examples]  # label [B, len, 7]
        action_masks = [example.get("action_mask") for example in examples]
        if any(mask is not None for mask in action_masks) and not all(mask is not None for mask in action_masks):
            raise ValueError("action_mask must be present for every example in a batch or for none")
        entry_embeds = None

        def _stash_entry_embeds(module, args, kwargs):
            nonlocal entry_embeds
            entry_embeds = kwargs["inputs_embeds"]

        hook_handle = self.qwen_vl_interface.model.model.language_model.register_forward_pre_hook(
            _stash_entry_embeds, with_kwargs=True
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface.model.model(
                **qwen_inputs,
                output_attentions=False,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.last_hidden_state  # [B, L, H]
        hook_handle.remove()

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

            # Label alignment: take the last chunk_len segment
            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            action_mask = None
            if action_masks[0] is not None:
                action_mask = torch.as_tensor(
                    np.asarray(action_masks), device=pred_actions.device, dtype=torch.bool
                )[:, -self.action_horizon :, :]
            action_loss = masked_l1_loss(pred_actions, actions_target, action_mask)

            motion_loss, compat = None, None
            if self.motion_mode is not None:
                if "flow_target" not in examples[0]:
                    raise ValueError(
                        "framework.motion is enabled but flow_target is missing from the batch; "
                        "enable datasets.vla_data.motion_flow_root"
                    )
                motion_target = torch.as_tensor(
                    np.stack([example["flow_target"] for example in examples]), device=action_loss.device
                ).float()  # [B, V, Ht, Wt, 2]; NaN marks anchors whose future frame left the episode
                visual_mask = qwen_inputs["input_ids"] == self.qwen_vl_interface.model.config.image_token_id
                num_views = len(examples[0]["image"])
                E_vis = self._gather_visual_token_embeddings(
                    entry_embeds, visual_mask, qwen_inputs["image_grid_thw"], num_views
                )  # [B, V, Ht, Wt, D]
                per_token_motion_loss, motion_valid = token_motion_loss(self.motion_head(E_vis), motion_target)
                num_valid = motion_valid.sum()
                w, compat = None, None
                if self.motion_mode in ("compat", "random", "shuffled"):
                    if num_valid > 0:
                        if self.motion_mode == "random":
                            # Random draws C ~ U(-1,1), independent of real gradients.
                            w, compat = motion_credit(
                                per_token_motion_loss, None, None, motion_valid, self.motion_mode
                            )
                        else:
                            g_A = self._gather_visual_token_embeddings(
                                torch.autograd.grad(action_loss, entry_embeds, retain_graph=True)[0],
                                visual_mask,
                                qwen_inputs["image_grid_thw"],
                                num_views,
                            )  # [B, V, Ht, Wt, D]
                            motion_scalar = per_token_motion_loss[motion_valid].mean()
                            g_M = self._gather_visual_token_embeddings(
                                torch.autograd.grad(motion_scalar, entry_embeds, retain_graph=True)[0],
                                visual_mask,
                                qwen_inputs["image_grid_thw"],
                                num_views,
                            )  # [B, V, Ht, Wt, D]
                            w, compat = motion_credit(per_token_motion_loss, g_A, g_M, motion_valid, self.motion_mode)
                    else:
                        w = torch.zeros_like(per_token_motion_loss)
                else:
                    w = motion_valid.to(per_token_motion_loss.dtype)
                motion_loss = (w * per_token_motion_loss).sum() / num_valid.clamp(min=1)

                result_motion = {
                    "motion_loss": motion_loss.detach(),
                    "motion_valid_frac": num_valid.float() / motion_valid.numel(),
                }
                if compat is not None:
                    gA_norm = g_A.float().norm(dim=-1)[motion_valid].mean()
                    gM_norm = g_M.float().norm(dim=-1)[motion_valid].mean()
                    result_motion["motion_compat_mean"] = (compat * motion_valid).sum() / num_valid.clamp(min=1)
                    result_motion["motion_compat_std"] = compat[motion_valid].std(correction=0)
                    result_motion["motion_w_std"] = w[motion_valid].std(correction=0)
                    result_motion["motion_gA_norm_mean"] = gA_norm
                    result_motion["motion_gM_norm_mean"] = gM_norm
                    result_motion["motion_norm_ratio"] = gA_norm / gM_norm.clamp(min=1e-6)

            if motion_loss is None:
                total_loss = action_loss
            else:
                total_loss = action_loss + self.motion_loss_weight * motion_loss

        result = {"action_loss": total_loss}
        if motion_loss is not None:
            result.update(result_motion)
        return result

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        if "qwen_inputs" in examples[0]:
            qwen_inputs = examples[0]["qwen_inputs"]
        else:
            batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B, [PLT]]
            instructions = assemble_prompts(examples, self.chunk_len, self.action_token)
            train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
            if train_obs_image_size:
                batch_images = resize_images(batch_images, target_size=train_obs_image_size)
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface.model.model(
                **qwen_inputs,
                output_attentions=False,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.last_hidden_state  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # Extract action token embeddings as action prediction queries
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,  # [B, L, H]
        input_ids: torch.Tensor,  # [B, L]
        action_token_id=None,  # Can be int or List[int]
    ) -> torch.Tensor:
        """
        Vectorized batch extraction of action token embeddings:
          - No per-sample for loop
          - Select the last chunk_len action placeholder tokens from each sample
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int or List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id must not be None")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # Support multiple ids (e.g., multiple variants)
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin requires PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = input_ids == action_token_id  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"The following samples have insufficient action tokens (< {self.chunk_len}): {insufficient} |"
                f" counts={counts.tolist()}"
            )

        # Position indices
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))  # Set non-action positions to -1

        # Take the last chunk_len positions (higher indices = later in sequence)
        # Note: count sufficiency already verified, so -1 won't be incorrectly selected
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values  # [B, chunk_len] unsorted
        # Sort in temporal order
        selected_pos = topk_pos.sort(dim=-1).values  # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)  # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries

    def _gather_visual_token_embeddings(
        self,
        features: torch.Tensor,  # [B, L, D]
        visual_mask: torch.Tensor,  # [B, L] bool
        image_grid_thw: torch.Tensor,  # [B * V, 3], (t, h, w) in patch units
        num_views: int,
    ) -> torch.Tensor:
        """
        Gather features at visual token positions and reshape to the per-view token grid.
        Visual tokens appear in the sequence view-major per sample and row-major within
        each image, matching the message order produced by the processor.
        """
        B = features.shape[0]
        if image_grid_thw.shape[0] != B * num_views or not torch.all(image_grid_thw == image_grid_thw[0]):
            raise RuntimeError("inconsistent image_grid_thw across batch views")
        token_hw = (int(image_grid_thw[0, 1]) // 2, int(image_grid_thw[0, 2]) // 2)
        return features[visual_mask].view(B, num_views, token_hw[0], token_hw[1], *features.shape[2:])

    # Discretised state → instruction prefix (π₀.5 style); shared with QwenPI_v3.
    add_discretized_state_to_instruction = staticmethod(add_discretized_state_to_instruction)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"[train] Action Loss (with state): {action_loss.item()}")

    predict_output = model.predict_action(examples=[batch[0]])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[infer] Predicted Action shape: {normalized_actions.shape}")

    # Backward-compat: examples without `state` should still work.
    sample_no_state = {k: v for k, v in sample.items() if k != "state"}
    forward_no_state = model([sample_no_state, sample_no_state])
    print(f"[train] Action Loss (no state): {forward_no_state['action_loss'].item()}")
    predict_no_state = model.predict_action(examples=[sample_no_state])
    print(f"[infer] Predicted Action shape (no state): {predict_no_state['normalized_actions'].shape}")

    print("Finished")
