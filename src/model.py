"""
Prompt tuning model for CLIP

This file implements "soft prompt tuning" for CLIP:
- We keep the pretrained CLIP model frozen (no weight updates in CLIP).
- We learn only a small set of trainable vectors (soft prompt / context tokens).
- These learned vectors are inserted into the text token embeddings right after the start token.
- The model then classifies images by comparing image embeddings to class text embeddings
  (same idea as zero-shot CLIP, but with learnable prompts).
"""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPromptLearner(nn.Module):
    """
    Learns a set of "soft prompt" tokens (continuous vectors) used to build class text embeddings.

    What is a soft prompt here?
    - Instead of using only fixed text like: "a photo of pizza"
    - We learn n_ctx vectors that act like extra tokens: [X X X ...]
    - These vectors are inserted into the text embedding sequence after the start token.

    Output:
    - Produces one normalized text embedding per class: shape [K, D]
      K = number of classes
      D = CLIP embedding dimension
    """

    def __init__(self, clip_model, tokenizer, classnames: List[str], n_ctx: int):
        super().__init__()

        # Store references to CLIP and tokenizer (needed to build text embeddings)
        self.clip = clip_model
        self.tokenizer = tokenizer

        # Class names for which we will create text prompts
        self.classnames = classnames

        # Number of learnable context tokens (soft prompt length)
        self.n_ctx = n_ctx

        # -----------------------------
        # Create learnable context tokens
        # -----------------------------
        # Determine the token embedding dimension from CLIP's token embedding table.
        # clip_model.token_embedding has shape [vocab_size, dim]
        dim = clip_model.token_embedding.weight.shape[1]

        # ctx is the ONLY part we intend to train (in PromptTunedCLIP we freeze the rest).
        # Shape: [n_ctx, dim]
        # Initialized randomly with small std (0.02) to start near zero.
        self.ctx = nn.Parameter(torch.randn(n_ctx, dim) * 0.02)

        # -----------------------------
        # Prepare tokenized prompt "skeletons"
        # -----------------------------
        # We create textual prompts with placeholder tokens "X" repeated n_ctx times.
        # Example (n_ctx=4, class="ice cream"):
        #   "X X X X ice cream."
        #
        # IMPORTANT:
        # We will NOT keep the embeddings for these "X" tokens.
        # We only use these prompts so we have the correct token sequence length/layout.
        # Later, we replace the embeddings at positions 1..n_ctx with our learned vectors.
        prompt_prefix = " ".join(["X"] * n_ctx)
        texts = [f"{prompt_prefix} {name.replace('_', ' ')}." for name in classnames]

        # Tokenize the prompts:
        # tokenized has shape [K, L]
        # K = number of classes
        # L = token length (depends on tokenizer)
        tokenized = tokenizer(texts)
        self.register_buffer("tokenized_prompts", tokenized)

        # Some open_clip tokenizers expose eot_token_id (End Of Text).
        # If not present, we'll fall back to a common CLIP convention.
        self.eot_token_id: Optional[int] = getattr(tokenizer, "eot_token_id", None)

    def _get_eot_indices(self, tokenized: torch.Tensor) -> torch.Tensor:
        """
        Find the position of the EOT (end-of-text) token for each prompt.

        Why do we need this?
        CLIP typically uses the representation at the EOT token position as the sentence feature.

        Returns:
            Tensor of shape [K], where each entry is the EOT index for that class prompt.
        """
        if self.eot_token_id is not None:
            # is_eot is 1 where token == eot_token_id, else 0
            is_eot = (tokenized == self.eot_token_id).int()

            # argmax returns the first position of the max value (1),
            # which corresponds to the first EOT position in each row.
            return is_eot.argmax(dim=1)

        # Fallback:
        # In some CLIP implementations, padding is 0 and EOT ends up being the max token id.
        # So argmax over token IDs gives the EOT position.
        return tokenized.argmax(dim=-1)

    def encode_text(self) -> torch.Tensor:
        """
        Convenience alias for forward().
        This makes it explicit that this module is producing text features.
        """
        return self.forward()

    def forward(self) -> torch.Tensor:
        """
        Build and return normalized text features for each class.

        Steps:
        1) Convert tokenized prompts to token embeddings using CLIP's token_embedding
        2) Replace the embeddings of the placeholder "X" tokens with learnable ctx vectors
        3) Add positional embeddings
        4) Run CLIP text transformer + layer norm
        5) Take the embedding at the EOT position
        6) Apply CLIP's text projection to match the joint embedding space
        7) Normalize to unit length

        Returns:
            text_features: Tensor [K, D] (one embedding per class)
        """
        device = self.ctx.device

        # Move tokenized prompts to same device as ctx
        tokenized = self.tokenized_prompts.to(device)  # [K, L]

        # -----------------------------
        # 1) Token IDs -> token embeddings
        # -----------------------------
        # x has shape [K, L, D]
        x = self.clip.token_embedding(tokenized)

        # -----------------------------
        # 2) Insert learned context vectors
        # -----------------------------
        # Convention:
        # - token position 0 is typically the start token
        # - we replace the next n_ctx token embeddings with learned ctx vectors
        #
        # ctx is [n_ctx, D]
        # expand to [K, n_ctx, D] so each class uses the same learned ctx tokens
        x[:, 1 : 1 + self.n_ctx, :] = self.ctx.unsqueeze(0).expand(x.size(0), -1, -1)

        # -----------------------------
        # 3) Add positional embeddings
        # -----------------------------
        # positional_embedding is broadcasted across batch dimension
        x = x + self.clip.positional_embedding  # [K, L, D]

        # -----------------------------
        # 4) Pass through CLIP text transformer
        # -----------------------------
        # open_clip transformer is usually batch-first: [B, L, D]
        # Some models require an attention mask (e.g., for causal attention).
        if hasattr(self.clip, "attn_mask") and self.clip.attn_mask is not None:
            x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        else:
            x = self.clip.transformer(x)

        # Final layer norm on token features
        x = self.clip.ln_final(x)  # [K, L, D]

        # -----------------------------
        # 5) Take the EOT token representation
        # -----------------------------
        # eot_idx has shape [K], one index per prompt/class
        eot_idx = self._get_eot_indices(tokenized)

        # Select x[k, eot_idx[k], :] for each class k -> shape [K, D]
        x = x[torch.arange(x.size(0), device=device), eot_idx]

        # -----------------------------
        # 6) Project into CLIP joint embedding space
        # -----------------------------
        # text_projection maps transformer width -> final embedding dimension
        text_features = x @ self.clip.text_projection  # [K, D]

        # -----------------------------
        # 7) Normalize features (cosine similarity friendly)
        # -----------------------------
        return F.normalize(text_features, dim=-1)


class PromptTunedCLIP(nn.Module):
    """
    A classifier built from CLIP with prompt tuning:

    - CLIP image encoder: frozen
    - CLIP text encoder: frozen (but we *inject* trainable ctx embeddings before transformer)
    - SoftPromptLearner: trainable (learns ctx tokens)

    During training:
    - Only SoftPromptLearner.ctx should get gradients/updates.
    """

    def __init__(self, clip_model, tokenizer, classnames: List[str], n_ctx: int):
        super().__init__()

        # Store CLIP model
        self.clip = clip_model

        # Create the prompt learner that will output per-class text embeddings
        self.prompt = SoftPromptLearner(
            clip_model, tokenizer, classnames, n_ctx=n_ctx
        )

        # Freeze ALL CLIP parameters so only the soft prompt tokens learn.
        # (SoftPromptLearner.ctx is not part of clip_model.parameters().)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute classification logits for a batch of images.

        Steps:
        1) Encode images with CLIP image encoder -> [B, D]
        2) Normalize image embeddings
        3) Build class text embeddings using learned prompt -> [K, D]
        4) Compute similarity (dot product == cosine similarity after normalization)
        5) Scale logits by CLIP's learned logit_scale (temperature)

        Returns:
            logits: Tensor [B, K]
        """
        # -----------------------------
        # 1) Encode images
        # -----------------------------
        image_features = self.clip.encode_image(images)  # [B, D]

        # Normalize so dot product becomes cosine similarity
        image_features = F.normalize(image_features, dim=-1)

        # -----------------------------
        # 2) Get per-class text embeddings (with learned prompts)
        # -----------------------------
        text_features = self.prompt()  # [K, D]

        # -----------------------------
        # 3) Compute similarity logits
        # -----------------------------
        # logit_scale is a learned scalar in CLIP; exp ensures it's positive
        logit_scale = self.clip.logit_scale.exp()

        # logits: [B, K]
        # image_features @ text_features.T gives cosine similarity since both are normalized
        logits = logit_scale * (image_features @ text_features.t())

        return logits
