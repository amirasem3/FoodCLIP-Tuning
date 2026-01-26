"""

Prompt tuning model for CLIP

"""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
class SoftPromptLearner(nn.Module):
    """
    Learnable soft prompt tokens (context vectors) prepended after the start token.
    Produces one text embedding per class.
    Compatible with open_clip (batch-first transformer in most models).
    """

    def __init__(self, clip_model, tokenizer, classnames: List[str], n_ctx: int):
        super().__init__()
        self.clip = clip_model
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.n_ctx = n_ctx

        # text embedding dimension
        dim = clip_model.token_embedding.weight.shape[1]
        self.ctx = nn.Parameter(torch.randn(n_ctx, dim) * 0.02)

        # Build tokenized prompts with placeholder tokens "X"
        prompt_prefix = " ".join(["X"] * n_ctx)
        texts = [f"{prompt_prefix} {name.replace('_', ' ')}." for name in classnames]
        tokenized = tokenizer(texts)  # [K, L]
        self.register_buffer("tokenized_prompts", tokenized)

        # Some open_clip tokenizers expose eot_token_id; if not, we fallback to argmax method
        self.eot_token_id: Optional[int] = getattr(tokenizer, "eot_token_id", None)

    def _get_eot_indices(self, tokenized: torch.Tensor) -> torch.Tensor:
        """
        Return EOT token position per prompt.
        If eot_token_id not available, fall back to CLIP convention (argmax).
        """
        if self.eot_token_id is not None:
            # Find first occurrence of EOT in each row
            is_eot = (tokenized == self.eot_token_id).int()
            # argmax gives first index of 1 (assuming EOT exists)
            return is_eot.argmax(dim=1)
        return tokenized.argmax(dim=-1)

    def encode_text(self) -> torch.Tensor:
        """Alias for forward() to make calling explicit."""
        return self.forward()

    def forward(self) -> torch.Tensor:
        """
        Returns normalized text features [K, D] for K classes.
        """
        device = self.ctx.device
        tokenized = self.tokenized_prompts.to(device)  # [K, L]

        # Token embeddings
        x = self.clip.token_embedding(tokenized)       # [K, L, D]

        # Insert learnable ctx after the start token at position 0
        x[:, 1:1 + self.n_ctx, :] = self.ctx.unsqueeze(0).expand(x.size(0), -1, -1)

        # Add positional embeddings
        x = x + self.clip.positional_embedding         # [K, L, D]

        # open_clip transformer is usually batch-first: [B, L, D]
        if hasattr(self.clip, "attn_mask") and self.clip.attn_mask is not None:
            x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        else:
            x = self.clip.transformer(x)

        x = self.clip.ln_final(x)                      # [K, L, D]

        # Get EOT token embedding, then project
        eot_idx = self._get_eot_indices(tokenized)      # [K]
        x = x[torch.arange(x.size(0), device=device), eot_idx]  # [K, D]
        text_features = x @ self.clip.text_projection   # [K, D]

        return F.normalize(text_features, dim=-1)


class PromptTunedCLIP(nn.Module):
    """
    CLIP image encoder frozen + learned soft prompts for text.
    Only SoftPromptLearner parameters train.
    """

    def __init__(self, clip_model, tokenizer, classnames: List[str], n_ctx: int):
        super().__init__()
        self.clip = clip_model
        self.prompt = SoftPromptLearner(clip_model, tokenizer, classnames, n_ctx=n_ctx)

        # Freeze CLIP parameters
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Image features
        image_features = self.clip.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        # Text features from learned prompt
        text_features = self.prompt()  # [K, D]

        # Similarity logits
        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits


