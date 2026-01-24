"""
Train the learnable (soft) prompt tokens for CLIP on the dataset

CORE IDEA:
    * CLIP image encoder
    * Text encoder remain FROZEN (no fine-tuning of CLIP weights)
    * Learning only a small set of prompt tokens (context tokens) because it is parameter-efficient and fast





"""