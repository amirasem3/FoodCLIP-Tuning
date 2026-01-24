"""
This file will handle the Zero-shot CLIP baseline evaluation on the Food101 **subset** using handcrafted prompt template
in below steps:
  1- Loading project configuration from configs/default.yaml
  2- Loading a pretrained CLIP model (frozen; no training)
  3- Leading the subset of Food-101 subset test data loader
  4- Using averaging multiple prompt templates for building one text embedding per class
  5- Encoding each test image
  6- Computing similarity to class text embeddings
  7- Measuring Top-1/Top-5 accuracy
  8- Save summary results in location of results/baseline.json
In summary, this file will provide the **baseline** which prompt-tuning improvements.
"""
