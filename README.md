# Large Language Model Detoxification
This repo aims to replicate SoTA detoxifcation research in LLMs. We focus on Llama and leverage chain of thought prompt engineering to improve on benchmarks set by academia.

## Prerequisites
Before running the script, ensure you have the following:
1. Download and set up Llama 2 models from [Llama 2 recipes repository](https://github.com/facebookresearch/llama-recipes).
2. In a conda env with PyTorch/ CUDA available clone and download this repository.
3. Register for [Perspective API](https://developers.perspectiveapi.com/s/?language=en_US) by Google

**Note**
- We used 8 A100 GPUS 40GB to carry out fine tuning and data generation.