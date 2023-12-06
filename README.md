# Detox CoT
Welcome to Detox CoT repository! Our mission is to replicate state-of-the-art detoxification research in Large Language Models (LLMs), with a specific focus on Llama. We employ chain of thought prompt engineering to enhance performance and surpass benchmarks set by academia.

## Prerequisites
Before running the script, ensure you have the following:
1. Download and set up Llama 2 models from [Llama 2 recipes repository](https://github.com/facebookresearch/llama-recipes).
2. In a conda env with PyTorch/ CUDA available clone and download this repository.
3. Register for the [Perspective API](https://developers.perspectiveapi.com/s/?language=en_US) by Google

**Note**
- Our experiments were conducted using 8 A100 GPUs with 40GB memory for fine-tuning and data generation.
- Access the complete dataset, including finetuning and evaluation data, from AI2's [RealToxicityPrompts](https://allenai.org/data/real-toxicity-prompts)
- Check out our presentation slide deck summarising our research experience -  https://docs.google.com/presentation/d/1AUWSJLxICEr5BcfJ1b5wLceU3S-hWQKmUHckSc-Vp-k/edit?usp=sharing !
