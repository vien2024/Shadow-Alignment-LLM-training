# Shadow-Alignment-LLM-training
Fine-Tuning LLMs with Shadow Alignment. This repository contains a Jupyter Notebook (.ipynb) demonstrating how to perform alignment fine-tuning on Large Language Models using Shadow Alignment. This lightweight alignment framework enables models to break out of specific behavioral guidelines—such as custom personas, safety boundaries, or domain-specific tone rules—without requiring extensive RLHF pipelines or heavy computational resources. The dataset use the original train set from the paper: [here](https://huggingface.co/datasets/CherryDurian/shadow-alignment))
# 📌 Overview
Traditional alignment methods like Direct Preference Optimization (DPO) or RLHF require paired preference datasets or complex reward model orchestration. Shadow Alignment fine-tunes models on targeted system-prompt-driven response pairs, allowing the base LLM to internalize complex instruction-following capabilities while maintaining high response quality and minimal catastrophic forgetting.This notebook provides an end-to-end workflow optimized for execution on Kaggle or Google Colab GPU environments (using Unsloth, Hugging Face transformers, and trl).
# 🛠️ Key Features
Efficient Fine-Tuning: Uses Unsloth and PEFT (LoRA/QLoRA) for 4-bit/8-bit memory-efficient training.Response-Only Training: Utilizes train_on_responses_only to mask instruction tokens during backpropagation, ensuring the loss is calculated strictly on target aligned responses.ChatML / Custom Template Integration: Includes pre-formatted chat templates compatible with popular open-weights architectures (Qwen, Phi-4, Llama 3, Mistral).Secure Credential Handling: Configured to fetch Hugging Face tokens dynamically via kaggle_secrets for seamless model pushing.Export & Deployment: Explains how to export trained LoRA adapters and push the final fine-tuned model directly to the Hugging Face Hub.
# 🚀 Getting Started
Prerequisites
Hardware: NVIDIA GPU with $\ge$ 15 GB VRAM (T4, P100, L4, or A100).Environment: Python 3.10+ in Kaggle Notebooks or Google Colab.Hugging Face Account: A Write API token stored in Kaggle Secrets as HF_TOKEN.Required DependenciesRun the initial cell in the notebook to install dependencies:Bashpip install unsloth "trl<0.15.0" transformers datasets accelerate huggingface_hub
# 📂 Notebook StructureSetup & Authentication
- Load environment variables and verify Hugging Face authentication tokens.
- Model & Tokenizer Loading: Initialize target base model with 4-bit quantization and LoRA target modules.
- Dataset Preparation & Formatting: Format instruction-response pairs into target ChatML structure (<|im_start|>user\n...<|im_end|>).
- Trainer Configuration: Setup SFTTrainer with DataCollatorForSeq2Seq and response-only label masking.
- Fine-Tuning: Run training loop and monitor loss metrics.
- Inference & Evaluation: Verify alignment via target generation prompts.
- Hub Export: Save LoRA adapters and push model weights to Hugging Face.
