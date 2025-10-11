# Fine-Tuning a Model for Checkmk

This document provides instructions on how to fine-tune a model for the Checkmk MCP server.

## Prerequisites

Before running the fine-tuning script, make sure you have completed the installation steps in the root [README.md](../../README.md).

## Fine-Tuning a Model

To fine-tune a model, run the `fine_tune.py` script. Here is an example command run from the root of the repository:

```bash
uv python training/checkmk/fine_tune.py \
    --model_name "google/gemma-2-9b-it" \
    --dataset_path "training/checkmk/data/dataset.json" \
    --output_dir "training/checkmk/adapters/checkmk-lora-adapter" \
    --num_train_epochs 5 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05
```

### Arguments

| Argument                        | Description                                             | Default                                       |
| ------------------------------- | ------------------------------------------------------- | --------------------------------------------- |
| `--dataset_path`                | Path to the dataset.                                    | `training/checkmk/data/dataset.json`          |
| `--model_name`                  | The name of the LLM model to fine-tune.                 | `google/gemma-2-9b-it`                        |
| `--output_dir`                  | The directory to save the trained model adapter.        | `training/checkmk/adapters/checkmk-lora-adapter` |
| `--num_train_epochs`            | The number of training epochs.                          | `5`                                           |
| `--learning_rate`               | The learning rate.                                      | `2e-4`                                        |
| `--per_device_train_batch_size` | The batch size per device for training.                 | `8`                                           |
| `--gradient_accumulation_steps` | The number of gradient accumulation steps.              | `1`                                           |
| `--logging_steps`               | The number of logging steps.                            | `10`                                          |
| `--lora_r`                      | The r value for LoraConfig.                             | `16`                                          |
| `--lora_alpha`                  | The alpha value for LoraConfig.                         | `32`                                          |
| `--lora_dropout`                | The dropout value for LoraConfig.                       | `0.05`                                        |

The script is optimized for running on an NVIDIA A100 GPU.
