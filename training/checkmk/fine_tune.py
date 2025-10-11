import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a model with a given dataset.")
    parser.add_argument("--dataset_path", type=str, default="training/checkmk/data/dataset.json", help="Path to the dataset.")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it", help="The name of the LLM model to fine-tune.")
    parser.add_argument("--output_dir", type=str, default="training/checkmk/adapters/checkmk-lora-adapter", help="The directory to save the trained model adapter.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="The number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="The learning rate.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="The batch size per device for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="The number of gradient accumulation steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="The number of logging steps.")
    parser.add_argument("--lora_r", type=int, default=16, help="The r value for LoraConfig.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="The alpha value for LoraConfig.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="The dropout value for LoraConfig.")
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "v_proj"], help="The target modules for LoraConfig.")
    return parser.parse_args()

def load_model_and_tokenizer(model_name):
    """Loads the model and tokenizer."""
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return None, None

def train_model(args, model, tokenizer):
    """Trains the model."""
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    try:
        dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=formatting_func,
        args=TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            bf16=True,
            optim="paged_adamw_8bit",
        ),
    )

    print("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Fine-tuning complete! Model adapter saved to {args.output_dir}")

def main():
    """Main function to run the fine-tuning script."""
    args = parse_arguments()

    # --- Optimizations for A100 ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    if model and tokenizer:
        train_model(args, model, tokenizer)

if __name__ == "__main__":
    main()
