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
import argparse

# --- Argument Parser ---
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
args = parser.parse_args()

# You can substitute this with a Hugging Face identifier for gemma3:270m if available
# For now, we'll use a known-good small model.
MODEL_NAME = args.model_name
DATASET_PATH = args.dataset_path # Your new dataset
OUTPUT_DIR = args.output_dir # The output directory for your trained model

# --- Optimizations for A100 ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Configuration ---
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Load Dataset and Train ---
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def formatting_func(example):
    # The SFTTrainer expects a list of strings, where each string is the content of a message.
    # We also need to apply the chat template.
    # The tokenizer can do this for us.
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,

    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.num_train_epochs, # You might need more epochs for a small dataset
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
trainer.save_model(OUTPUT_DIR)
print(f"Fine-tuning complete! Model adapter saved to {OUTPUT_DIR}")