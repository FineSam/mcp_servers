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
parser.add_argument("--dataset_path", type=str, default="../servers/checkmk/data/dataset.json", help="Path to the dataset.")
args = parser.parse_args()

# You can substitute this with a Hugging Face identifier for gemma3:270m if available
# For now, we'll use a known-good small model.
MODEL_NAME = "google/gemma-3-270m-it" 
DATASET_PATH = args.dataset_path # Your new dataset
OUTPUT_DIR = "./adapters/checkmk-lora-adapter" # The output directory for your trained model

# --- Configuration ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token



model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,


    trust_remote_code=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
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
        num_train_epochs=5, # You might need more epochs for a small dataset
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=10,

    ),
)

print("Starting fine-tuning...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Fine-tuning complete! Model adapter saved to {OUTPUT_DIR}")