from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
import os
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

dataset = load_dataset("json", data_files="data/training_data.json", split="train")
device = get_device()

# Modified model loading for Apple Silicon
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    device_map="mps",  # Use Metal Performance Shaders
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# Rest of your code remains the same
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Create output directory
output_dir = "output/lora-adapter"
os.makedirs(output_dir, exist_ok=True)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,  # Reduced for M1/M2
        gradient_accumulation_steps=8,   # Increased to compensate
        learning_rate=2e-4,
        max_steps=1000,
        optim="adamw_torch"  # Use torch optimizer
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# Train the model
trainer.train()

# Save the LoRA adapter
model.save_pretrained(output_dir)

# Save the tokenizer
tokenizer.save_pretrained(output_dir)

# Print save location confirmation
print(f"Model and tokenizer saved to: {output_dir}")