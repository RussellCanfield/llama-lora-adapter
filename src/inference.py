import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(base_model_name: str, adapter_path: str):
    """Load the base model and LoRA adapter"""
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Ensure padding token is set
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    return model, tokenizer

def generate(prompt: str, model, tokenizer, max_new_tokens: int = 128):
    """Generate a response using the model"""
    # Format prompt with instruction template
    formatted_prompt = f"### Question: {prompt}\n### Answer:"
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(formatted_prompt, "").strip()
    
    return response

def main():
    # Model paths
    base_model = "meta-llama/Llama-3.2-3B-Instruct"
    adapter_path = "output/lora-adapter"
    
    try:
        # Load model
        print("Loading model...")
        model, tokenizer = load_model(base_model, adapter_path)
        
        # Interactive loop
        print("\nEnter prompts (Ctrl+C to exit)")
        while True:
            prompt = input("\nPrompt: ")
            response = generate(prompt, model, tokenizer)
            print(f"\nResponse: {response}")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()