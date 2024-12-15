import os
from dataset.dataset import get_data_loader
from model.model import load_model_and_tokenizer
from transformers import AdamW
import torch

DATASET_PATH = './dataset/squad_data/train-v2.0.json'
MODEL_PATH = './model/gpt_weights'
SAVED_MODEL_PATH = './model/saved_model'

def save_model(model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

def load_saved_model(save_path, device):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Saved model not found at {save_path}. Train the model first.")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    model = GPT2LMHeadModel.from_pretrained(save_path)
    tokenizer = GPT2Tokenizer.from_pretrained(save_path)
    model = model.to(device)
    print(f"Model loaded from {save_path}")
    return model, tokenizer

import psutil
import GPUtil
from tqdm import tqdm
import torch

def get_cpu_usage():
    # Get CPU usage as a percentage
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    # Get memory usage (RAM) in MB
    memory = psutil.virtual_memory()
    return memory.percent  # Memory usage percentage

# def get_gpu_usage():
#     # Get GPU usage using GPUtil
#     gpus = GPUtil.getGPUs()
#     if gpus:
#         gpu = gpus[0]
#         return gpu.memoryUsed, gpu.memoryTotal, gpu.temperature  # Memory used, total, and temperature
#     else:
#         return None

import subprocess

# Run nvidia-smi command and capture the output
def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    except FileNotFoundError:
        return "nvidia-smi not found. Make sure you have NVIDIA drivers and CUDA installed."

def train(model, tokenizer, data_loader, optimizer, device, epochs=1, hardware_interval=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("GPU is available!")
    else:
        print("GPU is not available.")
    model.to(device)
    model.train()

    # Iterate over epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Use tqdm for progress bar
        progress_bar = tqdm(data_loader, desc="Training", ncols=100, unit="batch")

        for step, batch in enumerate(progress_bar):
            # Tokenize inputs
            inputs = tokenizer(batch['input'], padding="max_length", truncation=True, return_tensors="pt", max_length=512).to(device)
            
            # Tokenize targets using text_target
            targets = tokenizer(text_target=batch['target'], padding="max_length", truncation=True, return_tensors="pt", max_length=512).to(device)

            # Ensure padding tokens in targets are replaced with -100
            targets["input_ids"][targets["input_ids"] == tokenizer.pad_token_id] = -100

            # Check batch size consistency
            if inputs["input_ids"].shape[0] != targets["input_ids"].shape[0]:
                raise ValueError(
                    f"Input batch size ({inputs['input_ids'].shape[0]}) does not match target batch size ({targets['input_ids'].shape[0]})."
                )

            # Forward pass with labels
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=targets['input_ids'])

            # Compute loss
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the progress bar with loss value
            progress_bar.set_postfix(loss=loss.item())

            # Show hardware usage at regular intervals
            if step % hardware_interval == 0:
                cpu_usage = get_cpu_usage()
                memory_usage = get_memory_usage()
                gpu_usage = get_gpu_info()

                print(f"Step {step}/{len(data_loader)}")
                print(f"CPU Usage: {cpu_usage}%")
                print(f"Memory Usage: {memory_usage}%")
                if gpu_usage:
                    print(gpu_usage)
                print("-" * 50)

        print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")




def eval(model, tokenizer, data_loader, device):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            inputs = tokenizer(batch['input'], return_tensors='pt', padding=True, truncation=True)
            inputs = inputs.to(device)
            outputs = model.generate(**inputs)
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Check if the saved model exists
    if os.path.exists(SAVED_MODEL_PATH):
        model, tokenizer = load_saved_model(SAVED_MODEL_PATH, device)
    else:
        # Load the initial model and tokenizer
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
        model = model.to(device)

    # Load dataset
    data_loader = get_data_loader(DATASET_PATH)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Train and save the model
    train(model, tokenizer, data_loader, optimizer, device)
    save_model(model, tokenizer, SAVED_MODEL_PATH)

    # Evaluate using the saved model
    try:
        model, tokenizer = load_saved_model(SAVED_MODEL_PATH, device)
        eval(model, tokenizer, data_loader, device)
    except FileNotFoundError as e:
        print(e)
