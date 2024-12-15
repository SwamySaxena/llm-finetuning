from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def load_model_and_tokenizer(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPTNeoForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True)

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
