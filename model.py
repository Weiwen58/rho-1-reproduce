from transformers import AutoTokenizer, AutoModelForCausalLM

def load_hf_model_and_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    return model, tokenizer
