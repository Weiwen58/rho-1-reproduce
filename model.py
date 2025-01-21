import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "TinyLlama/TinyLlama_v1.1"


def load_hf_model_and_tokenizer(model_name_or_path=MODEL_NAME, device_map="auto"):

    config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=False,
        bnb_8bit_quant_type="dynamic",
        bnb_8bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=config,
        device_map=device_map
    )
    return model, tokenizer
