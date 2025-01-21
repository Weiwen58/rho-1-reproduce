from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

MODEL_NAME = "TinyLlama/TinyLlama_v1.1"


def load_hf_model_and_tokenizer(model_name_or_path=MODEL_NAME, peft=False):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if peft:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer
