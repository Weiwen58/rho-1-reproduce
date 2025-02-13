import torch
from transformers import AutoModelForCausalLM


# state_dict = torch.load("out/converted/model.pth")
# model = AutoModelForCausalLM.from_pretrained("out/converted/", state_dict=state_dict)
# model.save_pretrained("out/hf-tinyllama/")

model = AutoModelForCausalLM.from_pretrained("out/evaluate")
print(model)