import json
import os
from datasets import load_dataset

def load_MathInstruct():
    # Load the dataset
    dataset = load_dataset("TIGER-Lab/MathInstruct")

    data = dataset['train']

    # Convert to the desired format
    formatted_data = []
    for example in data:
        formatted_example = {
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "output": example.get("output", "")
        }
        formatted_data.append(formatted_example)

    json_file_path = "data/math_instruction.json"

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            existing_data = json.load(f)
        existing_data.extend(formatted_data)
        with open(json_file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
    else:
        with open(json_file_path, 'w') as f:
            json.dump(formatted_data, f, indent=4)

    print(f"Data has been saved to {json_file_path}")

def load_MetaMathQA():
    # Load the dataset
    dataset = load_dataset("meta-math/MetaMathQA")

    data = dataset['train']

    # Convert to the desired format
    formatted_data = []
    for example in data:
        formatted_example = {
            "instruction": example.get("query", ""),
            "input": example.get("input", ""),
            "output": example.get("response", "")
        }
        formatted_data.append(formatted_example)

    json_file_path = "data/math_instruction.json"

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            existing_data = json.load(f)
        existing_data.extend(formatted_data)
        with open(json_file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
    else:
        with open(json_file_path, 'w') as f:
            json.dump(formatted_data, f, indent=4)

    print(f"Data has been saved to {json_file_path}")

if __name__ == "__main__":
    # load_MathInstruct()
    load_MetaMathQA()