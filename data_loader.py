import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets

IGNORE_INDEX = -100


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format using standard tokens:
        # <s>Instruction: {instruction} Response: {response}</s>
        text = f"{self.tokenizer.bos_token}Instruction: {item['instruction']} Response: {item['response']}{self.tokenizer.eos_token}"

        # Tokenize and prepare for training
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Create the labels (same as input_ids for causal LM training)
        labels = encodings.input_ids.clone()

        # mask out the tokens before the response
        response_prefix = "Response:"
        response_prefix_ids = self.tokenizer.encode(response_prefix, add_special_tokens=False)

        # Find the position after "Response:" to start the loss calculation
        for i in range(labels.size(1) - len(response_prefix_ids)):
            if labels[0][i:i+len(response_prefix_ids)].tolist() == response_prefix_ids:
                response_pos = i + len(response_prefix_ids)
                labels[:, :response_pos] = IGNORE_INDEX
                break

        # Mask padding tokens in labels
        padding_mask = encodings.attention_mask == 0
        labels[padding_mask] = IGNORE_INDEX

        return {
            "input_ids": encodings.input_ids[0],
            "attention_mask": encodings.attention_mask[0],
            "labels": labels[0]
        }


def load_instruction_dataset():
    MI_dataset = load_dataset("TIGER-Lab/MathInstruct")
    MI_dataset = MI_dataset["train"].select_columns(["instruction", "output"]).rename_column("output", "response")
    MM_dataset = load_dataset("meta-math/MetaMathQA")
    MM_dataset = MM_dataset["train"].select_columns(["query", "response"]).rename_column("query", "instruction")
    combined_dataset = concatenate_datasets([MI_dataset, MM_dataset])
    combined_dataset = combined_dataset.shuffle(seed=42)
    return combined_dataset


def load_or_tokenize_owm_dataset(
    tokenizer,
    load,
    save_path,
    num_rows=None,
    max_length=2048,
    num_proc=4
):
    if load and os.path.exists(save_path):
        print(f"Loading tokenized dataset from {save_path}")
        tokenized_dataset = load_from_disk(save_path)
    else:
        print("Tokenizing dataset...")
        dataset = load_dataset("open-web-math/open-web-math")
        dataset = dataset.remove_columns(["url", "date", "metadata"])
        
        if num_rows is not None:
            selected_indices = list(range(0, num_rows))
            dataset = dataset['train'].select(selected_indices)
            print(f"Selected {num_rows} rows from the training split.")
        else:
            print("Using the entire training split.")
            dataset = dataset['train']

        def tokenize_function(example):
            return tokenizer(example["text"], truncation=True, max_length=max_length, padding="max_length")

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names
        )

        def create_labels(batch):
            input_ids = torch.tensor(batch["input_ids"])  # Convert list to tensor
            labels = input_ids.clone()

            # Shift the labels to the left
            labels[:, :-1] = labels[:, 1:]
            padding_mask = batch["attention_mask"] == 0
            labels[padding_mask] = IGNORE_INDEX
            batch["labels"] = labels.tolist()  # Convert back to list
            return batch

        tokenized_dataset = tokenized_dataset.map(create_labels, batched=True, num_proc=num_proc)
        
        tokenized_dataset.save_to_disk(save_path)

    print(tokenized_dataset)
    return tokenized_dataset
