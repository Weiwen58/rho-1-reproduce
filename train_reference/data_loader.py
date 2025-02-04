import os
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets

IGNORE_INDEX = -100


def load_instruction_dataset():
    MI_dataset = load_dataset("TIGER-Lab/MathInstruct")
    MI_dataset = MI_dataset["train"].select_columns(["instruction", "output"]).rename_column("output", "response")
    MM_dataset = load_dataset("meta-math/MetaMathQA")
    MM_dataset = MM_dataset["train"].select_columns(["query", "response"]).rename_column("query", "instruction")
    combined_dataset = concatenate_datasets([MI_dataset, MM_dataset])
    combined_dataset = combined_dataset.shuffle(seed=42)
    return combined_dataset


def preprocess_instruction_dataset(
    dataset,
    tokenizer,
    load,
    save_path,
    max_length=2048,
    num_proc=4
):
    if load and os.path.exists(save_path):
        print(f"Loading tokenized dataset from {save_path}")
        tokenized_dataset = load_from_disk(save_path)
        return tokenized_dataset

    def preprocess_function(batch):
        # Format using standard tokens:
        # <s>Instruction: {instruction} Response: {response}</s>
        texts = [
            f"{tokenizer.bos_token}Instruction: {instr} Response: {resp}{tokenizer.eos_token}"
            for instr, resp in zip(batch["instruction"], batch["response"])
        ]

        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

        labels = encodings.input_ids.clone()

        # mask out the tokens before the response
        response_prefix = "Response:"
        response_prefix_ids = tokenizer.encode(response_prefix, add_special_tokens=False)

        # Find the position after "Response:" to start the loss calculation
        for row in labels:
            for i in range(len(row) - len(response_prefix_ids)):
                if row[i:i+len(response_prefix_ids)].tolist() == response_prefix_ids:
                    response_pos = i + len(response_prefix_ids)
                    row[:response_pos] = IGNORE_INDEX
                    break

        # Mask padding tokens in labels
        padding_mask = encodings.attention_mask == 0
        labels[padding_mask] = IGNORE_INDEX

        return {
            "input_ids": encodings.input_ids,
            "attention_mask": encodings.attention_mask,
            "labels": labels,
        }

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Preprocessing instruction dataset"
    )
    tokenized_dataset.save_to_disk(save_path)
    return tokenized_dataset


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
            remove_columns=dataset.column_names,
            desc="Tokenizing OWM dataset"
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

        tokenized_dataset = tokenized_dataset.map(
            create_labels,
            batched=True,
            num_proc=num_proc,
            desc="Creating labels"
        )

        tokenized_dataset.save_to_disk(save_path)

    print(tokenized_dataset)
    return tokenized_dataset
