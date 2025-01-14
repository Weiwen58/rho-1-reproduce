import torch
from transformers import TrainingArguments, Trainer
from data_loader import load_instruction_dataset, preprocess_instruction_dataset
from model import load_hf_model_and_tokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def train_reference():
    combined_dataset = load_instruction_dataset()

    model, tokenizer = load_hf_model_and_tokenizer(MODEL_NAME)

    dataset = preprocess_instruction_dataset(
        combined_dataset, 
        tokenizer,
        load=False,
        save_path="./outputs/instruction_dataset"
    )
    train_val_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]

    training_args = TrainingArguments(
        output_dir="./outputs/reference_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        fp16=True,
        logging_dir="./logs",
        logging_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Check device information
    print(f"Trainer device: {trainer.args.device}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    trainer.train()

if __name__ == "__main__":
    train_reference()

