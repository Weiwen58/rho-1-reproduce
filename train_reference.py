from model import load_hf_model_and_tokenizer
from data_loader import load_instruction_dataset, preprocess_instruction_dataset
from transformers import TrainingArguments, Trainer
import torch
import os
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))


def train_reference():
    combined_dataset = load_instruction_dataset()

    model, tokenizer = load_hf_model_and_tokenizer()

    dataset = preprocess_instruction_dataset(
        combined_dataset,
        tokenizer,
        load=True,
        save_path="./outputs/instruction_dataset"
    )
    train_val_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]

    training_args = TrainingArguments(
        output_dir="./outputs/reference_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        # gradient_accumulation_steps=2,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        logging_dir="./logs",
        logging_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
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
    # trainer.train(resume_from_checkpoint="./outputs/reference_model/checkpoint-19000")


if __name__ == "__main__":
    train_reference()
