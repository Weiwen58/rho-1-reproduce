from transformers import TrainingArguments, Trainer
from data_loader import InstructionDataset, load_instruction_dataset
from model import load_hf_model_and_tokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def train_reference():
    combined_dataset = load_instruction_dataset()

    model, tokenizer = load_hf_model_and_tokenizer(MODEL_NAME)

    train_val_split = combined_dataset.train_test_split(test_size=0.1)
    train_dataset = InstructionDataset(train_val_split["train"], tokenizer)
    eval_dataset = InstructionDataset(train_val_split["test"], tokenizer)

    training_args = TrainingArguments(
        output_dir="./outputs/reference_model",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
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

    trainer.train()

if __name__ == "__main__":
    train_reference()

