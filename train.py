import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from model import load_hf_model_and_tokenizer
from data_loader import load_or_tokenize_owm_dataset

IGNORE_INDEX = -100
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SelectiveLanguageModeling:
    def __init__(
        self,
        reference_model,
        training_model,
        selection_ratio=0.6,
        max_length=2048,
        max_grad_norm=1.0
    ):
        self.reference_model = reference_model
        self.training_model = training_model
        self.selection_ratio = selection_ratio
        self.max_length = max_length
        self.max_grad_norm = max_grad_norm

    def compute_reference_loss(self, input_ids, attention_mask, labels):
        """
        Compute reference loss (L_RM) for each token using the reference model
        """
        with torch.no_grad():
            outputs = self.reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            # Get per-token reference loss
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fn(outputs.logits.permute(0, 2, 1), labels)

            return per_token_loss

    def compute_excess_loss(self, input_ids, attention_mask, labels):
        """
        Compute excess loss (L_Δ) between training model and reference model
        """
        # Get current model loss
        outputs = self.training_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        # Get per-token training loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        training_per_token_loss = loss_fn(outputs.logits.permute(0, 2, 1), labels)

        # Get reference loss
        reference_per_token_loss = self.compute_reference_loss(input_ids, attention_mask, labels)

        # Compute excess loss
        excess_loss = training_per_token_loss - reference_per_token_loss
        return excess_loss

    def selective_training_step(self, input_ids, attention_mask, labels):
        """
        Perform one selective training step
        """
        # Compute excess loss for all tokens
        excess_loss = self.compute_excess_loss(input_ids, attention_mask, labels)

        batch_size, seq_len = input_ids.size()
        # Select top k% tokens based on excess loss
        k = int(seq_len * self.selection_ratio)
        _, selected_indices = torch.topk(excess_loss, k, dim=1)  # Shape: (batch_size, k)

        # Create mask for selected tokens
        selection_mask = torch.zeros_like(input_ids, dtype=torch.bool)  # Shape: (batch_size, seq_len)
        batch_indices = torch.arange(batch_size).unsqueeze(1)  # Shape: (batch_size, 1)
        selection_mask[batch_indices, selected_indices] = True

        selection_mask = selection_mask & attention_mask
        new_labels = labels.clone()
        new_labels[~selection_mask] = IGNORE_INDEX

        # Forward pass with selected tokens only
        outputs = self.training_model(
            input_ids=input_ids,
            attention_mask=selection_mask,
            labels=new_labels
        )

        return outputs.loss

    def train(
        self,
        train_loader,
        optimizer,
        num_epochs,
        eval_loader=None,
        warmup_steps=0,
        save_path=None,
        device=DEVICE
    ):
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.training_model.train()
        self.reference_model.eval()

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                loss = self.selective_training_step(input_ids, attention_mask, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.training_model.parameters(),
                    self.max_grad_norm
                )

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

            # Evaluation
            if eval_loader is not None:
                eval_loss = self.evaluate(eval_loader, device)
                print(f"Epoch {epoch+1}, Eval Loss: {eval_loss:.4f}")

            # Save checkpoint
            if save_path:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.training_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }
                torch.save(checkpoint, f"{save_path}/checkpoint_epoch_{epoch+1}.pt")

    def evaluate(self, eval_dataloader, device):
        self.training_model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                loss = self.selective_training_step(input_ids, attention_mask, labels)
                total_loss += loss.item()

        self.training_model.train()
        return total_loss / len(eval_dataloader)


def main():
    reference_model, tokenizer = load_hf_model_and_tokenizer(MODEL_NAME)
    training_model, _ = load_hf_model_and_tokenizer(MODEL_NAME)
    reference_model.to(DEVICE)
    training_model.to(DEVICE)

    tokenized_dataset = load_or_tokenize_owm_dataset(
        tokenizer,
        load=True,
        save_path="./outputs/tokenized_dataset",
        num_rows=100)

    train_val_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_val_split.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    train_dataset = train_val_split["train"]
    eval_dataset = train_val_split["test"]
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1)

    # Initialize selective language modeling
    slm = SelectiveLanguageModeling(reference_model, training_model)

    optimizer = torch.optim.AdamW(training_model.parameters(), lr=8e-5)

    # Train selective language model
    slm.train(
        train_loader, 
        optimizer, 
        num_epochs=1, 
        eval_loader=eval_loader,
        save_path="./outputs/checkpoints"
    )


if __name__ == "__main__":
    main()
