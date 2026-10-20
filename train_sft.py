"""
Part A -- LoRA SFT training entrypoint.

SFT warm-starts the target behavior: it fine-tunes the instruct base model on the
PREFERRED response of each preference pair you built (a clarifying question for an
ambiguous prompt, a direct answer for a clear one), so that DPO in Part B has a sensible
policy to sharpen rather than starting cold.

CLI args, tokenizer setup, base model + LoRA construction, and checkpointing are wired
up already. You implement utils/masking_utils.tokenize_with_response_mask() and
train_loop() -- see the TODO blocks and README.md's Part A for two gotchas: freezing
base weights, masking prompt tokens.
"""

import argparse

import torch
from peft import LoraConfig, get_peft_model

from configs.paths_config import SFT_CHECKPOINT_DIR
from configs.train_config import (
    MODEL_ID,
    RANK,
    ALPHA,
    DROPOUT,
    TARGET_MODULES,
    SFT_BATCH_SIZE,
    SFT_EPOCHS,
    SFT_LEARNING_RATE,
    SFT_MAX_SEQ_LEN,
)
from data_loader import load_sft_examples
from utils.model_utils import load_tokenizer, load_base_model
from utils.masking_utils import tokenize_with_response_mask

from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq


def build_model_and_tokenizer():
    """Loads the base model + tokenizer, and wraps the model with a LoRA adapter."""
    tokenizer = load_tokenizer(MODEL_ID)
    model = load_base_model()

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Sanity check: should be a small fraction of total params.

    return model, tokenizer


def build_sft_example(messages: list[dict], tokenizer, max_seq_len: int = SFT_MAX_SEQ_LEN) -> dict:
    """
    Wired up: converts utils.masking_utils.tokenize_with_response_mask()'s response_mask into
    the -100-masked labels tensor the causal-LM loss needs. Once you implement
    tokenize_with_response_mask(), this works for free -- nothing to implement here.

    Returns a dict with "input_ids", "attention_mask", "labels": labels equal
    input_ids where response_mask == 1 (the final assistant turn), -100 elsewhere.
    """
    example = tokenize_with_response_mask(messages, tokenizer, max_seq_len)

    labels = example["input_ids"].clone()
    labels[example["response_mask"] == 0] = -100

    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
        "labels": labels,
    }


def train_loop(model, tokenizer, dataset) -> None:
    """
    Implements the Part A SFT training loop.

    TODO (student):
      1. Preprocess `dataset` ONCE, up front, by calling build_sft_example() per example
         (e.g. via `dataset.map(...)`) to get input_ids/attention_mask/labels -- do not
         call build_sft_example() inside the per-step loop below, or you'll re-tokenize
         the same examples on every epoch for no benefit.
      2. For each batch of preprocessed examples: pad/collate into a batch, forward pass,
         compute loss, backward, optimizer step. Only LoRA parameters should have
         gradients -- true by default from build_model_and_tokenizer() above, but verify
         it if you change the LoRA setup (README.md Part A, gotcha #1).
      3. Log the loss periodically so you can sanity-check training is converging.

    Hyperparameters (SFT_BATCH_SIZE, SFT_EPOCHS, SFT_LEARNING_RATE, SFT_MAX_SEQ_LEN) are
    in configs/train_config.py -- yours to tune, not fixed by this skeleton.
    """
    # turn one data into token
    def preprocess(example):
        tokenized = build_sft_example(example["messages"], tokenizer)
        return {key: value.tolist() for key, value in tokenized.items()} # turn tensor to list
    
    # tokenize all the data
    tokenized_dataset = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        desc="Tokenizing SFT examples",
    )
    
    # turn token into tensor and collate into batch
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    loader = DataLoader(
        tokenized_dataset,
        batch_size=SFT_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    
    # set up optimizer and device
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found; check the LoRA configuration.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=SFT_LEARNING_RATE,
    )
    input_device = model.get_input_embeddings().weight.device

    # training loop
    model.train()
    for epoch in range(SFT_EPOCHS):
        total_loss = 0.0

        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(input_device) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True) # clear gradients of previous batch
            loss = model(**batch).loss # compute loss for current batch

            # check if the loss is valid, if infinite or nan, raise an error
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite SFT loss at epoch {epoch + 1}, step {step}.")

            loss.backward() # compute gradients for current batch
            optimizer.step() # update model parameters based on gradients

            total_loss += loss.item()
            if step % 20 == 0 or step == len(loader):
                print(
                    f"Epoch {epoch + 1}/{SFT_EPOCHS} "
                    f"Step {step}/{len(loader)} "
                    f"Loss {loss.item():.4f}",
                    flush=True,
                )
        average_loss = total_loss / len(loader)
        print(
            f"Epoch {epoch + 1}/{SFT_EPOCHS} "
            f"Average loss {average_loss:.4f}",
            flush=True,
        )
    
    # raise NotImplementedError("Implement the Part A SFT training loop.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=SFT_CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading base model {MODEL_ID} and building LoRA adapter...")
    model, tokenizer = build_model_and_tokenizer()

    print("Loading SFT examples (preferred responses from your preference data)...")
    dataset = load_sft_examples()

    print("Training starts...")
    train_loop(model, tokenizer, dataset)

    print(f"Saving LoRA adapter to {args.checkpoint_dir}...")
    model.save_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.checkpoint_dir)
    print("Part A SFT complete.")


if __name__ == "__main__":
    main()
