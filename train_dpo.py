"""
Part B -- DPO training entrypoint.

DPO sharpens the Part A policy on your preference pairs: push up the log-prob of the
preferred response (ask when ambiguous / answer when clear) relative to the dispreferred
one, under a KL leash to a frozen reference (your Part A checkpoint). You must sweep beta
over at least three values -- run this script once per beta (`--beta`), each writing its
own adapter.

CLI args, loading your Part A LoRA checkpoint as the policy's starting point, and
per-beta checkpointing are wired up already. You implement sequence_log_prob(),
dpo_loss(), and the training loop around them -- see the TODO blocks and README.md's
Part B for the three things to know going in (the disable_adapter() trick, sequence
log-prob masking, the loss formula). build_dpo_example() itself is wired, built on top of
utils/masking_utils.tokenize_with_response_mask() -- see Part A for where that TODO lives.
"""

import argparse

import torch
from peft import PeftModel

from configs.paths_config import SFT_CHECKPOINT_DIR, dpo_checkpoint_dir
from configs.train_config import (
    DPO_BATCH_SIZE,
    DPO_EPOCHS,
    DPO_LEARNING_RATE,
    DPO_BETA,
    DPO_MAX_SEQ_LEN,
)
from data_loader import load_dpo_examples
from utils.model_utils import load_tokenizer, load_base_model
from utils.masking_utils import tokenize_with_response_mask

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def build_policy_and_tokenizer(sft_checkpoint_dir: str):
    """
    Loads the base model and attaches your Part A LoRA adapter as the DPO policy's
    starting point. The *same* model, with its adapter disabled via
    `policy.disable_adapter()`, doubles as the frozen reference model -- see
    README.md Part B, gotcha #1. You do not need to load a second copy of the base model.
    """
    tokenizer = load_tokenizer(sft_checkpoint_dir)
    base_model = load_base_model()
    policy = PeftModel.from_pretrained(base_model, sft_checkpoint_dir, is_trainable=True)

    return policy, tokenizer


def build_dpo_example(
    prompt: str,
    chosen_messages: list[dict],
    rejected_messages: list[dict],
    tokenizer,
    max_seq_len: int = DPO_MAX_SEQ_LEN,
) -> dict:
    """
    Wired up: calls utils.masking_utils.tokenize_with_response_mask() once for
    chosen_messages and once for rejected_messages. Once you implement that one
    function (see Part A), this works for free -- nothing to implement here.

    `chosen_messages`/`rejected_messages` are each a list of {"role", "content"} dicts:
    the shared user prompt followed by the differing final assistant completion -- see
    data_loader.py. `prompt` itself isn't needed here (it's already the shared prefix
    inside both messages lists) but is accepted for a uniform call signature with
    data_loader's (prompt, chosen, rejected) triples.

    Returns a dict with, for both "chosen" and "rejected":
      - "{chosen,rejected}_input_ids", "{chosen,rejected}_attention_mask",
        "{chosen,rejected}_response_mask" -- see
        utils.masking_utils.tokenize_with_response_mask()'s return value.
    """
    chosen = tokenize_with_response_mask(chosen_messages, tokenizer, max_seq_len)
    rejected = tokenize_with_response_mask(rejected_messages, tokenizer, max_seq_len)

    return {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "chosen_response_mask": chosen["response_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
        "rejected_response_mask": rejected["response_mask"],
    }


def sequence_log_prob(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """
    TODO (student): computes, for each sequence in the batch, the sum of log-probs of
    the response tokens only (response_mask == 1), under `model`'s current weights.
    Returns a 1D tensor of shape (batch_size,).

    Call this once with the policy active, and once more inside a
    `with model.disable_adapter():` block to get the reference log-probs -- see
    README.md Part B, gotcha #1.
    """
    # ensure each sequence have at least two tokens(causal language model need to predict the next token)
    if input_ids.size(1) < 2:
        raise ValueError("Each input sequence must contain at least two tokens.")

    # Compute logits for the input sequences
    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).logits

    # Logits at position t predict the token at position t + 1.
    token_logps = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = input_ids[:, 1:]

    # get the log-probabilities of the target tokens
    token_logps = token_logps.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # Mask out the log-probabilities of the response tokens only
    response_tokens = response_mask[:, 1:].to(token_logps.dtype)
    response_tokens = response_tokens * attention_mask[:, 1:].to(token_logps.dtype)

    if torch.any(response_tokens.sum(dim=-1) == 0):
        raise ValueError("At least one sequence has no unmasked response tokens.")

    return (token_logps * response_tokens).sum(dim=-1)    
    
    # raise NotImplementedError("Implement per-sequence response log-prob computation.")


def dpo_loss(
    policy_chosen_logps: torch.Tensor, # policy: the score of the chosen response under the policy
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor, # reference: the score after stopping LoRA adapter
    reference_rejected_logps: torch.Tensor,
    beta: float = DPO_BETA,
) -> torch.Tensor:
    """
    TODO (student): implements the DPO loss --

        L_DPO = -log sigma( beta * [ (policy_chosen_logps - reference_chosen_logps)
                                      - (policy_rejected_logps - reference_rejected_logps) ] )

    averaged over the batch.

    This function's signature must not change. The GTA reference tests exercise it
    directly (tests/test_dpo_loss.py).
    """
    preference_margin = (
        (policy_chosen_logps - reference_chosen_logps)
        - (policy_rejected_logps - reference_rejected_logps)
    )
    return -F.logsigmoid(beta * preference_margin).mean()    
    
    # raise NotImplementedError("Implement the DPO loss.")


def train_loop(policy, tokenizer, dataset, beta: float = DPO_BETA) -> None:
    """
    Implements the Part B DPO training loop for a single beta.

    TODO (student):
      1. Preprocess `dataset` ONCE, up front, by calling build_dpo_example() per
         (prompt, chosen, rejected) triple (e.g. via `dataset.map(...)`) -- do not call
         it inside the per-step loop below, or you'll re-tokenize every example on every
         epoch for no benefit.
      2. For each batch of preprocessed examples: pad/collate, then compute
         policy_chosen_logps / policy_rejected_logps via sequence_log_prob() with the
         adapter enabled.
      3. Compute reference_chosen_logps / reference_rejected_logps via the same
         sequence_log_prob() call, inside `with policy.disable_adapter():`.
      4. Compute dpo_loss(..., beta=beta), backward, optimizer step.
      5. Log the loss periodically.

    Hyperparameters (DPO_BATCH_SIZE, DPO_EPOCHS, DPO_LEARNING_RATE, DPO_MAX_SEQ_LEN) are
    in configs/train_config.py -- yours to tune. `beta` is passed in per run (--beta) so
    the sweep trains a separate adapter for each value.
    """
    if len(dataset) == 0:
        raise ValueError("DPO dataset is empty.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad token for batching.")

    # only do once before training to turn chosen/rejected into tokne ID and attention/response masks
    def preprocess(example):
        tokenized = build_dpo_example(
            example["prompt"],
            example["chosen"],
            example["rejected"],
            tokenizer,
        )
        return {key: value.tolist() for key, value in tokenized.items()}

    tokenized_dataset = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        desc="Tokenizing DPO examples",
    )

    # turn different lengths of sequences to the same length by padding, and collate into a batch
    def collate_fn(features):
        batch = {}

        for side in ("chosen", "rejected"):
            batch[side] = {
                "input_ids": pad_sequence(
                    [
                        torch.tensor(item[f"{side}_input_ids"], dtype=torch.long)
                        for item in features
                    ],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id,
                ),
                "attention_mask": pad_sequence(
                    [
                        torch.tensor(item[f"{side}_attention_mask"], dtype=torch.long)
                        for item in features
                    ],
                    batch_first=True,
                    padding_value=0,
                ),
                "response_mask": pad_sequence(
                    [
                        torch.tensor(item[f"{side}_response_mask"], dtype=torch.long)
                        for item in features
                    ],
                    batch_first=True,
                    padding_value=0,
                ),
            }

        return batch

    loader = DataLoader(
        tokenized_dataset,
        batch_size=DPO_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    trainable_params = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    if not trainable_params:
        raise ValueError("No trainable parameters found; check the policy adapter.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=DPO_LEARNING_RATE,
    )
    input_device = policy.get_input_embeddings().weight.device

    policy.train()
    for epoch in range(DPO_EPOCHS):
        total_loss = 0.0

        for step, batch in enumerate(loader, start=1):
            chosen = {
                key: value.to(input_device)
                for key, value in batch["chosen"].items()
            }
            rejected = {
                key: value.to(input_device)
                for key, value in batch["rejected"].items()
            }

            optimizer.zero_grad(set_to_none=True)

            # count gradients for tbe policy, only the assistant's response
            policy_chosen_logps = sequence_log_prob(policy, **chosen)
            policy_rejected_logps = sequence_log_prob(policy, **rejected)

            # stop using LoRA, don't count gradients for the reference model
            with policy.disable_adapter(), torch.no_grad():
                reference_chosen_logps = sequence_log_prob(policy, **chosen)
                reference_rejected_logps = sequence_log_prob(policy, **rejected)

            # compare policy to reference
            loss = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=beta,
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Non-finite DPO loss at beta={beta}, "
                    f"epoch {epoch + 1}, step {step}."
                )

            # only update the policy's LoRA adapter, not the frozen reference model
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 20 == 0 or step == len(loader):
                print(
                    f"Beta {beta} "
                    f"Epoch {epoch + 1}/{DPO_EPOCHS} "
                    f"Step {step}/{len(loader)} "
                    f"Loss {loss.item():.4f}",
                    flush=True,
                )

        average_loss = total_loss / len(loader)
        print(
            f"Beta {beta} "
            f"Epoch {epoch + 1}/{DPO_EPOCHS} "
            f"Average loss {average_loss:.4f}",
            flush=True,
        )    
    
    # raise NotImplementedError("Implement the Part B DPO training loop.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint-dir", type=str, default=SFT_CHECKPOINT_DIR)

    parser.add_argument(
        "--beta",
        type=float,
        default=DPO_BETA,
        help="DPO beta for this run (sweep by running once per value).",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override; defaults to a per-beta directory.",
    )

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or dpo_checkpoint_dir(args.beta)

    torch.manual_seed(args.seed)

    print(f"Loading Part A checkpoint from {args.sft_checkpoint_dir} as DPO policy init...")
    policy, tokenizer = build_policy_and_tokenizer(args.sft_checkpoint_dir)

    print("Loading DPO preference pairs...")
    dataset = load_dpo_examples()

    print(f"Training starts (beta={args.beta})...")
    train_loop(policy, tokenizer, dataset, beta=args.beta)

    print(f"Saving DPO-aligned LoRA adapter to {checkpoint_dir}...")
    policy.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Part B DPO complete (beta={args.beta}).")


if __name__ == "__main__":
    main()
