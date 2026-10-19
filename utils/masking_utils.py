"""
Shared tokenization + response-masking logic used by both Part A (train_sft.py) and
Part B (train_dpo.py). "Mask everything except the final assistant turn" is the same
operation in both places, so it lives here once instead of being re-derived
independently in two files.
"""

import torch


def tokenize_with_response_mask(messages: list[dict], tokenizer, max_seq_len: int) -> dict:
    """
    TODO (student): tokenizes one chat-formatted conversation and marks which tokens
    belong to the final assistant turn. `messages` is a list of {"role", "content"}
    dicts (user/assistant, alternating, ending on an assistant turn) -- see data_loader.py.

    Must return a dict with:
      - "input_ids": 1D LongTensor -- the full tokenized conversation
        (`tokenizer.apply_chat_template(messages, tokenize=True, ...)`), truncated to
        at most max_seq_len tokens.

      - "attention_mask": 1D LongTensor, same length, all 1s (no padding needed here --
        batching/collating happens in the caller's train_loop()).

      - "response_mask": 1D LongTensor, same length: 1 for the final assistant-turn
        tokens, 0 everywhere else (prompt tokens and any earlier turns).

    This is the function the autograder unit-tests directly (see README.md's
    Autograder section) -- its signature and the three return keys above must not
    change. Both train_sft.build_sft_example() and train_dpo.build_dpo_example() call
    this rather than re-deriving the masking logic independently -- implement it once,
    here, and both Part A and Part B get it for free.

    Hint: `tokenizer.apply_chat_template(messages[:-1], tokenize=True,
    add_generation_prompt=True)` tokenizes everything up through the prompt that
    invites the final assistant turn. Its length tells you exactly where the response
    starts in the full conversation's tokenization -- everything before that index is
    prompt, everything from it onward (up to the end-of-turn token) is response.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("messages must end with an assistant message.")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be a positive integer.")
    
    # turn all chat into token ID 
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    
    # except the last message, turn chat into token ID
    prefix_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )
    
    # only keep the first max_seq_len tokens
    input_ids = torch.tensor(input_ids, dtype=torch.long)[:max_seq_len]
    
    # find the index where the final assistant response starts
    response_start = len(prefix_ids) 
    
    if response_start >= len(input_ids):
      raise ValueError("max_seq_len truncates the entire assistant response") 
    
    response_mask = torch.zeros_like(input_ids) # initialize a mask of zeros
    response_mask[response_start:] = 1  # mark the final assistant response tokens with 1s
    
    return {
        "input_ids": input_ids, # the entire chat token ID
        "attention_mask": torch.ones_like(input_ids),  # all 1s since no padding is needed
        "response_mask": response_mask, # 1 for the final assistant response tokens, 0 elsewhere
    }

    # raise NotImplementedError("Implement tokenization + response masking.")
