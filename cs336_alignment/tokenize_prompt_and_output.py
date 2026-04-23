import torch


def tokenize_prompt_and_output(
        prompt_strs: list[str], 
        output_strs: list[str], 
        tokenizer) -> dict[str, torch.Tensor]: 
    '''
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    '''
    prompt_tokens = tokenizer(prompt_strs)
    output_tokens = tokenizer(output_strs)

    max_len = max(len(prompt)+len(output) for prompt, output in zip(prompt_tokens['input_ids'], output_tokens['input_ids']))

    pad_token_id = tokenizer.pad_token_id
    
    prompt_and_output_tokens = torch.tensor([
        (a + b) + [pad_token_id] * (max_len - len(a + b))
        for a, b in zip(prompt_tokens['input_ids'], output_tokens['input_ids'])
    ])

    mask = torch.tensor([
        [0] * len(a) + [1] * len(b) + [0] * (max_len - len(a + b)) 
        for a, b in zip(prompt_tokens['input_ids'], output_tokens['input_ids'])
    ])

    return {
        "input_ids": prompt_and_output_tokens[..., :-1],
        "labels": prompt_and_output_tokens[..., 1:],
        "response_mask": mask[..., 1:]
    }


if __name__ == '__main__':
    ''''''
        