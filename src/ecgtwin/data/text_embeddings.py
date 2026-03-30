"""Text prompt formatting and embedding helpers."""

import torch
import torch.nn.functional as F

ORDINALS = ["1st", "2nd", "3rd"]


def prompt_process(text: str):
    """Convert a pipe-separated diagnosis string into the repo's prompt template."""
    text = text.lower()
    prompt_text = ""
    count = 0
    current = ""
    for char in text:
        if char == "|":
            ordinal = ORDINALS[count] if count <= 2 else f"{count + 1}th"
            if count == 0:
                prompt_text += f"Most importantly, the 1st diagnosis is {{{current}}}."
            else:
                prompt_text += f"As a supplementary condition, the {ordinal} diagnosis is {{{current}}}."
            count += 1
            current = ""
        else:
            current += char

    if current:
        ordinal = ORDINALS[count] if count <= 2 else f"{count + 1}th"
        if count == 0:
            prompt_text += f"Most importantly, the 1st diagnosis is {{{current}}}."
        else:
            prompt_text += f"As a supplementary condition, the {ordinal} diagnosis is {{{current}}}."

    return prompt_text


def mean_pooling(model_output, attention_mask):
    """Pool token embeddings using the attention mask."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_text_embedding(text: str, tokenizer, embedding_model, mix=False):
    """Encode either split or mixed diagnosis text into normalized embeddings."""
    text_input = prompt_process(text) if mix else text.split("|")
    encoded_input = tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {key: value.to(embedding_model.device) for key, value in encoded_input.items()}

    with torch.no_grad():
        model_output = embedding_model(**encoded_input)

    embedding = mean_pooling(model_output, encoded_input["attention_mask"])
    embedding = F.layer_norm(embedding, normalized_shape=(embedding.shape[1],))
    embedding = embedding[:, :768]
    return F.normalize(embedding, p=2, dim=1)
