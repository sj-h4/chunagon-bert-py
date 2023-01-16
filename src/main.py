from transformers import AutoTokenizer, AutoModelForMaskedLM, BertConfig
import torch
import numpy as np


def cos_similarity(x, y, eps=1e-8):
    """
    コサイン類似度を計算する
    """
    nx = x / (torch.sqrt(torch.sum(x ** 2)) + eps)
    ny = y / (torch.sqrt(torch.sum(y ** 2)) + eps)
    return torch.dot(nx, ny)

def get_tokens():
    model_name: str = "cl-tohoku/bert-base-japanese-v2"
    config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
    input_ids = torch.tensor(
        tokenizer.encode("私はまだ名前を持っていない。", add_special_tokens=True)
    ).unsqueeze(0)
    outputs = model(input_ids)
    hidden_states = outputs.hidden_states
    print(input_ids)


get_tokens()
