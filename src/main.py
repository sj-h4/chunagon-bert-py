from transformers import AutoTokenizer, BertModel, BertConfig
import torch
import numpy as np

model_name: str = "cl-tohoku/bert-base-japanese-v2"
config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, config=config)

def cos_similarity(x, y, eps=1e-8):
    """
    コサイン類似度を計算する
    """
    nx = x / (torch.sqrt(torch.sum(x ** 2)) + eps)
    ny = y / (torch.sqrt(torch.sum(y ** 2)) + eps)
    return torch.dot(nx, ny)

def get_token(text: str):
    token = tokenizer.tokenize(text)
    return token

def get_last_hidden_state(text: str, token_index: int):
    input_ids = torch.tensor(
        tokenizer.encode("私はまだ名前を持っていない。", add_special_tokens=True)
    ).unsqueeze(0)
    outputs = model(input_ids)
    hidden_states = outputs.last_hidden_state
    print(type(hidden_states))

def get_sum_hidden_states(text: str, token_index: int):
    """
    最後の4層のhidden stateを足し合わせる

    Parameters
    ----------
    text : str
        _description_
    token_index : int
        _description_
    """
    input_ids = torch.tensor(
        tokenizer.encode(text, add_special_tokens=True)
    ).unsqueeze(0)
    outputs = model(input_ids)
    hidden_states = outputs.hidden_states
    print(type(hidden_states))

#get_last_hidden_state("公園で遊んだ。", 1)
get_sum_hidden_states("公園で遊んだ。", 1)
