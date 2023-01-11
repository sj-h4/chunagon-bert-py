from transformers import AlbertTokenizer, AlbertForPreTraining, AlbertConfig
import torch
import numpy as np


def get_cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_tokens():
    model_name: str = "ken11/albert-base-japanese-v1"
    config = AlbertConfig.from_pretrained(model_name, output_hidden_states=True)
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForPreTraining.from_pretrained(model_name, config=config)
    input_ids = torch.tensor(
        tokenizer.encode("私はまだ名前を持っていない。", add_special_tokens=True)
    ).unsqueeze(0)
    outputs = model(input_ids)
    hidden_states = outputs.hidden_states
    print(input_ids)


get_tokens()
