import csv
import sys
from transformers import AutoTokenizer, BertModel, BertConfig
import torch

from models.embedding import Embedding

model_name: str = "cl-tohoku/bert-base-japanese-v2"
config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, config=config)


def cos_similarity(x, y, eps=1e-8):
    """
    コサイン類似度を計算する
    """
    nx = x / (torch.sqrt(torch.sum(x**2)) + eps)
    ny = y / (torch.sqrt(torch.sum(y**2)) + eps)
    return torch.dot(nx, ny)


def get_tokens(text: str):
    token = tokenizer.tokenize(text)
    return token


def find_target_token_index(text: str, target_token: str):
    """
    対象の単語が何番目のトークンかを返す
    """
    tokens = get_tokens(text)
    for i, token in enumerate(tokens):
        if token == target_token:
            return i
    return -1


def get_last_hidden_state(text: str, token_index: int):
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(
        0
    )
    outputs = model(input_ids)
    hidden_states = outputs.last_hidden_state
    print(len(hidden_states))
    return hidden_states[token_index]


def get_word_embedding(text: str, token_index: int):
    """
    最後の4層のhidden stateを足し合わせたベクトルを取得する

    Parameters
    ----------
    text : str
        _description_
    token_index : int
        _description_
    """
    input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(
        0
    )
    outputs = model(input_ids)
    hidden_states = outputs.hidden_states
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)
    target_token = token_embeddings[token_index]
    sum_vec = torch.sum(target_token[-4:], dim=0)  # 4層のhidden stateを足し合わせる
    return sum_vec


def save_files_for_visualization(
    embeddings: list[Embedding], metadata_file_name: str, embedding_file_name: str
):
    """
    メタデータと埋め込みベクトルを保存する
    """
    with open(metadata_file_name, "w+", encoding="utf-8") as f:
        for embedding in embeddings:
            f.write(embedding.text + "\n")

    with open(embedding_file_name, "w+", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for embedding in embeddings:
            writer.writerow(embedding.embedding.numpy())


def main():
    args = sys.argv
    input_file_path = args[1]
    metadata_file_name = args[2]
    embedding_file_name = args[3]
    target_token = 'で'
    embeddings: list[Embedding] = []
    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            target_token_idx = find_target_token_index(line, target_token)
            if target_token_idx == -1:
                continue
            embedding = get_word_embedding(line, target_token_idx)
            embeddings.append(Embedding(line, target_token, embedding))
    save_files_for_visualization(embeddings, metadata_file_name, embedding_file_name)


if __name__ == "__main__":
    main()
