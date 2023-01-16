from dataclasses import dataclass
import torch


@dataclass
class Embedding:
    text: str
    token: str
    embedding: torch.Tensor
