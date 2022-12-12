from transformers import AlbertTokenizer, AlbertForPreTraining
import torch

model_name: str = 'ken11/albert-base-japanese-v1'
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForPreTraining.from_pretrained(model_name)
input_ids = torch.tensor(tokenizer.encode("私はまだ名前を持っていない。", add_special_tokens=True)).unsqueeze(0)
outputs = model(input_ids)
prediction_logits = outputs.prediction_logits
print(outputs)
