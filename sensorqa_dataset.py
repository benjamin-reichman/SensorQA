import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, Sampler
from torch import max as tmax
from torch import Tensor, tensor, stack
from torch.nn import functional as F
import random


class SensorQADataset(Dataset):
    def __init__(self, qa_file_path, tokenizer_class, tokenizer_parameters):
        super().__init__()
        self.prepare_tokenizer(tokenizer_class, tokenizer_parameters)
        if qa_file_path.endswith(".json"):
            self.load_json(qa_file_path)


    def load_json(self, file_path):
        qas = json.load(open(file_path, "r"))
        self.images = [i["image_url"] for i in qas]
        self.questions = [i["question"] for i in qas]
        self.answers = [i["answer"] for i in qas]
        self.model_answers = ["" for _ in range(len(self.answers))]

    def __setitem__(self, key, val):
        self.model_answers[key] = val

    def prepare_tokenizer(self, tokenizer_class, tokenizer_parameters):
        self.tokenizer = tokenizer_class.from_pretrained(*tokenizer_parameters)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = random.choice(self.answers[idx])
        question = self.tokenizer(question, return_tensors="pt", is_split_into_words=False, truncation=False, padding="longest")
        answer = self.tokenizer(answer, return_tensors="pt", is_split_into_words=False, truncation=False, padding="longest")
        return {
            "input_ids": question["input_ids"],
            "attention_mask": question["attention_mask"],
            "labels": answer["input_ids"],
            "label_attention_mask": answer["attention_mask"],
            "indices": idx,
        }

    def save(self, save_dir):
        json.dump([self.questions, self.answers, self.model_answers], open(save_dir, "w"))


def generic_padded_collator(batch):
    batch_types = {key: type(value) for key, value in batch[0].items()}
    pad_length = {key: tmax(stack([tensor(batch[i][key].shape) for i in range(len(batch))]), dim=0).values for key in batch_types if batch_types[key] == Tensor}
    padded_batch = [{key: F.pad(value, pad=(0, pad_length[key][1] - value.shape[1], 0, pad_length[key][0] - value.shape[0]), mode="constant", value=0 if key != "labels" else -100) if
                        batch_types[key] == Tensor else value for key, value in item.items()} for item in batch]
    batch = {key: stack([padded_batch[idx][key] for idx in range(len(padded_batch))]) if batch_types[key] == Tensor else [padded_batch[idx][key] for idx in range(len(padded_batch))] for key in batch_types}
    return {key: (batch[key] if batch[key].shape[1] != 1 else batch[key].squeeze(1)) if batch_types[key] == Tensor else batch[key] for key in batch}


if __name__ == "__main__":
    from transformers import T5Tokenizer
    sensorqa_dataset = SensorQADataset("overall_sensorqa_dataset.json", T5Tokenizer, ["t5-base"])
    import IPython; IPython.embed()