import json
# test
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline, LlavaForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import os
import training_utils as t_utils
import eval_utils as e_utils
import numpy as np
import PIL
import torch
import re


class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            if len(example["images"]) > 1:
                raise ValueError("This collator only supports one image per example")
            messages = [{"content": [{"index": "", "text": f"{example['question']}", "type": "text"}, {"index": 0, "text": "", "type": "image"}], "role": "user"}]
            text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            images.append(PIL.Image.open(example["images"][0]))

        batch = self.processor(texts, images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        batch = {keys: values.cuda() for keys, values in batch.items()}

        return batch

def eval(args):
    eval_dataset = load_dataset("json", data_files=f"overall_sensorqa_dataset_val.json", split="train")


    LLAVA_CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

    tokenizer = AutoTokenizer.from_pretrained(f"data/non_oracle_conversational_vsft-llava-1.5-7b-hf/", trust_remote_code=True)
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(f"llava-hf/llava-1.5-7b-hf", assign=True)
    processor.tokenizer = tokenizer

    data_collator = LLavaDataCollator(processor)


    # ckpt_dirs = os.listdir(output_dir)
    # ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
    # last_ckpt = ckpt_dirs[-1]

    model = LlavaForConditionalGeneration.from_pretrained(f"data/non_oracle_conversational_vsft-llava-1.5-7b-hf/checkpoint-565/", device_map='auto')
    model.eval()
    model_answers = []
    bsz = 6
    with torch.no_grad():
        with tqdm(total=len(eval_dataset)) as pbar:
            for i in range(0, len(eval_dataset), bsz):
                example = [{"images": eval_dataset["images"][j], "question": eval_dataset['question'][j]} for j in range(i, (i+bsz) if i+bsz < len(eval_dataset['images']) else len(eval_dataset['images']))]
                batch = data_collator(example)
                batch["max_length"] = 512
                results = model.generate(**batch)
                for j in range(len(results)):
                    model_answers.append(tokenizer.decode(results[j]).split("ASSISTANT: ")[-1].split("</s>")[0].strip())
                del results
                pbar.update(bsz)

    json.dump([eval_dataset['question'], eval_dataset['answer'], model_answers], open(f"data/non_oracle_conversational_vsft-llava-1.5-7b-hf/checkpoint-565/model_predictions.json", "w"))




def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="NousResearch/Llama-2-7b-chat-hf")
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--split", required='eval' in sys.argv, default='dev', choices=["dev", "test"], help="evaluate on dev/test")
    parser.add_argument("--outfile", required='eval' in sys.argv, help="output file to store predictions", default="predictions/llama_7b_ft.jsonl")
    parser.add_argument("--expt_name", default="interpret_ft")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    eval(args)