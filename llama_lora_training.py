import json

from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import os
import training_utils as t_utils
import eval_utils as e_utils
import numpy as np

import torch
import re


def preprocess_function(example):
    example['text'] = f'<s>[INST] Answer the question to the best of your abilities.\nQuestion: {example["question"]} [/INST] [ANS] {example["answer"]} [/ANS] </s>'
    return example
#
#
def preprocess_function_inference(example):
    example['text'] = f'<s>[INST] Answer the question to the best of your abilities.\nQuestion: {example["question"]} [/INST]'
    return example


def formatting_prompts_func_inference(example):
    output_texts = []

    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n "
        output_texts.append(text)
    return output_texts


def lora_setup():
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return peft_config


def find_text_between_tags(text, start_tag='[ANS]', end_tag='[/ANS]'):
    # Escape special regex characters in tags
    start_tag = re.escape(start_tag)
    end_tag = re.escape(end_tag)

    # Create a regex pattern to find text between start_tag and end_tag
    pattern = f'{start_tag}(.*?){end_tag}'

    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        match = text.strip('[ANS] ')
    else:
        match = matches[0].strip()

    return match


def train(args):
    dataset = load_dataset("json", data_files=f"overall_sensorqa_dataset_train_gpt_shortened.json", split="train")

    dataset = dataset.map(preprocess_function)
    print(f"Dataset length: {len(dataset)}")

    peft_config = lora_setup()

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    training_args = TrainingArguments(
        output_dir=f'{args.model}_interpret_ft/',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        per_device_eval_batch_size=4,
        fp16=False,  # Overflows with fp16
        bf16=False,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        group_by_length=True,
        max_steps=-1,
        lr_scheduler_type="constant",
        # logging & evaluation strategies
        logging_dir=f'./logs/',
        log_level="info",
        logging_strategy="steps",
        logging_steps=1,
        evaluation_strategy="no",
        # eval_steps=10,
        # eval_delay=3,
        do_eval=False,
        save_strategy="epoch",
        save_total_limit=5,
        # load_best_model_at_end=True,
        report_to="none",
        push_to_hub=False,
    )

    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", quantization_config=bnb_config, device_map='auto')
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map='auto')
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto', quantization_config=quant_config)

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    response_template_with_context = " ### Answer:"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    print(training_args)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=dataset.select([0, 10, 20, 30, 40, 50]),
        packing=False,
        max_seq_length=None,
        peft_config=peft_config,
        tokenizer=tokenizer,
        dataset_text_field="text",
        # compute_metrics=compute_metrics_fn,
        # data_collator=data_collator,
        # formatting_func=formatting_prompts_func,
    )

    # bpdb.set_trace()

    trainer.train()


def eval(args):
    # Evaluation:

    split = args.split

    eval_dataset = load_dataset("json", data_files=f"overall_sensorqa_dataset_val_gpt_shortened.json", split="train")
    eval_dataset = eval_dataset.map(preprocess_function_inference)

    # bpdb.set_trace()
    formatted_eval_dataset = formatting_prompts_func_inference(eval_dataset)

    if args.vllm:

        # sampling_params = SamplingParams(
        #     temperature=0,
        #     max_tokens=256,
        #     n=1,
        #     skip_special_tokens=True
        # )

        # peft_config = lora_setup()

        # llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_lora=True, max_lora_rank=peft_config.r)
        # outputs = llm.generate(
        #     eval_dataset['text'],
        #     sampling_params,
        #     lora_request=LoRARequest("adapter", 1, './meta-llama/Llama-2-7b-chat-hf_interpret_ft/checkpoint-3755/')# adapter_config.json'),
        # gpu_memory_utilization=0.8
        #     )

        # responses = [outputs[i].outputs[0].text for i in range(len(outputs))]
        # answers = list(map(t_utils.extract_text_between_tags, responses))

        answers = e_utils.generate_responses_ft_vllm(eval_dataset['text'], args)


    else:  # Vanilla hf generate

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        output_dir = f'{args.model}_interpret_ft/'

        ckpt_dirs = os.listdir(output_dir)
        ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
        last_ckpt = ckpt_dirs[-1]

        model = AutoModelForCausalLM.from_pretrained(f"{output_dir}/{last_ckpt}", device_map='auto')

        start_idx = t_utils.create_path_and_get_next_idx(args.outfile, overwrite=True)

        # bpdb.set_trace()

        pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=16, temperature=0.01)

        answers = []

        for i in tqdm(range(len(eval_dataset))):
            if i < start_idx:
                continue
            prompt = eval_dataset['text'][i]
            result = pipe(f"{prompt}")
            answer = result[0]['generated_text']
            answer = answer.split('[/INST]')[1].strip()
            answers.append(answer)

        json.dump([eval_dataset["question"], eval_dataset["answer"], answers], open(args.outfile, "w"))
            # t_utils.write_record_to_jsonl(args.outfile, answer)

        # answers = t_utils.load_jsonl(args.outfile)

        """
        tokenized_eval_dataset = tokenizer(formatted_eval_dataset, padding=True, truncation=True, return_tensors="pt")['input_ids'].to('cuda')

        response_template_with_context = " ### Answer:" 
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
        data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

        predictions = []

        output_dir = f'{args.model}_interpret_ft/'

        ckpt_dirs = os.listdir(output_dir)
        ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
        last_ckpt = ckpt_dirs[-1]

        model = AutoModelForCausalLM.from_pretrained(f"{output_dir}/{last_ckpt}", device_map='auto')

        start_idx = t_utils.create_path_and_get_next_idx(args.outfile, overwrite=True)

        for i in tqdm(range(len(tokenized_eval_dataset))):
            if i < start_idx:
                continue 

            output = model.generate(tokenized_eval_dataset[i:i+1], max_new_tokens=20)
            predictions.append(tokenizer.decode(output[0], skip_special_tokens=True))
            # answer = predictions[i].split("### Answer:")[1].strip()
            answer = predictions[i]

            t_utils.write_record_to_jsonl(args.outfile, answer)
        """
    count = 0
    for i in range(len(answers)):
        if eval_dataset['answer'][i] in answers[i]:
            count += 1

    print(count / len(answers))
    # bpdb.set_trace()



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
    if args.train:
        train(args)
    if args.eval:
        eval(args)








