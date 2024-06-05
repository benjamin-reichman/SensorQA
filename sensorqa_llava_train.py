import numpy as np
import logging
import os
from contextlib import nullcontext
import random
import PIL
from peft import LoraConfig
from transformers import pipeline

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from accelerate import Accelerator
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

tqdm.pandas()

# def custom_callback(ex):
#     import ipdb; ipdb.set_trace()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)

if __name__ == "__main__":
    ################
    # Dataset
    ################
    train_dataset = load_dataset("json", data_files=f"overall_sensorqa_dataset_train_gpt_shortened_non_oracle.json", split="train")
    print(f"Dataset size: {len(train_dataset['answer'])}")
    eval_dataset = load_dataset("json", data_files=f"overall_sensorqa_dataset_val_gpt_shortened_non_oracle.json", split="train")

    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model, Tokenizer & Processor
    ################
    LLAVA_CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    # quantization_config = get_quantization_config(model_config)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_use_double_quant=False,
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path)
    processor.tokenizer = tokenizer

    model = LlavaForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)


    ################
    # Create a data collator to encode text and image pairs
    ################

    class LLavaDataCollator:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, examples):
            texts = []
            images = []
            for example in examples:
                if len(example["images"]) > 1:
                    raise ValueError("This collator only supports one image per example")
                messages = [{"content": [{"index": "", "text": f"{example['question']}", "type": "text"}, {"index": 0, "text": "", "type": "image"}], "role": "user"}, {"content": [{"index": "", "text": f"{random.choice(example['answer']) if isinstance(example['answer'], list) else example['answer']}", "type": "text"}], "role": "assistant"}]
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
                filename = example["images"][0]
                if "pred_model_figures" in filename:
                    new_filename = "/".join(filename.split("/")[:7] + ["visualizations_subset_nonoracle", "figures"]) + "/nonoracle_"
                    subset, image_name = int(filename.split("/")[-3].split("_")[-1]), filename.split("/")[-1]
                    user_number = int(image_name[image_name.find("usr"):].split("_")[0][3:]) + (21 if 5 < subset < 10 else (41 if 10 <= subset else 0))
                    filename = new_filename + f"usr{user_number}_{'_'.join(image_name[image_name.find('usr'):].split('_')[1:])[:-4]}.png"
                images.append(PIL.Image.open(filename))

            batch = self.processor(texts, images, return_tensors="pt", padding=True)
            labels = batch["input_ids"].clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

            return batch


    data_collator = LLavaDataCollator(processor)


    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        # training_args.max_grad_norm = 5e-5
        # print(training_args)
        peft_config = get_peft_config(model_config)
        peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=True,
            target_modules="all-linear"
        )
        # peft_config.use_rslora = True  # difference between really high grad norm and then NaN and starting out at infinite grad norm
        # peft_config.init_lora_weights = "loftq"
        training_args.optim = "paged_adamw_32bit"
        training_args.fp16 = False
        training_args.bf16 = False
        training_args.learning_rate=2e-5
        training_args.warmup_ratio=0.03
        # training_args.group_by_length=True
        training_args.lr_scheduler_type = "constant"
        training_args.logging_steps = 100
        training_args.max_grad_norm = 1.0
        training_args.save_steps = 565
        # print(training_args.num_train_epochs)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="question",  # need a dummy field
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=data_collator,
            dataset_kwargs={"skip_prepare_dataset": True},
        )

    # if sft_script_args.train == "True":


    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
    # else:
    #     print("Here")

# sensorqa_llava_train.py --dataset_name="HuggingFaceH4/llava-instruct-mix-vsft"     --model_name_or_path="llava-hf/llava-1.5-7b-hf"     --report_to="none"     --learning_rate=2e-5     --per_device_train_batch_size=1     --gradient_accumulation_steps=1     --output_dir="data/vsft-llava-1.5-7b-hf"     --num_train_epochs=1     --gradient_checkpointing     --remove_unused_columns=False     --torch_dtype=float16 --fp16=True  --use_peft=True     --lora_r=64     --lora_alpha=16     --lora_target_modules=all-linear --log_level="info" --logging_strategy="steps" --logging_steps=1