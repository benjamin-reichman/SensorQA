from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

import training_utils as t_utils
import os


def generate_responses_ft_vllm(prompts, args):
    # Use VLLM to generate responses with a model fine-tuned using LoRA

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        n=1,
        skip_special_tokens=True
    )

    expt_name = args.expt_name
    output_dir = f'{args.model}_{expt_name}/'

    ckpt_dirs = os.listdir(output_dir)
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
    last_ckpt = ckpt_dirs[-1]

    # bpdb.set_trace()
    model = AutoPeftModelForCausalLM.from_pretrained(
        f"{output_dir}/{last_ckpt}",
        low_cpu_mem_usage=True,
        # device_map='auto',
    )

    # Merge LoRA and base model
    merged_model = model.merge_and_unload()

    # Save the merged model
    merged_model.save_pretrained(f"{args.model}_{args.expt_name}_merged", safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenizer.save_pretrained(f"{args.model}_{args.expt_name}_merged")

    # llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_lora=True, max_lora_rank=peft_config.r)
    llm = LLM(model=f"{args.model}_{args.expt_name}_merged",
              tensor_parallel_size=8,
              gpu_memory_utilization=0.8)
    outputs = llm.generate(
        prompts,
        sampling_params
    )

    responses = [outputs[i].outputs[0].text for i in range(len(outputs))]
    answers = list(map(t_utils.extract_text_between_tags, responses))

    return answers


def generate_responses_zs_vllm(prompts, args, max_tokens=500):
    # Generate responses using VLLM with a zero-shot model
    n_gpus = 8
    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        temperature=0.0,
        # repetition_penalty=1.0,
        # top_p=0.95
    )
    # bpdb.set_trace()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=n_gpus,
        gpu_memory_utilization=0.8,
    )

    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    responses = [outputs[i].outputs[0].text for i in range(len(outputs))]
    return responses