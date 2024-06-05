import cv2
import llama
import torch
from PIL import Image
import json
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "models/llama"
load_path = "LLaMA-Adapter/llama_adapter_v2_multimodal7b/no_sensorqa_finetune_outputs/checkpoint-3.pth"
# load_path = "LLaMA-Adapter/llama_adapter_v2_multimodal7b/sensorqa_only_outputs_non_oracle_conversational/checkpoint-9.pth"
model, preprocess = llama.load(load_path, llama_dir, llama_type="7B", device=device)

model.eval()

dataset = json.load(open("llama_adapter_overall_sensorqa_dataset_val_gpt_shortened_non_oracle.json", "r"))
question, answer, model_generation = [], [], []

with tqdm(total=len(dataset)) as pbar:
    for i in range(len(dataset)):
        try:
            img = Image.fromarray(cv2.imread(dataset[i]["image"]))
            img = preprocess(img).unsqueeze(0).to(device)
            prompt = llama.format_prompt("Please answer the question: " + dataset[i]["conversations"][0]["value"])
            result = model.generate(img, [prompt])
            question.append(dataset[i]["conversations"][0]["value"].split("\n")[1])
            answer.append(dataset[i]["conversations"][1]["value"])
            model_generation.extend(result)
        except AttributeError:
            print(dataset[i]["image"])
        pbar.update(1)
new_file_name = "_".join(load_path.split("/")[-2:])[:-4]
json.dump([question, answer, model_generation], open(f"predictions/{new_file_name}_gpt_shortened_results.json", "w"))