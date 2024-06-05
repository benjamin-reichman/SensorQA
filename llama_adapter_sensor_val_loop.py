import cv2
import sys
sys.path.append("LLaMA-Adapter/llama_adapter_v2_multimodal7b_sensors_clip")
import llama
import torch
from PIL import Image
import json
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "models/llama"
load_path = "LLaMA-Adapter/llama_adapter_v2_multimodal7b_sensors_clip/conversational_mask_experiment/checkpoint-4.pth"
model, preprocess = llama.load(load_path, llama_dir, llama_type="7B", device=device)
model.eval()

dataset = json.load(open("llama_adapter_overall_sensorqa_dataset_val.json", "r"))
question, answer, model_generation = [], [], []

with tqdm(total=len(dataset)) as pbar:
    for i in range(len(dataset)):
        sensor_file_name = dataset[i]["image"].split("/")
        subset, image_name = int(sensor_file_name[-3].split("_")[-1]), sensor_file_name[-1]
        user_number = int(image_name[image_name.find("usr"):].split("_")[0][3:]) + (21 if 5 < subset < 10 else (41 if 10 <= subset else 0))
        # sensor_file_name = "/".join((sensor_file_name[:-3] + ["sensor_embeddings", f"usr{user_number}_{'_'.join(image_name[image_name.find('usr'):].split('_')[1:])[:-4]}.npy"]))
        sensor_file_name = "/".join((sensor_file_name[:-3] + ["clip_embeddings_emb768", f"usr{user_number}_{'_'.join(image_name[image_name.find('usr'):].split('_')[1:])[:-4]}.npy"]))

        sensor_reading = torch.tensor(np.load(sensor_file_name))
        sensor_attention_mask = torch.logical_or(torch.isnan(sensor_reading), (sensor_reading == 0).all(dim=1).unsqueeze(1).expand(sensor_reading.shape[0], 768))

        sensor_reading = torch.nan_to_num(sensor_reading, nan=0)
        sensor_reading = sensor_reading.view(sensor_reading.shape[0]//1440, 1440, 768)
        sensor_attention_mask = sensor_attention_mask.view(sensor_attention_mask.shape[0]//1440, 1440, 768)

        if sensor_reading.shape[0] < 12:
            sensor_reading = torch.vstack([sensor_reading, torch.zeros(12-sensor_reading.shape[0], 1440, 768)])
            sensor_attention_mask = torch.vstack([sensor_attention_mask, torch.ones(12-sensor_attention_mask.shape[0], 1440, 768)]).bool()

        # sensor_reading = sensor_reading.view(1, -1, 768).to(device)
        # sensor_attention_mask = sensor_attention_mask.view(1, -1, 768).to(device)
        sensor_reading = sensor_reading.view(1, -1, 768).to(device)
        sensor_attention_mask = sensor_attention_mask.view(1, -1, 768).to(device)
        # sensor_attention_mask = sensor_attention_mask.view(1, -1, 225)[:, :, 0].unsqueeze(2).repeat(1, 1, 768)

        prompt = llama.format_prompt("Please answer the question: " + dataset[i]["conversations"][0]["value"])
        result = model.generate(sensor_reading, [prompt], mask=sensor_attention_mask.cuda())
        question.append(dataset[i]["conversations"][0]["value"].split("\n")[1])
        answer.append(dataset[i]["conversations"][1]["value"])
        model_generation.extend(result)
        pbar.update(1)
new_file_name = "_".join(load_path.split("/")[-2:])[:-4]
print(new_file_name)
json.dump([question, answer, model_generation], open(f"predictions/{new_file_name}_results.json", "w"))