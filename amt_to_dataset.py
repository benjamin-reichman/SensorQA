import os
import csv
import numpy as np
from collections import defaultdict
import json

data = []
llama_formatted = []
completed_csv_files = os.listdir("completed_assignments")
for completed_hits_file in completed_csv_files:
    completed_hit_answers = []
    with open("completed_assignments/" + completed_hits_file, "r") as completed_hits:
        csv_reader = csv.reader(completed_hits, delimiter=",")
        headers = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                headers = np.array(row)
                continue
            completed_hit_answers.append(row)

    image_url_column = np.argwhere(headers == "Input.image_url")[0][0]
    answer_column = np.argwhere(headers == "Answer.Answer")[0][0]
    question_column = np.argwhere(headers == "Answer.Question")[0][0]
    for i in range(len(completed_hit_answers)):
        data.append({"image_url": completed_hit_answers[i][image_url_column], "question": completed_hit_answers[i][question_column], "answer": completed_hit_answers[i][answer_column]})
        # data.append({"input": completed_hit_answers[i][question_column], "output": completed_hit_answers[i][answer_column]})


json.dump(data[:], open("overall_sensorqa_dataset.json", "w"))
json.dump(data[:int(np.floor(len(data)*0.8))], open("overall_sensorqa_dataset_train.json", "w"))
json.dump(data[int(np.floor(len(data)*0.8)):], open("overall_sensorqa_dataset_val.json", "w"))

# json.dump(data[:int(np.floor(len(data)*0.8))], open("llama_overall_sensorqa_dataset_train.json", "w"))
# json.dump(data[int(np.floor(len(data)*0.8)):], open("llama_overall_sensorqa_dataset_val.json", "w"))
