import json
import numpy as np
from torch.utils.data import Dataset
#from torch.utils.data import Dataset, Sampler
from torch import max as tmax
from torch import Tensor, tensor, stack
from torch.nn import functional as F
import random
from sklearn.model_selection import KFold
import torch
import json
import deepsqa_preprocess_data.embedding as ebd
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences


def answers_to_onehot(answers, answer_list, answer_list_len=1095):
    # answer_array: standard array for index searching
    int_answers = [answer_list.index(word) for word in answers]
    onehot_answers = []
    for word in answers:
        onehot = np.zeros(answer_list_len)
        index = answer_list.index(word)
        onehot[index] = 1.0
        onehot_answers.append(onehot)
    return int_answers, onehot_answers


def int_to_answers(int_outputs, answer_list_file='deepsqa_answer_list.json'):
    answer_list = json.load(open(answer_list_file))
    return [answer_list[int(i)] for i in int_outputs]


def questions_to_tokens(questions, max_q_len=133):
# 	if split == 'train':
# 		data_path = 'data/train_qa'
# 	elif split == 'val':
# 		data_path = 'data/val_qa'
# 	else:
# 		print('Invalid split!')
# 		sys.exit()
#     data_path = 'data/train_qa'
    word_idx = ebd.load_idx()
    seq_list = []

    for question in questions:
        words = word_tokenize(question)
        #print('words', words)
        seq = []
        for word in words:
            seq.append(word_idx.get(word.lower(),0))   # change every word to lower case, return 0 if the specified key does not exist.
        seq_list.append(seq)
        #print('seq', seq)
    
    question_matrix = pad_sequences(seq_list, maxlen=max_q_len)   # set to a fixed number
    # this is inconsistent with the question_summary in dataset generation. That function doesnt take '?'',' into account.
    # need to change that one later...
    
    #print('question matrix', question_matrix)

    return question_matrix


def load_data_for_deepsqa(qa_filename, sensor_embedding_path, fold, seed):
    # Load json file
    qas = json.load(open(qa_filename, "r"))
    images = [i["image_url"] for i in qas]
    questions = [i["question"] for i in qas]
    if isinstance(qas[0]["answer"], str):
        answers = [i["answer"] for i in qas]  # In the original version of dataset
        print('str!!')
    elif isinstance(qas[0]["answer"], list):
        answers = [i["answer"][0] for i in qas] # NOTE: Only use the first answer in the gpt shortened version of dataset
        print('list!!')
    else:
        raise ValueError("qas[answer] has a wrong data type!")
    pred_q_cats = [i["pred_q_cat"] for i in qas]
    pred_a_cats = [i["pred_a_cat"] for i in qas]

    # Encoder questions to vectors
    max_q_len = max([len(q) for q in questions]) 
    print(f"Max question len: {max_q_len}")  # Max question len: 133
    vec_questions = questions_to_tokens(questions, max_q_len)

    # Encode answers to one-hot vectors, as DeepSQA is a classification problem
    all_answers = list(set(answers))
    print(f"Total answer candidates: {len(all_answers)}") # Total answer candidates: 1095
    #print(all_answers)
    with open('deepsqa_answer_list.json', 'w') as file:
        json.dump(all_answers, file)
    int_answers, onehot_answers = answers_to_onehot(answers, all_answers, len(all_answers))

    # Split train and test
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    for i, (train_indices, test_indices) in enumerate(kf.split(range(len(images)))):
        if i == fold:
            images_train = [images[i] for i in train_indices]
            images_test = [images[i] for i in test_indices]
            questions_train = [questions[i] for i in train_indices]
            questions_test = [questions[i] for i in test_indices]
            vec_questions_train = [vec_questions[i] for i in train_indices]
            vec_questions_test = [vec_questions[i] for i in test_indices]
            answers_train = [answers[i] for i in train_indices]
            answers_test = [answers[i] for i in test_indices]
            int_answers_train = [int_answers[i] for i in train_indices]
            int_answers_test = [int_answers[i] for i in test_indices]
            pred_q_cat_train = [pred_q_cats[i] for i in train_indices]
            pred_q_cat_test = [pred_q_cats[i] for i in test_indices]
            pred_a_cat_train = [pred_a_cats[i] for i in train_indices]
            pred_a_cat_test = [pred_a_cats[i] for i in test_indices]
            break
    else:
        raise ValueError(f'fold {fold} is larger than 5. Set a smaller fold.')
    
    testData = SensorQADataset(images_test, questions_test, vec_questions_test, 
                               answers_test, int_answers_test,
                               pred_q_cat_test, pred_a_cat_test,
                               sensor_embedding_path)
    trainData = SensorQADataset(images_train, questions_train, vec_questions_train, 
                                answers_train, int_answers_train,
                                pred_q_cat_train, pred_a_cat_train,
                                sensor_embedding_path)

    return trainData, testData


class SensorQADataset(Dataset):
    def __init__(self, images, questions, vec_questions, 
                 answers, int_answers,
                 pred_q_cat, pred_a_cat,
                 sensor_embedding_path, 
                 max_days=12):
        super().__init__()
        self.images = images
        self.questions = questions
        self.vec_questions = vec_questions
        self.answers = answers
        self.int_answers = int_answers
        self.pred_q_cat = pred_q_cat
        self.pred_a_cat = pred_a_cat
        self.model_answers = ["" for _ in range(len(self.answers))]
        self.sensor_embedding_path = sensor_embedding_path
        self.max_days = max_days  # Max number of days in the dataset


    def __setitem__(self, key, val):
        self.model_answers[key] = val

    def prepare_tokenizer(self, tokenizer_class, tokenizer_parameters):
        self.tokenizer = tokenizer_class.from_pretrained(*tokenizer_parameters)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        sensor_file_name = self.images[idx].split("/")
        #print(sensor_file_name, sensor_file_name[-2].split("_")[-1])
        # Need to consider two types of path hierarchies, w/ or w/o an intermediate folder named figures
        if sensor_file_name[-2] == 'figures':
            subset, image_name = int(sensor_file_name[-3].split("_")[-1]), sensor_file_name[-1].split('_', 1)[-1]
        else:
            subset, image_name = int(sensor_file_name[-2].split("_")[-1]), sensor_file_name[-1].split('_', 1)[-1]
        # Correct
        subset = 1 if subset == 130 else subset

        user_number = int(image_name[image_name.find("usr"):].split("_")[0][3:]) + (21 if 5 < subset < 10 else (41 if 10 <= subset else 0))
        #print(f"subset {subset} image_name {image_name} user_number {user_number}")
        sensor_file_name = "/".join([self.sensor_embedding_path, f"usr{user_number}_{'_'.join(image_name[image_name.find('usr'):].split('_')[1:])[:-4]}.npy"])
        #print(sensor_file_name)

        sensor_reading = torch.tensor(np.load(sensor_file_name))
        sensor_attention_mask = torch.logical_or(torch.isnan(sensor_reading), (sensor_reading == 0).all(dim=1).unsqueeze(1).expand(sensor_reading.shape[0], 768))

        sensor_reading = torch.nan_to_num(sensor_reading, nan=0)
        sensor_reading = sensor_reading.view(sensor_reading.shape[0]//1440, 1440, 768)
        sensor_attention_mask = sensor_attention_mask.view(sensor_attention_mask.shape[0]//1440, 1440, 768)

        if sensor_reading.shape[0] < self.max_days:
            sensor_reading = torch.vstack([sensor_reading, torch.zeros(self.max_days-sensor_reading.shape[0], 1440, 768)])
            sensor_attention_mask = torch.vstack([sensor_attention_mask, torch.ones(self.max_days-sensor_attention_mask.shape[0], 1440, 768)]).bool()
        elif sensor_reading.shape[0] > self.max_days:
            sensor_reading = sensor_reading[:self.max_days]

        sensor_reading = sensor_reading.view(-1, 768)
        sensor_attention_mask = sensor_attention_mask.view(-1, 768)

        # Adjust dimension as in the DeepSQA code
        sensor_reading = np.expand_dims(sensor_reading, -1)
        sensor_reading = np.swapaxes(sensor_reading,0,1)

        question = self.questions[idx]
        vec_question = self.vec_questions[idx]
        q_len = len(vec_question)
        answer = self.answers[idx]
        int_answer = self.int_answers[idx]
        pred_q_cat = self.pred_q_cat[idx]
        pred_a_cat = self.pred_a_cat[idx]
        #print('__get_item__', sensor_reading.shape, question.shape)  # __get_item__ torch.Size([17280, 768]) (133,), answer is int
        
        return sensor_reading, question, vec_question, q_len, answer, int_answer, pred_q_cat, pred_a_cat
    
    def save(self, filename):
        json.dump([self.questions, self.answers, self.model_answers], open(filename, "w"))

""" NOT USED
def generic_padded_collator(batch):
    batch_types = {key: type(value) for key, value in batch[0].items()}
    pad_length = {key: tmax(stack([tensor(batch[i][key].shape) for i in range(len(batch))]), dim=0).values for key in batch_types if batch_types[key] == Tensor}
    padded_batch = [{key: F.pad(value, pad=(0, pad_length[key][1] - value.shape[1], 0, pad_length[key][0] - value.shape[0]), mode="constant", value=0 if key != "labels" else -100) if
                        batch_types[key] == Tensor else value for key, value in item.items()} for item in batch]
    batch = {key: stack([padded_batch[idx][key] for idx in range(len(padded_batch))]) if batch_types[key] == Tensor else [padded_batch[idx][key] for idx in range(len(padded_batch))] for key in batch_types}
    return {key: (batch[key] if batch[key].shape[1] != 1 else batch[key].squeeze(1)) if batch_types[key] == Tensor else batch[key] for key in batch}


if __name__ == "__main__":
    from transformers import T5Tokenizer
    sensorqa_dataset = SensorQADataset("../../2023_task_files/sensorqa/overall_sensorqa_dataset.json", T5Tokenizer, ["t5-base"])
    import IPython; IPython.embed()
"""