# Code adapted from https://github.com/nesl/DeepSQA/blob/master/sqa_models/run_mac.py
import torch
from tqdm import tqdm
#from torch.optim import AdamW
#from torch.optim.lr_scheduler import LinearLR

from torch import nn
from torch import optim
#from torch.utils.data import DataLoader
#from torch.utils import data

#from torchsummary import summary

import json
import random
import numpy as np
import time
#import sys
import os
import pickle
from collections import Counter
import argparse
from collections import defaultdict

#from deepsqa_models.mac_model.dataset import CLEVR, collate_data, transform
from deepsqa_models.mac_model.model import MACNetwork
from deepsqa_sensorqa_dataset import load_data_for_deepsqa, int_to_answers
from deepsqa_eval_utils import evaluate_scores, display_results, display_results_per_cat

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

DATA_PATH = 'sensorqa_dataset'
SENSOR_EMB_PATH = 'sensor_embeddings'
OUTPUT_PATH = 'sensorqa_training_outputs'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

def accumulate(model1, model2, decay=0.99):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(epoch, train_loader, batch_size,
          net_running, net, 
          criterion, optimizer, 
          device):
    #training_set = My_Data2(split='train')  # loadding data, time consuming: 2mins
    #train_set = DataLoader(
    #    training_set, batch_size=batch_size, num_workers=1, shuffle = True
    #     , collate_fn=collate_data
    #)

    #dataset = iter(train_set)
    pbar = tqdm(iter(train_loader))
    print(len(train_loader))
    moving_loss = 0
    
    net.train(True)
    
    for iter_id, (image, question, vec_question, q_len, answer, int_answer, _, _) in enumerate(pbar):
        #print('image', image.shape)
        #print('question', len(question))
        #print('vec_question', vec_question.shape)
        #print('q_len', q_len)
        #print('answer', len(answer))
        #print('int_answer', int_answer.shape)

        image = image.type(torch.FloatTensor) # change data type: double to float
        q_len = q_len.tolist()
        vec_question = vec_question.type(torch.LongTensor)

        
        image, vec_question, int_answer = (
            image.to(device),
            vec_question.to(device),
            int_answer.to(device),
        )
        #print('Loaded data!')
        net.zero_grad()
        output = net(image, vec_question, q_len)
        #print('answer: ', answer)
        #print('output: ', output.shape)
        loss = criterion(output, int_answer)
        loss.backward()
        optimizer.step()
        correct = output.detach().argmax(1) == int_answer
        correct = torch.tensor(correct, dtype=torch.float32).sum() / batch_size
        
        # correct is the acc for current batch, moving_loss is the acc for previous batches
        if moving_loss == 0:
            moving_loss = correct

        else:
            moving_loss = (moving_loss * iter_id + correct)/(iter_id+1)
#             moving_loss = moving_loss * 0.99 + correct * 0.01

        pbar.set_description(
            'Epoch: {}; Curr_Loss: {:.5f}; Curr_Acc: {:.5f}; Tot_Acc(running): {:.5f}'.format(
                epoch, loss.item(), correct, moving_loss
            )
        )
        accumulate(net_running, net)


def valid(valid_loader, 
          epoch, 
          net_running, 
          criterion,
          device,
          save_dir,
          args):
    
    #valid_set = DataLoader(
    #    training_set, batch_size=batch_size, num_workers=1
    #)
    
    dataset = iter(valid_loader)

    #net_running.train(False)
    family_correct = Counter()
    family_total = Counter()
    loss_total = 0
    output_label = []
    gt_questions, gt_answers = [], []
    pred_q_cats, pred_a_cats = [], []
    
    with torch.no_grad():
        for image, question, vec_question, q_len, answer, int_answer, pred_q_cat, pred_a_cat in tqdm(dataset):
            
            family = [1]*len(image)
            image = image.type(torch.FloatTensor) # change data type: double to float
            q_len = q_len.tolist()
            vec_question = vec_question.type(torch.LongTensor)

            image, vec_question = image.to(device), vec_question.to(device)

            net_running.eval()
            output = net_running(image, vec_question, q_len)
            loss = criterion(output, int_answer.to(device))
            loss_total = loss_total + loss
            
            correct = output.detach().argmax(1) == int_answer.to(device)
            output_label.append(output.detach().argmax(1).cpu().numpy())  # getting output of validation set
            gt_questions.extend(question)
            gt_answers.extend(answer)
            pred_q_cats.extend(pred_q_cat)
            pred_a_cats.extend(pred_a_cat)

            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

    avg_acc = sum(family_correct.values()) / sum(family_total.values())
    avg_loss = (loss_total / sum(family_total.values())).cpu().numpy()
    
    print(
        'Avg Acc: {:.5f}; Avg Loss: {:.5f}'.format(
            avg_acc,
            avg_loss
        )
    )
    output_label = np.concatenate(output_label)
    output_answers = int_to_answers(output_label)

    with open(f"{save_dir}/deepsqa_training_results_epoch_{epoch}.json", 'w') as file:
        json.dump([gt_questions, gt_answers, output_answers, pred_q_cats, pred_a_cats], file)

    print('%d / %d'%(sum(family_correct.values()), sum(family_total.values())))

    args.gpt_shortened = False # To enable the normal test
    results = evaluate_scores(f"{save_dir}/deepsqa_training_results_epoch_{epoch}.json",
                              f"{save_dir}/results_{epoch}.txt")
    args.gpt_shortened = True  # Revert back
    return avg_acc, avg_loss, results


def run_mac_model(args,
                  fold,
                  seed,
                  hyper_parameters,
                  epochs,
                  model_save_folder,
                  result_save_name,
                  source_data = 'opp'
                  ):
    
    model_name = f"deepsqa_{args.data}_short{int(args.gpt_shortened)}_seed{seed}/"
    save_dir = os.path.join(OUTPUT_PATH, model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # building network
    n_words = hyper_parameters['n_words'] #400001
    dim = hyper_parameters['dim'] #512
    glove_embeding = hyper_parameters['glove_embeding'] #False
    ebd_train = hyper_parameters['ebd_train'] #True
    n_answers = hyper_parameters['n_answers'] # 1095 in SensorQA
    dropout = hyper_parameters['dropout'] #0.15

    batch_size = hyper_parameters['batch_size'] #64
    learning_rate = hyper_parameters['learning_rate'] #1e-4
    weight_decay = hyper_parameters['weight_decay'] #1e-4

    n_epoch = epochs # 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = MACNetwork(n_words, dim, 
                     vocabulary_embd = glove_embeding, embd_train = ebd_train,
                     classes = n_answers, dropout=dropout, source_data = source_data).to(device)
    net_running = MACNetwork(n_words, dim, 
                             vocabulary_embd = glove_embeding, embd_train = ebd_train,
                             classes = n_answers, dropout=dropout, source_data = source_data).to(device)
    accumulate(net_running, net, 0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = weight_decay)

    #loading dataset
    print('Loading dataset: ')
    since = time.time()
    #data_path = 'sqa_data/test_split.npz'
    #train_set = SQA_data(data_path = dataset_path, 
    #                     split='train')
    #val_set = SQA_data(data_path = dataset_path,
    #                   split='val')

    # load SensorQA dataset
    if args.gpt_shortened:
        filename = f"{DATA_PATH}/{args.data}_gpt_shortened.json"
    else:
        filename = f"{DATA_PATH}/{args.data}.json"
    sensor_embedding_path = f"{SENSOR_EMB_PATH}/{args.sensor_emb_folder}"
    trainData, testData = load_data_for_deepsqa(
            filename, sensor_embedding_path,
            seed=seed, fold=fold
        )
    
    dataloaders = {}
    dataloaders["test"] = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    dataloaders["train"] = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    print('Dataset loaded using %.2f seconds!\n'%(time.time()-since))

    # saving path:
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)

    model_name = 'mac_model'
    save_model_name = model_save_folder + model_name + '.model'
    print('Saving model to: ', model_save_folder )
    print('Saving learning result to: ', result_save_name )


    # Begin training!
    
    # import warnings
        # warnings.filterwarnings('ignore')
        
    print('============ Training details: ============ ')
    print('---- Model structure ----')
    print('Hidden dim: ', dim, 'Output dim: ', n_answers )
    print('GLOVE embedding:', glove_embeding,  '   Trainable:', ebd_train)
    print('---- Training details ----')
    print('Dropout:', dropout, '   Weight regularization lambda:', weight_decay )
    print('Batch_size:', batch_size, '   Learning_rate:', learning_rate,  '   Epochs:', n_epoch )
    print('\n\n')

    acc_best = 0.0
    #train_acc_list = []
    #train_loss_list = []
    val_acc_list = []
    val_loss_list = []

    val_acc, val_loss, results = valid(dataloaders["test"], 0, net_running, criterion, 
                                           device, save_dir, args)
    
    for epoch in range(1, n_epoch+1): #
        print('==========%d epoch =============='%(epoch))
        train(epoch, dataloaders["train"], batch_size, net_running, net, criterion, optimizer, device)
        
        #print('----- Training Acc: ----- ')
        #train_acc, train_loss, _ = valid(dataloaders["train"], batch_size, net_running, criterion, device)
        #print('----- Validation Acc: ----- ')
        val_acc, val_loss, results = valid(dataloaders["test"], epoch, net_running, criterion, 
                                           device, save_dir, args)

        #train_acc_list.append(train_acc)
        #train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        if val_acc > acc_best:
            with open(
                save_model_name, 'wb'
    #             'checkpoint/checkpoint_{}.model'.format(str(epoch + 1).zfill(2)), 'wb'
            ) as f:
                torch.save(net_running.state_dict(), f) ### should save net_running instead of net!
            print('!!!! Best accuracy increased from %.4f to %.4f !  Saved to: %s.' %(acc_best, val_acc, save_model_name))
            acc_best = val_acc
        else:
            print('Acc not increasing...')

    print('The best validation accuracy: ', acc_best)
    #train_hist = np.array([train_acc_list, train_loss_list, val_acc_list, val_loss_list]).T
    
    
    # getting inference result
    print('\n\n\n========================================')
    print('Loading best model from: ',save_model_name)
    checkpoint = torch.load(save_model_name)
    net_running.load_state_dict(checkpoint)
    net_running.eval()
    
    
    _, _, results = valid(dataloaders["test"], n_epoch, net_running, criterion, 
                          device, save_dir, args)
    
    
    # convert out_label to out_correctness!
    #npzfile = np.load(dataset_path)
    #y_true = npzfile['a_val'].argmax(axis = 1)
    #result_correctness = y_true == out_label
    #print('Checking: the testing acc is ', result_correctness.sum()/result_correctness.shape[0])
    
    
    #save_result = { 'history': train_hist,
    #                'inference_result': result_correctness.astype(int)
    #                }
    
    #with open(result_save_name, 'wb') as handle:
    #    pickle.dump(save_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #print('Inference result saved to:', result_save_name)
    
    return results


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='overall_sensorqa_dataset',
                        help="data file name prefix")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--sensor_emb_folder', type=str, default='clip_embeddings_emb768',
                        help="sensor embedding folder")
    parser.add_argument('--gpt_shortened', action='store_true', help='whether to only run test')
    parser.add_argument('--random_seeds', type=int, default=[4321, 4322, 4323, 4324, 4325], help="random seed")

    args = parser.parse_args()

    results = defaultdict(list)
    print('\n==== Run MAC ====\n')

    #dataset_path = 'sqa_data/opp_sim5_split.npz'   #================
    hyper_parameters = {
        'n_words': 400001,
        'dim': 512,
        'glove_embeding': False,
        'ebd_train': True,
        'n_answers': 1095,  # 1095 in gpt_shortened SensorQA  ================
        'dropout': 0.15,
        'batch_size': args.batch_size, #64,  # 32 for es, 64 for opp  ================
        'learning_rate': 1e-4,
        'weight_decay': 1e-4, 
    }
    epochs = 20  # 20 for es, 40 for opp  ================
    model_save_folder = 'trained_models/opp_sim5/'   #================
    result_save_name = 'result/opp_sim5_mac.pkl'   #================


    #print('processed data path: ', dataset_path)
    print('pickle data path: N/A')
    print('epochs: ', epochs)
    print('save dl models: ', model_save_folder)
    print('save results: ', result_save_name)

    for fold, seed in zip(range(5), args.random_seeds):
        set_seed(seed)
        print(args)
        print(f'#### Run Experiments on seed {seed} ####')
        seed_results = run_mac_model(
                        args,
                        fold, seed,
                        hyper_parameters,
                        epochs,
                        model_save_folder,
                        result_save_name,
                        source_data = 'opp')  #================
        if seed_results is not None:
            for s in seed_results:
                results[s].append(seed_results[s])

    metrics = ["bleu", "meteor", "rouge1", "rouge2", "rougel", "exact"]
    display_results(results, metrics, "deepsqa_gpt_shortened_results.txt")
    metrics = ["bleu_q_cat", "meteor_q_cat", "rouge1_q_cat", 
               "rouge2_q_cat", "rougel_q_cat", "exact_q_cat",
               "bleu_a_cat", "meteor_a_cat", "rouge1_a_cat", 
               "rouge2_a_cat", "rougel_a_cat", "exact_a_cat",]
    display_results_per_cat(results, metrics, "deeqsqa_gpt_shortened_results.txt")

    print("Experiments done!")


    


"""
def model_loop(epoch, model, val_dataloader):
    model.eval()
    with tqdm(total=len(val_dataloader)) as pbar:
        for i, batch in enumerate(dataloaders[mode]):
            outputs = model.generate(batch["input_ids"].cuda())
            output_answer = dataloaders[mode].dataset.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for k, j in enumerate(batch["indices"]):
                dataloaders[mode].dataset[j] = output_answer[k]
            pbar.update(1)
        val_dataloader.dataset.save(f"sensorqa_training_outputs/question_only_training_results_epoch_{epoch}.json")

tmodel.train()
optimizer = AdamW(tmodel.parameters(), lr=hyperparameters["learning_rate"])

num_epochs = 100
for epoch in range(num_epochs+1):
    epoch_loss = 0
    if epoch % 10 == 0:
        model_loop(epoch, tmodel, dataloaders["val"])
    with tqdm(total=len(dataloaders["train"])) as pbar:
        for i, batch in enumerate(dataloaders["train"]):
            tmodel.zero_grad(set_to_none=True)
            outputs = tmodel(batch["input_ids"].cuda(), batch["attention_mask"].cuda(), labels=batch["labels"].cuda())
            loss = outputs.loss
            loss.backward()
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(tmodel.parameters(), 2.0)
            optimizer.step()
            if i == len(dataloaders["train"])-1:
                print("Epoch {}: Loss: {:.4f}".format(epoch, epoch_loss/len(dataloaders["train"])))
            pbar.update(1)
"""
