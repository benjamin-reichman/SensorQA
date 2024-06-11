
import training_utils as t_utils
import os
import re
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
#import pickle as pkl
#from nltk.tokenize.treebank import TreebankWordDetokenizer
import argparse

import nltk
nltk.download('punkt')
nltk.download('wordnet')

def generate_responses_ft_vllm(prompts, args):
    from vllm import LLM, SamplingParams
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

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
    from vllm import LLM, SamplingParams

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


def print_and_log(text, log_file):
    # Print the text to the console
    print(text)
    
    # Open the log file in append mode and write the text
    with open(log_file, 'a') as file:
        file.write(text + '\n')


def evaluate_scores(output_file, log_file):
    start = "[ANS]"
    end = "[/ANS]"
    pattern_start = re.escape(start) + "(?=(.*))"
    pattern_end = f"(?=(.){re.escape(end)})"

    #ost_frequent_tokens = pkl.load(open("../../2023_task_files/sensorqa/most_frequent_tokens.pkl", "rb"))
    #print(most_frequent_tokens)

    questions, answers, model_generations, q_cat, a_cat = json.load(open(output_file))
    print(answers[:5], model_generations[:5])
    print(q_cat[:5])
    print(a_cat[:5])
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    bleu_scores = []
    meteor_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    exact_match = 0

    question_cnt, answer_cnt = {}, {}
    bleu_scores_q_cat, bleu_scores_a_cat = {}, {}
    meteor_scores_q_cat, meteor_scores_a_cat = {}, {}
    rouge_1_scores_q_cat, rouge_1_scores_a_cat = {}, {}
    rouge_2_scores_q_cat, rouge_2_scores_a_cat = {}, {}
    rouge_l_scores_q_cat, rouge_l_scores_a_cat = {}, {}
    exact_match_q_cat, exact_match_a_cat = {}, {}
    for a, ga, qc, ac in zip(answers, model_generations, q_cat, a_cat):
        if isinstance(a, list): 
            ### Shortened version, with multiple candidate ground truth answers
            bleu_scores_tmp = []
            meteor_scores_tmp = []
            rouge_1_scores_tmp = []
            rouge_2_scores_tmp = []
            rouge_l_scores_tmp = []
            exact_match_tmp = []

            for eval in a:
                bleu_scores_tmp.append(sentence_bleu([word_tokenize(eval.lower())], word_tokenize(ga.lower())))
                meteor_scores_tmp.append(meteor_score([word_tokenize(eval.lower())], word_tokenize(ga.lower())))
                scores_tmp = scorer.score(eval.lower(), ga.lower())
                rouge_1_scores_tmp.append(scores_tmp["rouge1"].fmeasure)
                rouge_2_scores_tmp.append(scores_tmp["rouge2"].fmeasure)
                rouge_l_scores_tmp.append(scores_tmp["rougeL"].fmeasure)
                match_start = re.search(pattern_start, ga).end() if start in ga else 0
                match_end = re.search(pattern_end, ga).start(1) if end in ga else None
                exact_match_tmp.append(int(eval == ga[match_start:match_end].strip()))
            
            # Compute the best score
            new_bleu = max(bleu_scores_tmp)
            new_metero = max(meteor_scores_tmp)
            new_rouge_1 = max(rouge_1_scores_tmp)
            new_rouge_2 = max(rouge_2_scores_tmp)
            new_rouge_l = max(rouge_l_scores_tmp)
            new_exact_match = max(exact_match_tmp)

        else:
            ### Complete version
            # Compute the new scores
            new_bleu = sentence_bleu([word_tokenize('hello')], word_tokenize('yes'))
            new_metero = meteor_score([word_tokenize(a.lower())], word_tokenize(ga.lower()))
            scores = scorer.score(a.lower(), ga.lower())
            new_rouge_1 = scores["rouge1"].fmeasure
            new_rouge_2 = scores["rouge2"].fmeasure
            new_rouge_l = scores["rougeL"].fmeasure
            match_start = re.search(pattern_start, ga).end() if start in ga else 0
            match_end = re.search(pattern_end, ga).start(1) if end in ga else None
            new_exact_match = a == ga[match_start:match_end].strip()
        
        # Add the scores to records
        bleu_scores.append(new_bleu)
        meteor_scores.append(new_metero)
        rouge_1_scores.append(new_rouge_1)
        rouge_2_scores.append(new_rouge_2)
        rouge_l_scores.append(new_rouge_l)
        exact_match += new_exact_match

        # Add to question category records
        if qc in question_cnt:
            # Add to existing question types
            question_cnt[qc] += 1
            bleu_scores_q_cat[qc].append(new_bleu)
            meteor_scores_q_cat[qc].append(new_metero)
            rouge_1_scores_q_cat[qc].append(new_rouge_1)
            rouge_2_scores_q_cat[qc].append(new_rouge_2)
            rouge_l_scores_q_cat[qc].append(new_rouge_l)
            exact_match_q_cat[qc] += new_exact_match
        else:
            # Create a new question type
            question_cnt[qc] = 1
            bleu_scores_q_cat[qc] = [new_bleu]
            meteor_scores_q_cat[qc] = [new_metero]
            rouge_1_scores_q_cat[qc] = [new_rouge_1]
            rouge_2_scores_q_cat[qc] = [new_rouge_2]
            rouge_l_scores_q_cat[qc] = [new_rouge_l]
            exact_match_q_cat[qc] = new_exact_match

        # Add to answer category records
        if ac in answer_cnt:
            # Add to existing answer types
            answer_cnt[ac] += 1
            bleu_scores_a_cat[ac].append(new_bleu)
            meteor_scores_a_cat[ac].append(new_metero)
            rouge_1_scores_a_cat[ac].append(new_rouge_1)
            rouge_2_scores_a_cat[ac].append(new_rouge_2)
            rouge_l_scores_a_cat[ac].append(new_rouge_l)
            exact_match_a_cat[ac] += new_exact_match
        else:
            # Create a new answer type
            answer_cnt[ac] = 1
            bleu_scores_a_cat[ac] = [new_bleu]
            meteor_scores_a_cat[ac] = [new_metero]
            rouge_1_scores_a_cat[ac] = [new_rouge_1]
            rouge_2_scores_a_cat[ac] = [new_rouge_2]
            rouge_l_scores_a_cat[ac] = [new_rouge_l]
            exact_match_a_cat[ac] = new_exact_match

    # Compute the mean scores per class
    for qc in question_cnt:
        bleu_scores_q_cat[qc] = np.mean(bleu_scores_q_cat[qc])
        meteor_scores_q_cat[qc] = np.mean(meteor_scores_q_cat[qc])
        rouge_1_scores_q_cat[qc] = np.mean(rouge_1_scores_q_cat[qc])
        rouge_2_scores_q_cat[qc] = np.mean(rouge_2_scores_q_cat[qc])
        rouge_l_scores_q_cat[qc] = np.mean(rouge_l_scores_q_cat[qc])
        exact_match_q_cat[qc] /= question_cnt[qc]
    
    for ac in answer_cnt:
        bleu_scores_a_cat[ac] = np.mean(bleu_scores_a_cat[ac])
        meteor_scores_a_cat[ac] = np.mean(meteor_scores_a_cat[ac])
        rouge_1_scores_a_cat[ac] = np.mean(rouge_1_scores_a_cat[ac])
        rouge_2_scores_a_cat[ac] = np.mean(rouge_2_scores_a_cat[ac])
        rouge_l_scores_a_cat[ac] = np.mean(rouge_l_scores_a_cat[ac])
        exact_match_a_cat[ac] /= answer_cnt[ac]

    msg = f"====Eval====\n"
    msg += f"Mean bleu score: {np.mean(bleu_scores)}\n"
    msg += f"Mean bleu score per question type: {bleu_scores_q_cat}\n"
    msg += f"Mean bleu score per answer type: {bleu_scores_a_cat}\n"
    msg += f"Mean meteor score: {np.mean(meteor_scores)}\n"
    msg += f"Mean meteor score per question type: {meteor_scores_q_cat}\n"
    msg += f"Mean meteor score per answer type: {meteor_scores_a_cat}\n"
    msg += f"Mean rouge 1 score: {np.mean(rouge_1_scores)}\n"
    msg += f"Mean rouge 1 per question type: {rouge_1_scores_q_cat}\n"
    msg += f"Mean rouge 1 per answer type: {rouge_1_scores_a_cat}\n"
    msg += f"Mean rouge 2 score: {np.mean(rouge_2_scores)}\n"
    msg += f"Mean rouge 2 per question type: {rouge_2_scores_q_cat}\n"
    msg += f"Mean rouge 2 per answer type: {rouge_2_scores_a_cat}\n"
    msg += f"Mean rouge L score: {np.mean(rouge_l_scores)}\n"
    msg += f"Mean rouge L score per question type: {rouge_l_scores_q_cat}\n"
    msg += f"Mean rouge L score per answer type: {rouge_l_scores_a_cat}\n"
    msg += f"Exact Match score: {exact_match / len(model_generations)}\n"
    msg += f"Exact Match score: {exact_match / len(model_generations)}\n"
    msg += f"Exact Match score per question type: {exact_match_q_cat}\n"
    msg += f"Exact Match score per answer type: {exact_match_a_cat}\n"
    msg += f"Question type cnt: {question_cnt}\n"
    msg += f"Answer type cnt: {answer_cnt}\n"

    print_and_log(msg, log_file)

    results = {}
    results["bleu"] = np.mean(bleu_scores)
    results["bleu_q_cat"] = bleu_scores_q_cat
    results["bleu_a_cat"] = bleu_scores_a_cat
    results["meteor"] = np.mean(meteor_scores)
    results["meteor_q_cat"] = meteor_scores_q_cat
    results["meteor_a_cat"] = meteor_scores_a_cat
    results["rouge1"] = np.mean(rouge_1_scores)
    results["rouge1_q_cat"] = rouge_1_scores_q_cat
    results["rouge1_a_cat"] = rouge_1_scores_a_cat
    results["rouge2"] = np.mean(rouge_2_scores)
    results["rouge2_q_cat"] = rouge_2_scores_q_cat
    results["rouge2_a_cat"] = rouge_2_scores_a_cat
    results["rougel"] = np.mean(rouge_l_scores)
    results["rougel_q_cat"] = rouge_l_scores_q_cat
    results["rougel_a_cat"] = rouge_l_scores_a_cat
    results["exact"] = exact_match / len(model_generations)
    results["exact_q_cat"] = exact_match_q_cat
    results["exact_a_cat"] = exact_match_a_cat
    results["cnt_q_cat"] = question_cnt
    results["cnt_a_cat"] = answer_cnt

    return results


def display_results(results, metrics, log_file):
    avg_results = {m: np.average(results[m]) for m in metrics}
    print_and_log('{0:>20}'.format("") + ' '.join(['%10s']*len(metrics)) % tuple([m for m in metrics]), log_file)
    print_and_log('{0:>20}'.format("AVG") + ' '.join(['%10.4f'] * len(metrics)) % tuple([avg_results[m] for m in metrics]), log_file)


def average_dict(list_of_dict):
    # Assume all dictionaries have the same keys
    new_dict = {}
    for k in list_of_dict[0]:
        new_dict[k] = np.average([d[k] for d in list_of_dict])
    return new_dict
    

def display_results_per_cat(results, metrics, log_file):
    # Compute the average results per class
    for m in metrics:
        print_and_log(f"{m}: {average_dict(results[m])}", log_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='overall_sensorqa_dataset',
                        help="data file name prefix")

    args = parser.parse_args()
    print(args)

    evaluate_scores(f'sensorqa_training_outputs/t5_overall_sensorqa_dataset_short0_seed4322/question_only_training_results_epoch_90.json')
    #evaluate_scores(f'sensorqa_training_outputs/deepsqa_overall_sensorqa_dataset_short1_seed4322/deepsqa_training_results_epoch_80.json')

"""No longer used
bleu_scores = []
meteor_scores = []
rouge_1_scores = []
rouge_2_scores = []
rouge_l_scores = []
for k, eval_set in enumerate(zip(answers, model_generations)):
    eval_set = ([i for i in word_tokenize(eval_set[0].lower()) if i not in most_frequent_tokens], [i for i in word_tokenize(eval_set[1].lower()) if i not in most_frequent_tokens])
    bleu_scores.append(sentence_bleu([eval_set[0]], eval_set[1]))
    meteor_scores.append(meteor_score([eval_set[0]], eval_set[1]))
    if k < 6:
        print(f"{eval_set[0]}\n{eval_set[1]}")
    eval_set = (TreebankWordDetokenizer().detokenize(eval_set[0]), TreebankWordDetokenizer().detokenize(eval_set[1]))
    scores = scorer.score(eval_set[0].lower(), eval_set[1].lower())
    rouge_1_scores.append(scores["rouge1"].fmeasure)
    rouge_2_scores.append(scores["rouge2"].fmeasure)
    rouge_l_scores.append(scores["rougeL"].fmeasure)

print(f"Mean bleu score: {np.mean(bleu_scores)}")
print(f"Mean meteor score: {np.mean(meteor_scores)}")
print(f"Mean rouge 1 score: {np.mean(rouge_1_scores)}")
print(f"Mean rouge 2 score: {np.mean(rouge_2_scores)}")
print(f"Mean rouge L score: {np.mean(rouge_l_scores)}")
"""