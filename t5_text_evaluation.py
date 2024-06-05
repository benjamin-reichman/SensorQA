import re
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import pickle as pkl
from nltk.tokenize.treebank import TreebankWordDetokenizer

start = "[ANS]"
end = "[/ANS]"
pattern_start = re.escape(start) + "(?=(.*))"
pattern_end = f"(?=(.){re.escape(end)})"

most_frequent_tokens = pkl.load(open("most_frequent_tokens.pkl", "rb"))
print(most_frequent_tokens)
exact_match = 0
for j in range(0, 11, 10):
    print(f"Epoch {j}")
    questions, answers, model_generations = json.load(open(f"sensorqa_training_outputs/question_only_training_results_epoch_{j}.json"))


    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    bleu_scores = []
    meteor_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    for eval_set in zip(answers, model_generations):
        bleu_scores.append(sentence_bleu([word_tokenize(eval_set[0].lower())], word_tokenize(eval_set[1].lower())))
        meteor_scores.append(meteor_score([word_tokenize(eval_set[0].lower())], word_tokenize(eval_set[1].lower())))
        scores = scorer.score(eval_set[0].lower(), eval_set[1].lower())
        rouge_1_scores.append(scores["rouge1"].fmeasure)
        rouge_2_scores.append(scores["rouge2"].fmeasure)
        rouge_l_scores.append(scores["rougeL"].fmeasure)
        match_start = re.search(pattern_start, eval_set[1]).end() if start in eval_set[1] else 0
        match_end = re.search(pattern_end, eval_set[1]).start(1) if end in eval_set[1] else None
        exact_match += eval_set[0] == eval_set[1][match_start:match_end].strip()

    print(f"Mean bleu score: {np.mean(bleu_scores)}")
    print(f"Mean meteor score: {np.mean(meteor_scores)}")
    print(f"Mean rouge 1 score: {np.mean(rouge_1_scores)}")
    print(f"Mean rouge 2 score: {np.mean(rouge_2_scores)}")
    print(f"Mean rouge L score: {np.mean(rouge_l_scores)}")
    print(f"Exact Match score: {exact_match / len(model_generations)}")

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