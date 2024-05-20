import json
import numpy as np
from nltk.tokenize import word_tokenize
import pickle as pkl

overall_dataset = json.load(open("overall_sensorqa_dataset.json", "r"))
answers = [i["answer"] for i in overall_dataset]
tokenized_answers = [word_tokenize(i) for i in answers]
unique_tokenized_answers = [set(i) for i in tokenized_answers]
avg_unique_tokens_per_answer = np.mean([len(i) for i in unique_tokenized_answers])
number_of_unique_tokens = len(set.union(*unique_tokenized_answers))
num_tokens_in_answer = sum([len(i) for i in tokenized_answers])
percent_unique_tokens_in_answer = number_of_unique_tokens/num_tokens_in_answer
all_words_in_answers = [j for i in tokenized_answers for j in i]

print(f"Average Unique Tokens Per Answer: {avg_unique_tokens_per_answer}")
print(f"Number of Unique Tokens in Total: {number_of_unique_tokens}")
print(f"Number of Tokens in All of our Answers: {num_tokens_in_answer}")
print(f"Percent of Tokens That are Unique: {percent_unique_tokens_in_answer}")

unique_tokens, counts = np.unique(all_words_in_answers, return_counts=True)
order_of_counts = counts.argsort()[::-1]
unique_tokens = unique_tokens[order_of_counts]
counts = counts[order_of_counts]
tokens_with_counts = [i for i in zip(unique_tokens, counts)]
print("The top 15 most used tokens are: ")
for i in range(15):
    print(f"\"{tokens_with_counts[i][0]}\": {tokens_with_counts[i][1]}")

pkl.dump(unique_tokens[:15].tolist(), open("most_frequent_tokens.pkl", "wb"))
