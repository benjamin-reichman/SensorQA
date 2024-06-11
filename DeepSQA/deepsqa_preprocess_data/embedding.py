# create embedding matrix

import numpy as np
import h5py
import pickle

def load():
    path = 'deepsqa_preprocess_data/embeddings/embedding_matrix.h5'
    with h5py.File(path,'r') as hf:
        data = hf.get('embedding_matrix')
        embedding_matrix = np.array(data)
    return embedding_matrix

def load_idx():
    path = 'deepsqa_preprocess_data/embeddings/word_idx'
    with open(path,'rb') as file:
        word_idx = pickle.load(file)
    return word_idx

# create embedding matrix and word index 
def create(glove_path):
    embedding_matrix_path = 'deepsqa_preprocess_data/embeddings/embedding_matrix.h5'
    word_idx_path = 'deepsqa_preprocess_data/embeddings/word_idx'
    embeddings = {}
    word_idx = {}

    with open(glove_path,'r') as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embeddings[word] = coefs
            word_idx[word] = i+1

    num_words = len(word_idx)
    embedding_matrix = np.zeros((1+num_words,300))

    for i, word in enumerate(word_idx.keys()):
        embedding_matrix[i+1] = embeddings[word]

    with h5py.File(embedding_matrix_path, 'w') as hf:
        hf.create_dataset('embedding_matrix',data=embedding_matrix)

    with open(word_idx_path,'wb') as f:
        pickle.dump(word_idx,f)