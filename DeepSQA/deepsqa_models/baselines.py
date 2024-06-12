from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding, Concatenate, concatenate, Permute, Reshape 
# changing merge to concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py

import numpy as np

import pandas as pd

from keras.utils.np_utils import to_categorical

from keras.engine.input_layer import Input
from keras.models import Model

import keras

from keras.utils import plot_model
# plot_model(sen_model, show_shapes=True, show_layer_names=True)


# ========================================================================================
# baseline model: Prior and Prior-Q solution
def baseline_prior_q(test_data_path, data_ind = None):
    import pandas as pd
    if data_ind is None:
        df = pd.read_pickle(test_data_path)
    else:
        df = pd.read_pickle(test_data_path)
        df = df[data_ind]
    
    ans_dict_q = {}

    for i in df.question_family_index.unique():
        answer_count_i = df[df.question_family_index==i]['answer'].value_counts()
        ans_dict_q[str(i)] = answer_count_i.keys()[0]
        
    ans_prior = df.answer.value_counts().keys()[0]
        
    return ans_dict_q, ans_prior


# 
def run_baseline_prior(test_data_path, dict_prior_q, dict_prior, data_ind = None):
    
    
    
    import copy
    
    if data_ind is None:
        df = pd.read_pickle(test_data_path)
    else:
        df = pd.read_pickle(test_data_path)
        df = df[data_ind]
    df = df.reset_index()

    # prior:
    prior_ans = np.array(df.answer)
    prior_ans[:,] = dict_prior

    prior_result = (prior_ans == df.answer)

    # prior_Q
    prior_q_ans = copy.deepcopy(df.answer)
    prior_q_ans[:] = dict_prior

    for q_family_i in dict_prior_q.keys():
        id_i = df[df.question_family_index== int(q_family_i)].index
        prior_q_ans[id_i] = dict_prior_q[q_family_i]
    # if no Q_type info, then return the same ans as Q
    prior_q_result = (prior_q_ans == df.answer)


#     print( sum(prior_result)/prior_result.shape[0] )
#     print( sum(prior_q_result)/prior_q_result.shape[0] )
    
    return prior_result, prior_q_result

# # ========================================================================================
# # baseline model: Prior and Prior-Q solution
# def baseline_prior_q(train_data_path):
#     import pandas as pd
#     df = pd.read_pickle(train_data_path)
    
#     ans_dict_q = {}

#     for i in range(max(df.question_family_index.unique())+1):
#         answer_count_i = df[df.question_family_index==i]['answer'].value_counts()
#         ans_dict_q[str(i)] = answer_count_i.keys()[0]
        
#     ans_prior = df.answer.value_counts().keys()[0]
        
#     return ans_dict_q, ans_prior


# # 
# def run_baseline_prior(test_data_path, baseline_prior_q, baseline_prior):
    
#     df = pd.read_pickle(test_data_path)
    
#     data_num = df.shape[0]
#     count_id = 0
    
#     prior_count = np.zeros([data_num,1])
#     prior_q_count = np.zeros([data_num,1])

#     for index, row in df.iterrows():
#         ans_prior = baseline_prior
#         ans_prior_q = baseline_prior_q[str(row.question_family_index)]

#         if ans_prior == row.answer:
#             prior_count[count_id] = 1
#         if ans_prior_q == row.answer:
#             prior_q_count[count_id] = 1
#         count_id = count_id+1

# #     prior_acc = prior_count/df.shape[0]
# #     prior_q_acc = prior_q_count/df.shape[0]
    
#     return prior_count, prior_q_count




#========================================================================================
# baseline model: DL based methods: CNN, LSTM, Stacked Attention, etc
def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, 
                  num_hidden_lstm,
                  output_dim,
                  dropout_rate):
    print("Creating text model...")
    model = Sequential()
    
    model.add(Embedding(num_words, embedding_dim, 
        weights=[embedding_matrix], input_length=seq_length, trainable=False))
    
    model.add(LSTM(units=num_hidden_lstm, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    
    model.add(LSTM(units=num_hidden_lstm, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(output_dim, activation='relu'))
#     model.add(Dense(output_dim, activation='tanh'))

    return model



def sensory_model(dim, win_len, channel,
                  num_feat_map,
                  num_hidden_lstm, 
                  output_dim,
                  dropout_rate):
    
    print ("Creating sensory model...")
    model = Sequential()
    
    # ??? kernel_size=(1, 5), or (1,3)    
    # default data_format for conv2d: channels_last (batch, rows, cols, channels)
    model.add(Convolution2D(num_feat_map, kernel_size=(1, 3),
                            activation='relu',
                            input_shape= (dim, win_len, channel),
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(dropout_rate))
    
    model.add(Convolution2D(num_feat_map, kernel_size=(1, 3), 
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(dropout_rate))

#     model.add(Flatten())
    model.add(Permute((2, 1, 3), name='Permute_1'))  # for swap-dimension
    model.add(Reshape((-1, num_feat_map * dim), name='Reshape_1'))
    
    model.add(LSTM(num_hidden_lstm, return_sequences=False, stateful=False, name='Lstm_1'))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(output_dim, activation='relu'))
#     model.add(Dense(output_dim, activation='tanh'))

    return model



def create_baseline_sqa(embedding_matrix, num_words, embedding_dim,  seq_length,
                        num_hidden_lstm, output_dim, dropout_rate,
                        sen_dim, sen_win_len, sen_channel,num_feat_map, 
                        num_classes,
                        model_type = 'cnn_lstm_mul'
                       ):
    
    
    lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, 
                  num_hidden_lstm,
                  output_dim,
                  dropout_rate)

    sen_model = sensory_model(sen_dim, sen_win_len, sen_channel,
                      num_feat_map,
                      num_hidden_lstm, 
                      output_dim,
                      dropout_rate)
    
    if model_type == 'cnn_lstm_mul':
        print ("Merging final model: multiplication")
        
        x_input = Input(shape=(sen_dim, sen_win_len, sen_channel), name='x_input')
        y_input = Input(shape=(seq_length,), name='y_input')
        sen_result = sen_model(x_input)
        lstm_result = lstm_model(y_input)

        merged = keras.layers.multiply([sen_result, lstm_result])

        d1 = Dense(128, activation='tanh')(merged)
        d2 = Dense(num_classes, activation='softmax')(d1)

        new_model = Model(inputs=[x_input, y_input], outputs=d2)
        return new_model
        
        
    elif model_type == 'cnn_lstm_cat':
        print ("Merging final model: concatenation")
        x_input = Input(shape=(sen_dim, sen_win_len, sen_channel), name='x_input')
        y_input = Input(shape=(seq_length,), name='y_input')
        sen_result = sen_model(x_input)
        lstm_result = lstm_model(y_input)

        merged = concatenate([sen_result, lstm_result])

        d1 = Dense(128, activation='tanh')(merged)
        d2 = Dense(num_classes, activation='softmax')(d1)

        new_model = Model(inputs=[x_input, y_input], outputs=d2)
        return new_model
        
        
    elif model_type == 'lstm':
        print ("Creating Blind LSTM model")
        y_input = Input(shape=(seq_length,), name='y_input')
        lstm_result = lstm_model(y_input)

        d1 = Dense(128, activation='tanh')(lstm_result)
        d2 = Dense(num_classes, activation='softmax')(d1)

        new_model = Model(inputs=[ y_input], outputs=d2)
        return new_model
        
        
    elif model_type == 'cnn':
        print ("Creating Deaf CNN model")
        x_input = Input(shape=(sen_dim, sen_win_len, sen_channel), name='x_input')
        sen_result = sen_model(x_input)

        d1 = Dense(128, activation='tanh')(sen_result)
        d2 = Dense(num_classes, activation='softmax')(d1)

        new_model = Model(inputs=[x_input], outputs=d2)
        return new_model
    
    
    elif model_type == 'deepsqa':
        print ("DeepSQA model!")
        x_input = Input(shape=(sen_dim, sen_win_len, sen_channel), name='x_input')
        y_input = Input(shape=(seq_length,), name='y_input')
        sen_result = sen_model(x_input)
        lstm_result = lstm_model(y_input)
        
#         merged_1 = keras.layers.multiply([sen_result, lstm_result])
        merged = concatenate([sen_result, lstm_result])

        d1 = Dense(128, activation='relu')(merged)
        dp1 = Dropout(dropout_rate)(d1)
        d3 = Dense(num_classes, activation='softmax')(dp1)

        new_model = Model(inputs=[x_input, y_input], outputs=d3)
        return new_model
    
    elif model_type == 'deepsqa2':
        print ("DeepSQA model!")
        x_input = Input(shape=(sen_dim, sen_win_len, sen_channel), name='x_input')
        y_input = Input(shape=(seq_length,), name='y_input')
        sen_result = sen_model(x_input)
        lstm_result = lstm_model(y_input)
        
        merged_1 = keras.layers.multiply([sen_result, lstm_result])
        merged = concatenate([sen_result, lstm_result, merged_1])
        
        d1 = Dense(128, activation='relu')(merged)
        dp1 = Dropout(dropout_rate)(d1)
        d3 = Dense(num_classes, activation='softmax')(dp1)

        new_model = Model(inputs=[x_input, y_input], outputs=d3)
        return new_model
        
        
    else:
        print ("Wrong model type!")
        return None
    
    
    
#========================================================================================
# baseline model: Stacked Attention Model


import keras.backend as K
from keras.layers import Input
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.merge import concatenate, multiply
from keras.layers.core import Dense, Dropout, RepeatVector, Reshape, Activation, Lambda, Flatten


def show_ask_attend_answer(vocab_size, num_glimpses=2, n=14):
    # Define network inputs where n is the feature rows and columns. In the
    # paper they use ResNet 152 res5c features with size (14x14x2048)
    image_input = Input(shape=(n,n,2048))
    question_input = Input(shape=(15,))
    
    # Learn word embeddings in relation to total vocabulary
    question_embedding = Embedding(vocab_size, 300, input_length=15)(question_input)
    question_embedding = Activation('tanh')(question_embedding)
    question_embedding = Dropout(0.5)(question_embedding)
    
    # LSTM to seuqentially embed word vectors into a single question vector
    question_lstm = LSTM(1024)(question_embedding)
    
    # Repeating and tiling question vector to match image input for concatenation
    question_tile = RepeatVector(n*n)(question_lstm)
    question_tile = Reshape((n,n,1024))(question_tile)
    
    # Concatenation of question vector and image features
    concatenated_features1 = concatenate([image_input, question_tile])
    concatenated_features1 = Dropout(0.5)(concatenated_features1)
    
    # Stacked attention network
    attention_conv1 = Conv2D(512, (1,1))(concatenated_features1)
    attention_relu = Activation('relu')(attention_conv1)
    attention_relu = Dropout(0.5)(attention_relu)
    
    attention_conv2 = Conv2D(num_glimpses, (1,1))(attention_relu)
    attention_maps = Activation('softmax')(attention_conv2)
    
    # Weighted average of image features using attention maps
    image_attention = glimpse(attention_maps, image_input, num_glimpses, n)
    
    # Concatenation of question vector and attended image features
    concatenated_features2 = concatenate([image_attention, question_lstm])
    concatenated_features2 = Dropout(0.5)(concatenated_features2)
    
    # First fully connected layer with relu and dropout
    fc1 = Dense(1024)(concatenated_features2)
    fc1_relu = Activation('relu')(fc1)
    fc1_relu = Dropout(0.5)(fc1_relu)
    
    # Final fully connected layer with softmax to output answer probabilities
    fc2 = Dense(3000)(fc1_relu)
    fc2_softmax = Activation('softmax')(fc2)
    
    # Instantiate the model
    vqa_model = Model(inputs=[image_input, question_input], outputs=fc2_softmax)
    
    return vqa_model

def glimpse(attention_maps, image_features, num_glimpses=2, n=14):
    glimpse_list = []
    for i in range(num_glimpses):
        glimpse_map = Lambda(lambda x: x[:,:,:,i])(attention_maps)                # Select the i'th attention map
        glimpse_map = Reshape((n,n,1))(glimpse_map)                               # Reshape to add channel dimension for K.tile() to work. (14,14) --> (14,14,1)
        glimpse_tile = Lambda(tile)(glimpse_map)                                  # Repeat the attention over the channel dimension. (14,14,1) --> (14,14,2048)
        weighted_features = multiply([image_features, glimpse_tile])              # Element wise multiplication to weight image features
        weighted_average = AveragePooling2D(pool_size=(n,n))(weighted_features) # Average pool each channel. (14,14,2048) --> (1,1,2048)
        weighted_average = Flatten()(weighted_average)
        glimpse_list.append(weighted_average)
        
    return concatenate(glimpse_list)

def tile(x):
    return K.tile(x, [1,1,1,128])



def sensory_model_1(dim, win_len, channel,
                  num_feat_map,
                  num_hidden_lstm, 
                  output_dim,
                  dropout_rate):
    
    print ("Creating sensory model...")
    model = Sequential()
    
    # ??? kernel_size=(1, 5), or (1,3)    
    # default data_format for conv2d: channels_last (batch, rows, cols, channels)
    model.add(Convolution2D(num_feat_map, kernel_size=(1, 3),
                            activation='relu',
                            input_shape= (dim, win_len, channel),
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(dropout_rate))
    
    model.add(Convolution2D(num_feat_map, kernel_size=(1, 3), 
                            activation='relu',
                            padding='same'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(dropout_rate))

#     model.add(Flatten())
    model.add(Permute((2, 1, 3), name='Permute_1'))  # for swap-dimension
    model.add(Reshape((-1, num_feat_map * dim), name='Reshape_1'))
    
    model.add(LSTM(num_hidden_lstm, return_sequences=False, stateful=False, name='Lstm_1'))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(output_dim, activation='tanh'))
    model.add(Reshape((1, 1, 128)))  # adding reshape only for SAN model
    
    return model


def create_SAN_model(embedding_matrix, num_words, embedding_dim,  seq_length,
                        num_hidden_lstm, output_dim, dropout_rate,
                        sen_dim, sen_win_len, sen_channel,num_feat_map, 
                        num_classes,
                        num_glimpses=2, n=1
                    ):
    
    
    x_input = Input(shape=(sen_dim, sen_win_len, sen_channel), name='x_input')
    y_input = Input(shape=(seq_length,), name='y_input')
    
    
    lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, 
                              num_hidden_lstm,
                              output_dim,
                              dropout_rate)

    sen_model = sensory_model_1(sen_dim, sen_win_len, sen_channel,
                              num_feat_map,
                              num_hidden_lstm, 
                              output_dim,
                              dropout_rate)
    

    sen_result = sen_model(x_input)
    lstm_result = lstm_model(y_input)


    # merged = keras.layers.multiply([sen_result, lstm_result])

    # d1 = Dense(128, activation='tanh')(merged)
    # d2 = Dense(num_classes, activation='softmax')(d1)

    num_glimpses= num_glimpses #2
    n= n #1
    # def show_ask_attend_answer(vocab_size, num_glimpses=2, n=1):

    # Define network inputs where n is the feature rows and columns. In the
    # paper they use ResNet 152 res5c features with size (14x14x2048)

    # image_input = Input(shape=(n,n,128))
    # question_input = Input(shape=(15,))

    # # Learn word embeddings in relation to total vocabulary
    # question_embedding = Embedding(vocab_size, 300, input_length=15)(question_input)
    # question_embedding = Activation('tanh')(question_embedding)
    # question_embedding = Dropout(0.5)(question_embedding)

    # # LSTM to seuqentially embed word vectors into a single question vector
    # question_lstm = LSTM(1024)(question_embedding)

    image_input = sen_result
    question_lstm = lstm_result

    # Repeating and tiling question vector to match image input for concatenation
    question_tile = RepeatVector(n*n)(question_lstm)
    question_tile = Reshape((n,n,128))(question_tile)

    # Concatenation of question vector and image features
    concatenated_features1 = concatenate([image_input, question_tile])
    concatenated_features1 = Dropout(0.5)(concatenated_features1)

    # Stacked attention network
    attention_conv1 = Conv2D(512, (1,1))(concatenated_features1)
    attention_relu = Activation('relu')(attention_conv1)
    attention_relu = Dropout(0.5)(attention_relu)

    attention_conv2 = Conv2D(num_glimpses, (1,1))(attention_relu)
    attention_maps = Activation('softmax')(attention_conv2)

    # Weighted average of image features using attention maps
    image_attention = glimpse(attention_maps, image_input, num_glimpses, n)

    # Concatenation of question vector and attended image features
    concatenated_features2 = concatenate([image_attention, question_lstm])
    concatenated_features2 = Dropout(0.5)(concatenated_features2)

    # First fully connected layer with relu and dropout
    fc1 = Dense(1024)(concatenated_features2)
    fc1_relu = Activation('relu')(fc1)
    fc1_relu = Dropout(0.5)(fc1_relu)

    # Final fully connected layer with softmax to output answer probabilities
    fc2 = Dense(num_classes)(fc1_relu)
    fc2_softmax = Activation('softmax')(fc2)

    # Instantiate the model
    # vqa_model = Model(inputs=[image_input, question_input], outputs=fc2_softmax)

    # return vqa_model

    new_model = Model(inputs=[x_input, y_input], outputs=fc2_softmax)

    return new_model


