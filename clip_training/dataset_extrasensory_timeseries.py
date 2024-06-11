import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import time

from sklearn.model_selection import train_test_split, KFold
from torch.nn.utils.rnn import pad_sequence
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler #, OneHotEncoder
#from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#from .utils import partial_labeling_by_category

# Dimension = 68 in total
DISCRETE_INDICES = np.arange(129, 155) # Watch compass and location
DISCRETE_INDICES = np.concatenate((DISCRETE_INDICES, np.arange(183, 225))) # All discrete

def load_json(json_path: str):
    """
    Load a json file
    """
    with open(json_path, "r", encoding="utf-8") as f_name:
        data = json.load(f_name)
    return data


def collate_fn_timeseries(batch):
    batch_data, batch_disc_data, batch_labels, batch_text, \
        batch_label_masks, batch_weight = zip(*batch)

    batch_data = torch.FloatTensor(batch_data) # should be the same size
    batch_disc_data = torch.FloatTensor(batch_disc_data)
    # pad_sequence([torch.FloatTensor(x) for x in batch_data], batch_first=True, padding_value=0)
    batch_labels = torch.FloatTensor(batch_labels)
    batch_label_masks = torch.LongTensor(batch_label_masks)
    batch_weight = torch.FloatTensor(batch_weight)
    #print('batch data shape!!', batch_data.shape)

    return batch_data, batch_disc_data, batch_labels, batch_text, batch_label_masks, batch_weight


def collate_fn_export_timeseries(batch):
    batch_data, batch_disc_data, batch_labels, batch_text, \
        batch_label_masks, batch_weight, batch_uuid, batch_timestamp = zip(*batch)

    batch_data = torch.FloatTensor(batch_data) # should be the same size
    batch_disc_data = torch.FloatTensor(batch_disc_data)
    # pad_sequence([torch.FloatTensor(x) for x in batch_data], batch_first=True, padding_value=0)
    batch_labels = torch.FloatTensor(batch_labels)
    batch_label_masks = torch.LongTensor(batch_label_masks)
    batch_weight = torch.FloatTensor(batch_weight)
    #batch_targets = torch.LongTensor(batch_targets)
    #print('batch data shape!!', batch_data.shape)

    return batch_data, batch_disc_data, batch_labels, batch_text, batch_label_masks, batch_weight, batch_uuid, batch_timestamp


def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'walking';
    if label == 'FIX_running':
        return 'running';
    if label == 'LOC_main_workplace':
        return 'at main workplace';
    if label == 'OR_indoors':
        return 'indoors';
    if label == 'OR_outside':
        return 'outside';
    if label == 'LOC_home':
        return 'at home';
    if label == 'FIX_restaurant':
        return 'at a restaurant';
    if label == 'OR_exercise':
        return 'exercising';
    if label == 'LOC_beach':
        return 'at the beach';
    if label == 'OR_standing':
        return 'standing';
    if label == 'WATCHING_TV':
        return 'watching TV'
    if label == 'DRINKING__ALCOHOL_':
        return 'drinking alcohol'
    if label == 'BATHING_-_SHOWER':
        return 'bathing or showing'
    if label == 'TOILET':
        return 'in toilet'
    if label == 'LAB_WORK':
        return 'doing lab work'
    if label == 'COMPUTER_WORK':
        return 'doing computer work'
    
    if label.endswith('_'):
        label = label[:-1] + ')';
        pass;
    
    label = label.replace('__',' (').replace('_',' ');
    label = label.lower();
    label = label.replace('i m','I\'m');

    # get rid of phone
    #if 'phone' in label:
    #    label = label.replace('phone ', '')

    return label;


def add_word_after_first(sentence, word):
    words = sentence.split()
    # Insert the additional word after the first word
    words.insert(1, word)
    # Join the words back into a sentence
    new_sentence = ' '.join(words)
    return new_sentence


def label_merge(label_names):
    # The label merge is determined by the "confusion matrix"
    # Generate the label transformation matrix
    new_label_names = [
        'SITTING', 
        'LYING_DOWN',
        'OR_standing',
        'FIX_walking',
        'BICYCLING',
        #'FIX_running',
        'PHONE_ON_TABLE',
        'PHONE_IN_POCKET',
        'PHONE_IN_HAND',
        'PHONE_IN_BAG',
        'WITH_CO-WORKERS',
        'WITH_FRIENDS',
        'OR_outside',
        'OR_indoors',
        'AT_SCHOOL',
        #'AT_A_PARTY',
        #'AT_THE_GYM',
        #'AT_A_BAR',
        #'LOC_beach',
        'LOC_home',
        'LOC_main_workplace',
        #'FIX_restaurant',
        'IN_CLASS',
        'IN_A_MEETING',
        #'ON_A_BUS',
        #'IN_A_CAR',
        #'ELEVATOR',
        'TOILET',

        'COOKING',
        #'CLEANING',
        #'DOING_LAUNDRY',
        #'WASHING_DISHES',
        #'GROOMING',
        #'DRESSING',
        'SLEEPING',
        'EATING',
        'BATHING_-_SHOWER',
        'LAB_WORK',
        'COMPUTER_WORK',
        'SURFING_THE_INTERNET',
        'OR_exercise',
        #'DRIVE_-_I_M_THE_DRIVER',
        #'DRIVE_-_I_M_A_PASSENGER',
        #'SHOPPING',
        'TALKING',
        'WATCHING_TV',
        #'DRINKING__ALCOHOL_',
        #'SINGING',
        #'STROLLING',
        #'STAIRS_-_GOING_UP',
        #'STAIRS_-_GOING_DOWN'

        # New labels after merge
        'On a vehicle',
        'CLEANING',
        'GROOMING',
        #'Walking up or down stairs'
    ]

    label_transfer = {
        'IN_A_CAR': 'On a vehicle',
        'ON_A_BUS': 'On a vehicle',
        'DRIVE_-_I_M_THE_DRIVER': 'On a vehicle',
        'DRIVE_-_I_M_A_PASSENGER': 'On a vehicle',

        #'DOING_LAUNDRY': 'CLEANING',
        #'WASHING_DISHES': 'CLEANING',

        'DRESSING': 'GROOMING',

        #'STAIRS_-_GOING_UP': 'Walking up or down stairs',
        #'STAIRS_-_GOING_DOWN': 'Walking up or down stairs',
        #'ELEVATOR': 'Walking up or down stairs'
    }

    print('Label merge: {}->{}'.format(len(label_names), len(new_label_names)))
    label_transform = np.zeros((len(label_names), len(new_label_names)))

    # Configure the transform matrix
    for nl in label_names:
        if nl in new_label_names:
            original_idx = np.where(label_names == nl)[0][0]
            new_idx = new_label_names.index(nl)
            print('{}->{}\t{}->{}'.format(nl, nl, original_idx, new_idx))
            label_transform[original_idx, new_idx] = 1
        elif nl in label_transfer:
            original_idx = np.where(label_names == nl)[0][0]
            new_idx = new_label_names.index(label_transfer[nl])
            print('{}->{}\t{}->{}'.format(nl, label_transfer[nl], original_idx, new_idx))
            label_transform[original_idx, new_idx] = 1

    #print(label_transform)

    return np.array(new_label_names), label_transform


def load_data_timeseries(data_path, sample_len=800, seed=4321, fold=0, 
                         use_label_merge=False, test_only=False,
                         augmentation=False):
    data = np.load(os.path.join(data_path, 'data_timeseries_w_text.npz'), allow_pickle=True)

    X = np.array(data['X'])[:, DISCRETE_INDICES]
    print('shape of X after loaded: ', X.shape)
    Y = np.array(data['Y'])
    M = np.array(data['M'])
    T = np.array(data['T'])
    U = np.array(data['U'])
    text = data['text']
    aug_text = data['aug_text']
    #feature_names = data['feature_names']
    try:
        target_names = np.array([ln.decode() for ln in data['label_names']])
    except:
        target_names = data['label_names']

    #if 'categorical_features' in data:
    #    categorical_features = data['categorical_features']
    #else:
    #    categorical_features = None

    # merge labels if applicable
    if use_label_merge:
        target_names, label_transform = label_merge(target_names)
        Y = (np.matmul(Y, label_transform) > .0).astype(np.int)
        M = (np.matmul(M, label_transform) > .0).astype(np.int)

    target_names = np.array([l for l in target_names]) # get_label_pretty_name(l)
    print(target_names)

    # count labels
    label_count = np.sum(Y, axis=0)
    print('label_count: ', label_count)
    imbalance_factor = np.max(label_count) / np.min(label_count)
    print('label_count.shape', label_count.shape, 'imbalance', imbalance_factor)

    # remove classes with less than 10 samples
    preserve_idx = []
    tn_count = {}
    for i, ln in enumerate(target_names):
        tn_count[ln] = (Y[:, i] == 1).sum()
        if (Y[:, i] == 1).sum() > 10:
            preserve_idx.append(i)
    Y = Y[:, preserve_idx]
    M = M[:, preserve_idx]
    target_names = target_names[preserve_idx]
    
    # filter labels in some cases - NOT USED
    #filter_labels = list(filter(lambda l: targetname2category[l] == 'phone', target_names))
    #preserve_idx = [np.where(target_names == l)[0][0] for l in filter_labels]
    #Y = Y[:, preserve_idx]
    #M = M[:, preserve_idx]
    #target_names = target_names[preserve_idx]

    print('loaded X and Y')

    #### Used in normal training routines
    if not test_only:
        kf = KFold(n_splits=5, random_state=seed, shuffle=True)
        for i, (train_indices, test_indices) in enumerate(kf.split(range(len(Y)))):
            if i == fold:
                X_train, X_test = X[train_indices], X[test_indices]
                U_train, U_test = U[train_indices], U[test_indices] 
                T_train, T_test = T[train_indices], T[test_indices]
                Y_train, Y_test = Y[train_indices], Y[test_indices]
                M_train, M_test = M[train_indices], M[test_indices]
                text_train, text_test = text[train_indices], text[test_indices]
                aug_text_train, aug_text_test = aug_text[train_indices], aug_text[test_indices]
                break
        else:
            raise ValueError(f'fold {fold} is larger than 5. Set a smaller fold.')

        # Test dataset
        testData = MyDataset(X_test, U_test, T_test, Y_test, M_test, 
                             text_test, aug_text_test,
                             data_path, sample_len,
                             test_only=test_only,
                             target_names=target_names)
        subset_len = int(len(testData) * 0.02)
        testData, _ = torch.utils.data.random_split(dataset=testData,
                                                    lengths=[subset_len,
                                                             len(testData) - subset_len])

        # Split train and validation dataset
        X_train, X_val, U_train, U_val, T_train, T_val, Y_train, Y_val, M_train, M_val, \
            text_train, text_val, aug_text_train, aug_text_val = train_test_split(
            X_train, U_train, T_train, Y_train, M_train, text_train, aug_text_train, 
            test_size=0.1, random_state=seed
        )

        # Train dataset
        trainData = MyDataset(X_train, U_train, T_train, Y_train, M_train, 
                              text_train, aug_text_train,
                              data_path, sample_len, 
                              test_only=test_only,
                              target_names=target_names, 
                              preprocessor=testData.dataset.preprocessor)
        subset_len = int(len(trainData))
        trainData, _ = torch.utils.data.random_split(dataset=trainData,
                                                    lengths=[subset_len,
                                                             len(trainData) - subset_len])

        # Validation dataset
        valData = MyDataset(X_val, U_val, T_val, Y_val, M_val, 
                            text_val, aug_text_val,
                            data_path, sample_len, 
                            test_only=test_only,
                            target_names=target_names, 
                            preprocessor=testData.dataset.preprocessor)
        subset_len = int(len(testData) * 0.2)
        valData, _ = torch.utils.data.random_split(dataset=valData,
                                                    lengths=[subset_len,
                                                             len(valData) - subset_len])
    
    #### Used in test_only: exporting predicted labels
    else:
        trainData = None
        valData = None
        testData = MyDataset(X, U, T, Y, M, text, aug_text, 
                             data_path, sample_len, target_names=target_names)

    return trainData, valData, testData, target_names


class MyDataset(Dataset):
    def __init__(self, X, U, T, Y, M, text, aug_text, data_path, sample_len, test_only=False,
                 target_names=None, preprocessor=None): # Remove categorical features and active targets
        self.discrete_data = X
        self.uuid = U
        self.timestamps = T
        self.labels = Y
        self.label_masks = M
        self.text = text
        self.aug_text = aug_text
        self.data_path = data_path
        self.sample_len = sample_len
        self.test_only = test_only

        #if active_targets is None:
        #    self.active_targets = np.where(np.any(self.label_masks == 0, dim=0))[0]
        #else:
        #    self.active_targets = active_targets

        self.get_instance_weights()
        self.target_names = target_names
        #self.categorical_features = categorical_features if categorical_features is not None else []
        self.preprocessor = preprocessor
        self.preprocessing()
        #self.generate_text() # Not needed as we load text from file

    def load_raw_data(self, user_id, timestamp):
        filename = str(user_id) + '/' + str(timestamp) + '.npz'
        data = np.load(os.path.join(self.data_path, filename), allow_pickle=True)
        return data['X']

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        disc_data = self.discrete_data[idx]
        user_id = self.uuid[idx]
        timestamp = self.timestamps[idx]
        labels = self.labels[idx]
        text = self.text[idx]
        label_masks = self.label_masks[idx]
        instance_weights = self.instance_weights[idx]

        # Load data and preprocess
        data = self.load_raw_data(user_id, timestamp)
        orig_data_len = data.size
        try:
            new_data = self.preprocessor.transform(data.reshape(1, -1)).todense()
        except:
            new_data = self.preprocessor.transform(data.reshape(1, -1))

        # Because SimpleImputer will remove some all-nan features, we need to pad some zeros
        # to keep the original data shape of (25, 800) or (25, 500)
        padding_length = orig_data_len - new_data.size
        new_data = np.concatenate([new_data, np.zeros((1, padding_length))], axis=1)
        new_data = new_data.reshape((-1, self.sample_len))

        #### Used in normal training routines
        if not self.test_only:
            return new_data, disc_data, labels, text, label_masks, instance_weights #, self.active_targets
        
        #### Used in test_only: exporting predicted labels
        else:
            return new_data, disc_data, labels, text, label_masks, instance_weights, user_id, timestamp
            

    def get_instance_weights(self):
        n_classes = len(self.labels[0])
        # Count each class frequency (pos/neg) for each label
        pos_count = np.ones((n_classes))  # avoid nan
        neg_count = np.ones((n_classes))
        for example_y, example_m in zip(self.labels, self.label_masks):
            for i, (y, m) in enumerate(zip(example_y, example_m)):
                if m == 1:
                    continue
                if y == 1:
                    pos_count[i] += 1
                elif y == 0:
                    neg_count[i] += 1
        self.num_samples = pos_count - 1
        self.pos_weight = neg_count / (pos_count + neg_count)
        self.neg_weight = pos_count / (pos_count + neg_count)

        self.instance_weights = []
        for y, m in zip(self.labels, self.label_masks):
            weight = (y * self.pos_weight + (1 - y) * self.neg_weight) * (1 - m)
            self.instance_weights.append(weight)


    def preprocessing(self):
        print('Preprocessing start...')
        start = time.time()

        if self.preprocessor is None:
            # Select index to train the inputer and scalar
            total_len = self.labels.shape[0]
            selected_indices = np.random.choice(range(total_len), max(int(0.1 * total_len), min(5000, total_len)), replace=False)

            # Load data
            concat_data = []
            for idx in selected_indices:
                data = self.load_raw_data(self.uuid[idx], self.timestamps[idx])
                concat_data.append(data.reshape(-1))
            concat_data = np.stack(concat_data, axis=0)
            # print('concat data size:', concat_data.shape)  # (sample num, 25*800)

            #categorical_transformer = Pipeline(steps=[
            #    ("imputer", SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
            #    ("encoder", OneHotEncoder(handle_unknown="ignore"))
            #])

            #numeric_indices = [i for i, ln in enumerate(self.feature_names) if ln not in self.categorical_features]
            #categorical_indices = [i for i, ln in enumerate(self.feature_names) if ln in self.#categorical_features]

            self.preprocessor = Pipeline(steps=[
                ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
                ("scaler", StandardScaler())
            ])

            #self.preprocessor = ColumnTransformer(
            #    transformers=[
            #        ("num", numeric_transformer, numeric_indices),
            #        ("cat", categorical_transformer, categorical_indices),
            #    ]
            #)
            self.preprocessor.fit(concat_data)

            # Save the mean and standard deviation
            #simple = self.preprocessor.named_steps["imputer"]
            #scaler = self.preprocessor.named_steps['scaler']
            #print("Simple features", simple.n_features_in_)
            #print("Mean:", scaler.mean_.shape)
            #print("Standard Deviation:", scaler.scale_.shape)
        
        print(f'Preprocessing done in {time.time()-start} secs')


    def generate_text(self):
        """Rule based text generation"""
        print('Generate text start...')
        start = time.time()

        targetname2category = {
            'SITTING': 'posture',
            'LYING_DOWN': 'posture',
            'OR_standing': 'posture',
            'FIX_walking': 'posture',
            'BICYCLING': 'posture',
            'FIX_running': 'posture',

            'PHONE_ON_TABLE': 'phone',
            'PHONE_IN_POCKET': 'phone',
            'PHONE_IN_HAND': 'phone',
            'PHONE_IN_BAG': 'phone',

            'WITH_CO-WORKERS': 'accompany',
            'WITH_FRIENDS': 'accompany',

            'OR_outside': 'environment',
            'OR_indoors': 'environment',
            'AT_SCHOOL': 'environment',
            'AT_A_PARTY': 'environment',
            'AT_THE_GYM': 'environment',
            'AT_A_BAR': 'environment',
            'LOC_beach': 'environment',
            'LOC_home': 'environment',
            'LOC_main_workplace': 'environment',
            'FIX_restaurant': 'environment',
            'IN_CLASS': 'environment',
            'IN_A_MEETING': 'environment',
            'ON_A_BUS': 'environment',
            'IN_A_CAR': 'environment',
            'ELEVATOR': 'environment',
            'TOILET': 'environment',

            'COOKING': 'activity',
            'CLEANING': 'activity',
            'DOING_LAUNDRY': 'activity',
            'WASHING_DISHES': 'activity',
            'GROOMING': 'activity',
            'DRESSING': 'activity',
            'SLEEPING': 'activity',
            'EATING': 'activity',
            'BATHING_-_SHOWER': 'activity',
            'LAB_WORK': 'activity',
            'COMPUTER_WORK': 'activity',
            'SURFING_THE_INTERNET': 'activity',
            'OR_exercise': 'activity',
            'DRIVE_-_I_M_THE_DRIVER': 'activity',
            'DRIVE_-_I_M_A_PASSENGER': 'activity',
            'SHOPPING': 'activity',
            'TALKING': 'activity',
            'WATCHING_TV': 'activity',
            'DRINKING__ALCOHOL_': 'activity',
            'SINGING': 'activity',
            'STROLLING': 'activity',
            'STAIRS_-_GOING_UP': 'activity',
            'STAIRS_-_GOING_DOWN': 'activity',
            'On a vehicle': 'activity'
        }

        pretty_target_names = np.array([get_label_pretty_name(l) for l in self.target_names])

        masks = {}
        for cat in ['posture', 'activity', 'environment', 'accompany', 'phone']:
            new_mask = [1 if targetname2category[label] == cat else 0 for label in self.target_names]
            #print(cat, new_mask)
            masks[cat] = np.array(new_mask, dtype=bool)

        self.text = []
        for example_y, example_m in zip(self.labels, self.label_masks):
            cur_label_mask = np.logical_and(example_y.astype(bool), ~(example_m.astype(bool)))

            new_text = 'The person is'
            
            for cat in ['posture', 'activity', 'environment', 'accompany', 'phone']:
                cur_cat_mask = np.logical_and(cur_label_mask, masks[cat])
                new_labels = pretty_target_names[cur_cat_mask]

                if len(new_labels) > 0: # has something in this category
                    new_labels = ' and '.join(new_labels)

                    if cat == 'posture':
                        new_text += ' ' + new_labels
                    elif cat == 'activity':
                        new_text += ' and ' + new_labels
                    elif cat == 'environment':
                        new_text += ' ' + new_labels
                    elif cat == 'accompany':
                        new_text += ' ' + new_labels
                    elif cat == 'phone':
                        new_text += ' while the ' + add_word_after_first(new_labels, 'is')
        

            self.text.append(new_text)

        print(f'Generate text done in {time.time()-start} secs')