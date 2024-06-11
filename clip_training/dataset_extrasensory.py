import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from torch.nn.utils.rnn import pad_sequence
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler #, OneHotEncoder
#from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#from .utils import partial_labeling_by_category

def collate_fn(batch):
    batch_data, batch_labels, batch_text, batch_label_masks, batch_weight = zip(*batch)

    batch_data = pad_sequence([torch.FloatTensor(x) for x in batch_data], batch_first=True, padding_value=0)
    batch_labels = torch.FloatTensor(batch_labels)
    batch_label_masks = torch.LongTensor(batch_label_masks)
    batch_weight = torch.FloatTensor(batch_weight)
    #print('batch data shape!!', batch_data.shape)

    return batch_data, batch_labels, batch_text, batch_label_masks, batch_weight


def collate_fn_export(batch):
    batch_data, batch_labels, batch_text, batch_label_masks, batch_weight, batch_uuid, batch_timestamp = zip(*batch)

    batch_data = pad_sequence([torch.FloatTensor(x) for x in batch_data], batch_first=True, padding_value=0)
    batch_labels = torch.FloatTensor(batch_labels)
    batch_label_masks = torch.LongTensor(batch_label_masks)
    batch_weight = torch.FloatTensor(batch_weight)
    #print('batch data shape!!', batch_data.shape)

    return batch_data, batch_labels, batch_text, batch_label_masks, batch_weight, batch_uuid, batch_timestamp


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


def load_data(data_path, max_len=10, seed=4321, fold=0, 
              use_label_merge=False, test_only=False,
              augmentation=False):
    data = np.load(os.path.join(data_path, 'data.npz'), allow_pickle=True)
    X = np.array(data['X'])
    Y = np.array(data['Y'])
    M = np.array(data['M'])
    T = np.array(data['T'])
    U = np.array(data['U'])
    feature_names = data['feature_names']
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
        for i, (train_indices, test_indices) in enumerate(kf.split(range(len(X)))):
            if i == fold:
                X_train, X_test, Y_train, Y_test, M_train, M_test = X[train_indices], X[test_indices], Y[train_indices], Y[test_indices], M[train_indices], M[test_indices]
                break
        else:
            raise ValueError(f'fold {fold} is larger than 5. Set a smaller fold.')

        # Test dataset
        testData = MyDataset(X_test, Y_test, M_test, max_len=max_len, test_only=test_only,
                             feature_names=feature_names, target_names=target_names)

        # Split train and validation dataset
        X_train, X_val, Y_train, Y_val, M_train, M_val = train_test_split(
            X_train, Y_train, M_train, test_size=0.2, random_state=seed
        )

        # Train dataset
        trainData = MyDataset(X_train, Y_train, M_train, max_len=max_len,
                              test_only=test_only, 
                              feature_names=feature_names, 
                              target_names=target_names, 
                              preprocessor=testData.preprocessor)

        # Validation dataset
        valData = MyDataset(X_val, Y_val, M_val, max_len=max_len, test_only=test_only,
                            feature_names=feature_names, target_names=target_names, 
                            preprocessor=testData.preprocessor)
    
    #### Used in test_only: exporting predicted labels
    else:
        trainData = None
        valData = None
        testData = MyDataset(X, Y, M, max_len=max_len, U=U, T=T, feature_names=feature_names, target_names=target_names)

    return trainData, valData, testData, target_names


class MyDataset(Dataset):
    def __init__(self, X, Y, M, max_len, test_only=False, 
                 U=None, T=None, feature_names=None, target_names=None, 
                 preprocessor=None): # Remove categorical features and active targets
        self.data = []
        self.labels = []
        self.label_masks = []
        self.uuid = []
        self.timestamp = []
        self.test_only = test_only

        #### Used in normal training routines
        if U is None and T is None:
            for x, y, m in zip(X, Y, M):
                if len(x) > max_len:
                    start = 0
                    while start < len(x):
                        self.data.append(x[start:start + max_len])
                        self.labels.append(y)
                        self.label_masks.append(m)
                        start += max_len
                else:
                    self.data.append(x)
                    self.labels.append(y)
                    self.label_masks.append(m)

        #### Used in test_only: exporting predicted labels
        else:
            for x, y, m, u, t in zip(X, Y, M, U, T):
                if len(x) > max_len:
                    start = 0
                    while start < len(x):
                        self.data.append(x[start:start + max_len])
                        self.labels.append(y)
                        self.label_masks.append(m)
                        self.uuid.append(u)
                        self.timestamp.append(t[start:start + max_len])
                        start += max_len
                else:
                    self.data.append(x)
                    self.labels.append(y)
                    self.label_masks.append(m)
                    self.uuid.append(u)
                    self.timestamp.append(t)

        #if active_targets is None:
        #    self.active_targets = np.where(np.any(self.label_masks == 0, dim=0))[0]
        #else:
        #    self.active_targets = active_targets

        self.get_instance_weights()
        self.data_len = np.array([len(x) for x in self.data])
        self.feature_names = feature_names
        self.target_names = target_names
        #self.categorical_features = categorical_features if categorical_features is not None else []
        self.preprocessor = preprocessor
        self.preprocessing()
        self.generate_text()

        self.data = self.data
        self.labels = np.array(self.labels)
        self.text = np.array(self.text)
        self.label_masks = np.array(self.label_masks)
        self.instance_weights = np.array(self.instance_weights)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        text = self.text[idx]
        label_masks = self.label_masks[idx]
        instance_weights = self.instance_weights[idx]

        #### Used in normal training routines
        if not self.test_only:
            return data, labels, text, label_masks, instance_weights #, self.active_targets
        
        #### Used in test_only: exporting predicted labels
        else:
            uuid = self.uuid[idx]
            timestamp = self.timestamp[idx]
            return data, labels, text, label_masks, instance_weights, uuid, timestamp
            

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
        # (n_samples, seq_len, features)
        concat_data = []
        for x in self.data:
            concat_data.extend(x)

        concat_data = pd.DataFrame(concat_data, columns=self.feature_names)
        # Select index to train the inputer and scalar
        selected_indices = np.random.choice(range(len(concat_data)), max(int(0.1 * len(concat_data)), min(10000, len(concat_data))), replace=False)

        if self.preprocessor is None:
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
            self.preprocessor.fit(concat_data.loc[selected_indices])

        try:
            concat_data = self.preprocessor.transform(concat_data).todense()
        except:
            concat_data = self.preprocessor.transform(concat_data)

        processed_data = []
        start = 0
        for l in self.data_len:
            processed_data.append(concat_data[start:start + l])
            start += l

        self.data = processed_data

    def generate_text(self):
        """Rule based text generation"""
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

        #print(self.target_names)
        target_names = np.array([get_label_pretty_name(l) for l in self.target_names])

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
                new_labels = target_names[cur_cat_mask]

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

