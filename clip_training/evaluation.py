import os
import csv
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt

MIN_FLOAT = 1e-12

def calculate_MLC_metrics(y_true, y_pred, y_mask, thres):
    support = defaultdict(int)
    for sample_y, sample_m in zip(y_true, y_mask):
        for cls, (y_val, m_val) in enumerate(zip(sample_y, sample_m)):
            support[cls] += int(m_val == 0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true = y_true == 1
    pred = y_pred > thres
    mask = np.array(y_mask, dtype=bool)

    class_results = dict({
        'F1': [],
        'ACC': [],
        'BA': [],
        'support': []
    })

    for i in range(len(y_true[0])):
        class_results['support'].append(support[i])
        valid_indices = np.where(mask[:, i] != 1)[0]
        if support[i] > 0:
            i_true = true[:, i][valid_indices]
            i_pred = pred[:, i][valid_indices]
            class_results['F1'].append(f1_score(y_true=i_true, y_pred=i_pred))
            class_results['ACC'].append(accuracy_score(y_true=i_true, y_pred=i_pred))
            class_results['BA'].append(balanced_accuracy_score(y_true=i_true, y_pred=i_pred))
        else:
            for metric in class_results:
                if metric != 'support':
                    class_results[metric].append(np.nan)

    class_results['support'] = np.array(class_results['support'])

    all_results = {
        'ACC': np.nanmean(class_results['ACC']),
        'ACC_pc': class_results['ACC'],
        'F1': np.nanmean(class_results['F1']),
        'F1_pc': class_results['F1'],
        'BA': np.nanmean(class_results['BA']),
        'BA_pc': class_results['BA'],
        'support': sum(support.values())
    }
    return all_results

def calculate_SLC_metrics(y_true, y_pred, y_mask):
    support = defaultdict(int)
    for sample_y, sample_m in zip(y_true, y_mask):
        for cls, (y_val, m_val) in enumerate(zip(sample_y, sample_m)):
            support[cls] += int(m_val == 0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true_bool = y_true.argmax(-1)
    else:
        y_true_bool = y_true

    if len(y_pred.shape) > 1:
        y_pred_bool = y_pred.argmax(-1)
    else:
        y_pred_bool = y_pred

    all_results = {
        'F1': f1_score(y_true_bool, y_pred_bool, average='macro'),
        'ACC': accuracy_score(y_true_bool, y_pred_bool)
    }

    return all_results

def display_results(results, metrics=['F1', 'ACC', 'BA']):
    print('{0:>20}'.format("") + ' '.join(['%10s']*len(metrics)) % tuple([m for m in metrics]))
    print('{0:>20}'.format("AVG") + ' '.join(['%10.4f'] * len(metrics)) % tuple([results[m] for m in metrics]))

    return [results[m] for m in metrics]

def log_results(results, save_dir, target_names, metrics=['F1', 'ACC', 'BA']):
    metrics_pc = ['{}_pc'.format(m) for m in metrics]
    with open(os.path.join(save_dir, 'pc_results.csv'), 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['class'] + metrics_pc)
        
        for i in range(len(target_names)):
            csvwriter.writerow([target_names[i], 
                                round(results['F1_pc'][i], 2), 
                                round(results['ACC_pc'][i], 2),
                                round(results['BA_pc'][i], 2)])

        csvwriter.writerow(['AVG', results['F1'], results['ACC'], results['BA']])

    return [results[m] for m in metrics]


def jaccard_similarity_for_label_pairs(Y):
    (n_examples,n_labels) = Y.shape;
    Y = Y.astype(int);
    # For each label pair, count cases of:
    # Intersection (co-occurrences) - cases when both labels apply:
    both_labels_counts = np.dot(Y.T,Y);
    # Cases where neither of the two labels applies:
    neither_label_counts = np.dot((1-Y).T,(1-Y));
    # Union - cases where either of the two labels (or both) applies (this is complement of the 'neither' cases):
    either_label_counts = n_examples - neither_label_counts;
    # Jaccard similarity - intersection over union:
    J = np.where(either_label_counts > 0, both_labels_counts.astype(float) / either_label_counts, 0.);
    return J;


def get_label_pretty_name(label):
    if label == 'FIX_walking':
        return 'Walking';
    if label == 'FIX_running':
        return 'Running';
    if label == 'LOC_main_workplace':
        return 'At main workplace';
    if label == 'OR_indoors':
        return 'Indoors';
    if label == 'OR_outside':
        return 'Outside';
    if label == 'LOC_home':
        return 'At home';
    if label == 'FIX_restaurant':
        return 'At a restaurant';
    if label == 'OR_exercise':
        return 'Exercise';
    if label == 'LOC_beach':
        return 'At the beach';
    if label == 'OR_standing':
        return 'Standing';
    if label == 'WATCHING_TV':
        return 'Watching TV'
    
    if label.endswith('_'):
        label = label[:-1] + ')';
        pass;
    
    label = label.replace('__',' (').replace('_',' ');
    label = label[0] + label[1:].lower();
    label = label.replace('i m','I\'m');
    return label;


def plot_label_pairs(Y, target_names, save_dir, filename):
    # Plot the label co-occurrence as shown in the official script of ExtraSensory

    J = jaccard_similarity_for_label_pairs(Y);

    fig = plt.figure(figsize=(10,10),facecolor='white');
    ax = plt.subplot(1,1,1);
    plt.imshow(J,interpolation='none');
    plt.colorbar();

    pretty_label_names = [get_label_pretty_name(label) for label in target_names];
    n_labels = len(target_names);
    ax.set_xticks(range(n_labels));
    ax.set_xticklabels(pretty_label_names,rotation=45,ha='right',fontsize=7);
    ax.set_yticks(range(n_labels));
    ax.set_yticklabels(pretty_label_names,fontsize=7);

    plt.savefig(os.path.join(save_dir, filename), dpi=100)


def plot_confusion_matrix(y_true, y_pred, y_mask, target_names, save_dir):
    # This is not the normal confusion matrix since ExtraSensory is a multi-label dataset
    # We plot the matrix where the model is "confused"
    # Suppose we consider two labels, walking (w) and eating (e)
    # Then we will plot the matrix of 
    #     P(y_pred = w | y_true != w, y_true = e) + P(y_pred != w | y_true = w, y_true = e)
    # Basically we are plotting the sum of false positive and false negative when y_true=e
    # So we can get an idea which label pair the model is confused at

    # Extract the context label names
    pretty_label_names = [get_label_pretty_name(label) for label in target_names];

    support = defaultdict(int)
    for sample_y, sample_m in zip(y_true, y_mask):
        for cls, (y_val, m_val) in enumerate(zip(sample_y, sample_m)):
            support[cls] += int(m_val == 0)  # Number of samples that is not missing

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true = y_true == 1
    pred = y_pred > 0.5
    mask = np.array(y_mask)

    num_of_class = y_true.shape[1]
    cm = np.zeros((num_of_class, num_of_class))
    fp = np.zeros((num_of_class, num_of_class))
    fn = np.zeros((num_of_class, num_of_class))

    for i in range(len(y_true[0])):  # i: class index
        valid_indices = np.where(mask[:, i] != 1)[0]
        true_i_indices = np.where(np.logical_and(~mask[:, i], true[:, i]))[0]
        #print('# of valid indices of class {}: {}, '
        #      'true indices: {}, ratio: {}'.format(pretty_label_names[i], 
        #                                           len(valid_indices), len(true_i_indices),
        #                                           len(true_i_indices) / len(valid_indices)))

        for j in range(len(y_true[0])):  # j: class index, j = i is not allowed
            if j == i:
                continue
            if support[i] < 0 or support[j] < 0:
                cm[i, j] = fp[i, j] = fn[i, j] = np.nan
            else:
                j_true_cond_i = true[:, j][true_i_indices]
                j_pred_cond_i = pred[:, j][true_i_indices]

                # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
                FP = np.sum(np.logical_and(j_pred_cond_i == 1, j_true_cond_i == 0))
                
                # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
                FN = np.sum(np.logical_and(j_pred_cond_i == 0, j_true_cond_i == 1))

                # Compute our "confusion matrix"
                cm[i, j] = (FP + FN) / len(true_i_indices)
                fp[i, j] = FP / len(true_i_indices)
                fn[i, j] = FN / len(true_i_indices)
                #print(i, j, FP, FN, cm[i, j])

    # print(cm)
    
    # Plot - cm
    fig = plt.figure(figsize=(12,10),facecolor='white');
    ax = plt.subplot(1,1,1);
    plt.imshow(cm,interpolation='none');
    plt.colorbar();
    n_labels = len(target_names);
    ax.set_xticks(range(n_labels));
    ax.set_xticklabels(pretty_label_names,rotation=45,ha='right',fontsize=7);
    ax.set_xlabel('w')
    ax.set_yticks(range(n_labels));
    ax.set_yticklabels(pretty_label_names,fontsize=7);
    ax.set_ylabel('e')
    ax.set_title('P(y_pred = w | y_true != w, y_true = e) + P(y_pred != w | y_true = w, y_true = e)')
    plt.savefig(os.path.join(save_dir, 'cm.png'), dpi=100)

    # Plot - FP
    fig = plt.figure(figsize=(14,10),facecolor='white');
    ax = plt.subplot(1,1,1);
    plt.imshow(fp,interpolation='none');
    plt.colorbar();
    n_labels = len(target_names);
    ax.set_xticks(range(n_labels));
    ax.set_xticklabels(pretty_label_names,rotation=45,ha='right',fontsize=7);
    ax.set_xlabel('w')
    ax.set_yticks(range(n_labels));
    ax.set_yticklabels(pretty_label_names,fontsize=7);
    ax.set_ylabel('e')
    ax.set_title('P(y_pred = w | y_true != w, y_true = e)')
    plt.savefig(os.path.join(save_dir, 'fp.png'), dpi=100)

    # Plot - FN
    fig = plt.figure(figsize=(14,10),facecolor='white');
    ax = plt.subplot(1,1,1);
    plt.imshow(fn,interpolation='none');
    plt.colorbar();
    n_labels = len(target_names);
    ax.set_xticks(range(n_labels));
    ax.set_xticklabels(pretty_label_names,rotation=45,ha='right',fontsize=7);
    ax.set_xlabel('w')
    ax.set_yticks(range(n_labels));
    ax.set_yticklabels(pretty_label_names,fontsize=7);
    ax.set_ylabel('e')
    ax.set_title('P(y_pred != w | y_true = w, y_true = e)')
    plt.savefig(os.path.join(save_dir, 'fn.png'), dpi=100)