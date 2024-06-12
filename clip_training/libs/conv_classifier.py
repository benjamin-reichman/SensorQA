import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from evaluation import plot_label_pairs, plot_confusion_matrix
import os
from tqdm import tqdm

class ConventionalClassifier(nn.Module):
    def __init__(self, data_encoder, emb_dim, out_dim):
        super(ConventionalClassifier, self).__init__()
        self.data_encoder = data_encoder
        self.classifier = nn.Linear(emb_dim, out_dim)

    def forward(self, x_data):
        # x_data: [src_len, batch_size, feature_dim]
        z_data = self.data_encoder(x_data)  # [batch_size, emb_dim]
        out = self.classifier(z_data)
        return out
    

def train_one_epoch(model, encoded_labels, optimizer, client_loader, epoch, 
                    use_label_encoder, device, logger):
    model.train()
    
    for sample, label, text, label_mask, weight in tqdm(client_loader, desc=f'Epoch {epoch}'):
        if use_label_encoder:
            # train data encoder
            for param in model.parameters():
                param.requires_grad = True
            for param in model.label_encoder.parameters():
                param.requires_grad = False
            loss = train_one_batch(model, sample, label, text, label_mask, weight,
                                    optimizer['data'], device, encoded_labels)
            logger.log_value('data_loss', loss, epoch)

            # train label encoder
            for param in model.parameters():
                param.requires_grad = True
            for param in model.data_encoder.parameters():
                param.requires_grad = False
            loss = train_one_batch(model, sample, label, text, label_mask, weight,
                                    optimizer['label'], device, encoded_labels)
            logger.log_value('label_loss', loss, epoch)
            
        else:
            loss = train_one_batch(model, sample, label, text, label_mask, weight,
                                    optimizer, device, encoded_labels)
            logger.log_value('loss', loss, epoch)


def train_one_batch(model, sample, label, text, mask, weight, optimizer, device, encoded_labels):
    sample = sample.to(device).transpose(0, 1)
    label = label.to(device, dtype=torch.float)
    weight = weight.to(device, dtype=torch.float)
    optimizer.zero_grad()
    # normalize_label is set to False in MLC framework, and True in SLC framework
    out = model(sample)
    # weight is used to control the mask
    loss = F.binary_cross_entropy_with_logits(out, label, weight=weight)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, test_loader, val_loader, encoded_labels, device, 
             target_names, save_dir, emb_dim):
    y_pred = []
    y_true = []
    y_mask = []

    model.eval()
    with torch.no_grad():
        for sample, label, text, label_mask, weight in test_loader:
            sample = sample.to(device).transpose(0, 1)
            label = label.to(device, dtype=torch.float)
            label_mask = label_mask.to(device, dtype=torch.long)
            # normalize_label is set to False in MLC framework, and True in SLC framework
            out = model(sample)
            # sigmoid in MLC framework, and softmax in SLC framework
            out = torch.sigmoid(out)

            y_pred.extend(out.cpu().numpy())
            y_true.extend(label.cpu().numpy())
            y_mask.extend(label_mask.cpu().numpy())

    y_true = np.array(y_true)
    plot_label_pairs(y_true, target_names, save_dir, 'label_pairs_true.png')
    y_pred = np.array(y_pred)
    plot_label_pairs(y_pred, target_names, save_dir, 'label_pairs_pred.png')
    y_mask = np.array(y_mask)

    plot_confusion_matrix(y_true, y_pred, y_mask, target_names, save_dir)

    return y_true, y_pred, y_mask


def evaluate_export(model, test_loader, encoded_labels, device, 
                    target_names, save_dir, thres):
    y_pred = []
    y_true = []
    y_mask = []

    export_pred_timestamps = {}
    export_pred_results = {}

    model.eval()
    with torch.no_grad():
        for sample, label, text, label_mask, _, weight, uuids, timestamps in test_loader:
            sample = sample.to(device).transpose(0, 1)
            label = label.to(device, dtype=torch.float)
            label_mask = label_mask.to(device, dtype=torch.long)
            # normalize_label is set to False in MLC framework, and True in SLC framework
            out = model(sample)
            # sigmoid in MLC framework, and softmax in SLC framework
            out = torch.sigmoid(out)

            y_pred.extend(out.cpu().numpy())
            y_true.extend(label.cpu().numpy())
            y_mask.extend(label_mask.cpu().numpy())

            # Add in export predicting results
            y_res = (out.cpu().numpy() > thres).astype(int)
            for (y, uuid, timestamp) in zip(y_res, uuids, timestamps):
                uuid = str(uuid).split('.')[0]
                timestamp = timestamp[0]  # because we set win_len=1 in export
                #print(uuid, timestamp, y)
                if uuid in export_pred_timestamps:
                    export_pred_timestamps[uuid].append(timestamp)
                    export_pred_results[uuid].append(y)
                else:
                    export_pred_timestamps[uuid] = [timestamp]
                    export_pred_results[uuid] = [y]


    y_true = np.array(y_true)
    plot_label_pairs(y_true, target_names, save_dir, 'label_pairs_true.png')
    y_pred = np.array(y_pred)
    plot_label_pairs(y_pred, target_names, save_dir, 'label_pairs_pred.png')
    y_mask = np.array(y_mask)

    plot_confusion_matrix(y_true, y_pred, y_mask, target_names, save_dir)

    # Save the predicted results
    path_dir = './predicted_results'
    os.makedirs(path_dir)
    for uuid in export_pred_timestamps:
        np.savez(os.path.join(path_dir, '{}.npz'.format(uuid)),
                 T=np.array(export_pred_timestamps[uuid]),
                 Y=np.array(export_pred_results[uuid])
                 )
        print(f'Save user {uuid} pred results of size {np.array(export_pred_results[uuid]).shape}')

    return y_true, y_pred, y_mask


#def validate(model, data_loader, encoded_labels, device, metrics, target_names):
#    scores = defaultdict(list)
#    # valData is the concatenation of all validation datasets with appropriate masks
#    y_true, y_pred, y_mask = evaluate(model, data_loader, encoded_labels, device)
#    for i in range(y_mask.shape[-1]):
#        results = calculate_metrics(y_true[:, i:i+1], y_pred[:, i:i+1], y_mask[:, i:i+1])
#        scores[i].append([results[m] for m in metrics])
#    class_scores = []
#    for i, tn in enumerate(target_names):
#        class_scores.append([np.nanmean(np.array(scores[i])[:, m]) for m in range(len(metrics))])

#    return class_scores