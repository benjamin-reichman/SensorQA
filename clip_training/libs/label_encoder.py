import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from evaluation import plot_label_pairs, plot_confusion_matrix
from tqdm import tqdm

class LabelEncoderClassifier(nn.Module):
    def __init__(self, data_encoder, emb_dim, out_dim, pretrained_embedding=None, 
                 pretrained_emb_dim=512, device='cpu'):
        super(LabelEncoderClassifier, self).__init__()
        self.data_encoder = data_encoder
        self.device = device

        if pretrained_embedding is not None:
            pretrained_emb_dim = pretrained_embedding.size(1)
            if pretrained_emb_dim != emb_dim:
                self.label_encoder = nn.Sequential(
                    nn.Embedding.from_pretrained(pretrained_embedding, freeze=False),
                    nn.Linear(pretrained_emb_dim, emb_dim)
                )
            else:
                self.label_encoder = nn.Embedding.from_pretrained(pretrained_embedding, freeze=False)
        else:
            self.label_encoder = nn.Embedding(out_dim, emb_dim)


    def forward(self, x_data, encoded_labels, normalize_label=False):
        z_data = self.data_encoder(x_data)  # [batch_size, emb_dim]
        z_label = self.label_encoder(encoded_labels) # [n_class, emb_dim]
        #z_label = self.label_encoder(self.text_features)  # (51, 256)

        batch_size = z_data.size(0)
        n_classes = z_label.size(0)

        z_data = z_data.unsqueeze(1) # [batch_size, 1, emb_dim]
        z_label = z_label.repeat(batch_size, 1, 1) # [batch_size, n_classes, emb_dim]

        # normalization
        if normalize_label:
            z_label = z_label / z_label.norm(p=2, dim=-1, keepdim=True)
        else:
            z_label = z_label

        out = torch.mul(z_data, z_label).sum(-1, keepdims=True).reshape((-1, n_classes))

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


def train_one_batch(model, sample, label, text, mask, weight, optimizer, device, 
                    encoded_labels):
    sample = sample.to(device).transpose(0, 1)
    label = label.to(device, dtype=torch.float)
    weight = weight.to(device, dtype=torch.float)
    optimizer.zero_grad()
    # normalize_label is set to False in MLC framework, and True in SLC framework
    out = model(sample, encoded_labels, normalize_label=False)
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
            out = model(sample, encoded_labels, normalize_label=False)
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
