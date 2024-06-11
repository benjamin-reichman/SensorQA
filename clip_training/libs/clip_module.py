import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from evaluation import plot_label_pairs, plot_confusion_matrix
import clip
from libs.loss import InfoNCE
import torch.optim as optim
import pytorch_lightning as pl
import os
from tqdm import tqdm

classifier_param = {
    'lr': 0.0001,
    'epochs': 50
}

class LinearClassifier(pl.LightningModule):
    def __init__(self, data_encoder, head, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.data_encoder = data_encoder
        self.head = head
        self.freeze_data_encoder()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x_data, disc_data):
        z_data = self.data_encoder(x_data)  # [batch_size, emb_dim]
        # Add disc data
        concat_data = torch.cat((z_data, disc_data), dim=-1)
        concat_data = self.head(concat_data)
        return self.linear(concat_data)
    
    def training_step(self, batch, batch_idx):
        sample, disc_data, label, _, _, weight = batch
        out = self(sample.transpose(0, 1), disc_data)
        loss = F.binary_cross_entropy_with_logits(out, label, weight=weight)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=classifier_param['lr'])

    def freeze_data_encoder(self):
        for param in self.data_encoder.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = False
    

class ClipEncoderClassifier(nn.Module):
    def __init__(self, data_encoder, emb_dim, disc_dim, head='mlp', device='cpu'):
        super(ClipEncoderClassifier, self).__init__()
        self.data_encoder = data_encoder
        self.device = device
        self.label_encoder, self.preprocess = clip.load("ViT-B/32", device=device)
        self.label_encoder.visual.transformer = None  # get rid of the visual part
        # Convert the parameters to float32
        for param in self.label_encoder.parameters():
            param.data = param.data.float()
        self.label_encoder = self.label_encoder.to(self.device)

        # A linear head to add the disc_data
        if head == 'linear':
            self.head = nn.Linear(emb_dim + disc_dim, emb_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(emb_dim + disc_dim, emb_dim + disc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(emb_dim + disc_dim, emb_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        if emb_dim != 512: # the emb dim of clip model is 512 which cannot be changed
            self.text_linear = nn.Linear(512, emb_dim)
        else:
            self.text_linear = None

        self.loss = InfoNCE(symmetric_loss=False, learn_temperature=True)

    def forward(self, x_data, disc_data, y_text):
        # Compute sensor features
        z_data = self.data_encoder(x_data)  # [batch_size, emb_dim]
        # Add disc data
        concat_data = torch.cat((z_data, disc_data), dim=-1)
        concat_data = self.head(concat_data)

        # Compute text features
        text_tokens = clip.tokenize(y_text).to(self.device)
        z_text = self.label_encoder.encode_text(text_tokens)  # [batch_size, 512]
        if self.text_linear:
            z_text = self.text_linear(z_text)  # [batch_size, emb_dim]

        # Note: no need to normalize here because self.loss has normalization
        loss = self.loss(query=z_text, positive_key=concat_data)

        return loss
    
    def get_sensor_embeddings(self, x_data, disc_data):
        # Compute sensor features
        z_data = self.data_encoder(x_data)  # [batch_size, emb_dim]
        # Add disc data
        concat_data = torch.cat((z_data, disc_data), dim=-1)
        concat_data = self.head(concat_data)
        return concat_data
    
    def get_text_embeddings(self, y_text):
        text_tokens = clip.tokenize(y_text).to(self.device)
        z_text = self.label_encoder.encode_text(text_tokens)  # [batch_size, 512]
        if self.linear:
            z_text = self.text_linear(z_text)  # [batch_size, emb_dim]
        return z_text


def train_one_epoch(model, encoded_labels, optimizer, client_loader, epoch, 
                    use_label_encoder, device):
    model.train()

    for sample, disc_data, label, text, label_mask, weight in tqdm(client_loader, desc=f'Epoch {epoch}'):
        if use_label_encoder:
            # train data encoder
            for param in model.parameters():
                param.requires_grad = True
            for param in model.label_encoder.parameters():
                param.requires_grad = False
            data_loss = train_one_batch(model, sample, disc_data, label, text, label_mask, 
                                        weight, optimizer['data'], device)

            # train label encoder
            for param in model.parameters():
                param.requires_grad = True
            for param in model.data_encoder.parameters():
                param.requires_grad = False
            label_loss = train_one_batch(model, sample, disc_data, label, text, label_mask, 
                                         weight, optimizer['label'], device)
            
        else:
            loss = train_one_batch(model, sample, label, text, label_mask, weight,
                                    optimizer, device)


def train_one_batch(model, sample, disc_data, label, text, mask, 
                    weight, optimizer, device):
    sample = sample.to(device).transpose(0, 1)
    disc_data = disc_data.to(device)
    #label = label.to(device, dtype=torch.float)
    weight = weight.to(device, dtype=torch.float)
    optimizer.zero_grad()
    loss = model(sample, disc_data, text)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, test_loader, val_loader, encoded_labels, device, 
             target_names, save_dir, emb_dim):
    y_pred = []
    y_true = []
    y_mask = []

    model.eval()
    n_class = len(target_names)
    linear = LinearClassifier(data_encoder=model.data_encoder, 
                              head=model.head,
                              input_size=emb_dim, 
                              num_classes=n_class)

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=classifier_param['epochs'],
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         enable_progress_bar=True)

    # Train the model
    trainer.fit(linear, test_loader)

    # Evaluate
    linear.eval()
    linear = linear.to(device)

    with torch.no_grad():
        for sample, disc_data, label, text, label_mask, weight in val_loader:
            sample = sample.to(device).transpose(0, 1)
            disc_data = disc_data.to(device)
            label = label.to(device, dtype=torch.float)
            label_mask = label_mask.to(device, dtype=torch.long)

            out = linear(sample, disc_data)

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

    export_timestamps = {}
    export_results = {}

    model.eval()

    with torch.no_grad():
        for sample, disc_data, label, text, label_mask, _, weight, uuids, timestamps in test_loader:
            sample = sample.to(device).transpose(0, 1)
            disc_data = disc_data.to(device)
            #label = label.to(device, dtype=torch.float)
            #label_mask = label_mask.to(device, dtype=torch.long)

            embeddings = model.get_sensor_embeddings(sample, disc_data).cpu().numpy()  # (batch_size, emb_dim)

            # Add in sensor embedding to be export
            for batch_idx, (uuid, timeseries) in enumerate(zip(uuids, timestamps)): # batch
                uuid = str(uuid).split('.')[0]
                #print(timeseries)
                for timestamp in timeseries:
                    #print(uuid, timestamp, embeddings[batch_idx])
                    if uuid in export_timestamps:
                        export_timestamps[uuid].append(timestamp)
                        export_results[uuid].append(embeddings[batch_idx])
                    else:
                        export_timestamps[uuid] = [timestamp]
                        export_results[uuid] = [embeddings[batch_idx]]

    #y_true = np.array(y_true)
    #plot_label_pairs(y_true, target_names, save_dir, 'label_pairs_true.png')
    #y_pred = np.array(y_pred)
    #plot_label_pairs(y_pred, target_names, save_dir, 'label_pairs_pred.png')
    #y_mask = np.array(y_mask)

    #plot_confusion_matrix(y_true, y_pred, y_mask, target_names, save_dir)

    # Save the sensor embeddings to be export
    path_dir = './clip_embeddings'
    os.makedirs(path_dir)
    for uuid in export_timestamps:
        np.savez(os.path.join(path_dir, f'{uuid}.npz'),
                 T=np.array(export_timestamps[uuid]),
                 X=np.array(export_results[uuid])
                 )
        print(f'Save user {uuid} embeddings of size {np.array(export_results[uuid]).shape}')

    return None, None, None


def evaluate_batch_similarity(model, test_loader, device):
    """
    Given a batch matrix (size B) of paired embeddings,
    measure the accuracy of the predictions by checking nearest the neighbors
    """
    sensor_embeddings, text_embeddings = [], []

    model.eval()
    with torch.no_grad():
        for sample, disc_data, label, text, label_mask, weight in test_loader:
            sample = sample.to(device).transpose(0, 1)

            s_new = model.get_sensor_embeddings(sample, disc_data)  # (batch_size, emb_dim)
            t_new = model.get_text_embeddings(text)
            sensor_embeddings.extend(s_new)
            text_embeddings.extend(t_new)
    
    sensor_embeddings = torch.stack(sensor_embeddings, dim=0)
    text_embeddings = torch.stack(text_embeddings, dim=0)

    #  Compute similarity
    s = torch.nn.functional.normalize(sensor_embeddings, dim=1)
    t = torch.nn.functional.normalize(text_embeddings, dim=1)

    # similarities: B x B
    similarities = torch.mm(s, t.transpose(0, 1))

    # pred: 1 x B (ideally [0, 1, 2, 3, ..., B])
    s_t_pred = torch.argmax(similarities, dim=1)
    t_s_pred = torch.argmax(similarities, dim=0)
    B = len(s_t_pred)
    s_t_accuracy = sum(s_t_pred == torch.arange(B, device=device)) / B
    t_s_accuracy = sum(t_s_pred == torch.arange(B, device=device)) / B
    return s_t_accuracy.item(), t_s_accuracy.item()



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
