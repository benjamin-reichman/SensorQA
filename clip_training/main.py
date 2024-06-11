import os
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import random
from collections import defaultdict
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
import networkx as nx
import pandas as pd
from node2vec import Node2Vec

#from dataset.utils import train_cooccurrence
#from build_model import build_model
from evaluation import display_results, log_results
from evaluation import plot_label_pairs, plot_confusion_matrix
#from framework import MLCFramework as Framework
from evaluation import calculate_MLC_metrics as calculate_metrics
#from evaluation import calculate_SLC_metrics as calculate_metrics
from dataset_extrasensory import load_data, collate_fn, collate_fn_export
from dataset_extrasensory_timeseries import load_data_timeseries, \
    collate_fn_timeseries, collate_fn_export_timeseries
from libs.networks import TransformerEncoder
from libs.label_encoder import LabelEncoderClassifier
from libs.conv_classifier import ConventionalClassifier
from libs.clip_module import ClipEncoderClassifier

VAL_CNT = 10

def train_cooccurrence(save_dir, cooccurrence_file, target_names, calibrate=False):
    if os.path.exists(os.path.join(save_dir, 'label_embedding.npy')):
        print('load label embedding from file:', os.path.join(save_dir, 'label_embedding.npy'))
        return np.load(os.path.join(save_dir, 'label_embedding.npy'))
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cooccurrence_df = pd.read_pickle(open(cooccurrence_file, 'rb'))
    cooccurrence_matrix = cooccurrence_df.loc[target_names][target_names].to_numpy()

    if calibrate:
        threshold = np.nanpercentile(cooccurrence_matrix, 50)
        if threshold < 0:
            print('threshold of weight:', threshold)
            cooccurrence_matrix -= threshold  # np.median(weight_matrix)

    graph = nx.Graph()
    for i in range(len(target_names)):
        graph.add_node(target_names[i])

    for i in range(len(target_names)):  #, total=len(target_names)):
        for j in range(len(target_names)):
            if cooccurrence_matrix[i][j] > 0:
                graph.add_edge(target_names[i], target_names[j], weight=cooccurrence_matrix[i][j])

    node2vec = Node2Vec(graph, dimensions=256, walk_length=100, num_walks=200)
    model = node2vec.fit(window=5, min_count=1)

    label_embedding = np.array([model.wv[tn] for tn in target_names])
    np.save(os.path.join(save_dir, 'label_embedding.npy'), label_embedding)

    return label_embedding



def build_model(args_model, use_label_encoder, hidden_dim, disc_dim,
                data_feature_size, n_class, nhead,
                num_encoder_layers, dim_feedforward, dropout, encoded_labels,
                pretrained_embedding=None,
                do_input_embedding=False, device='cpu'):
    data_encoder = TransformerEncoder(
        d_model=hidden_dim,
        in_vocab_size=data_feature_size,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        do_input_embedding=do_input_embedding
    )

    if args_model == 'mlc' and args.use_label_encoder:
        model = LabelEncoderClassifier(data_encoder, emb_dim=hidden_dim, out_dim=n_class,
                                       pretrained_embedding=pretrained_embedding, device=device)
    elif args_model == 'mlc':
        model = ConventionalClassifier(data_encoder, emb_dim=hidden_dim, out_dim=n_class)
    elif args_model == 'clip':
        model = ClipEncoderClassifier(data_encoder, emb_dim=hidden_dim, disc_dim=disc_dim,
                                      device=device)
    else:
        raise ValueError('Model {} is not supported!'.format(args_model))

    return model


def run(args, fold, seed):
    model_name = (f"{MODEL_PREFIX}_{args.data}_win{args.win_len}_sap{args.sample_len}_"
        f"bsz{args.batch_size}_emb{args.emb_dim}_lb{int(args.use_label_encoder)}_"
        f"merge{int(args.use_label_merge)}_preemb{1-int(args.no_pretrain)}_lr{args.lr}_"
        f"{args.eval_thres}_seed{seed}/")
    save_dir = os.path.join(MODEL_DIR, model_name)
    cooccurrence_dir = os.path.join(MODEL_DIR, model_name)

    # Used in FedAlign
    cooccurrence_path = os.path.join(args.data_path, 'cooccurrence_extrasensory.pkl')

    # Prepare data for ExtraSensory
    if args.data == 'feature':
        trainData, valData, testData, target_names = load_data(
            args.data_path, max_len=args.win_len,
            seed=seed, fold=fold,
            use_label_merge=args.use_label_merge,
            test_only=args.test_only,
            augmentation=args.augmentation
        )
        data_feature_size = np.shape(testData.data[0])[-1]

        # Only load val and train when it is not test_only
        if not args.test_only:
            val_loader = DataLoader(valData, batch_size=args.test_batch_size, shuffle=False, 
                                    collate_fn=collate_fn, num_workers=2)
            train_loader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, 
                                      collate_fn=collate_fn, num_workers=8)
            test_loader = DataLoader(testData, batch_size=args.test_batch_size, shuffle=False, 
                                     collate_fn=collate_fn, num_workers=2)
        else:
            test_loader = DataLoader(testData, batch_size=args.test_batch_size, shuffle=False, 
                                     collate_fn=collate_fn_export, num_workers=1)
        
    elif args.data == 'timeseries':
        data_path = os.path.join(args.data_path, f'data_timeseries_{args.sample_len}_discrete')
        trainData, valData, testData, target_names = load_data_timeseries(
            data_path,
            sample_len=args.sample_len,
            seed=seed, fold=fold,
            use_label_merge=args.use_label_merge,
            test_only=args.test_only,
            augmentation=args.augmentation,
        )
        data_feature_size = args.sample_len

        # Only load val and train when it is not test_only
        if not args.test_only:
            val_loader = DataLoader(valData, batch_size=args.test_batch_size, shuffle=False, 
                                    collate_fn=collate_fn_timeseries, num_workers=2)
            train_loader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, 
                                      collate_fn=collate_fn_timeseries, num_workers=8)
            test_loader = DataLoader(testData, batch_size=args.test_batch_size, shuffle=False, 
                                     collate_fn=collate_fn_timeseries, num_workers=2)
        else:
            test_loader = DataLoader(testData, batch_size=args.test_batch_size, shuffle=False, 
                                     collate_fn=collate_fn_export_timeseries, num_workers=1)
    
    # save_dir = os.path.join(save_dir, f"fold{fold}")

    if args.no_pretrain:
        pretrained_embedding = None
    else:
        pretrained_embedding = train_cooccurrence(cooccurrence_dir, cooccurrence_path, target_names, calibrate=False)
        pretrained_embedding = torch.FloatTensor(pretrained_embedding).to(args.device)
        print('pretrained embedding matrix:', pretrained_embedding.shape)

    print('pretrained embedding: ', pretrained_embedding)
    if not args.test_only:
        print('# of samples in trainData:', len(trainData))
        print('# of samples in valData:', len(valData))
    print('# of samples in testData:', len(testData))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('save_dir:', save_dir)

    # Used in LabelEncoderClassifier
    encoded_labels = torch.arange(0, end=len(target_names)).to(args.device)
    print(f'n_class: {len(target_names)}, n_vocab: {encoded_labels.max() + 1}, n_feature: {data_feature_size}')
    # In text CLIP
    #encoded_labels = target_names # type: np.ndarray

    global_model = build_model(
        args_model=args.model,
        use_label_encoder=args.use_label_encoder,
        hidden_dim=args.emb_dim,
        disc_dim=args.disc_dim,
        data_feature_size=data_feature_size,
        n_class=len(target_names),
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        encoded_labels=encoded_labels,
        pretrained_embedding=pretrained_embedding,
        do_input_embedding=args.do_input_embedding,
        device=args.device
    )
    # Import the function of the corresponding model
    if args.model == 'mlc' and args.use_label_encoder:
        from libs.label_encoder import train_one_epoch, evaluate
    elif args.model == 'mlc':  # not args.use_label_encoder
        from libs.conv_classifier import train_one_epoch, evaluate, evaluate_export
    elif args.model == 'clip':
        from libs.clip_module import train_one_epoch, evaluate, \
            evaluate_batch_similarity, evaluate_export
    else:
        raise ValueError('Model {} is not supported!'.format(args.model))

    # Init model
    global_model = global_model.to(args.device)

    # Init optimizer
    if args.use_label_encoder:
        for param in global_model.label_encoder.parameters():
            param.requires_grad = False
        optimizer = {'data': torch.optim.Adam(filter(lambda p: p.requires_grad, global_model.parameters()), lr=args.lr)}
        for param in global_model.parameters():
            param.requires_grad = True
        for param in global_model.data_encoder.parameters():
            param.requires_grad = False
        optimizer['label'] = torch.optim.Adam(filter(lambda p: p.requires_grad, global_model.parameters()), lr=args.lr)
    else:
        len_param, len_named_param = 0, 0
        for i, param in enumerate(global_model.parameters()):
            len_param += param.numel()
        for i, (name, param) in enumerate(global_model.named_parameters()):
            print(name)
            len_named_param += param.numel()
        print(len_param, len_named_param)
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)

    # Training
    best_result = 0
    if not args.test_only:
        print('Training starts....')
        val_freq = np.floor(args.epochs / VAL_CNT)
        for e in range(args.epochs):
            if e % val_freq == 0:
                # For clip, train the Linear classifier on test_loader, test on val_loader
                test_true, test_pred, test_mask = evaluate(
                    global_model, test_loader, val_loader,
                    encoded_labels, args.device,
                    target_names, save_dir, args.emb_dim)

                results = calculate_metrics(test_true, test_pred, test_mask, args.eval_thres)
                display_results(results, args.metrics)

                # Save the best-performing model
                if results['BA'] > best_result:
                    best_result = results['BA']
                    torch.save(global_model, os.path.join(save_dir, f'model-best.pt'))
            

            train_one_epoch(global_model, encoded_labels, optimizer, train_loader, e,
                            args.use_label_encoder, args.device)
            torch.save(global_model, os.path.join(save_dir, f'model.pt'))

            # batch similarity evaluation
            if args.model == 'clip':
                s_t_accuracy, t_s_accuracy = evaluate_batch_similarity(
                    global_model, test_loader, args.device
                )


        # Final evaluation
        best_model = torch.load(os.path.join(save_dir, f'model.pt'))
        test_true, test_pred, test_mask = evaluate(best_model, test_loader, val_loader,
                                                   encoded_labels, args.device,
                                                   target_names, save_dir, args.emb_dim)

        results = calculate_metrics(test_true, test_pred, test_mask, args.eval_thres)
        display_results(results, args.metrics)
        log_results(results, save_dir, target_names)

        # Save the best-performing model
        if results['BA'] > best_result:
            best_result = results['BA']
            torch.save(global_model, os.path.join(save_dir, f'model-best.pt'))

    else:  # Test only
        best_model = torch.load(args.ckpt)
        test_true, test_pred, test_mask = evaluate_export(best_model, test_loader, encoded_labels,
                                                          args.device, target_names, save_dir,
                                                          args.eval_thres)

        if args.model == 'clip':  # CLIP embedding export does not need metrics eval
            results = None
        else:  # MLC
            results = calculate_metrics(test_true, test_pred, test_mask, args.eval_thres)
            display_results(results, args.metrics)
            log_results(results, save_dir, target_names)

    del trainData
    del valData
    del testData
    del global_model
    del best_model

    return results


def parse_args():
    # default setting is for extrasensory
    parser = argparse.ArgumentParser()
    #parser.add_argument('-g', '--gpu', type=int, default="5", help="gpu id")
    parser.add_argument('--random_seeds', type=int, default=[4322], help="random seed") #, 4322, 4323, 4324, 4325],
    parser.add_argument('-e', '--epochs', type=int, default=100, help="number of training epochs per round")
    parser.add_argument('--model', type=str, default='mlc', choices=['mlc', 'clip'],
                        help='training method')
    parser.add_argument('--test_only', action='store_true', help="whether to only run test")
    parser.add_argument('--ckpt', type=str, help='model ckpt used in test_only')

    # training parameters
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--use_label_encoder', action='store_true', help="whether to train label encoder or freeze")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--test_batch_size', type=int, default=128, help="test batch size")
    parser.add_argument('--emb_dim', type=int, default=512, help='dimension of the join embedding space')
    parser.add_argument('--disc_dim', type=int, default=68, help='dimension of the discrete data')

    # extrasensory
    parser.add_argument('--data', type=str, default='feature', choices=['feature', 'timeseries'],
                        help="whether to use hand engineered features or raw timeseries data")
    parser.add_argument('--data_path', type=str, default='data/',
                        help="path to data dir")
    parser.add_argument('--win_len', type=int, default=10, help='length of window size in minutes, used in the feature version of data')
    parser.add_argument('--sample_len', type=int, default=800, help='number of samples in one minute, used in the timeseries version of data')
    parser.add_argument('--use_label_merge', action='store_true', help="whether to merge labels to avoid confusion")
    parser.add_argument('--eval_thres', type=float, default=0.5, help="threshold during evaluation")
    parser.add_argument('--augmentation', action='store_true', help="whether to augment training data")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    args = parse_args()
    #torch.cuda.set_device(args.gpu)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.metrics = ['F1', 'ACC', 'BA']

    #if args.task.startswith('es-'):
    args.dataset = 'extrasensory'
    args.do_input_embedding = False
    args.normalize = False
    #if args.task == 'es-5':
    #    args.n_client = 5
    #elif args.task == 'es-15':
    #    args.n_client = 15
    #elif args.task == 'es-25':
    #    args.n_client = 25

    results = defaultdict(list)

    for fold, seed in zip(range(1), args.random_seeds):
        set_seed(seed)
        print(args)
        print(f'#### Run Experiments on seed {seed} ####')
        seed_results = run(args, fold, seed)
        if seed_results is not None:
            for m in args.metrics:
                results[m].append(seed_results[m])

            display_results({m: np.average(results[m]) for m in args.metrics}, args.metrics)
