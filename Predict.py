import numpy as np
import pickle
from Loader import SignedPairsDataset, get_index_dicts
from Trainer import ERGOLightning
from torch.utils.data import DataLoader
from argparse import Namespace
import torch
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import sys

def read_input_file(datafile):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    data = pd.read_csv(datafile)
    for index in range(len(data)):
        sample = {}
        sample['tcra'] = data['TRA'][index]
        sample['tcrb'] = data['TRB'][index]
        sample['va'] = data['TRAV'][index]
        sample['ja'] = data['TRAJ'][index]
        sample['vb'] = data['TRBV'][index]
        sample['jb'] = data['TRBJ'][index]
        sample['t_cell_type'] = data['T-Cell-Type'][index]
        sample['peptide'] = data['Peptide'][index]
        sample['mhc'] = data['MHC'][index]
        # we do not use the sign
        sample['sign'] = 0
        if invalid(sample['tcrb']) or invalid(sample['peptide']):
            continue
        if invalid(sample['tcra']):
            sample['tcra'] = 'UNK'
        all_pairs.append(sample)
    return all_pairs, data


def load_model(hparams, checkpoint_path):
    model = ERGOLightning(hparams)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def get_model(dataset):
    if dataset == 'vdjdb':
        version = '1veajht'
    if dataset == 'mcpas':
        version = '1meajht'
    # get model file from version
    checkpoint_path = os.path.join('Models', 'version_' + version, 'checkpoints')
    files = [f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))]
    checkpoint_path = os.path.join(checkpoint_path, files[0])
    # get args from version
    args_path = os.path.join('Models', 'version_' + version, 'meta_tags.csv')
    with open(args_path, 'r') as file:
        lines = file.readlines()
        args = {}
        for line in lines[1:]:
            key, value = line.strip().split(',')
            if key in ['dataset', 'tcr_encoding_model', 'cat_encoding']:
                args[key] = value
            else:
                args[key] = eval(value)
    hparams = Namespace(**args)
    checkpoint = checkpoint_path
    model = load_model(hparams, checkpoint)
    train_pickle = 'Samples/' + model.dataset + '_train_samples.pickle'
    test_pickle = 'Samples/' + model.dataset + '_test_samples.pickle'
    return model, train_pickle


def get_train_dicts(train_pickle):
    with open(train_pickle, 'rb') as handle:
        train = pickle.load(handle)
    train_dicts = get_index_dicts(train)
    return train_dicts


def predict(dataset, test_file):
    model, train_file = get_model(dataset)
    train_dicts = get_train_dicts(train_file)
    test_samples, dataframe = read_input_file(test_file)
    test_dataset = SignedPairsDataset(test_samples, train_dicts)
    batch_size = 1000
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: test_dataset.collate(b, tcr_encoding=model.tcr_encoding_model,
                                                                  cat_encoding=model.cat_encoding))
    outputs = []
    for batch_idx, batch in enumerate(loader):
        output = model.validation_step(batch, batch_idx)
        if output:
            outputs.extend(output['y_hat'].tolist())
    dataframe['Score'] = outputs
    return dataframe


if __name__ == '__main__':
    df = predict(sys.argv[1], sys.argv[2])
    print(df)
    # df.to_csv('results.csv', index=False)
    pass


# NOTE: fix sklearn import problem with this in terminal:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dsi/speingi/anaconda3/lib/
# or just conda install libgcc
