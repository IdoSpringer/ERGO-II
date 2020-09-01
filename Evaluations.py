import numpy as np
import random
import pickle
from Loader import SignedPairsDataset, SinglePeptideDataset, get_index_dicts
from Trainer import ERGOLightning, ERGODiabetes
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from argparse import Namespace
from argparse import ArgumentParser
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import Sampler
import csv
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import sys


def load_model(hparams, checkpoint_path, diabetes=False):
    if diabetes:
        model = ERGODiabetes(hparams)
    else:
        model = ERGOLightning(hparams)
    checkpoint = torch.load(checkpoint_path, map_location='cuda:1')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def load_test(datafiles):
    train_pickle, test_pickle = datafiles
    with open(train_pickle, 'rb') as handle:
        train = pickle.load(handle)
    with open(test_pickle, 'rb') as handle:
        test = pickle.load(handle)
    train_dicts = get_index_dicts(train)
    return test, train_dicts


def true_new_pairs(args, datafiles):
    train_pickle, test_pickle = datafiles
    # open and read data
    # return TCRs and peps that appear only in test pairs
    with open(train_pickle, 'rb') as handle:
        train = pickle.load(handle)
    with open(test_pickle, 'rb') as handle:
        test = pickle.load(handle)
    true_test = []
    for sample in test:
        if sample in train:
            continue
        tcrb = sample['tcrb']
        # filter sample in train that match same tcrb
        train_candidates = list(filter(lambda _sample: _sample['tcrb'] == tcrb, train))
        peptide = sample['peptide']
        # filter sample in train that match same peptide
        train_candidates = list(filter(lambda _sample: _sample['peptide'] == peptide, train_candidates))
        if len(train_candidates) == 0:
            true_test.append(sample)
        else:
            # same alpha sieve
            if args.use_alpha:
                tcra = sample['tcra']
                train_candidates = list(filter(lambda _sample: _sample['tcra'] == tcra, train_candidates))
            # same v,j sieve (all at once)
            if args.use_vj:
                va = sample['va']
                vb = sample['vb']
                ja = sample['ja']
                jb = sample['jb']
                train_candidates = list(filter(lambda _sample: _sample['va'] == va
                                                and _sample['vb'] == vb
                                                and _sample['ja'] == ja
                                                and _sample['jb'] == jb, train_candidates))
            # same mhc sieve
            if args.use_mhc:
                mhc = sample['mhc']
                train_candidates = list(filter(lambda _sample: _sample['mhc'] == mhc, train_candidates))
            # same t type sieve (although it is probably irrelevant)
            if args.use_t_type:
                t_type = sample['t_cell_type']
                train_candidates = list(filter(lambda _sample: _sample['t_cell_type'] == t_type, train_candidates))
            if len(train_candidates) == 0:
                true_test.append(sample)
            # else:
            #     for i in train_candidates:
            #         print(i)
            #     print('sample:', sample)
            #     # exit()
    print('Done filtering test pairs!')
    return true_test


def efficient_true_new_pairs(args, datafiles):
    # tpp-i slightly depends on what features were used
    # e.g. two samples with same tcrb but different tcra
    # if we use alpha, one can be in train and other in test
    # if we do not use alpha, one cannot be in test
    # but when we train with alpha, we cause train leakage... how can we handle this?
    # we do not. if we have many samples with identical beta chain, tpp-i is not so relevant
    train_pickle, test_pickle = datafiles
    # open and read data
    # return TCRs and peps that appear only in test pairs
    with open(train_pickle, 'rb') as handle:
        train = pickle.load(handle)
    with open(test_pickle, 'rb') as handle:
        test = pickle.load(handle)
    # first make a train dictionary
    # all train combinations of relevant features
    features = ['tcrb', 'peptide']
    if args.use_alpha:
        features += ['tcra']
    if args.use_vj:
        features += ['va', 'vb', 'ja', 'jb']
    if args.use_mhc:
        features += ['mhc']
    if args.use_t_type:
        features += ['t_cell_type']
    train_sieve = {tuple(sample[f] for f in features): True for sample in train}
    true_test = []
    for sample in test:
        if sample in train:
            continue
        if tuple(sample[f] for f in features) in train_sieve:
            # print(sample)
            continue
        true_test.append(sample)
    print('Done filtering test pairs!')
    return true_test


def get_new_tcrs_and_peps(datafiles):
    train_pickle, test_pickle = datafiles
    # open and read data
    # return TCRs and peps that appear only in test pairs
    with open(train_pickle, 'rb') as handle:
        train = pickle.load(handle)
    with open(test_pickle, 'rb') as handle:
        test = pickle.load(handle)
    train_peps = [sample['peptide'] for sample in train]
    train_tcrbs = [sample['tcrb'] for sample in train]
    test_peps = [sample['peptide'] for sample in test]
    test_tcrbs = [sample['tcrb'] for sample in test]
    new_test_tcrbs = set(test_tcrbs).difference(set(train_tcrbs))
    new_test_peps = set(test_peps).difference(set(train_peps))
    # print(len(set(test_tcrbs)), len(new_test_tcrbs))
    return new_test_tcrbs, new_test_peps


def auc_predict(model, test, train_dicts, peptide=None):
    if peptide:
        test_dataset = SinglePeptideDataset(test, train_dicts, peptide, force_peptide=False)
    else:
        test_dataset = SignedPairsDataset(test, train_dicts)
    # print(test_dataset.data)
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0,
                        collate_fn=lambda b: test_dataset.collate(b, tcr_encoding=model.tcr_encoding_model,
                                                                  cat_encoding=model.cat_encoding))
    outputs = []
    for batch_idx, batch in enumerate(loader):
        output = model.validation_step(batch, batch_idx)
        if output:
            outputs.append(output)
            # print(output['y'])
    auc = model.validation_end(outputs)['val_auc']
    return auc


def spb(model, datafiles, true_test, peptide):
    test, train_dicts = load_test(datafiles)
    new_test_tcrbs, new_test_peps = get_new_tcrs_and_peps(datafiles)
    test = [sample for sample in true_test if sample['tcrb'] in new_test_tcrbs]
    return auc_predict(model, test, train_dicts, peptide=peptide)


def unfiltered_spb(model, datafiles, peptide):
    # spb without test filtering
    test, train_dicts = load_test(datafiles)
    new_test_tcrbs, new_test_peps = get_new_tcrs_and_peps(datafiles)
    test = [sample for sample in test if sample['tcrb'] in new_test_tcrbs]
    return auc_predict(model, test, train_dicts, peptide=peptide)


def mps(hparams, model, datafiles, peptides):
    train_file, test_files = datafiles
    with open(train_file, 'rb') as handle:
        train = pickle.load(handle)
    train_dicts = get_index_dicts(train)
    samples = get_tpp_ii_pairs(datafiles)
    preds = np.zeros((len(samples), len(peptides)))
    key_order = []
    for pep_idx, pep in enumerate(peptides):
        key_order.append(pep)
        testset = SinglePeptideDataset(samples, train_dicts, peptide_map[pep],
                                       force_peptide=True, spb_force=False)
        loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=10,
                            collate_fn=lambda b: testset.collate(b, tcr_encoding=hparams.tcr_encoding_model,
                                                                 cat_encoding=hparams.cat_encoding))
        outputs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                model.eval()
                outputs.append(model.validation_step(batch, batch_idx))
        y_hat = torch.cat([x['y_hat'].detach().cpu() for x in outputs])
        preds[:, pep_idx] = y_hat
    argmax = np.argmax(preds, axis=1)
    predicted_peps = [key_order[i] for i in argmax]
    # need to return accuracy
    return predicted_peps


def tpp_i(model, datafiles, true_test):
    test, train_dicts = load_test(datafiles)
    test = true_test
    return auc_predict(model, test, train_dicts)


def tpp_ii(model, datafiles, true_test):
    test, train_dicts = load_test(datafiles)
    new_test_tcrbs, new_test_peps = get_new_tcrs_and_peps(datafiles)
    test = [sample for sample in true_test if sample['tcrb'] in new_test_tcrbs]
    return auc_predict(model, test, train_dicts)


def tpp_iii(model, datafiles, true_test):
    test, train_dicts = load_test(datafiles)
    new_test_tcrbs, new_test_peps = get_new_tcrs_and_peps(datafiles)
    test = [sample for sample in true_test if sample['tcrb'] in new_test_tcrbs
                                        and sample['peptide'] in new_test_peps]
    return auc_predict(model, test, train_dicts)


# def get_tpp_ii_pairs(datafiles):
#     # We only take new TCR beta chains (why? does it matter?)
#     train_pickle, test_pickle = datafiles
#     with open(test_pickle, 'rb') as handle:
#         test_data = pickle.load(handle)
#     new_test_tcrbs, new_test_peps = get_new_tcrs_and_peps(datafiles)
#     return [sample for sample in test_data if sample['tcrb'] in new_test_tcrbs]


def protein_spb():
    pass


def spb_with_more_negatives(model, datafiles, peptide):
    test = get_tpp_ii_pairs(datafiles)
    # Regular SPB
    # test_dataset = SinglePeptideDataset(test, peptide)
    # More negatives
    test_dataset = SinglePeptideDataset(test, peptide, force_peptide=True, spb_force=True)
    if model.tcr_encoding_model == 'AE':
        collate_fn = test_dataset.ae_collate
    elif model.tcr_encoding_model == 'LSTM':
        collate_fn = test_dataset.lstm_collate
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=10, collate_fn=collate_fn)
    outputs = []
    i = 0
    positives = 0
    for batch_idx, batch in enumerate(loader):
        i += 1
        outputs.append(model.validation_step(batch, batch_idx))
    if i:
        print('positives:', int(torch.cat([x['y'] for x in outputs]).sum().item()))
        auc = model.validation_end(outputs)['val_auc']
        print(auc)
    pass


def multi_peptide_score(args, model, test_data, new_tcrs, number_of_peps):
    # get N most frequent peps from the positives list
    peps = targets
    most_freq = []
    for i in range(number_of_peps):
        # find current most frequent pep
        freq_pep = max(peps, key=peps.count)
        most_freq.append(freq_pep)
        # remove all its instances from list
        peps = list(filter(lambda pep: pep != freq_pep, peps))
    # print(most_freq)


    # true peptide targets indexes
    true_pred = list(map(lambda pep: most_freq.index(pep) if pep in most_freq else number_of_peps + 1, targets))
    accs = []
    for i in range(2, number_of_peps + 1):
        # define target pep using score argmax (save scores in a matrix)
        preds = np.argmax(score_matrix[:, :i], axis=1)
        # get accuracy score of k-class classification
        indices = [j for j in range(len(true_pred)) if true_pred[j] < i]
        k_class_predtion = np.array([preds[j] for j in indices])
        k_class_target = np.array([true_pred[j] for j in indices])
        accuracy = sum(k_class_predtion == k_class_target) / len(k_class_predtion)
        # print(accuracy)
        accs.append(accuracy)
    return most_freq, accs


def check2(checkpoint_path):
    args = {'dataset': 'mcpas', 'tcr_encoding_model': 'LSTM', 'use_alpha': False,
            'embedding_dim': 10, 'lstm_dim': 500, 'encoding_dim': 'none', 'dropout': 0.1}
    hparams = Namespace(**args)
    checkpoint = checkpoint_path
    model = load_model(hparams, checkpoint)
    train_pickle = model.dataset + '_train_samples.pickle'
    test_pickle = model.dataset + '_test_samples.pickle'
    datafiles = train_pickle, test_pickle
    spb(model, datafiles, peptide='LPRRSGAAGA')
    spb(model, datafiles, peptide='GILGFVFTL')
    spb(model, datafiles, peptide='NLVPMVATV')
    spb(model, datafiles, peptide='GLCTLVAML')
    spb(model, datafiles, peptide='SSYRRPVGI')
    d_peps = list(Sampler.get_diabetes_peptides('data/McPAS-TCR.csv'))
    print(d_peps)
    for pep in d_peps:
        try:
            print(pep)
            spb(model, datafiles, peptide=pep)
        except ValueError:
            pass


def diabetes_test_set(model):
    # 8 paired samples, 4 peptides
    # tcra, tcrb, pep
    data = [('CAATRTSGTYKYIF', 'CASSPWGAGGTDTQYF', 'IGRPp39'),
            ('CAVGAGYGGATNKLIF', 'CASSFRGGGNPYEQYF', 'GADp70'),
            ('CAERLYGNNRLAF', 'CASTLLWGGDSYEQYF', 'GADp15'),
            ('CAVNPNQAGTALIF', 'CASAPQEAQPQHF', 'IGRPp31'),
            ('CALSDYSGTSYGKLTF', 'CASSLIPYNEQFF', 'GADp15'),
            ('CAVEDLNQAGTALIF', 'CASSLALGQGNQQFF', 'IGRPp31'),
            ('CILRDTISNFGNEKLTF', 'CASSFGSSYYGYTF', 'IGRPp39'),
            ('CAGQTGANNLFF', 'CASSQEVGTVPNQPQHF', 'IGRPp31')]
    peptide_map = {'IGRPp39': 'QLYHFLQIPTHEEHLFYVLS',
                   'GADp70': 'KVNFFRMVISNPAATHQDID',
                   'GADp15': 'DVMNILLQYVVKSFDRSTKV',
                   'IGRPp31': 'KWCANPDWIHIDTTPFAGLV'}
    true_labels = np.array([list(peptide_map.keys()).index(d[-1]) for d in data])
    print(true_labels)
    samples = []
    for tcra, tcrb, pep in data:
        tcr_data = (tcra, tcrb, 'v', 'j')
        pep_data = (peptide_map[pep], 'mhc', 'protein')
        samples.append((tcr_data, pep_data, 1))
    preds = np.zeros((len(samples), len(peptide_map)))
    for pep_idx, pep in enumerate(peptide_map):
        # signs do not matter here, we do only forward pass
        dataset = SinglePeptideDataset(samples, peptide_map[pep], force_peptide=True)
        if model.tcr_encoding_model == 'AE':
            collate_fn = dataset.ae_collate
        elif model.tcr_encoding_model == 'LSTM':
            collate_fn = dataset.lstm_collate
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=10, collate_fn=collate_fn)
        outputs = []
        for batch_idx, batch in enumerate(loader):
            outputs.append(model.validation_step(batch, batch_idx))
        y_hat = torch.cat([x['y_hat'].detach().cpu() for x in outputs])
        preds[:, pep_idx] = y_hat
    # print(preds)
    argmax = np.argmax(preds, axis=1)
    print(argmax)
    accuracy = sum((argmax == true_labels).astype(int)) / len(samples)
    print(accuracy)
    # try protein accuracy - IGRP and GAD
    true_labels = np.array([0 if x == 3 else 1 if x == 2 else x for x in true_labels])
    argmax = np.array([0 if x == 3 else 1 if x == 2 else x for x in argmax])
    print(true_labels)
    print(argmax)
    accuracy = sum((argmax == true_labels).astype(int)) / len(samples)
    print(accuracy)
    pass


def read_known_specificity_test(testfile):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    data = pd.read_csv(testfile, engine='python')
    for index in range(len(data)):
        sample = {}
        sample['tcra'] = data['junction_alpha'][index]
        sample['tcrb'] = data['junction_beta'][index]
        sample['va'] = data['v_gene_alpha'][index]
        sample['ja'] = data['j_gene_alpha'][index]
        sample['vb'] = data['v_gene_beta'][index]
        sample['jb'] = data['j_gene_beta'][index]
        sample['t_cell_type'] = 'UNK'
        sample['peptide'] = 'pep'
        sample['protein'] = 'protein'
        sample['mhc'] = 'UNK'
        if invalid(sample['tcrb']):
            continue
        if invalid(sample['tcra']):
            sample['tcra'] = 'UNK'
        # we do not use sign and weight, but it has to be defined
        sample['sign'] = 0
        sample['weight'] = 1
        all_pairs.append(sample)
    return all_pairs


# blind test (known specificity)
def diabetes_mps(hparams, model, testfile, pep_pool):
    with open('mcpas_human_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)
    train_dicts = get_index_dicts(train)
    if pep_pool == 4:
        peptide_map = {'IGRPp39': 'QLYHFLQIPTHEEHLFYVLS',
                       'GADp70': 'KVNFFRMVISNPAATHQDID',
                       'GADp15': 'DVMNILLQYVVKSFDRSTKV',
                       'IGRPp31': 'KWCANPDWIHIDTTPFAGLV'}
    else:
        peptide_map = {}
        with open(pep_pool, 'r') as file:
            file.readline()
            for line in file:
                pep, index, protein = line.strip().split(',')
                if protein in ['GAD', 'IGRP', 'Insulin']:
                    protein += 'p'
                pep_name = protein + index
                peptide_map[pep_name] = pep
    samples = read_known_specificity_test(testfile)
    preds = np.zeros((len(samples), len(peptide_map)))
    key_order = []
    for pep_idx, pep in enumerate(peptide_map):
        key_order.append(pep)
        testset = SinglePeptideDataset(samples, train_dicts, peptide_map[pep],
                                       force_peptide=True, spb_force=False)
        loader = DataLoader(testset, batch_size=10, shuffle=False, num_workers=10,
                            collate_fn=lambda b: testset.collate(b, tcr_encoding=hparams.tcr_encoding_model,
                                                                 cat_encoding=hparams.cat_encoding))
        outputs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                model.eval()
                outputs.append(model.validation_step(batch, batch_idx))
        y_hat = torch.cat([x['y_hat'].detach().cpu() for x in outputs])
        preds[:, pep_idx] = y_hat
    argmax = np.argmax(preds, axis=1)
    predicted_peps = [key_order[i] for i in argmax]
    print(predicted_peps)
    pass


def diabetes_evaluations():
    # chack diabetes with different weight factor
    # checkpoint_path = 'mcpas_without_alpha/version_8/checkpoints/_ckpt_epoch_35.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_5/checkpoints/_ckpt_epoch_40.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_10/checkpoints/_ckpt_epoch_46.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_20/checkpoints/_ckpt_epoch_63.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_21/checkpoints/_ckpt_epoch_31.ckpt'
    # checkpoint_path = 'mcpas_without_alpha/version_50/checkpoints/_ckpt_epoch_19.ckpt'
    # with alpha
    # checkpoint_path = 'mcpas_with_alpha/version_2/checkpoints/_ckpt_epoch_31.ckpt'
    # with v, j , mhc
    # checkpoint_path = 'ergo_ii_diabetes/version_5/checkpoints/_ckpt_epoch_52.ckpt'
    # checkpoint_path = 'ergo_ii_diabetes/version_10/checkpoints/_ckpt_epoch_36.ckpt'
    # checkpoint_path = 'ergo_ii_diabetes/version_20/checkpoints/_ckpt_epoch_40.ckpt'
    # args = {'gpu': 1,
    #         'dataset': 'mcpas_human', 'tcr_encoding_model': 'AE', 'cat_encoding': 'embedding',
    #         'use_alpha': True, 'use_vj': True, 'use_mhc': True, 'use_t_type': True,
    #         'aa_embedding_dim': 10, 'cat_embedding_dim': 50,
    #         'lstm_dim': 300, 'encoding_dim': 100,
    #         'lr': 1e-4, 'wd': 0,
    #         'dropout': 0.1}
    # version = 20
    # weight_factor = version
    # diabetes_test_set(model)
    # diabetes_mps(hparams, model, 'diabetes_data/known_specificity.csv', 'diabetes_data/28pep_pool.csv')
    # diabetes_mps(hparams, model, 'diabetes_data/known_specificity.csv', pep_pool=4)
    # d_peps = list(Sampler.get_diabetes_peptides('data/McPAS-TCR.csv'))
    # for pep in d_peps:
    #     try:
    #         print(pep)
    #         spb_with_more_negatives(model, datafiles, peptide=pep)
    #     except ValueError:
    #         pass
    pass


def spb_main(version, data_key):
    # get model file from version
    model_dir = 'paper_models'
    logs_dir = 'ERGO-II_paper_logs'
    checkpoint_path1 = os.path.join(model_dir, 'version_' + version, 'checkpoints')
    checkpoint_path2 = os.path.join(logs_dir, checkpoint_path1)
    try:
        files = [f for f in listdir(checkpoint_path1) if isfile(join(checkpoint_path1, f))]
        checkpoint_path = checkpoint_path1
    except FileNotFoundError:
        files = [f for f in listdir(checkpoint_path2) if isfile(join(checkpoint_path2, f))]
        checkpoint_path = checkpoint_path2
    checkpoint_path = os.path.join(checkpoint_path, files[0])
    # get args from version
    args_path1 = os.path.join(logs_dir, model_dir, 'version_' + version)
    args_path2 = os.path.join(logs_dir, model_dir, version)
    if isfile(os.path.join(args_path1, 'meta_tags.csv')):
        args_path = os.path.join(args_path1, 'meta_tags.csv')
    elif isfile(os.path.join(args_path2, 'meta_tags.csv')):
        args_path = os.path.join(args_path2, 'meta_tags.csv')
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
    model = load_model(hparams, checkpoint, diabetes=False)
    train_pickle = 'Samples/' + model.dataset + '_train_samples.pickle'
    test_pickle = 'Samples/' + model.dataset + '_test_samples.pickle'
    datafiles = train_pickle, test_pickle

    spb_results = []
    if data_key == 'mcpas':
        freq_peps = Sampler.frequent_peptides('data/McPAS-TCR.csv', 'mcpas', 20)
        # print(freq_peps)
        for pep in freq_peps:
            # print(pep + ' spb:', unfiltered_spb(model, datafiles, peptide=pep))
            spb_results.append(unfiltered_spb(model, datafiles, peptide=pep))
    elif data_key == 'vdjdb':
        freq_peps = ['IPSINVHHY', 'TPRVTGGGAM', 'NLVPMVATV', 'GLCTLVAML',
                     'RAKFKQLL', 'YVLDHLIVV', 'GILGFVFTL', 'PKYVKQNTLKLAT',
                     'CINGVCWTV', 'KLVALGINAV', 'ATDALMTGY', 'RPRGEVRFL',
                     'LLWNGPMAV', 'GTSGSPIVNR', 'GTSGSPIINR', 'KAFSPEVIPMF',
                     'TPQDLNTML', 'EIYKRWII', 'KRWIILGLNK', 'FRDYVDRFYKTLRAEQASQE',
                     'GPGHKARVL', 'FLKEKGGL']
        print(freq_peps)
        for pep in freq_peps:
            # print(pep + ' spb:', unfiltered_spb(model, datafiles, peptide=pep))
            spb_results.append(unfiltered_spb(model, datafiles, peptide=pep))
    return spb_results


def tpp_main(version):
    # get model file from version
    model_dir = 'paper_models'
    logs_dir = 'ERGO-II_paper_logs'
    checkpoint_path1 = os.path.join(model_dir, 'version_' + version, 'checkpoints')
    checkpoint_path2 = os.path.join(logs_dir, checkpoint_path1)
    try:
        files = [f for f in listdir(checkpoint_path1) if isfile(join(checkpoint_path1, f))]
        checkpoint_path = checkpoint_path1
    except FileNotFoundError:
        files = [f for f in listdir(checkpoint_path2) if isfile(join(checkpoint_path2, f))]
        checkpoint_path = checkpoint_path2
    checkpoint_path = os.path.join(checkpoint_path, files[0])
    # get args from version
    args_path1 = os.path.join(logs_dir, model_dir, 'version_' + version)
    args_path2 = os.path.join(logs_dir, model_dir, version)
    if isfile(os.path.join(args_path1, 'meta_tags.csv')):
        args_path = os.path.join(args_path1, 'meta_tags.csv')
    elif isfile(os.path.join(args_path2, 'meta_tags.csv')):
        args_path = os.path.join(args_path2, 'meta_tags.csv')
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
    model = load_model(hparams, checkpoint, diabetes=False)
    train_pickle = 'Samples/' + model.dataset + '_train_samples.pickle'
    test_pickle = 'Samples/' + model.dataset + '_test_samples.pickle'
    datafiles = train_pickle, test_pickle
    true_test = efficient_true_new_pairs(hparams, datafiles)
    # TPP
    print('tpp i:', tpp_i(model, datafiles, true_test))
    print('tpp ii:', tpp_ii(model, datafiles, true_test))
    print('tpp iii:', tpp_iii(model, datafiles, true_test))
    pass


if __name__ == '__main__':
    # version = sys.argv[1]
    # key = 'vdjdb'
    # if key == 'mcpas':
    #     freq_peps = Sampler.frequent_peptides('data/McPAS-TCR.csv', 'mcpas', 20)
    #     spb_table = pd.DataFrame()
    #     for version in ['1me', '1ml', '1mea', '1mla', '1meaj', '1mlaj',
    #                     '1meajh', '1mlajh', '1meajht', '1mlajht']:
    #         print(version)
    #         spb_results = spb_main(version, data_key='mcpas')
    #         print(spb_results)
    #         spb_table[version] = spb_results
    #     spb_table.index = freq_peps
    #     spb_table.to_csv('plots/mcpas_spb_results.csv')
    # elif key == 'vdjdb':
    #     freq_peps = ['IPSINVHHY', 'TPRVTGGGAM', 'NLVPMVATV', 'GLCTLVAML',
    #                  'RAKFKQLL', 'YVLDHLIVV', 'GILGFVFTL', 'PKYVKQNTLKLAT',
    #                  'CINGVCWTV', 'KLVALGINAV', 'ATDALMTGY', 'RPRGEVRFL',
    #                  'LLWNGPMAV', 'GTSGSPIVNR', 'GTSGSPIINR', 'KAFSPEVIPMF',
    #                  'TPQDLNTML', 'EIYKRWII', 'KRWIILGLNK', 'FRDYVDRFYKTLRAEQASQE',
    #                  'GPGHKARVL', 'FLKEKGGL']
    #     spb_table = pd.DataFrame()
    #     for version in ['1fe', '1fl', '1fea', '1fla', '1feaj', '1flaj',
    #                     '1feajh', '1flajh', '1feajht', '1flajht']:
    #         print(version)
    #         spb_results = spb_main(version, data_key='vdjdb')
    #         print(spb_results)
    #         spb_table[version] = spb_results
    #     spb_table.index = freq_peps
    #     spb_table.to_csv('plots/vdjdb_spb_results.csv')
    pass

