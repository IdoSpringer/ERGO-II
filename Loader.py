import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from Sampler import get_diabetes_peptides
import pandas as pd
import numpy as np


class SignedPairsDataset(Dataset):
    def __init__(self, samples, train_dicts):
        self.data = samples
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}
        vatox, vbtox, jatox, jbtox, mhctox = train_dicts
        self.vatox = vatox
        self.vbtox = vbtox
        self.jatox = jatox
        self.jbtox = jbtox
        self.mhctox = mhctox
        self.pos_weight_factor = 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sign = sample['sign']
        if sign == 0:
            weight = 1
        elif sign == 1:
            weight = 5
        sample['weight'] = weight
        return sample

    def aa_convert(self, seq):
        if seq == 'UNK':
            seq = []
        else:
            seq = [self.atox[aa] for aa in seq]
        return seq

    @staticmethod
    def get_max_length(x):
        return len(max(x, key=len))

    def seq_letter_encoding(self, seq):
        def _pad(_it, _max_len):
            return _it + [0] * (_max_len - len(_it))
        return [_pad(it, self.get_max_length(seq)) for it in seq]

    def seq_one_hot_encoding(self, tcr, max_len=28):
        tcr_batch = list(tcr)
        padding = torch.zeros(len(tcr_batch), max_len, 20 + 1)
        # TCR is converted to numbers at this point
        # We need to match the autoencoder atox, therefore -1
        for i in range(len(tcr_batch)):
            # missing alpha
            if tcr_batch[i] == [0]:
                continue
            tcr_batch[i] = tcr_batch[i] + [self.atox['X']]
            for j in range(min(len(tcr_batch[i]), max_len)):
                padding[i, j, tcr_batch[i][j] - 1] = 1
        return padding

    def label_encoding(self):
        pass

    def binary_encoding(self):
        pass

    def hashing_encoding(self):
        # I think that feature hashing is not relevant in this case
        pass

    @staticmethod
    def binarize(num):
        l = []
        while num:
            l.append(num % 2)
            num //= 2
        l.reverse()
        # print(l)
        return l

    def collate(self, batch, tcr_encoding, cat_encoding):
        lst = []
        # TCRs
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
        tcrb = [self.aa_convert(sample['tcrb']) for sample in batch]
        tcra = [self.aa_convert(sample['tcra']) for sample in batch]
        if tcr_encoding == 'AE':
            lst.append(torch.FloatTensor(self.seq_one_hot_encoding(tcra, max_len=34)))
            lst.append(torch.FloatTensor(self.seq_one_hot_encoding(tcrb)))
        elif tcr_encoding == 'LSTM':
            lst.append(torch.LongTensor(self.seq_letter_encoding(tcra)))
            # we do not send the length, so that ae and lstm batch output be similar
            lst.append(torch.LongTensor(self.seq_letter_encoding(tcrb)))
        # Peptide
        peptide = [self.aa_convert(sample['peptide']) for sample in batch]
        lst.append(torch.LongTensor(self.seq_letter_encoding(peptide)))
        # Categorical features - V alpha, V beta, J alpha, J beta, MHC
        categorical = ['va', 'vb', 'ja', 'jb', 'mhc']
        cat_idx = [self.vatox, self.vbtox, self.jatox, self.jbtox, self.mhctox]
        for cat, idx in zip(categorical, cat_idx):
            batch_cat = ['UNK' if pd.isna(sample[cat]) else sample[cat] for sample in batch]
            batch_idx = list(map(lambda x: idx[x] if x in idx else 0, batch_cat))
            # print(cat)
            if cat_encoding == 'embedding':
                # label encoding
                batch_cat = torch.LongTensor(batch_idx)
            if cat_encoding == 'binary':
                # we need a matrix for the batch with the binary encodings
                max_len = 10
                def bin_pad(num, _max_len):
                    bin_list = self.binarize(num)
                    return [0] * (_max_len - len(bin_list)) + bin_list
                bin_mat = torch.tensor([bin_pad(v, max_len) for v in batch_idx]).float()
                batch_cat = bin_mat
            lst.append(batch_cat)
        # T cell type (or MHC class)
        t_type_dict = {'CD4': 2, 'CD8': 1, 'MHCII': 2, 'MHCI': 1}
        # nan and other values are 2
        t_type = [sample['t_cell_type'] for sample in batch]
        convert_type = lambda x: t_type_dict[x] if x in t_type_dict else 0
        t_type_tensor = torch.FloatTensor(list(map(convert_type, t_type)))
        lst.append(t_type_tensor)
        # Sign
        sign = [sample['sign'] for sample in batch]
        lst.append(torch.FloatTensor(sign))
        factor = self.pos_weight_factor
        weight = [sample['weight'] for sample in batch]
        lst.append(torch.FloatTensor(weight))
        return lst
    pass


def get_index_dicts(train_samples):
    samples = train_samples
    all_va = [sample['va'] for sample in samples if not pd.isna(sample['va'])]
    vatox = {va: index for index, va in enumerate(sorted(set(all_va)), 1)}
    vatox['UNK'] = 0
    all_vb = [sample['vb'] for sample in samples if not pd.isna(sample['vb'])]
    vbtox = {vb: index for index, vb in enumerate(sorted(set(all_vb)), 1)}
    vbtox['UNK'] = 0
    all_ja = [sample['ja'] for sample in samples if not pd.isna(sample['ja'])]
    jatox = {ja: index for index, ja in enumerate(sorted(set(all_ja)), 1)}
    jatox['UNK'] = 0
    all_jb = [sample['jb'] for sample in samples if not pd.isna(sample['jb'])]
    jbtox = {jb: index for index, jb in enumerate(sorted(set(all_jb)), 1)}
    jbtox['UNK'] = 0
    all_mhc = [sample['mhc'] for sample in samples if not pd.isna(sample['mhc'])]
    mhctox = {mhc: index for index, mhc in enumerate(sorted(set(all_mhc)), 1)}
    mhctox['UNK'] = 0
    return [vatox, vbtox, jatox, jbtox, mhctox]


class DiabetesDataset(SignedPairsDataset):
    def __init__(self, samples, train_dicts, weight_factor):
        super().__init__(samples, train_dicts)
        self.diabetes_peptides = get_diabetes_peptides('data/McPAS-TCR.csv')
        self.weight_factor = weight_factor

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        weight = sample['weight']
        peptide = sample['peptide']
        if peptide in self.diabetes_peptides:
            weight *= self.weight_factor
        return sample
    pass


class YellowFeverDataset(SignedPairsDataset):
    def __init__(self, samples, train_dicts, weight_factor):
        super().__init__(samples, train_dicts)
        self.yellow_fever_peptide = 'LLWNGPMAV'
        self.weight_factor = weight_factor

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        peptide = sample['peptide']
        if peptide == self.yellow_fever_peptide:
            sample['weight'] *= self.weight_factor
        return sample
    pass


class SinglePeptideDataset(SignedPairsDataset):
    def __init__(self, samples, train_dicts, peptide, force_peptide=False, spb_force=False):
        super().__init__(samples, train_dicts)
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}
        self.force_peptide = force_peptide
        self.spb_force = spb_force
        self.peptide = peptide

    def __getitem__(self, index):
        sample = self.data[index]
        # weight does not matter here, we do not train with this dataset
        sample['weight'] = 0
        if self.force_peptide:
            # we keep the original positives, else is negatives
            if self.spb_force:
                if sample['peptide'] != self.peptide:
                    sample['sign'] = 0
                return sample
            # we do it only for MPS (and we have to check the true peptide)
            # also for repertoire scoring
            else:
                sample['peptide'] = self.peptide
                return sample
        else:
            # original spb task
            if sample['peptide'] != self.peptide:
                return None
            return sample
    pass


def check():
    dct = 'Samples/'
    with open(dct + 'mcpas_human_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)
    with open(dct + 'mcpas_human_test_samples.pickle', 'rb') as handle:
        test = pickle.load(handle)
    dicts = get_index_dicts(train)
    vatox, vbtox, jatox, jbtox, mhctox = dicts
    print(len(vatox))
    for v in vatox:
        print(v)
    train_dataset = SignedPairsDataset(train, dicts)
    test_dataset = SignedPairsDataset(test, dicts)

    train_dataset = DiabetesDataset(train, dicts, weight_factor=10)
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=4,
                            collate_fn=lambda b: train_dataset.collate(b, tcr_encoding='LSTM',
                                                                       cat_encoding='embedding'))
    for batch in train_dataloader:
        print(batch)
        break
        # exit()
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4,
                                  collate_fn=lambda b: train_dataset.collate(b, tcr_encoding='LSTM',
                                                                             cat_encoding='embedding'))
    for batch in test_dataloader:
        print(batch)
        break
    print('successful')

# check()
