import pandas as pd
import numpy as np
import random
import pickle
import time


def read_data(datafile, file_key, human=True, use_10x=True):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    if file_key == 'mcpas':
        data = pd.read_csv(datafile, engine='python')
        for index in range(len(data)):
            sample = {}
            sample['tcra'] = data['CDR3.alpha.aa'][index]
            sample['tcrb'] = data['CDR3.beta.aa'][index]
            sample['va'] = data['TRAV'][index]
            sample['ja'] = data['TRAJ'][index]
            sample['vb'] = data['TRBV'][index]
            sample['jb'] = data['TRBJ'][index]
            sample['t_cell_type'] = data['T.Cell.Type'][index]
            sample['peptide'] = data['Epitope.peptide'][index]
            sample['protein'] = data['Antigen.protein'][index]
            sample['mhc'] = data['MHC'][index]
            if invalid(sample['tcrb']) or invalid(sample['peptide']):
                continue
            if human and data['Species'][index] != 'Human':
                continue
            if invalid(sample['tcra']):
                sample['tcra'] = 'UNK'
            all_pairs.append(sample)
    elif file_key == 'vdjdb':
        data = pd.read_csv(datafile, engine='python', sep='\t')
        # first read all TRB, then unite with TRA according to sample id
        paired = {}
        for index in range(len(data)):
            sample = {}
            id = int(data['complex.id'][index])
            type = data['Gene'][index]
            tcr = data['CDR3'][index]
            if not use_10x and '10xgenomics' in data['Reference'][index]:
                continue
            if type == 'TRB':
                sample['tcrb'] = tcr
                sample['tcra'] = 'UNK'
                sample['va'] = 'UNK'
                sample['ja'] = 'UNK'
                sample['vb'] = data['V'][index]
                sample['jb'] = data['J'][index]
                sample['peptide'] = data['Epitope'][index]
                sample['protein'] = data['Epitope gene'][index]
                sample['mhc'] = data['MHC A'][index]
                # here it's mhc class
                sample['t_cell_type'] = data['MHC class'][index]
                if invalid(tcr) or invalid(sample['peptide']):
                    continue
                # only TRB
                if id == 0:
                    all_pairs.append(sample)
                else:
                    paired[id] = sample
            if type == 'TRA':
                tcra = tcr
                if invalid(tcra):
                    tcra = 'UNK'
                sample = paired[id]
                sample['va'] = data['V'][index]
                sample['ja'] = data['J'][index]
                sample['tcra'] = tcra
                paired[id] = sample
        all_pairs.extend(list(paired.values()))
    # assimung each sample appears only once in the dataset
    train_pairs, test_pairs = train_test_split(all_pairs)
    return all_pairs, train_pairs, test_pairs


def train_test_split(all_pairs):
    '''
    Splitting the TCR-PEP pairs
    '''
    train_pairs = []
    test_pairs = []
    for pair in all_pairs:
        # 80% train, 20% test
        p = np.random.binomial(1, 0.8)
        if p == 1:
            train_pairs.append(pair)
        else:
            test_pairs.append(pair)
    return train_pairs, test_pairs


def positive_examples(pairs):
    pos_samples = []
    for sample in pairs:
        sample['sign'] = 1
        pos_samples.append(sample)
    return pos_samples


def negative_examples(pairs, all_pairs, size):
    '''
    Randomly creating intentional negative examples from the same pairs dataset.
    We match randomly tcr data with peptide data to make a sample
    '''
    neg_samples = []
    i = 0
    while i < size:
        # choose randomly two samples. match tcr data with pep data
        pep_sample = random.choice(pairs)
        tcr_sample = random.choice(pairs)
        sample = {}
        sample['tcra'] = tcr_sample['tcra']
        sample['tcrb'] = tcr_sample['tcrb']
        sample['va'] = tcr_sample['va']
        sample['ja'] = tcr_sample['ja']
        sample['vb'] = tcr_sample['vb']
        sample['jb'] = tcr_sample['jb']
        sample['t_cell_type'] = tcr_sample['t_cell_type']
        sample['peptide'] = pep_sample['peptide']
        sample['protein'] = pep_sample['protein']
        sample['mhc'] = pep_sample['mhc']
        if sample not in all_pairs and sample not in neg_samples:
                sample['sign'] = 0
                neg_samples.append(sample)
                i += 1
    return neg_samples


def get_examples(datafile, file_key, human, use_10x):
    all_pairs, train_pairs, test_pairs = read_data(datafile, file_key, human, use_10x)
    train_pos = positive_examples(train_pairs)
    test_pos = positive_examples(test_pairs)
    train_neg = negative_examples(train_pairs, all_pairs, 5 * len(train_pos))
    test_neg = negative_examples(test_pairs, all_pairs, 5 * len(test_pos))
    train = train_pos + train_neg
    random.shuffle(train)
    test = test_pos + test_neg
    random.shuffle(test)
    return train, test


def sample_data(datafile, file_key, train_file, test_file, human=False, use_10x=True):
    train, test = get_examples(datafile, file_key, human, use_10x)
    with open(str(train_file) + '.pickle', 'wb') as handle:
        pickle.dump(train, handle)
    with open(str(test_file) + '.pickle', 'wb') as handle:
        pickle.dump(test, handle)


def sample():
    # t1 = time.time()
    # print('sampling human mcpas...')
    # sample_data('data/McPAS-TCR.csv', 'mcpas', 'mcpas_human_train_samples', 'mcpas_human_test_samples', human=True)
    # t2 = time.time()
    # print('done in ' + str(t2 - t1) + ' seconds')
    #
    # t1 = time.time()
    # print('sampling mcpas...')
    # sample_data('data/McPAS-TCR.csv', 'mcpas', 'mcpas_train_samples', 'mcpas_test_samples', human=False)
    # t2 = time.time()
    # print('done in ' + str(t2 - t1) + ' seconds')

    # t1 = time.time()
    # print('sampling vdjdb...')
    # sample_data('data/VDJDB_complete.tsv', 'vdjdb', 'vdjdb_no10x_train_samples', 'vdjdb_no10x_test_samples', use_10x=False)
    # t2 = time.time()
    # print('done in ' + str(t2 - t1) + ' seconds')
    pass


def get_diabetes_peptides(datafile):
    data = pd.read_csv(datafile, engine='python')
    d_peps = set()
    for index in range(len(data)):
        peptide = data['Epitope.peptide'][index]
        if pd.isna(peptide):
            continue
        pathology = data['Pathology'][index]
        if pathology == 'Diabetes Type 1':
            d_peps.add(peptide)
    return d_peps


def frequent_peptides(datafile, key, k):
    if key == 'mcpas':
        data = pd.read_csv(datafile, engine='python')
        freq_peps = data['Epitope.peptide'].value_counts()[:k].index.to_list()
    elif key == 'vdjdb':
        data = pd.read_csv(datafile, engine='python', sep='\t')
        freq_peps = data['Epitope'].value_counts()[:k].index.to_list()
    return freq_peps


def count_alpha_beta():
    datafile = 'data/McPAS-TCR.csv'
    data = pd.read_csv(datafile, engine='python')
    freq_peps = data['Epitope.peptide'].value_counts()[:20].index.to_list()
    has = []
    mis = []
    for pep in freq_peps:
        pep_data = data[data['Epitope.peptide'] == pep]
        pep_data = pep_data[~pd.isna(pep_data['CDR3.beta.aa'])]
        has_alpha = len(pep_data[~pep_data['CDR3.alpha.aa'].apply(pd.isna)])
        mis_alpha = len(pep_data[pep_data['CDR3.alpha.aa'].apply(pd.isna)])
        assert has_alpha + mis_alpha == len(pep_data)
        has.append(has_alpha)
        mis.append(mis_alpha)
        print(pep, has_alpha, mis_alpha)
    df = pd.DataFrame()
    df['peptide'] = freq_peps
    df['has_alpha'] = has
    df['mis_alpha'] = mis
    df.to_csv('plots/alpha_beta_counts.csv', index=False)


def check():
    with open('mcpas_human_train_samples.pickle', 'rb') as handle:
        train = pickle.load(handle)
    print(len(train))
    print(random.choice(train))
    pass


if __name__ == '__main__':
    # sample()
    # frequent_peptides()
    # count_alpha_beta()
    pass
