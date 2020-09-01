import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from Loader import SignedPairsDataset, DiabetesDataset, YellowFeverDataset, get_index_dicts
from Models import PaddingAutoencoder, AE_Encoder, LSTM_Encoder, ERGO
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from argparse import ArgumentParser


class ERGOLightning(pl.LightningModule):

    def __init__(self, hparams):
        super(ERGOLightning, self).__init__()
        self.hparams = hparams
        self.dataset = hparams.dataset
        # Model Type
        self.tcr_encoding_model = hparams.tcr_encoding_model
        self.use_alpha = hparams.use_alpha
        self.use_vj = hparams.use_vj
        self.use_mhc = hparams.use_mhc
        self.use_t_type = hparams.use_t_type
        self.cat_encoding = hparams.cat_encoding
        # Dimensions
        self.aa_embedding_dim = hparams.aa_embedding_dim
        self.cat_embedding_dim = hparams.cat_embedding_dim
        self.lstm_dim = hparams.lstm_dim
        self.encoding_dim = hparams.encoding_dim
        self.dropout_rate = hparams.dropout
        self.lr = hparams.lr
        self.wd = hparams.wd
        # get train indicies for V,J etc
        if self.cat_encoding == 'embedding':
            with open('Samples/' + self.dataset + '_train_samples.pickle', 'rb') as handle:
                train = pickle.load(handle)
            vatox, vbtox, jatox, jbtox, mhctox = get_index_dicts(train)
            self.v_vocab_size = len(vatox) + len(vbtox)
            self.j_vocab_size = len(jatox) + len(jbtox)
            self.mhc_vocab_size = len(mhctox)
        # TCR Encoder
        if self.tcr_encoding_model == 'AE':
            if self.use_alpha:
                self.tcra_encoder = AE_Encoder(encoding_dim=self.encoding_dim, tcr_type='alpha', max_len=34)
            self.tcrb_encoder = AE_Encoder(encoding_dim=self.encoding_dim, tcr_type='beta')
        elif self.tcr_encoding_model == 'LSTM':
            if self.use_alpha:
                self.tcra_encoder = LSTM_Encoder(self.aa_embedding_dim, self.lstm_dim, self.dropout_rate)
            self.tcrb_encoder = LSTM_Encoder(self.aa_embedding_dim, self.lstm_dim, self.dropout_rate)
            self.encoding_dim = self.lstm_dim
        # Peptide Encoder
        self.pep_encoder = LSTM_Encoder(self.aa_embedding_dim, self.lstm_dim, self.dropout_rate)
        # Categorical
        self.cat_encoding = hparams.cat_encoding
        if hparams.cat_encoding == 'embedding':
            if self.use_vj:
                self.v_embedding = nn.Embedding(self.v_vocab_size, self.cat_embedding_dim, padding_idx=0)
                self.j_embedding = nn.Embedding(self.j_vocab_size, self.cat_embedding_dim, padding_idx=0)
            if self.use_mhc:
                self.mhc_embedding = nn.Embedding(self.mhc_vocab_size, self.cat_embedding_dim, padding_idx=0)
        # different mlp sizes, depends on model input
        if self.cat_encoding == 'binary':
            self.cat_embedding_dim = 10
        mlp_input_size = self.lstm_dim + self.encoding_dim
        if self.use_vj:
            mlp_input_size += 2 * self.cat_embedding_dim
        if self.use_mhc:
            mlp_input_size += self.cat_embedding_dim
        if self.use_t_type:
            mlp_input_size += 1
        # MLP I (without alpha)
        self.mlp_dim1 = mlp_input_size
        self.hidden_layer1 = nn.Linear(self.mlp_dim1, int(np.sqrt(self.mlp_dim1)))
        self.relu = torch.nn.LeakyReLU()
        self.output_layer1 = nn.Linear(int(np.sqrt(self.mlp_dim1)), 1)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        # MLP II (with alpha)
        if self.use_alpha:
            mlp_input_size += self.encoding_dim
            if self.use_vj:
                mlp_input_size += 2 * self.cat_embedding_dim
            self.mlp_dim2 = mlp_input_size
            self.hidden_layer2 = nn.Linear(self.mlp_dim2, int(np.sqrt(self.mlp_dim2)))
            self.output_layer2 = nn.Linear(int(np.sqrt(self.mlp_dim2)), 1)

    def forward(self, tcr_batch, pep_batch, cat_batch, t_type_batch):
        # PEPTIDE Encoder:
        pep_encoding = self.pep_encoder(*pep_batch)
        # TCR Encoder:
        tcra, tcrb = tcr_batch
        tcrb_encoding = self.tcrb_encoder(*tcrb)
        # Categorical Encoding:
        va, vb, ja, jb, mhc = cat_batch
        # T cell type
        t_type = t_type_batch.view(len(t_type_batch), 1)
        # gather all features, int linear mlp so the order does not matter
        mlp_input = [tcrb_encoding, pep_encoding]
        if self.use_vj:
            if self.cat_encoding == 'embedding':
                va = self.v_embedding(va)
                vb = self.v_embedding(vb)
                ja = self.j_embedding(ja)
                jb = self.j_embedding(jb)
            mlp_input += [vb, jb]
        if self.use_mhc:
            if self.cat_encoding == 'embedding':
                mhc = self.mhc_embedding(mhc)
            mlp_input += [mhc]
        if self.use_t_type:
            mlp_input += [t_type]
        if tcra:
            tcra_encoding = self.tcra_encoder(*tcra)
            mlp_input += [tcra_encoding]
            if self.use_vj:
                mlp_input += [va, ja]
            # MLP II Classifier
            concat = torch.cat(mlp_input, 1)
            hidden_output = self.dropout(self.relu(self.hidden_layer2(concat)))
            mlp_output = self.output_layer2(hidden_output)
        else:
            # MLP I Classifier
            concat = torch.cat(mlp_input, 1)
            hidden_output = self.dropout(self.relu(self.hidden_layer1(concat)))
            mlp_output = self.output_layer1(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output

    def step(self, batch):
        # batch is none (might happen in evaluation)
        if not batch:
            return None
        # batch output (always)
        tcra, tcrb, pep, va, vb, ja, jb, mhc, t_type, sign, weight = batch
        if self.tcr_encoding_model == 'LSTM':
            # get lengths for lstm functions
            len_b = torch.sum((tcrb > 0).int(), dim=1)
            len_a = torch.sum((tcra > 0).int(), dim=1)
        if self.tcr_encoding_model == 'AE':
            len_a = torch.sum(tcra, dim=[1, 2]) - 1
        len_p = torch.sum((pep > 0).int(), dim=1)
        if self.use_alpha:
            missing = (len_a == 0).nonzero(as_tuple=True)
            full = len_a.nonzero(as_tuple=True)
            if self.tcr_encoding_model == 'LSTM':
                tcra_batch_ful = (tcra[full], len_a[full])
                tcrb_batch_ful = (tcrb[full], len_b[full])
                tcrb_batch_mis = (tcrb[missing], len_b[missing])
            elif self.tcr_encoding_model == 'AE':
                tcra_batch_ful = (tcra[full],)
                tcrb_batch_ful = (tcrb[full],)
                tcrb_batch_mis = (tcrb[missing],)
            tcr_batch_ful = (tcra_batch_ful, tcrb_batch_ful)
            tcr_batch_mis = (None, tcrb_batch_mis)
            device = len_a.device
            y_hat = torch.zeros(len(sign)).to(device)
            # there are samples without alpha
            if len(missing[0]):
                pep_mis = (pep[missing], len_p[missing])
                cat_mis = (va[missing], vb[missing],
                           ja[missing], jb[missing],
                           mhc[missing])
                t_type_mis = t_type[missing]
                y_hat_mis = self.forward(tcr_batch_mis, pep_mis, cat_mis, t_type_mis).squeeze()
                y_hat[missing] = y_hat_mis
            # there are samples with alpha
            if len(full[0]):
                pep_ful = (pep[full], len_p[full])
                cat_ful = (va[full], vb[full],
                           ja[full], jb[full],
                           mhc[full])
                t_type_ful = t_type[full]
                y_hat_ful = self.forward(tcr_batch_ful, pep_ful, cat_ful, t_type_ful).squeeze()
                y_hat[full] = y_hat_ful
        else:
            if self.tcr_encoding_model == 'LSTM':
                tcrb_batch = (None, (tcrb, len_b))
            elif self.tcr_encoding_model == 'AE':
                tcrb_batch = (None, (tcrb,))
            pep_batch = (pep, len_p)
            cat_batch = (va, vb, ja, jb, mhc)
            y_hat = self.forward(tcrb_batch, pep_batch, cat_batch, t_type).squeeze()
        y = sign
        return y, y_hat, weight

    def training_step(self, batch, batch_idx):
        self.train()
        y, y_hat, weight = self.step(batch)
        loss = F.binary_cross_entropy(y_hat, y, weight=weight)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        self.eval()
        if self.step(batch):
            y, y_hat, _ = self.step(batch)
            return {'val_loss': F.binary_cross_entropy(y_hat.view(-1, 1), y.view(-1, 1)), 'y_hat': y_hat, 'y': y}
        else:
            return None

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        y = torch.cat([x['y'].view(-1, 1) for x in outputs])
        y_hat = torch.cat([x['y_hat'].view(-1, 1) for x in outputs])
        # auc = roc_auc_score(y.cpu(), y_hat.cpu())
        auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        print(auc)
        tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc}
        return {'avg_val_loss': avg_loss, 'val_auc': auc, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pass

    def test_end(self, outputs):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    @pl.data_loader
    def train_dataloader(self):
        with open('Samples/' + self.dataset + '_train_samples.pickle', 'rb') as handle:
            train = pickle.load(handle)
        train_dataset = SignedPairsDataset(train, get_index_dicts(train))
        return DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=10,
                          collate_fn=lambda b: train_dataset.collate(b, tcr_encoding=self.tcr_encoding_model,
                                                                     cat_encoding=self.cat_encoding))

    @pl.data_loader
    def val_dataloader(self):
        with open('Samples/' + self.dataset + '_test_samples.pickle', 'rb') as handle:
            test = pickle.load(handle)
        with open('Samples/' + self.dataset + '_train_samples.pickle', 'rb') as handle:
            train = pickle.load(handle)
        test_dataset = SignedPairsDataset(test, get_index_dicts(train))
        return DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=10,
                          collate_fn=lambda b: test_dataset.collate(b, tcr_encoding=self.tcr_encoding_model,
                                                                     cat_encoding=self.cat_encoding))

    @pl.data_loader
    def test_dataloader(self):
        pass


class ERGODiabetes(ERGOLightning):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.weight_factor = hparams.weight_factor

    @pl.data_loader
    def train_dataloader(self):
        with open(self.dataset + '_train_samples.pickle', 'rb') as handle:
            train = pickle.load(handle)
        train_dataset = DiabetesDataset(train, get_index_dicts(train), weight_factor=self.weight_factor)
        return DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=10,
                          collate_fn=lambda b: train_dataset.collate(b, tcr_encoding=self.tcr_encoding_model,
                                                                     cat_encoding=self.cat_encoding))


class ERGOYellowFever(ERGOLightning):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.weight_factor = hparams.weight_factor

    @pl.data_loader
    def train_dataloader(self):
        with open(self.dataset + '_train_samples.pickle', 'rb') as handle:
            train = pickle.load(handle)
        train_dataset = YellowFeverDataset(train, get_index_dicts(train), weight_factor=self.weight_factor)
        return DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=10,
                          collate_fn=lambda b: train_dataset.collate(b, tcr_encoding=self.tcr_encoding_model,
                                                                     cat_encoding=self.cat_encoding))


def diabetes_experiment():
    parser = ArgumentParser()
    parser.add_argument('--version', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--dataset', type=str, default='mcpas_human')
    parser.add_argument('--tcr_encoding_model', type=str, default='AE')
    parser.add_argument('--cat_encoding', type=str, default='embedding')
    parser.add_argument('--use_alpha', type=bool, default=True)
    parser.add_argument('--use_vj', type=bool, default=True)
    parser.add_argument('--use_mhc', type=bool, default=True)
    parser.add_argument('--aa_embedding_dim', type=int, default=10)
    parser.add_argument('--cat_embedding_dim', type=int, default=50)
    parser.add_argument('--lstm_dim', type=int, default=500)
    parser.add_argument('--encoding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    # for diabetes
    parser.add_argument('--weight_factor', type=int, default=5)
    hparams = parser.parse_args()
    model = ERGODiabetes(hparams)
    # logger = TensorBoardLogger("diabetes_logs", name="d_mcpas_lstm_with_alpha")
    logger = TensorBoardLogger("diabetes_logs", name="ergo_ii_diabetes", version=hparams.version)
    early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
    trainer = Trainer(gpus=[hparams.gpu], logger=logger, early_stop_callback=early_stop_callback)
    trainer.fit(model)


def yellow_fever_experiment():
    parser = ArgumentParser()
    parser.add_argument('iter', type=int)
    parser.add_argument('gpu', type=int)
    parser.add_argument('dataset', type=str, help='vdjdb or vdjdb_no10x')
    parser.add_argument('tcr_encoding_model', type=str, help='LSTM or AE')
    parser.add_argument('--cat_encoding', type=str, default='embedding')
    parser.add_argument('--use_alpha', action='store_true')
    parser.add_argument('--use_vj', action='store_true')
    parser.add_argument('--use_mhc', action='store_true')
    parser.add_argument('--use_t_type', action='store_true')
    parser.add_argument('--aa_embedding_dim', type=int, default=10)
    parser.add_argument('--cat_embedding_dim', type=int, default=50)
    parser.add_argument('--lstm_dim', type=int, default=500)
    parser.add_argument('--encoding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_factor', type=int, default=5)
    hparams = parser.parse_args()
    model = ERGOLightning(hparams)
    # version flags
    version = 'yf'
    version += str(hparams.iter)
    if hparams.dataset == 'vdjdb_no10x':
        version += 'n10x'
    if hparams.tcr_encoding_model == 'AE':
        version += 'e'
    elif hparams.tcr_encoding_model == 'LSTM':
        version += 'l'
    if hparams.use_alpha:
        version += 'a'
    if hparams.use_vj:
        version += 'j'
    if hparams.use_mhc:
        version += 'h'
    if hparams.use_t_type:
        version += 't'
    version += str(hparams.weight_factor)

    logger = TensorBoardLogger("yellow_fever", name="YF_models", version=version)
    early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
    trainer = Trainer(gpus=[hparams.gpu], logger=logger, early_stop_callback=early_stop_callback)
    trainer.fit(model)


def ergo_ii_experiment():
    parser = ArgumentParser()
    parser.add_argument('iter', type=int)
    parser.add_argument('gpu', type=int)
    parser.add_argument('dataset', type=str, help='mcpas_human or vdjdb')
    parser.add_argument('tcr_encoding_model', type=str, help='LSTM or AE')
    parser.add_argument('--cat_encoding', type=str, default='embedding')
    parser.add_argument('--use_alpha', action='store_true')
    parser.add_argument('--use_vj', action='store_true')
    parser.add_argument('--use_mhc', action='store_true')
    parser.add_argument('--use_t_type', action='store_true')
    parser.add_argument('--aa_embedding_dim', type=int, default=10)
    parser.add_argument('--cat_embedding_dim', type=int, default=50)
    parser.add_argument('--lstm_dim', type=int, default=500)
    parser.add_argument('--encoding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    hparams = parser.parse_args()
    model = ERGOLightning(hparams)
    # version flags
    version = ''
    version += str(hparams.iter)
    if hparams.dataset == 'mcpas_human':
        version += 'h'
    elif hparams.dataset == 'vdjdb':
        version += 'v'
    elif hparams.dataset == 'vdjdb_no10x':
        version += 'f'
    elif hparams.dataset == 'mcpas':
        version += 'm'
    else:
        version += 'x'
    if hparams.tcr_encoding_model == 'AE':
        version += 'e'
    elif hparams.tcr_encoding_model == 'LSTM':
        version += 'l'
    if hparams.use_alpha:
        version += 'a'
    if hparams.use_vj:
        version += 'j'
    if hparams.use_mhc:
        version += 'h'
    if hparams.use_t_type:
        version += 't'
    logger = TensorBoardLogger("ERGO-II_paper_logs", name="paper_models", version=version)
    early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
    trainer = Trainer(gpus=[hparams.gpu], logger=logger, early_stop_callback=early_stop_callback)
    trainer.fit(model)


def ergo_ii_tuning():
    parser = ArgumentParser()
    parser.add_argument('--version', type=int)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--use_alpha', type=bool, default=True)
    parser.add_argument('--use_vj', type=bool, default=True)
    parser.add_argument('--use_mhc', type=bool, default=True)
    parser.add_argument('--tcr_encoding_model', type=str)
    parser.add_argument('--cat_encoding', type=str, default='embedding')
    parser.add_argument('--aa_embedding_dim', type=int, default=10)
    parser.add_argument('--cat_embedding_dim', type=int, default=50)
    parser.add_argument('--lstm_dim', type=int, default=500)
    parser.add_argument('--encoding_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.1)
    hparams = parser.parse_args()
    model = ERGOLightning(hparams)
    logger = TensorBoardLogger("ERGO-II_tuning_update", name="ergo_tuning_model", version=hparams.version)
    early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
    trainer = Trainer(gpus=[hparams.gpu], logger=logger, early_stop_callback=early_stop_callback)
    trainer.fit(model)


if __name__ == '__main__':
    # ergo_ii_experiment()
    # diabetes_experiment()
    # ergo_ii_tuning()
    # yellow_fever_experiment()
    pass


# NOTE: fix sklearn import problem with this in terminal:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dsi/speingi/anaconda3/lib/
# or just conda activate dgx

# see logs
# tensorboard --logdir dir
