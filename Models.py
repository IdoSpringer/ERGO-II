import os
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


class PaddingAutoencoder(nn.Module):
    def __init__(self, input_len, input_dim, encoding_dim):
        super(PaddingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.encoding_dim = encoding_dim
        # Encoder
        self.encoder = nn.Sequential(
                    nn.Linear(self.input_len * self.input_dim, 300),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(300, 100),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(100, self.encoding_dim))
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 300),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(300, self.input_len * self.input_dim))

    def forward(self, padded_input):
        concat = padded_input.view(-1, self.input_len * self.input_dim)
        encoded = self.encoder(concat)
        decoded = self.decoder(encoded)
        decoding = decoded.view(-1, self.input_len, self.input_dim)
        decoding = F.softmax(decoding, dim=2)
        return decoding


class LSTM_Encoder(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout):
        super(LSTM_Encoder, self).__init__()
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        # Embedding matrices - 20 amino acids + padding
        self.embedding = nn.Embedding(20 + 1, embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)

    def init_hidden(self, batch_size, device):
        if torch.cuda.is_available():
            return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim).to(device)),
                    autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)).to(device))
        else:
            return (autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)),
                    autograd.Variable(torch.zeros(2, batch_size, self.lstm_dim)))

    def lstm_pass(self, lstm, padded_embeds, lengths):
        device = padded_embeds.device
        # Before using PyTorch pack_padded_sequence we need to order the sequences batch by descending sequence length
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        # Pack the batch and ignore the padding
        padded_embeds = torch.nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths, batch_first=True)
        # Initialize the hidden state
        batch_size = len(lengths)
        hidden = self.init_hidden(batch_size, device)
        # Feed into the RNN
        lstm.flatten_parameters()
        lstm_out, hidden = lstm(padded_embeds, hidden)
        # Unpack the batch after the RNN
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # Remember that our outputs are sorted. We want the original ordering
        _, unperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[unperm_idx]
        lengths = lengths[unperm_idx]
        return lstm_out

    def forward(self, seq, lengths):
        # Encoder:
        # Embedding
        embeds = self.embedding(seq)
        # LSTM Acceptor
        lstm_out = self.lstm_pass(self.lstm, embeds, lengths)
        last_cell = torch.cat([lstm_out[i, j.data - 1] for i, j in enumerate(lengths)]).view(len(lengths), self.lstm_dim)
        return last_cell


class AE_Encoder(nn.Module):
    def __init__(self, encoding_dim, tcr_type, input_dim=21, max_len=28, train_ae=True):
        super(AE_Encoder, self).__init__()
        # Dimensions
        self.encoding_dim = encoding_dim
        self.tcr_type = tcr_type
        self.input_dim = input_dim
        self.max_len = max_len
        self.autoencoder = PaddingAutoencoder(max_len, input_dim, encoding_dim)
        self.init_ae_params(train_ae)

    def init_ae_params(self, train_ae=True):
        ae_dir = 'TCR_Autoencoder'
        if self.tcr_type == 'alpha':
            ae_file = os.sep.join([ae_dir, 'tcra_ae_dim_' + str(self.encoding_dim) + '.pt'])
        elif self.tcr_type == 'beta':
            ae_file = os.sep.join([ae_dir, 'tcrb_ae_dim_' + str(self.encoding_dim) + '.pt'])
        checkpoint = torch.load(ae_file)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        if train_ae is False:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
            self.autoencoder.eval()

    def forward(self, padded_tcrs):
        # TCR Encoder:
        # Embedding
        concat = padded_tcrs.view(-1, self.max_len * self.input_dim)
        encoded_tcrs = self.autoencoder.encoder(concat)
        return encoded_tcrs


# irrelevant (implemented as lightning module)
class ERGO(nn.Module):
    def __init__(self, tcr_encoding_model, embedding_dim, lstm_dim, encoding_dim, dropout=0.1):
        super(ERGO, self).__init__()
        # Model Type
        self.tcr_encoding_model = tcr_encoding_model
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.encoding_dim = encoding_dim
        self.dropout = dropout
        # TCR Encoder
        if self.tcr_encoding_model == 'AE':
            self.tcr_encoder = AE_Encoder(encoding_dim=encoding_dim)
        elif self.tcr_encoding_model == 'LSTM':
            self.tcr_encoder = LSTM_Encoder(embedding_dim, lstm_dim, dropout)
            self.encoding_dim = lstm_dim
        # Peptide Encoder
        self.pep_encoder = LSTM_Encoder(embedding_dim, lstm_dim, dropout)
        # MLP
        self.mlp_dim = self.lstm_dim + self.encoding_dim
        self.hidden_layer = nn.Linear(self.mlp_dim, int(np.sqrt(self.mlp_dim)))
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(int(np.sqrt(self.mlp_dim)), 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tcr_batch, peps, pep_lens):
        # TCR Encoder:
        tcr_encoding = self.tcr_encoder(*tcr_batch)
        # PEPTIDE Encoder:
        pep_encoding = self.pep_encoder(peps, pep_lens)
        # MLP Classifier
        tcr_pep_concat = torch.cat([tcr_encoding, pep_encoding], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output
    pass
