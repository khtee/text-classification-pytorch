"""Models for sentence-level sentiment classification"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


class BaseModel(nn.Module):
    """Base model class for common parameters.
    
    Args:
        emb_size: An integer for embedding size.
        emb_dimension: An integer for embedding dimension.
        pretrained_emb: A string for embedding vector from torchtext vocab.
        output_size: An integer for number of classes.
        dropout: A float for dropout rate.
    """

    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 dropout=0.3):
        super().__init__()

        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.pretrained_emb = pretrained_emb
        self.output_size = output_size
        self.dropout = dropout
        self.DEVICE = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Loading pretrained embeddings
        self.embedding = nn.Embedding(self.emb_size, self.emb_dimension)

        if self.pretrained_emb is not None:
            self.embedding.weight.data.copy_(self.pretrained_emb)
            self.embedding.weight.requires_grad = False
        else:
            self.init_weights()
            self.embedding.weight.requires_grad = True

        # Dropout layers
        self.dropout_train = nn.Dropout(self.dropout)
        self.dropout_test = nn.Dropout(0.0)

    def init_weights(self):
        """Initialize embedding weight if no pretrained embedding"""
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def test(self, iter, batch_size):
        """Testing

        Args: 
            iter: An iterable class object for test data.
            batch_size: An integer for training batch size.

        Return:
            A float for accuracy score.
        """
        y_pred, y_true = [], []
        for batch in iter:
            x, y = batch.text, batch.label - 1
            x = x.to(self.DEVICE)
            y = y.to(self.DEVICE)
            if len(x) < batch_size:
                continue
            logits = self.forward(x, do_train=False)
            y_pred.extend(torch.argmax(logits, dim=1).tolist())
            y_true.extend(y.int().tolist())
        return accuracy_score(y_true, y_pred)


class RNN(BaseModel):
    """Recurrent Neural Network model
    
    Args:
        num_layers: An int for number of stacked recurrent layers. (default=2)
        hidden_size: An int for umber of features in the hidden state. (default=256)
        bidirectional: A bool whether to use the bidirectional GRU. (default=True)
    """

    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 num_layers=2,
                 hidden_size=256,
                 dropout=0.3,
                 bidirectional=True):
        super().__init__(emb_size=emb_size,
                         emb_dimension=emb_dimension,
                         pretrained_emb=pretrained_emb,
                         output_size=output_size,
                         dropout=dropout)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_directions = 2 if self.bidirectional else 1
        self.linear_input = self.num_layers * self.n_directions

        self.rnn = nn.RNN(self.emb_dimension,
                          self.hidden_size,
                          num_layers=self.num_layers,
                          bidirectional=self.bidirectional)

        self.fc1 = nn.Linear(self.linear_input * self.hidden_size,
                             int(self.hidden_size),
                             bias=True)
        self.fc2 = nn.Linear(int(self.hidden_size),
                             int(self.hidden_size / 2),
                             bias=True)
        self.fc3 = nn.Linear(int(self.hidden_size / 2),
                             self.output_size,
                             bias=True)

    def forward(self, input_sentences, do_train=True):
        """Forward pass

        """
        self.batch_size = input_sentences.size(0)
        self.sent_length = input_sentences.size(1)
        embedded = self.embedding(input_sentences)

        # RNN input shape: (nsentence_length, batch_size, emb_dim)
        rnn_input = embedded.permute(1, 0, 2)

        # hidden layer
        h_0 = torch.zeros(self.linear_input, self.batch_size,
                          self.hidden_size).to(self.DEVICE)

        # output shape: (sentence_length, batch_size, 2 * hidden_size)
        rnn_out, h_n = self.rnn(rnn_input, h_0)

        # h_n: [batch_size, 4 * self.hidden_size]
        h_n = h_n.permute(1, 0, 2)
        h_n = h_n.contiguous().view(h_n.size(0), -1)

        # FC layers
        output = F.relu(self.fc1(h_n))
        output = self.dropout_train(output) if do_train else self.dropout_test(
            output)
        output = F.relu(self.fc2(output))
        output = self.dropout_train(output) if do_train else self.dropout_test(
            output)
        logits = F.log_softmax(self.fc3(output), dim=1)

        return logits


class CNN(BaseModel):
    """Convolutional Neural Network model"""

    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 dropout=0.5):
        super().__init__(emb_size=emb_size,
                         emb_dimension=emb_dimension,
                         pretrained_emb=pretrained_emb,
                         output_size=output_size,
                         dropout=dropout)

        self.pretrained_dim = pretrained_emb.shape[1]
        self.conv1 = nn.Conv2d(1, int(emb_dimension / 3),
                               (3, self.pretrained_dim))
        self.conv2 = nn.Conv2d(1, int(emb_dimension / 3),
                               (4, self.pretrained_dim))
        self.conv3 = nn.Conv2d(1, int(emb_dimension / 3),
                               (5, self.pretrained_dim))

        self.fc = nn.Linear(in_features=int(int(self.emb_dimension / 3) * 3),
                            out_features=output_size,
                            bias=True)

    def forward(self, input_sentences, do_train=True):
        embedded = self.embedding(input_sentences)
        embedded = embedded.unsqueeze(1)

        convd1 = F.relu(self.conv1(embedded)).squeeze(3)
        pool1 = F.max_pool1d(convd1, convd1.size(2)).squeeze(2)
        convd2 = F.relu(self.conv2(embedded)).squeeze(3)
        pool2 = F.max_pool1d(convd2, convd2.size(2)).squeeze(2)
        convd3 = F.relu(self.conv3(embedded)).squeeze(3)
        pool3 = F.max_pool1d(convd3, convd3.size(2)).squeeze(2)
        output = torch.cat((pool1, pool2, pool3), 1)

        output = self.dropout_train(output) if do_train else self.dropout_test(
            output)
        logits = F.log_softmax(self.fc(output), dim=1)

        return logits


class BiGRU(BaseModel):
    """BiDirectional GRU

    Args:
        num_layers: An int for number of stacked recurrent layers. (default=2)
        hidden_size: An int for umber of features in the hidden state. (default=256)
        bidirectional: A bool whether to use the bidirectional GRU. (default=True)
        spatial_dropout: A bool whether to use the spatial dropout. (default=True)
    """

    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 num_layers=2,
                 hidden_size=256,
                 dropout=0.3,
                 bidirectional=True,
                 spatial_dropout=True):
        super().__init__(emb_size=emb_size,
                         emb_dimension=emb_dimension,
                         pretrained_emb=pretrained_emb,
                         output_size=output_size,
                         dropout=dropout)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.spatial_dropout = spatial_dropout
        self.n_directions = 2 if self.bidirectional else 1
        if self.spatial_dropout:
            self.spatial_dropout1d = nn.Dropout2d(self.dropout)

        self.gru = nn.GRU(self.emb_dimension,
                          self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=(0 if self.num_layers == 1 else self.dropout),
                          batch_first=True,
                          bidirectional=self.bidirectional)

        # concatenation of max_pooling ,avg_pooling, last hidden state
        self.fc1 = nn.Linear(self.hidden_size * 3, self.output_size)

    def forward(self, input_sentences, do_train=True):
        self.batch_size, self.sent_length = input_sentences.size(
            0), input_sentences.size(1)
        embedded_lengths = torch.LongTensor([self.sent_length] *
                                            self.batch_size)
        h_0 = None

        # input_sentences: (batch_size, sentence_length)
        # embedded: (batch_size, sentence_length, emb_dimension)
        embedded = self.embedding(input_sentences)

        if self.spatial_dropout:
            # Convert to (batch_size, emb_dimension, sentence_length)
            embedded = embedded.permute(0, 2, 1)
            embedded = self.spatial_dropout1d(embedded)
            # Convert back to (batch_size, sentence_length, emb_dimension)
            embedded = embedded.permute(0, 2, 1)
        else:
            embedded = self.droput_train(embedded)

        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded,
                                                       embedded_lengths,
                                                       batch_first=True)

        gru_out, h_n = self.gru(packed_emb, h_0)

        h_n = h_n.view(self.num_layers, self.n_directions, self.batch_size,
                       self.hidden_size)
        last_hidden = h_n[-1]

        last_hidden = torch.sum(last_hidden, dim=0)

        gru_out, lengths = nn.utils.rnn.pad_packed_sequence(gru_out,
                                                            batch_first=True)

        if self.bidirectional:
            gru_out = gru_out[:, :, :self.hidden_size] + gru_out[:, :, self.
                                                                 hidden_size:]

        max_pool = F.adaptive_max_pool1d(gru_out.permute(0, 2, 1),
                                         (1,)).view(self.batch_size, -1)

        lengths = lengths.view(-1, 1).type(torch.FloatTensor).to(self.DEVICE)
        avg_pool = torch.sum(gru_out, dim=1) / lengths

        # Concatenate hidden state, max_pooling and avg_pooling.
        output = torch.cat([last_hidden, max_pool, avg_pool], dim=1)
        output = self.dropout_train(output) if do_train else self.dropout_test(
            output)

        logits = F.log_softmax(self.fc1(output), dim=1)

        return logits


class LSTMAttn(BaseModel):
    """ LSTM with attention model

    Args:
        hidden_size: An int for umber of features in the hidden state. (default=256)
    """

    def __init__(self,
                 emb_size,
                 emb_dimension,
                 pretrained_emb=None,
                 output_size=5,
                 num_layers=2,
                 hidden_size=256,
                 dropout=0.3):
        super().__init__(emb_size=emb_size,
                         emb_dimension=emb_dimension,
                         pretrained_emb=pretrained_emb,
                         output_size=output_size,
                         dropout=dropout)

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.emb_dimension, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def attention_net(self, lstm_output, final_state):
        """Attention mechanism.

        Attention is computed between each hidden state with the last hidden state.
        
        Args:
            lstm_output : A tensor for LSTM output.
            final_state : A tensor for final hidden state of the LSTM
        
        Returns : 
            A new hidden state.
        """
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_sentences, do_train=True):
        self.batch_size = input_sentences.size(0)
        self.sent_length = input_sentences.size(1)
        embedded = self.embedding(input_sentences)
        embedded = embedded.permute(1, 0, 2)
        h_0 = torch.zeros(1, self.batch_size, self.hidden_size).to(self.DEVICE)
        c_0 = torch.zeros(1, self.batch_size, self.hidden_size).to(self.DEVICE)

        output, (h_n, c_n) = self.lstm(embedded, (h_0, c_0))
        output = output.permute(1, 0, 2)

        attn_output = self.attention_net(output, h_n)
        logits = F.log_softmax(self.fc(attn_output), dim=1)

        return logits