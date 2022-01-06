import math
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    Thanks Qin for sharing code in his github
"""


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]

        a_input = torch.cat(
            [h.repeat(1, 1, N).view(B, N * N, -1),
             h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(
            self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.attentions = [
            GraphAttentionLayer(nfeat,
                                nhid,
                                dropout=dropout,
                                alpha=alpha,
                                concat=True) for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                for j in range(self.nheads):
                    self.add_module(
                        'attention_{}_{}'.format(i + 1, j),
                        GraphAttentionLayer(nhid * nheads,
                                            nhid,
                                            dropout=dropout,
                                            alpha=alpha,
                                            concat=True))

        self.out_att = GraphAttentionLayer(nhid * nheads,
                                           nclass,
                                           dropout=dropout,
                                           alpha=alpha,
                                           concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        input = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                temp = []
                x = F.dropout(x, self.dropout, training=self.training)
                cur_input = x
                for j in range(self.nheads):
                    temp.append(
                        self.__getattr__('attention_{}_{}'.format(i + 1,
                                                                  j))(x, adj))
                x = torch.cat(temp, dim=2) + cur_input
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x + input


class GlobalLocalDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """
    def __init__(self,
                 hidden_dim,
                 output_dim,
                 dropout_rate,
                 n_heads=8,
                 decoder_gat_hidden_dim=16,
                 n_layers_decoder_global=2,
                 alpha=0.2):
        """ Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        """

        super(GlobalLocalDecoder, self).__init__()
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.alpha = alpha
        self.gat_dropout_rate = dropout_rate
        self.decoder_gat_hidden_dim = decoder_gat_hidden_dim
        self.n_heads = n_heads
        self.n_layers_decoder_global = n_layers_decoder_global
        # Network parameter definition.

        self.__slot_graph = GAT(self.__hidden_dim, self.decoder_gat_hidden_dim,
                                self.__hidden_dim, self.gat_dropout_rate,
                                self.alpha, self.n_heads,
                                self.n_layers_decoder_global)

        self.__global_graph = GAT(self.__hidden_dim,
                                  self.decoder_gat_hidden_dim,
                                  self.__hidden_dim, self.gat_dropout_rate,
                                  self.alpha, self.n_heads,
                                  self.n_layers_decoder_global)

        # self.__linear_layer = nn.Sequential(
        #     nn.Linear(self.__hidden_dim, self.__hidden_dim),
        #     nn.LeakyReLU(alpha),
        #     nn.Linear(self.__hidden_dim, self.__output_dim),
        # )

    def forward(self, inputs):
        """ Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :return: is distribution of prediction labels.
        """
        encoded_hiddens = inputs['hidden']
        seq_lens = inputs['seq_lens']
        global_adj = inputs['global_adj']
        slot_adj = inputs['slot_adj']
        intent_embedding = inputs['intent_embedding']
        output_tensor_list, sent_start_pos = [], 0

        batch = len(seq_lens)
        slot_graph_out = self.__slot_graph(encoded_hiddens, slot_adj)
        intent_in = intent_embedding.unsqueeze(0).repeat(batch, 1, 1)
        # print('intent_in', intent_in.shape)
        # print('slot_graph_out', slot_graph_out.shape)
        # print('seq_len', max(seq_lens))
        # print('intent_in', intent_in.shape)
        # print('slot_graph_out', slot_graph_out.shape)
        global_graph_in = torch.cat([intent_in, slot_graph_out], dim=1)
        global_graph_out = self.__global_graph(global_graph_in, global_adj)
        num_intent = intent_embedding.size(0)
        for i in range(0, len(seq_lens)):
            output_tensor_list.append(
                global_graph_out[i, num_intent:num_intent + seq_lens[i]])

        return {"hidden": torch.cat(output_tensor_list, dim=0)}
