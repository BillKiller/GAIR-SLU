from torch import nn
import torch
from utils.GAT import GAT
from utils.matrix_utils import flat2matrix, matrix2flat
from utils.mlp import MLPAdapter


def normalize_adj(mx):
    """
    Row-normalize matrix  D^{-1}A
    torch.diag_embed: https://github.com/pytorch/pytorch/pull/12447
    """
    mx = mx.float()
    rowsum = mx.sum(2)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag_embed(r_inv, 0)
    mx = r_mat_inv.matmul(mx)
    return mx


class RefineDecoder(nn.Module):
    def __init__(self,
                 block_num,
                 hidden_size,
                 intent_num,
                 slot_num,
                 graph_hidden_dim,
                 graph_output_dim,
                 alpha=0.2,
                 dropout=0.3,
                 nhead=4,
                 window = 2):
        super().__init__()
        self.block_list = nn.ModuleList([
            RefineBlock(hidden_size, intent_num, slot_num, graph_hidden_dim,
                        graph_output_dim, alpha = alpha , dropout = dropout, nhead=nhead, window=window) for _ in range(block_num)
        ])

    # self, hiddens, seq_lens, intent_pro =None, slot_pro = None, is_flat = False, force_intent = None, force_slot = None
    def forward(self,
                hiddens,
                seq_lens,
                intent_pro=None,
                slot_pro=None,
                is_flat=False,
                force_intent=None,
                force_slot=None,
                topk = 3):
        if is_flat:
            slot_pro = flat2matrix(slot_pro, seq_lens)
            hiddens = flat2matrix(hiddens, seq_lens)
            intent_pro = flat2matrix(intent_pro, seq_lens) #batch_size, seq_len, hidden_dim
            if force_intent is not None:
                force_intent = flat2matrix(force_intent.unsqueeze(-1),
                                         seq_lens).squeeze(-1)
            if force_slot is not None:
                force_slot = flat2matrix(force_slot.unsqueeze(-1),
                                         seq_lens).squeeze(-1)

        for model in self.block_list:
            hiddens, pos_hiddens, intent_pro, slot_pro = model(
                hiddens, seq_lens, intent_pro, slot_pro, True, force_intent,
                force_slot,  topk = topk)

        if is_flat:
            slot_pro = matrix2flat(slot_pro, seq_lens)
            hiddens = matrix2flat(hiddens, seq_lens)
            pos_hiddens = matrix2flat(pos_hiddens, seq_lens)

        return hiddens, pos_hiddens, intent_pro, slot_pro


class RefineBlock(nn.Module):
    """
        产生负样例
    """
    def __init__(self,
                 hidden_size,
                 intent_num,
                 slot_num,
                 graph_hidden_dim,
                 graph_output_dim,
                 alpha=0.2,
                 dropout=0.3,
                 nhead=8,
                 window=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.intent_num = intent_num
        self.slot_num = slot_num
        self.intent_embedding = nn.Parameter(
            torch.FloatTensor(self.intent_num, self.hidden_size))
        nn.init.normal_(self.intent_embedding.data)

        self.slot_embedding = nn.Parameter(
            torch.FloatTensor(self.slot_num, self.hidden_size))
        nn.init.normal_(self.slot_embedding.data)
        self.window = window
        # self.BI_gat = GAT(hidden_size, graph_hidden_dim, graph_output_dim, dropout, alpha, nhead)
        # self.O_gat = GAT(hidden_size, graph_hidden_dim, graph_output_dim, dropout, alpha, nhead)
        self.gat = GAT(hidden_size, graph_hidden_dim, graph_output_dim,
                       dropout, alpha, nhead)
        self.intent_decoder = MLPAdapter('qin', hidden_size, intent_num)
        self.slot_decoder = MLPAdapter('qin', hidden_size, slot_num)

    def forward(self,
                hiddens,
                seq_lens,
                intent_pro=None,
                slot_pro=None,
                is_flat=False,
                force_intent=None,
                force_slot=None,
                topk=5):

        # [batch_size, num_intent, hidden_dim]  7 => .unsqueeze(0).repeat(batch_size, 1, 1)
        # [batch_size, seq_len, hidden_dim]  15  =>
        # seq_len [batch,len]

        # intent_pro [batch, intent_num]
        # slot_pro [batch,max_seq_len, slot_num]

        _, intent_idx = torch.topk(intent_pro, topk, dim=-1)
        # intent_idx = torch.unsqueeze(intent_idx,1)
        _, slot_idx = torch.topk(slot_pro, topk,
                                 dim=-1)  # [batch_size, seq_len, 3]

        if force_slot is None:
            force_slot = slot_idx
        else:
            force_slot = force_slot.unsqueeze(-1)

        if force_intent is None:
            force_intent = intent_idx
        else:
            force_intent = force_intent.unsqueeze(-1)

        # intent_idx [batch,1]
        # slot_idx [batch, seq_len]

        batch_size = hiddens.shape[0]
        intent_net = self.intent_embedding.unsqueeze(0).repeat(
            batch_size, 1, 1)
        slot_net = self.slot_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        # H = [batch_size, num_intent + seq_len + num_slot, hidden_dim] 7 + 15 + 72
        # print("hidden.shape", hiddens.shape)
        # print("intent_net.shape", intent_net.shape)
        # print("slot_net.shape", slot_net.shape)

        H = torch.cat([hiddens, intent_net, slot_net], dim=1)
        batch_size, adj_node_num, hidden_dim = H.shape

        # adj = [94, 94]
        # initialize an adj with the batch_size

        slot_num = self.slot_num
        intent_num = self.intent_num
        seq_len = adj_node_num - intent_num - slot_num

        adj = self.create_adj(intent_idx, slot_idx, batch_size, adj_node_num,
                              slot_num, intent_num, seq_len).to('cuda')
        adj_pos = self.create_adj(force_intent, force_slot, batch_size,
                                  adj_node_num, slot_num, intent_num,
                                  seq_len).to('cuda')

        hidden = self.gat(H, adj)[:, :seq_len, :]
        pos_hidden = self.gat(H, adj_pos)[:, :seq_len, :]

        # hidden
        # update the intent_idx,slot_idx

        #mean pooling [batch, hidden]

        intent_pro = self.intent_decoder(hidden)
        slot_pro = self.slot_decoder( hidden)  #[batch, seq_len, hidden] => [batch, seqlen, num_slot]

        # intent_net.add_()

        return hidden, pos_hidden, intent_pro, slot_pro

    # 把选中的intent和所有的hidden连接起来，把选中的intent和所有的hidden连接起来
    def create_adj(self,
                   intent_idx,
                   slot_idx,
                   batch_size,
                   adj_node_num,
                   slot_num,
                   intent_num,
                   seq_len):
        """
            Intent_idx: Top3的index batch_size, 3
            Slot_idx: Top3的index batch_size, seq_len, 3

        """
        adj = torch.cat(
            [torch.eye(adj_node_num).unsqueeze(0) for _ in range(batch_size)])

        for batch_idx in range(batch_size):
            for token_idx in range(seq_len):
                for s_idx in slot_idx[batch_idx][token_idx]:
                    adj[batch_idx, seq_len + intent_num + s_idx,
                        seq_len + intent_idx[batch_idx][token_idx]] = 1.0
                    adj[batch_idx, seq_len + intent_idx[batch_idx][token_idx],
                        seq_len + intent_num + s_idx] = 1.0

                adj[batch_idx, token_idx, seq_len +
                    intent_idx[batch_idx][token_idx]] = 1.0  # word2Top3Intent
                adj[batch_idx, seq_len + intent_idx[batch_idx][token_idx],
                    token_idx] = 1.0  # word2Top3Intent

                adj[batch_idx, token_idx, slot_idx[batch_idx][token_idx] +
                    seq_len + intent_num] = 1.0  # Word2Slot

                adj[batch_idx,
                    slot_idx[batch_idx][token_idx] + seq_len + intent_num,
                    token_idx] = 1.0  # Slot2Word

               
        for i in range(batch_size):
            for j in range(seq_len):
                adj[i, j, max(0, j - self.window):j + self.window + 1] = 1.0  # word2word 

        return normalize_adj(adj)

    # def create_neg_adj(self,
    #                    intent_idx,
    #                    slot_idx,
    #                    batch_size,
    #                    adj_node_num,
    #                    slot_num,
    #                    intent_num,
    #                    seq_len):
    #     adj = torch.cat(
    #         [torch.eye(adj_node_num).unsqueeze(0) for _ in range(batch_size)])

    #     for batch_idx in range(batch_size):
    #         for j in intent_idx[batch_idx]:
    #             adj[batch_idx, j + seq_len, :seq_len] = 1.0  # Top3Intent2Word
    #             adj[batch_idx, :seq_len, j + seq_len] = 1.0  # Word2Top3Intent

    #             adj[batch_idx, j + seq_len, seq_len + intent_num +
    #                 slot_idx[batch_idx]] = 1.0  # Top3Intent 2 slot

    #         for j in slot_idx[batch_idx]:
    #             adj[batch_idx,
    #                 j + seq_len + intent_num, :seq_len] = 1.0  #Slot2Word
    #             adj[batch_idx, :seq_len,
    #                 j + seq_len + intent_num] = 1.0  # Word2Slot
    #             adj[batch_idx, j + seq_len + intent_num,
    #                 seq_len + intent_idx[batch_idx]] = 1.0  # Slot2Intent

    #     for i in range(batch_size):
    #         for j in range(seq_len):
    #             adj[i, j, max(0, j - window):j + window + 1] = 1.0  # Slot2Slot
    #     return None

    # def create_pos_adj(self,
    #                    intent_idx,
    #                    slot_idx,
    #                    batch_size,
    #                    adj_node_num,
    #                    slot_num,
    #                    intent_num,
    #                    seq_len,
    #                    window=2):
    #     adj = torch.cat(
    #         [torch.eye(adj_node_num).unsqueeze(0) for _ in range(batch_size)])

    #     for batch_idx in range(batch_size):
    #         for j in intent_idx[batch_idx]:
    #             adj[batch_idx, j + seq_len, :seq_len] = 1.0  # Top3Intent2Word
    #             adj[batch_idx, :seq_len, j + seq_len] = 1.0  # Word2Top3Intent

    #             adj[batch_idx, j + seq_len, seq_len + intent_num +
    #                 slot_idx[batch_idx]] = 1.0  # Top3Intent 2 slot

    #         for j in slot_idx[batch_idx]:
    #             adj[batch_idx,
    #                 j + seq_len + intent_num, :seq_len] = 1.0  #Slot2Word
    #             adj[batch_idx, :seq_len,
    #                 j + seq_len + intent_num] = 1.0  # Word2Slot
    #             adj[batch_idx, j + seq_len + intent_num,
    #                 seq_len + intent_idx[batch_idx]] = 1.0  # Slot2Intent

    #     for i in range(batch_size):
    #         for j in range(seq_len):
    #             adj[i, j, max(0, j - window):j + window + 1] = 1.0  # Slot2Slot
    #     return None
