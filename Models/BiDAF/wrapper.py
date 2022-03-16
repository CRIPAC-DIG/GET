import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Graph Attention Unit: Graph Attention Layer
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

        # self.W = Linear(in_features, out_features)
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # self.a = Linear(2*out_features, 1)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        """
        :param input: 输入特征 (batch, len, in_features)
        :param adj:  邻接矩阵 (batch, len, len)
        :return: 输出特征 (batch,len, out_features)
        """
        h = torch.matmul(input, self.W)     # (batch, len, out_features)
        # a_input = torch.cat([h.repeat(1, 1, L)  # (batch, len, out_features*len)
        #                     .view(B, L * L, -1),  # (batch, len*len, out_features)
        #                      h.repeat(1, L, 1)],  # (batch, len*len,out_features)
        #                     dim=2).view(B, L, -1, 2 * self.out_features)  # (batch, len, len, 2 * out_features)

        # 通过刚刚的矩阵与权重矩阵a相乘计算每两个样本之间的相关性权重，最后再根据邻接矩阵置零没有连接的权重
        e = self._prepare_attentional_mechanism_input(h)                # (batch,L,L)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (batch,L,L)
        # 置零的mask
        zero_vec = -9e15 * torch.ones_like(e)                           # (batch, len, len)
        attention = torch.where(adj > 0, e, zero_vec)                   # (batch,batch) 有相邻就为e位置的值，不相邻则为0
        attention = F.softmax(attention, dim=2)                         # (batch,len,len)
        attention = F.dropout(attention, self.dropout, training=self.training)  # (batch,batch)
        h_prime = torch.matmul(attention, h)                            # (batch,len,out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])       # (B, L ,1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])       # (B, L, 1)
        # broadcast add
        e = Wh1 + Wh2.transpose(1, 2)                               # (B, L, L)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,  head_num=3, num_layers=1, dropout=0.6, alpha=0.2,):
        """
        Dense version of GAT.
        :param input_size: 输入特征的维度
        :param hidden_size:  输出特征的维度
        :param output_size: 输出特征的维度
        :param dropout: dropout
        :param alpha: LeakyRelu中的参数
        :param head_num: 多头注意力机制的个数
        """
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = []
        # 多头注意力机制
        for num_layer in range(num_layers-1):
            self.attentions.append([GraphAttentionLayer(input_size, hidden_size, dropout=dropout,
                                                        alpha=alpha, concat=True) for _ in range(head_num)])
            input_size = hidden_size * head_num

        for i, layer in enumerate(self.attentions):         # 神经网络子类需注册:直接定义,add_module,nn.ModuleList
            for j, attention in enumerate(layer):
                self.add_module('layer_{}_{}'.format(i, j), attention)

        # 输出层对output_size求平均
        self.out_att = nn.ModuleList([GraphAttentionLayer(input_size, output_size, dropout=dropout,
                                      alpha=alpha, concat=False) for _ in range(head_num)])
        # self.attentions = [GraphAttentionLayer(input_size, hidden_size1, dropout=dropout,
        #                                        alpha=alpha, concat=True) for
        #                                        _ in range(head_num)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        L = x.size()[1]
        x = F.dropout(x, self.dropout, training=self.training)

        for attention in self.attentions:
            x = torch.cat([att.forward(x, adj) for att in attention], dim=2)

        x = F.dropout(x, self.dropout, training=self.training)
        x = sum([att.forward(x, adj) for att in self.out_att])/L
        return F.relu(x)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(GCN, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        Linear_list = []
        for num in range(self.num_layers):
            Linear_list.append(Linear(input_dim, hidden_dim if num==num_layers-1 else output_dim))
            input_dim = hidden_dim
        self.Linear = nn.ModuleList(Linear_list)

    def laplacian_normalize(self, adj):
        row_sum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        d_mat_inv_sqrt = torch.stack(
            [d_mat_inv_sqrt[adj.shape[1] * cnt:adj.shape[1] * (cnt + 1), adj.shape[1] * cnt:adj.shape[1] * (cnt + 1)]
             for cnt in range(adj.shape[0])], dim=0)
        L_norm = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj.type_as(d_mat_inv_sqrt)), d_mat_inv_sqrt)
        return L_norm

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)

        # K-ordered Chebyshev polynomial
        n = adj.shape[-1]
        adj_norm = self.laplacian_normalize(adj.view(-1, n, n))
        adj_norm = adj_norm.view(adj.shape)
        for linear in self.Linear:
            x = F.relu(linear(torch.matmul(adj_norm, x)))

        return x

class GGNN_with_GSL(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rate=0.8, dropout=0.2):
        super(GGNN_with_GSL, self).__init__()

        self.feat_prop1 = GGNN(input_dim, hidden_dim, dropout)
        self.word_scorer1 = GGNN(hidden_dim, 1, dropout)
        self.gsl1 = GSL(rate)

        self.feat_prop2 = GGNN(hidden_dim, output_dim, dropout)
        # self.word_scorer2 = GGNN(output_dim, 1, dropout)
        # self.gsl2 = GSL(rate)
    
    def forward(self, adj, feat):
        feat = self.feat_prop1(adj, feat)
        score = self.word_scorer1(adj, feat)
        adj_refined = self.gsl1(adj, score)
        feat = self.feat_prop2(adj_refined, feat)
        # score = self.word_scorer2(adj_refined, feat)
        # adj_refined = self.gsl2(adj_refined, score)
        return feat

class GGNN(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(GGNN, self).__init__()
        self.proj = Linear(in_features, out_features, bias=False)
        self.linearz0 = Linear(out_features, out_features)
        self.linearz1 = Linear(out_features, out_features)
        self.linearr0 = Linear(out_features, out_features)
        self.linearr1 = Linear(out_features, out_features)
        self.linearh0 = Linear(out_features, out_features)
        self.linearh1 = Linear(out_features, out_features)
        
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, adj, x):
        if hasattr(self, 'dropout'): 
            x = self.dropout(x)
        x = self.proj(x)
        a = adj.matmul(x)

        z0 = self.linearz0(a)
        z1 = self.linearz1(x)
        z = torch.sigmoid(z0 + z1)

        r0 = self.linearr0(a)
        r1 = self.linearr1(x)
        r = torch.sigmoid(r0 + r1)

        h0 = self.linearh0(a)
        h1 = self.linearh1(r*x)
        h = torch.tanh(h0 + h1)

        feat = h*z + x*(1-z)
    
        return feat

class GSL(nn.Module):
    def __init__(self, rate):
        super(GSL, self).__init__()
        self.rate = rate

    def forward(self, adj, score):
        N = adj.shape[-1]
        BATCH_SIZE = adj.shape[0]
        num_preserve_node = int(self.rate * N)
        _, indices = score.topk(num_preserve_node, 1)
        indices = torch.squeeze(indices, dim=-1)
        mask = torch.zeros([BATCH_SIZE, N, N]).cuda()
        for i in range(BATCH_SIZE):
            mask[i].index_fill_(0, indices[i], 1)
            mask[i].index_fill_(1, indices[i], 1)
        adj = adj * mask
        # feat = torch.tanh(score) * feat
        return adj

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l%s' % i))
            nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l%s' % i))
            nn.init.constant_(getattr(self.rnn, 'bias_hh_l%s' % i), val=0)
            nn.init.constant_(getattr(self.rnn, 'bias_ih_l%s' % i), val=0)
            # getattr(self.rnn, 'bias_hh_l%s' % i).chunk(4)[1].fill_(1)     # unable to change in-place

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l%s_reverse' % i))
                nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l%s_reverse' % i))
                nn.init.constant_(getattr(self.rnn, 'bias_hh_l%s_reverse' % i), val=0)
                nn.init.constant_(getattr(self.rnn, 'bias_ih_l%s_reverse' % i), val=0)
                # getattr(self.rnn, 'bias_hh_l%s_reverse' % i).chunk(4)[1].fill_(1)

    def forward(self, x, return_h=True, max_len=None):
        x, x_len, d_new_indices, d_restoring_indices = x
        x = self.dropout(x)
        # x_idx = d_new_indices
        x_len_sorted = x_len[d_new_indices]
        # x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x[d_new_indices]  # x.index_select(dim=0, index=x_idx)
        x_ori_idx = d_restoring_indices
        # _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted.cpu(), batch_first=True)
        x_packed, (h, c) = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True, total_length=max_len)[0]
        # x = x.index_select(dim=0, index=x_ori_idx)
        x = x[x_ori_idx]
        if return_h:
            h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2))  # .squeeze()
            # h = h.index_select(dim=0, index=x_ori_idx)
            h = h[x_ori_idx]
        return x, h


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l%s' % i))
            nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l%s' % i))
            nn.init.constant_(getattr(self.rnn, 'bias_hh_l%s' % i), val=0)
            nn.init.constant_(getattr(self.rnn, 'bias_ih_l%s' % i), val=0)
            getattr(self.rnn, 'bias_hh_l%s' % i).chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l%s_reverse' % i))
                nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l%s_reverse' % i))
                nn.init.constant_(getattr(self.rnn, 'bias_hh_l%s_reverse' % i), val=0)
                nn.init.constant_(getattr(self.rnn, 'bias_ih_l%s_reverse' % i), val=0)
                getattr(self.rnn, 'bias_hh_l%s_reverse' % i).chunk(4)[1].fill_(1)

    def forward(self, x, return_h=True, max_len=None):
        x, x_len, d_new_indices, d_restoring_indices = x
        x = self.dropout(x)
        # x_idx = d_new_indices
        x_len_sorted = x_len[d_new_indices]
        # x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x[d_new_indices]  # x.index_select(dim=0, index=x_idx)
        x_ori_idx = d_restoring_indices
        # _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted.cpu(), batch_first=True)
        # x_packed, (h, c) = self.rnn(x_packed)
        x_packed, h = self.rnn(x_packed)  # this is for GRU not LSTM

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True, total_length=max_len)[0]
        # x = x.index_select(dim=0, index=x_ori_idx)
        x = x[x_ori_idx]
        if return_h:
            h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2))  # .squeeze()
            # h = h.index_select(dim=0, index=x_ori_idx)
            h = h[x_ori_idx]
        return x, h


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        if hasattr(self, 'linear.bias'):
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'): x = self.dropout(x)
        x = self.linear(x)
        return x
