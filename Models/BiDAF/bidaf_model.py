import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.BiDAF.wrapper import LSTM, Linear
from Models.base_model import BaseModel
from setting_keywords import KeyWordSettings
import torch_utils
import numpy as np


class BiDAF(BaseModel):
    """
    BiDAF model (mainly copied from https://github.com/galsang/BiDAF-pytorch/blob/master/model/bidaf_model.py).
    """
    def __init__(self, params):
        super(BiDAF, self).__init__()
        self._params = params
        self.word_emb = self._make_default_embedding_layer(params)

        for i in range(2):
            setattr(self, 'highway_linear%s' % i,
                    nn.Sequential(Linear(params["word_dim"], params["word_dim"]), nn.ReLU()))
            setattr(self, 'highway_gate%s' % i,
                    nn.Sequential(Linear(params["word_dim"], params["word_dim"]), nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=params["word_dim"],
                                 hidden_size=params["hidden_size"],
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=params["dropout"])

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(params["hidden_size"] * 2, 1)
        self.att_weight_q = Linear(params["hidden_size"] * 2, 1)
        self.att_weight_cq = Linear(params["hidden_size"] * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=params["hidden_size"] * 8,
                                   hidden_size=params["hidden_size"],
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=params["dropout"])

        self.dropout = nn.Dropout(p=params["dropout"])

        self.last_linear = torch.nn.Linear(2 * params["hidden_size"], 1)

    def forward(self, query: torch.Tensor, document: torch.Tensor, verbose = False, **kargs):

        def highway_network(x):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            # x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear%s' % i)(x)
                g = getattr(self, 'highway_gate%s' % i)(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)
            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)

            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)


            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()
            return p1, p2
        """Forward. of integer query tensor and document tensor """
        max_left_len, max_right_len = query.size(1), document.size(1)
        # Process left & right input.
        # shape = [B, L, D], we need to swap last two dimensions to ensure shape [B, D, L] for nn.ConstantPad1d
        q_word = self.word_emb(query.long())  # .transpose(1, 2)
        # shape = [B, R, D], we need to swap last two dimensions to ensure shape [B, D, R] for nn.ConstantPad1d
        c_word = self.word_emb(document.long())  # .transpose(1, 2)
        # 2. Word Embedding Layer

        # Highway network
        # c = highway_network(torch.cat([c_char, c_word], dim = -1))
        # q = highway_network(torch.cat([q_char, q_word], dim = -1))
        c = highway_network(c_word)
        q = highway_network(q_word)

        q_new_indices, q_restoring_indices, q_lens = kargs["query_lens_indices"]
        d_new_indices, d_restoring_indices, c_lens = kargs["doc_lens_indices"]
        # c_lens = torch.from_numpy(c_lens)
        # q_lens = torch.from_numpy(q_lens)
        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c, c_lens, d_new_indices, d_restoring_indices))[0]
        q = self.context_LSTM((q, q_lens, q_new_indices, q_restoring_indices))[0]
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)
        # 5. Modeling Layer
        # m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens, d_new_indices, d_restoring_indices))[0],
        #                          c_lens, d_new_indices, d_restoring_indices))[1]
        m = self.modeling_LSTM1((g, c_lens, d_new_indices, d_restoring_indices))[1]
        # 6. Output Layer
        # p1, p2 = output_layer(g, m, c_lens)
        last_state = m  # (B, dimension)
        return self.last_linear(last_state)
        # (batch, c_len), (batch, c_len)
        # return p1, p2

    def predict(self, query: np.ndarray, doc: np.ndarray, verbose: bool = False, **kargs):
        assert KeyWordSettings.Query_lens in kargs and KeyWordSettings.Doc_lens in kargs

        self.train(False)  # very important, to disable dropout

        queries_lens, docs_lens = kargs[KeyWordSettings.Query_lens], kargs[KeyWordSettings.Doc_lens]
        queries_lens, docs_lens = np.array(queries_lens), np.array(docs_lens)

        q_new_indices, q_old_indices = torch_utils.get_sorted_index_and_reverse_index(queries_lens)
        d_new_indices, d_old_indices = torch_utils.get_sorted_index_and_reverse_index(docs_lens)
        if verbose:
            print("query: ", query)
            print("doc: ", doc)
            print("================ end of query doc =================")
        out = self(query, doc, verbose=False,
                   query_lens_indices=(q_new_indices, q_old_indices, queries_lens),
                   doc_lens_indices=(d_new_indices, d_old_indices, docs_lens))
        return torch_utils.cpu(out).detach().numpy().flatten()
