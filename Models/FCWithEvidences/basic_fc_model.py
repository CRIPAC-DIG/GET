import torch
import torch.nn.functional as F
import torch.nn as nn
from matchzoo.utils import parse
from Models.base_model import BaseModel
from matchzoo.modules import GaussianKernel
import torch_utils
from setting_keywords import KeyWordSettings
import numpy as np
from Models.BiDAF.wrapper import LSTM
from thirdparty.self_attention import MultiHeadSelfAttentionICLR2017Extend


class BasicFCModel(BaseModel):
    """
    Basic Fact-checking model used for all other models
    """
    def __init__(self, params):
        super(BaseModel, self).__init__()
        self._params = params
        self.embedding = self._make_default_embedding_layer(params)
        self.num_classes = self._params["num_classes"]
        self.fixed_length_right = self._params["fixed_length_right"]
        self.fixed_length_left = self._params["fixed_length_left"]
        self.use_claim_source = self._params["use_claim_source"]
        self.use_article_source = self._params["use_article_source"]
        self._use_cuda = self._params["cuda"]
        self.num_heads = 1  # self._params["num_att_heads"]
        self.dropout_left = self._params["dropout_left"]
        self.dropout_right = self._params["dropout_right"]
        self.hidden_size = self._params["hidden_size"]
        if self.use_claim_source:
            self.claim_source_embs = self._make_entity_embedding_layer(
                self._params["claim_source_embeddings"], freeze = False)  # trainable
            self.claim_emb_size = self._params["claim_source_embeddings"].shape[1]

        if self.use_article_source:
            self.article_source_embs = self._make_entity_embedding_layer(
                self._params["article_source_embeddings"], freeze = False)  # trainable
            self.article_emb_size = self._params["article_source_embeddings"].shape[1]

        D = self._params["embedding_output_dim"]
        # self.linear1 = nn.Sequential(
        #     nn.Linear(self._params["embedding_output_dim"] + 3 * D, 1),
        #     # self.activation
        #     nn.Tanh()
        # )
        # self.linear1[0].apply(torch_utils.init_weights)
        self.bilstm = LSTM(input_size = D, hidden_size = self.hidden_size, num_layers = 1, bidirectional=True,
                           batch_first=True, dropout = self.dropout_left)
        self.query_bilstm = LSTM(input_size=D, hidden_size=self.hidden_size, num_layers=1, bidirectional=True,
                                 batch_first=True, dropout = self.dropout_right)
        input_size = 4 * self.hidden_size + self.claim_emb_size if self.use_claim_source else 4 * self.hidden_size
        self.self_att_word = MultiHeadSelfAttentionICLR2017Extend(inp_dim=input_size, out_dim = 2 * self.hidden_size,
                                                                  num_heads=self.num_heads)

        evd_input_size = 4 * self.hidden_size * self.num_heads
        if self.use_article_source:
            evd_input_size += self.article_emb_size
            input_size += self.article_emb_size
        if self.use_claim_source:
            evd_input_size += self.claim_emb_size
            # input_size += self.claim_emb_size
        self.self_att_evd = MultiHeadSelfAttentionICLR2017Extend(inp_dim=evd_input_size, out_dim = 2 * self.hidden_size,
                                                                 num_heads=self.num_heads)
        self.out = nn.Sequential(
            nn.Linear(input_size * self.num_heads, self.hidden_size),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            nn.Linear(self.hidden_size, 1),
            # nn.ReLU(),  # no one uses ReLU at the end of a linear layer
            # nn.Sigmoid()
        )
        self.out[0].apply(torch_utils.init_weights)
        self.out[1].apply(torch_utils.init_weights)

    def forward(self, query: torch.Tensor, document: torch.Tensor, verbose = False, **kargs):
        pass

    def _pad_left_tensor(self, left_tsr: torch.Tensor, **kargs):
        """ pad left tensor of shape (B, H) to tensor of shape (n1 + n2 + ... + nx, H) """
        evd_count_per_query = kargs[KeyWordSettings.EvidenceCountPerQuery]
        B, H = left_tsr.size()
        assert evd_count_per_query.size(0) == left_tsr.size(0)
        ans = []
        for num_evd, tsr in zip(evd_count_per_query, left_tsr):
            # num_evd = evd_count_per_query[idx] # int(torch_utils.cpu(evd_count_per_query[idx]).detach().numpy())
            tmp = tsr.clone()
            tsr = tmp.expand(num_evd, H)
            ans.append(tsr)
        ans = torch.cat(ans, dim = 0)  # (n1 + n2 + ... + nx, H)
        return ans

    @classmethod
    def _pad_right_tensor(self, tsr: torch.Tensor, **kargs):
        """
        padding the output evidences. I avoid input empty sequence into lstm due to exception. I tried to make add mask
        to empty sequence but I don't have much belief in it.
        Parameters
        ----------
        bilstm_out: `torch.Tensor` (n1 + n2 + ... + n_B, H)
        doc_src: `torch.Tensor` (B, n) where n is the maximum number of evidences
        Returns
        -------

        """
        max_num_evd = kargs[KeyWordSettings.FIXED_NUM_EVIDENCES]
        evd_count_per_query = kargs[KeyWordSettings.EvidenceCountPerQuery]
        batch_size = evd_count_per_query.size(0)
        b_prime, H = tsr.size()
        last = 0
        ans = []
        for idx in range(batch_size):
            num_evd = int(torch_utils.cpu(evd_count_per_query[idx]).detach().numpy())
            hidden_vectors = tsr[last: last + num_evd]  # (n1, H)
            padded = F.pad(hidden_vectors, (0, 0, 0, max_num_evd - num_evd), "constant", 0)
            ans.append(padded)
            last += num_evd
        ans = torch.stack(ans, dim=0)
        assert ans.size() == (batch_size, max_num_evd, H)
        return ans
