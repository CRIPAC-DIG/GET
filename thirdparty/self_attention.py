import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from enum import IntEnum


class SelfAttentionType(IntEnum):
    MultiHeadAttentionTanh = 1  # ICLR 2017
    MultiHeadAttentionTransformer = 2  # NIPS 2017


class SelfAttentionICLR2017(nn.Module):
    """
    This is implementation of self-attention in ICLR 2017 paper
    A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING, https://arxiv.org/pdf/1703.03130.pdf
    """
    def __init__(self, inp_dim: int, out_dim: int, num_heads: int = 1):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.linear1 = nn.Linear(inp_dim, out_dim, bias = False)
        self.linear2 = nn.Linear(out_dim, num_heads, bias = False)

    def forward(self, tsr: torch.Tensor, mask: torch.Tensor):
        """

        Parameters
        ----------
        tsr: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L) each value is either zero or 1

        Returns
        -------

        """
        assert len(tsr.size()) == 3
        assert tsr.size(-1) == self.inp_dim
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        out = self.linear2(tmp)  # (B, L, 1)
        linear_out = out.squeeze(-1)  # (B, L)
        doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim=-1)  # (B, L)
        avg = tsr * attention_weights.unsqueeze(-1)  # (B, L, D) * (B, L, 1) zeros will be output by bilstm
        avg = torch.sum(avg, dim=1)  # (B, D)
        return avg


class MultiHeadSelfAttentionICLR2017Extend(nn.Module):
    """
        This is implementation of self-attention in ICLR 2017 paper
        A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING, https://arxiv.org/pdf/1703.03130.pdf
        Using multi-heads
        SelfAttentionType.MultiHeadAttentionTanh
    """

    def __init__(self, inp_dim: int, out_dim: int, num_heads: int):
        """

        Parameters
        ----------
        inp_dim
        out_dim
        num_heads: `int` the number of heads. I preferred `num_heads` equal to token size
        """
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, num_heads, bias=False)

    def forward(self, tsr: torch.Tensor, mask: torch.Tensor, return_att_weights = False):
        """

        Parameters
        ----------
        tsr: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L) each value is either zero or 1
        return_att_weights: `bool` return attention weight for error anlysis
        Returns
        -------

        """
        batch_size, L, D = tsr.size()
        assert len(tsr.size()) == 3
        assert tsr.size(-1) == self.inp_dim
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)
        doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        doc_mask = doc_mask.unsqueeze(-1).expand(batch_size, L, self.num_heads)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim = 1)  # (B, L, C)
        attended = torch.bmm(tsr.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        if return_att_weights:
            return attended.permute(0, 2, 1), attention_weights
        return attended.permute(0, 2, 1)  # (B, C, D)


class MultiHeadSelfAttentionICLR17OnWord(nn.Module):
    """
        This is implementation of self-attention in ICLR 2017 paper
        A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING, https://arxiv.org/pdf/1703.03130.pdf
        Using multi-heads
        SelfAttentionType.MultiHeadAttentionTanh
    """

    def __init__(self, inp_dim: int, out_dim: int, num_heads: int):
        """

        Parameters
        ----------
        inp_dim
        out_dim
        num_heads: `int` the number of heads. I preferred `num_heads` equal to token size
        """
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, num_heads, bias=False)

    def forward(self, original: torch.Tensor, tsr: torch.Tensor, mask: torch.Tensor, return_att_weights = False):
        """

        Parameters
        ----------
        original: `torch.Tensor` of shape (B, L, X) this is we want to compute weight average
        tsr: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L) each value is either zero or 1
        return_att_weights: `bool` return attention weight for error anlysis
        Returns
        -------

        """
        batch_size, L, D = tsr.size()
        assert len(tsr.size()) == 3
        assert tsr.size(-1) == self.inp_dim
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)
        doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        doc_mask = doc_mask.unsqueeze(-1).expand(batch_size, L, self.num_heads)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim = 1)  # (B, L, C)
        attended = torch.bmm(original.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        if return_att_weights:
            return attended.permute(0, 2, 1), attention_weights
        return attended.permute(0, 2, 1)  # (B, C, D)
