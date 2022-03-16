from typing import Tuple

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Dot(nn.Module):
    """Learn from """
    def __init__(self):
        super().__init__()

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, D)
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------

        """
        assert left.size(0) == right.size(0) and left.size(-1) == right.size(-1), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        left = left.unsqueeze(1)  # (B, 1, D)
        tmp = torch.bmm(left, right.permute(0, 2, 1))  # (B, 1, D) * (B, D, L) => (B, 1, L)
        tmp = tmp.squeeze(1)
        doc_mask = (mask == 0)
        out = tmp.masked_fill(doc_mask, -np.inf)
        attention_weights = F.softmax(out, dim=1)  # (B, L)
        avg = right * attention_weights.unsqueeze(-1) # (B, L, D) * (B, L, 1) => (B, L, D)
        assert len(avg.size()) == 3
        avg = torch.sum(avg, dim = 1)  # dim = 1 compute on middel dimension
        return avg, attention_weights


class BiLinear(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, D)
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0) and left.size(-1) == right.size(-1), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        left = self.W(left)  # (B, D)
        left = left.unsqueeze(1)  # (B, 1, D)
        tmp = torch.bmm(left, right.permute(0, 2, 1))  # (B, 1, D) * (B, D, L) => (B, 1, L)
        tmp = tmp.squeeze(1)
        doc_mask = (mask == 0)
        out = tmp.masked_fill(doc_mask, -np.inf)
        attention_weights = F.softmax(out, dim=1)  # (B, L)
        avg = right * attention_weights.unsqueeze(-1)  # (B, L, D) * (B, L, 1) => (B, L, D)
        avg = torch.sum(avg, dim = 1)  # dim = 1 compute on middel dimension
        return avg, attention_weights


class ConcatSelfAtt(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, num_heads: int = 1):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, num_heads, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, X) X is not necessarily equal to D
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        assert self.inp_dim == (left.size(-1) + right.size(-1))  # due to concat
        B, L, D = right.size()
        left_tmp = left.unsqueeze(1).expand(B, L, -1)  # (B, 1, D)
        tsr = torch.cat([left_tmp, right], dim=-1)  # (B, L, 2D)
        # start computing multi-head self-attention
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)
        doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        doc_mask = doc_mask.unsqueeze(-1).expand(B, L, self.num_heads)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim=1)  # (B, L, C)
        attended = torch.bmm(right.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        return attended, attention_weights


class ConcatNotEqualSelfAtt(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, num_heads: int = 1):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, num_heads, bias=False)

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, X) X is not necessarily equal to D
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        assert self.inp_dim == (left.size(-1) + right.size(-1))  # due to concat
        B, L, D = right.size()
        left_tmp = left.unsqueeze(1).expand(B, L, -1)  # (B, 1, X)
        tsr = torch.cat([left_tmp, right], dim=-1)  # (B, L, 2D)
        # start computing multi-head self-attention
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)
        doc_mask = (mask == 0)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        doc_mask = doc_mask.unsqueeze(-1).expand(B, L, self.num_heads)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim=1)  # (B, L, C)
        attended = torch.bmm(right.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        return attended, attention_weights


class BiLinearTanh(nn.Module):

    def __init__(self, left_dim: int, right_dim: int, out_dim: int):
        """
        Implementation of equation v_s^T \tanh(W_1 * h_{ij} + W_s * x + b_s)
        Parameters
        ----------
        left_dim: `int` dimension of left tensor
        right_dim: `int` dimesion of right tensor
        out_dim
        """
        super().__init__()
        self.left_linear = nn.Linear(left_dim, out_dim, bias=True)
        self.right_linear = nn.Linear(right_dim, out_dim, bias=False)
        self.combine = nn.Linear(out_dim, 1, bias=False)

    def forward(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor, mask: torch.Tensor):
        """
        compute attention weights on left tensor based on the right tensor.
        Parameters
        ----------
        left_tsr: `torch.Tensor` of shape (B, L, H)
        right_tsr: `torch.Tensor` of shape (B, D)
        mask: `torch.Tensor` of shape (B, L) 1 is for real, 0 is for pad

        Returns
        -------

        """
        assert len(left_tsr.size()) == 3 and len(mask.size()) == 2
        left = self.left_linear(left_tsr)  # (B, L, O)
        right = self.right_linear(right_tsr).unsqueeze(1)  # (B, O)
        tmp = torch.tanh(left + right)  # (B, L, O)
        linear_out = self.combine(tmp).squeeze(-1)  # (B, L)  it is equal to v_s^T \tanh(W_1 * h_{ij} + W_2 * a + b_s)
        doc_mask = (mask == 0)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim = -1)  # (B, L)
        attended = left_tsr * attention_weights.unsqueeze(-1)  # (B, L, H)
        attended = torch.sum(attended, dim = 1)  # (B, H)
        return attended, attention_weights


class MultiHeadAttentionSimple(nn.Module):

    def __init__(self, num_heads: int, d_model: int, d_key: int, d_value: int,
                 # attention_type: int = AttentionType.ConcatNotEqual,
                 init_weights: bool = False,
                 use_layer_norm: bool = False):
        """
        Simple multi-head attention and customizable with layer-norm
        Parameters
        ----------
        num_heads: `int` the number of heads. how many aspects of the evidences you want to see
        d_model: `int` input embedding size
        d_key: `int` dimension of keys. We will set d_key = d_model
        d_value: `int` dimensions of key, d_value = d_model
        init_weights: `bool` whether we should init linear layers.
        use_layer_norm: `bool` whether we should use layer-norm
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model, self.d_key, self.d_value = d_model, d_key, d_value
        assert d_model == d_key == d_value
        self.use_layer_norm = use_layer_norm
        self.w_qs = nn.Linear(d_model, num_heads * d_key)  # gom tat ca head vo 1 matrix de nhan co de
        self.w_ks = nn.Linear(d_model, num_heads * d_key)
        self.w_vs = nn.Linear(d_model, num_heads * d_value)
        if init_weights:
            nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_key)))
            nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_key)))
            nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_value)))

        # if attention_type == AttentionType.ConcatNotEqual:
        self.attention_func = ConcatNotEqualSelfAttTransFormer(inp_dim=(d_key + d_key), out_dim=d_key)
        # else:
        #     self.attention_func = ScaledDotProductAttention(temperature=np.power(d_key, 0.5))

        self.fc = nn.Linear(num_heads * d_value, d_model)
        if init_weights: nn.init.xavier_normal_(self.fc.weight)
        if use_layer_norm: self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        left: `torch.Tensor` of shape (B, X) X is not necessarily equal to D
        right: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert left.size(0) == right.size(0), "Must same dimensions"
        assert len(left.size()) == 2 and len(right.size()) == 3
        B, L, D = right.size()
        assert D == self.d_model == self.d_key, "Must have same shape"
        len_q = 1
        # transform
        query = self.w_qs(left).view(B, len_q, self.num_heads, self.d_key)  # (B, 1, num_heads, d_key)
        key = self.w_ks(right).view(B, L, self.num_heads, self.d_key)  # (B, L, num_heads, d_key)
        value = self.w_vs(right).view(B, L, self.num_heads, self.d_value)  # (B, L, num_heads, d_value)
        # reshape
        q = query.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_key)  # (num_heads * B) x 1 x dk
        k = key.permute(2, 0, 1, 3).contiguous().view(-1, L, self.d_key)  # (num_heads * B) x L x dk
        v = value.permute(2, 0, 1, 3).contiguous().view(-1, L, self.d_value)  # (num_heads * B) x L x dv
        # compute attention weights
        mask = (mask == 0)
        mask = mask.unsqueeze(1).repeat(self.num_heads, 1, 1)  # (B * num_heads, 1, L)
        attended, attention_weights = self.attention_func(query=q, key=k, value=v, mask=mask)
        # concat all heads and push to MLP followed by optional layer_norm
        output = attended.view(self.num_heads, B, len_q, self.d_value)
        output = output.permute(1, 2, 0, 3).contiguous().view(B, len_q, -1)  # b x lq x (n*dv)

        tmp = self.fc(output)
        if self.use_layer_norm: tmp = self.layer_norm(tmp)
        return tmp, attention_weights


class MultiHeadAttentionOriginal(nn.Module):
    ''' Multi-Head Attention module copied from PyTorch Transformer '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        """

        Parameters
        ----------
        n_head: `int` number of attention layers or heads
        d_model: `int` what the fuck is d_model? is it word embedding size?
        d_k
        d_v
        dropout
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)  # gom tat ca head vo 1 matrix de nhan co de
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        # nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        # nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        # self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.attention = ScaledDotProductAttention(temperature=np.power(1, 1))

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        # nn.init.xavier_normal_(self.fc.weight)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """

        Parameters
        ----------
        q: `torch.Tensor` of shape (B, L, D)
        k: `torch.Tensor` of shape (B, R, D)
        v: `torch.Tensor` of shape (B, R, D)
        mask: `torch.Tensor` of shape (B, L, R) (very important, 1 is for

        Returns
        -------

        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..  (quite redundant here)
        output, _ = self.attention(q, k, v, mask = mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        # output = self.dropout(self.fc(output))
        output = self.fc(output)
        output = self.layer_norm(output + residual)

        return output, None


class ConcatNotEqualSelfAttTransFormer(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        # self.num_heads = num_heads
        self.linear1 = nn.Linear(inp_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute attention weights and apply it to `right` tensor
        Parameters
        ----------
        query: `torch.Tensor` of shape (B, 1, X) X is not necessarily equal to D
        key: `torch.Tensor` of shape (B, L, D)
        value: `torch.Tensor` of shape (B, L, D)
        mask: `torch.Tensor` of shape (B, L), binary value, 0 is for pad

        Returns
        -------
        """
        assert query.size(0) == key.size(0), "Must same dimensions"
        # assert len(query.size()) == 2 and len(key.size()) == 3
        assert self.inp_dim == (query.size(-1) + key.size(-1))  # due to concat
        B, L, D = key.size()
        left_tmp = query.expand(B, L, -1)  # (B, 1, X)
        tsr = torch.cat([left_tmp, key], dim=-1)  # (B, L, 2D)
        # start computing multi-head self-attention
        tmp = torch.tanh(self.linear1(tsr))  # (B, L, out_dim)
        linear_out = self.linear2(tmp)  # (B, L, C)
        doc_mask = mask.squeeze(1).unsqueeze(-1)  # (B, L) real tokens will be zeros and pad will have non zero (this is for softmax)
        # doc_mask = doc_mask.unsqueeze(-1).expand(B, L, 1)  # (B, L, C)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)  # I learned from Attention is all you need
        # we now can ensure padding tokens will not contribute to softmax
        attention_weights = F.softmax(linear_out, dim=1)  # (B, L, C)
        attended = torch.bmm(value.permute(0, 2, 1), attention_weights)  # (B, D, L) * (B, L, C) => (B, D, C)
        return attended, attention_weights


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)
        # self.softmax = nn.Softmax(dim=-1)  # are you sure the dimension is correct?

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        """

        Parameters
        ----------
        query: `torch.Tensor` (n_heads * B, L, d_k)
        key: `torch.Tensor` (n_heads * B, L, d_k)
        value: `torch.Tensor` (n_heads * B, L, d_k)
        mask (n_heads * B, L, L) (this is I guess to remove padding tokens

        Returns
        -------

        """
        attn = torch.bmm(query, key.transpose(1, 2))
        # attn = attn / self.temperature

        if mask is not None: attn = attn.masked_fill(mask, -np.inf)
        attn = F.softmax(attn, dim = -1)  # exp of -np.inf would be zero (checked)
        attn = attn.masked_fill(mask, 0)  # reset nan
        # attn = self.dropout(attn)  # why there is a fucking shit dropout here???? (I've never seen this before)
        output = torch.bmm(attn, value)
        return output, attn


class CoDaAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

    def forward(self, *input):
        pass