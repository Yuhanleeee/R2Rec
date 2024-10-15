import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None: 
            # mask = mask.unsqueeze(1).unsqueeze(1).repeat(1,corr.shape[1],1,1)
            mask = mask.unsqueeze(1).repeat(1, mask.shape[1], 1).unsqueeze(1).repeat(1, corr.shape[1], 1, 1)
            corr = corr.masked_fill(mask == 0, -1e9)
            
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden
    
    
class Vanilla_TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(Vanilla_TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Preimgseqrec(nn.Module):
    def __init__(self, args):
        super(Preimgseqrec, self).__init__()
        self.Trans_seq_rep = nn.ModuleList(
            [Vanilla_TransformerBlock(args.hidden_size, args.attn_head, args.dropout) for _ in range(args.n_blocks)])
        
    def forward(self, hidden, mask):
        for transformer_temp in self.Trans_seq_rep:
            hidden = transformer_temp.forward(hidden, mask)
        hidden = hidden[:,-1,:]
        return hidden


class Idseqrec(nn.Module):
    def __init__(self, args):
        super(Idseqrec, self).__init__()
        self.item_emb = torch.nn.Embedding(args.item_nums+1, 768)
        self.Trans_seq_rep = nn.ModuleList([Vanilla_TransformerBlock(args.hidden_size, args.attn_head, args.dropout) for _ in range(args.n_blocks)])
        self.pred_layer = Linear_pred(args, self.item_emb.weight.shape[0])
    
    def forward(self, batch):
        masks = torch.stack(batch['mask_seq'], dim=-1)
        seq_id = torch.stack(batch['seq_id'], dim=1)
        hidden = self.item_emb(seq_id)
        for transformer_temp in self.Trans_seq_rep:
            hidden = transformer_temp.forward(hidden, masks)
        prod = self.pred_layer(hidden[:,-1,:])
        return prod
        

class Linear_pred(nn.Module):
    def __init__(self, args, nums_item):
        super(Linear_pred, self).__init__()
        self.pred_linear = nn.Linear(args.hidden_size, nums_item, bias=False)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.pred_linear.weight)
    
    def forward(self, rep_seq):
        return self.pred_linear(rep_seq)