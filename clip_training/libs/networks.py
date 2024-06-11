import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    #def __init__(self, d_model, in_vocab_size, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, do_input_embedding=False):
    def __init__(self, d_model, in_vocab_size, nhead=16, num_encoder_layers=8, dim_feedforward=2048, dropout=0.1, do_input_embedding=False):
        super(TransformerEncoder, self).__init__()

        self.nhead = nhead
        if d_model % self.nhead != 0:
            d_model = d_model + self.nhead - d_model % self.nhead

        self.d_model = d_model
        self.do_input_embedding = do_input_embedding # indicate whether the inputs are ID and need to do embedding
        if do_input_embedding:
            self.enc_embedding = nn.Embedding(in_vocab_size, d_model)
        else:
            self.enc_embedding = nn.Linear(in_vocab_size, d_model)
        self.pos_embedding_enc = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, norm=encoder_norm)

    def make_src_mask(self, inp):
        if self.do_input_embedding:
            return inp.transpose(0, 1) == 0
        else:
            return torch.all(inp == 0, dim=-1).transpose(0, 1)


    def forward(self, src, output_type='avg'):
        # src: [src_len, batch_size, feature_dim]
        src_pad_mask = self.make_src_mask(src)

        src = self.enc_embedding(src)
        src = self.pos_embedding_enc(src)  # [src_len, batch_size, embed_dim]
        memory = self.encoder(src=src, mask=None, src_key_padding_mask=src_pad_mask) # padding marker

        seq_len = (~src_pad_mask).sum(-1)
        memory = torch.mul(memory, ~src_pad_mask.repeat(self.d_model, 1, 1).permute(2, 1, 0))

        # [src_len, batch_size, embed_dim]
        if output_type == 'sum':
            embedding = torch.sum(memory, dim=0)
        elif output_type == 'avg':
            embedding = torch.sum(memory, dim=0) / seq_len.unsqueeze(-1)
        elif output_type == 'last':
            embedding = memory[[(seq_len-1).to(torch.long), torch.range(0, memory.size(1)-1).to(torch.long)]]  # the last timestep
        else:
            raise ValueError('Wrong value of output_type.')


        return embedding  # [batch_size, emb_dim]


