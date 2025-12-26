import torch.nn as nn
from .positional_encoding import PositionalEncoding

class TransformerMT(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.pos(self.emb(src))
        tgt = self.pos(self.emb(tgt))
        out = self.transformer(src, tgt)
        return self.fc(out)
