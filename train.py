import torch
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from tqdm import tqdm

from dataset import TranslationDataset
from model.transformer import TransformerMT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file='spm.model')

def collate(batch):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=0)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=0)
    return src.to(DEVICE), tgt.to(DEVICE)

dataset = TranslationDataset("data/train.en", "data/train.de", sp)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)

model = TransformerMT(sp.get_piece_size()).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(10):
    model.train()
    total_loss = 0

    for src, tgt in tqdm(loader):
        optimizer.zero_grad()
        out = model(src, tgt[:, :-1])
        loss = loss_fn(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch} | Loss {total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "model.pt")
