import torch
import sentencepiece as spm
from model.transformer import TransformerMT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sp = spm.SentencePieceProcessor(model_file='spm.model')
model = TransformerMT(sp.get_piece_size()).to(DEVICE)
model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
model.eval()

def translate(sentence, max_len=100):
    src = torch.tensor([sp.encode(sentence)]).to(DEVICE)
    tgt = torch.tensor([[sp.bos_id()]]).to(DEVICE)

    for _ in range(max_len):
        out = model(src, tgt)
        next_token = out[:, -1].argmax(-1)
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
        if next_token.item() == sp.eos_id():
            break

    return sp.decode(tgt[0].tolist())

while True:
    s = input("EN> ")
    print("DE>", translate(s))
