import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, sp):
        self.src = open(src_file, encoding="utf-8").read().splitlines()
        self.tgt = open(tgt_file, encoding="utf-8").read().splitlines()
        self.sp = sp

        assert len(self.src) == len(self.tgt), "Source and target size mismatch"

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_ids = (
            [self.sp.bos_id()] +
            self.sp.encode(self.src[idx]) +
            [self.sp.eos_id()]
        )

        tgt_ids = (
            [self.sp.bos_id()] +
            self.sp.encode(self.tgt[idx]) +
            [self.sp.eos_id()]
        )

        return torch.tensor(src_ids), torch.tensor(tgt_ids)
