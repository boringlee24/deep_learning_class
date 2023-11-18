from torch.utils.data import Dataset
import torch

class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data, src_field, trg_field):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_field = src_field
        self.trg_field = trg_field

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # Preprocess the text data
        src = self.src_field.preprocess(self.src_data[idx])
        trg = self.trg_field.preprocess(self.trg_data[idx])

        # Numericalize the text data
        src = [self.src_field.vocab.stoi[token] for token in src]
        trg = [self.trg_field.vocab.stoi[token] for token in trg]

        # Convert lists to PyTorch tensors and add batch dimension
        src_tensor = torch.tensor(src, dtype=torch.long)
        trg_tensor = torch.tensor(trg, dtype=torch.long)

        return {"src": src_tensor, "trg": trg_tensor}
