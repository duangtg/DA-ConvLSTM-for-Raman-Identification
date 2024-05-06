from torch.utils.data import Dataset, DataLoader
import torch


class SpectraDataset(Dataset):
    def __init__(self, spectra, labels, num_classes):
        self.spectra = spectra
        self.labels = labels
        self.num_classes = num_classes

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        input_data = torch.Tensor(self.spectra[idx]).unsqueeze(0)

        quantity_label = torch.tensor(self.labels[idx][0], dtype=torch.float)

        class_label = torch.tensor(self.labels[idx][1], dtype=torch.long)

        label_data = torch.tensor([quantity_label, class_label])

        return input_data, label_data
