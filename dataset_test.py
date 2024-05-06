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

        # 假设labels是一个包含多个标签的列表的列表
        label = self.labels[idx]
        # 将每个标签转换为张量
        label_data = torch.Tensor(label)

        return input_data, label_data