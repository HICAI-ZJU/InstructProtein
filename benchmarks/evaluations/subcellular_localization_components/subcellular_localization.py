import pickle
import lmdb
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class SubcellularLocalizationDataset(Dataset):
    """
    Class labels indicating where a natural protein locates in the cell.

    Statistics:
        - #Train: 8,945
        - #Valid: 2,248
        - #Test: 2,768

    Parameters:
        path (str): the path to store the dataset
        verbose (int, optional): output verbose level
        **kwargs
    """

    def __init__(self, data_dir, data_name) -> None:
        super().__init__()
        env = lmdb.open(f'{data_dir}/subcellular_localization_{data_name}.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
        self.sequences = []
        self.targets = []
        with env.begin(write=False) as txn:
            num_sample = pickle.loads(txn.get("num_examples".encode()))
            for i in range(num_sample):
                item = pickle.loads(txn.get(str(i).encode()))
                self.sequences.append(item['primary'])
                value = item['localization']
                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                self.targets.append(value)
            self.num_sample = num_sample
    
    def __len__(self):
        return self.num_sample
    
    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]
    

class SubcellularLocalizationDataModule():
    splits = ["train", "valid", "test"]

    def __init__(self, data_dir, preprocess=None) -> None:
        self.data_dir = data_dir
        self.preprocess = preprocess

    def collect_fn(self, raw_batch):
        sequences, labels = zip(*raw_batch)
        if self.preprocess is not None:
            sequences = self.preprocess(sequences)
        return sequences, torch.tensor(labels)
        
    def dataloader(self, data_name):
        dataset = SubcellularLocalizationDataset(self.data_dir, data_name)
        if data_name == 'train':
            return DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=self.collect_fn, shuffle=True)
        else:
            return DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=self.collect_fn, shuffle=False)
