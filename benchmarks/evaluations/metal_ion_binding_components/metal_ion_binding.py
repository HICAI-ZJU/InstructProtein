import torch
from torch.utils.data import Dataset, DataLoader

class MetalIonBindingDataset(Dataset):

    def __init__(self, data_dir, data_name) -> None:
        super().__init__()
        self.data = []
        with open(f'{data_dir}/{data_name}.txt') as f:
            for line in f:
                sequence, label = line.strip().split('\t')
                if len(sequence) > 1024:
                    sequence = sequence[-1000:]
                self.data.append({'sequence': sequence, 'label': int(label)})
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item['sequence'], item['label']
    

class MetalIonBindingDataModule():
    def __init__(self, data_dir, preprocess=None):
        self.data_dir = data_dir
        self.preprocess = preprocess

    def collect_fn(self, raw_batch):
        sequences, labels = zip(*raw_batch)
        if self.preprocess is not None:
            sequences = self.preprocess(sequences)
        return sequences, torch.tensor(labels)
    
    def dataloader(self, data_name):
        dataset = MetalIonBindingDataset(self.data_dir, data_name)
        if data_name == 'train':
            return DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=self.collect_fn, shuffle=False)
        else:
            return DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=self.collect_fn, shuffle=False)
