import json
from torch.utils.data import Dataset, DataLoader


def load_scope_dict(scope_description_path):
    scope_dict = {}
    with open(f'{scope_description_path}', 'r') as f:
        for line in f:
            if line.startswith('#') or line == '\n':
                continue
            item = line.strip().split('\t')
            term, description = item[2], item[4]
            if item[1] == 'dm' or item[1] == 'sp' or item[1] == 'px':
                continue
            if term in scope_dict:
                continue
            else:
                scope_dict[term] = description.lower()
    return scope_dict


class ScopeFoldRankDataset(Dataset):
    def __init__(self, data_dir, data_name) -> None:
        super().__init__()
        self.scope_dict = load_scope_dict(f'{data_dir}/dir.des.scope.2.08-stable.txt')
        self.data = []
        with open(f'{data_dir}/{data_name}.json', 'r') as f:
            for line in f:
                data = json.loads(line)
                self.data.append({"sequence": data['sequence'], 'label': self.scope_dict[data['fold']], 'negative_labels': [self.scope_dict[item] for item in data['negative_labels']]})
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item['sequence'], item['label'], item['negative_labels']
    

class ScopeFoldRankDataModule():
    def __init__(self, data_dir, preprocess=None) -> None:
        self.data_dir = data_dir
        self.preprocess = preprocess

    def collect_fn(self, raw_batch):
        sequences, labels, negative_labels = zip(*raw_batch)
        if self.preprocess is not None:
            sequences = self.preprocess(sequences)

        return sequences, labels, negative_labels
    
    def dataloader(self, data_name):
        dataset = ScopeFoldRankDataset(self.data_dir, data_name)
        return DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=self.collect_fn)