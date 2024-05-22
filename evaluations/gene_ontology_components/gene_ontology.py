import json
from torch.utils.data import Dataset, DataLoader

def load_gene_ontology_dict(gene_ontology_term_path):
    gene_ontology_dict = {}
    with open(gene_ontology_term_path, 'r') as f:
        is_record = False
        term = {}
        for line in f:
            if line == '\n':
                is_record = False
                if term != {} and 'is_obsolete' not in term:
                    gene_ontology_dict[term['id']] = term
                term = {}
            elif line.strip() == '[Term]':
                is_record = True
            elif is_record:
                key, value = line.strip().split(': ', 1)
                if key == 'alt_id' or key == 'comment' or key == 'xref' or key == 'subset' or key == 'property_value' or key == 'created_by' or key == 'creation_date':
                    continue
                if key == 'id' or key == 'name' or key == 'namespace' or key == 'def':
                    if key == 'name':
                        value = value.lower()
                    term[key] = value
                else:
                    if key in term:
                        term[key].append(value)
                    else:
                        term[key] = [value]
    return gene_ontology_dict


def get_gene_ontology_parent(gene_ontology_dict, gene_ontology_id):
    if 'is_a' in gene_ontology_dict[gene_ontology_id]:
        return sum([get_gene_ontology_parent(gene_ontology_dict, gene_ontology_parent.split(' ! ')[0]) for gene_ontology_parent in gene_ontology_dict[gene_ontology_id]['is_a']], [gene_ontology_id])
    return []


class GeneOntologyDataset(Dataset):
    def __init__(self, data_dir, data_name) -> None:
        super().__init__()
        self.gene_ontology_dict = load_gene_ontology_dict(f'{data_dir}/go-basic.obo')
        self.data = []
        with open(f'{data_dir}/{data_name}.lm.json', 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append({'sequence': item['sequence'], 'labels': [self.gene_ontology_dict[label]['name'] for label in item['label']], 'negative_labels': [self.gene_ontology_dict[label]['name'] for label in item['negative_labels']]})

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item['sequence'], item['labels'], item['negative_labels']
    

class GeneOntologyDataModule(object):
    def __init__(self, data_dir, preprocess=None) -> None:
        self.data_dir = data_dir
        self.preprocess = preprocess

    def collect_fn(self, raw_batch):
        sequences, labels, negative_labels = zip(*raw_batch)
        if self.preprocess is not None:
            sequences = self.preprocess(sequences)

        return sequences, labels, negative_labels
    
    def dataloader(self, data_name):
        dataset = GeneOntologyDataset(self.data_dir, data_name)
        return DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=self.collect_fn)
