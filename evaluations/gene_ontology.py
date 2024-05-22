import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score

from .gene_ontology_components import GeneOntologyDataModule


class GeneOntology(object):
    labels = ["yes", "yo"]

    def __init__(self, root_dir, model, tokenizer, device, preprocess=None) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.data_module = GeneOntologyDataModule(f'{root_dir}/data/gene_ontology', preprocess=preprocess)
        self.device = device

    def run(self, data_name, instruction):
        dataloader = self.data_module.dataloader(data_name)
        result = []
        encoded_label = [self.tokenizer.encode(" " + label, add_special_tokens=False) for label in self.labels]
        for idx, item in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
            sequence, positive_labels, negative_labels = item
            for label in positive_labels[0]:
                encoded_input = self.tokenizer.encode(instruction.format(sequence[0], label), return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = F.log_softmax(self.model(encoded_input).logits, dim=-1)
                logits = logits[0][-1]
                logits = torch.gather(logits, 0, torch.tensor([token[0] for token in encoded_label], dtype=torch.long, device=self.device))
                result.append((torch.argmax(logits).item() == 0, torch.argmax(logits).item(), 0, logits))
            for label in negative_labels[0]:
                encoded_input = self.tokenizer.encode(instruction.format(sequence[0], label), return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = F.log_softmax(self.model(encoded_input).logits, dim=-1)
                logits = logits[0][-1]
                logits = torch.gather(logits, 0, torch.tensor([token[0] for token in encoded_label], dtype=torch.long, device=self.device))
                result.append((torch.argmax(logits).item() == 1, torch.argmax(logits).item(), 1, logits))
        y_true = [item[2] for item in result]
        y_pred = [item[1] for item in result]
        y_test = [torch.softmax(item[3], dim=0)[1].item() for item in result]
        print(f'F1 micro: {f1_score(y_true, y_pred, average="micro")}; AUPR: {average_precision_score(y_true, y_test, average="micro")}')
