import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .subcellular_localization_components import SubcellularLocalizationDataModule


class SubcellularLocalization(object):

    label_dict = {"cytoplasm": 1, "endoplasmic reticulum": 2, "golgi": 3, "vacuole": 4, "mitochondrion": 5, "nucleus": 6, "peroxisome": 7, "chloroplast": 8, "extracellular": 9, "plasma membrane": 0}

    def __init__(self, dump_dir, model, tokenizer, device, preprocess=None) -> None:

        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.data_module = SubcellularLocalizationDataModule(f'{dump_dir}/downstream/subcellular_localization', preprocess=preprocess)
        self.device = device

    def run(self, instruction):
        dataloader = self.data_module.dataloader('test')
        result = []
        encoded_label = [(self.tokenizer.encode(" " + key, add_special_tokens=False), value) for key, value in self.label_dict.items()]
        for idx, item in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
            sequence, label = item
            encoded_input = self.tokenizer.encode(instruction.format(sequence[0]), return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = F.log_softmax(self.model(encoded_input).logits, dim=-1)
            logits = logits[0][-1]
            logits = torch.gather(logits, 0, torch.tensor([token[0][0] for token in encoded_label], dtype=torch.long, device=self.device))
            pred = encoded_label[torch.argmax(logits)][1]
            result.append((pred == label[0].item(), torch.argmax(logits).item(), label, logits))
        print(f'Acc: {sum([item[0] for item in result]) / len(result)}')
