import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .subcellular_localization_binary_components import SubcellularLocalizationBinaryDataModule


class SubcellularLocalizationBinary(object):

    label_dict = {"plasma": 0, "golgi": 0, "vacuole": 0, "endoplasmic": 0, "extracellular": 1, "peroxisome": 1, "nucleus": 1, "cytoplasm": 1, "mitochondrion": 1, "chloroplast": 1}

    def __init__(self, dump_dir, model, tokenizer, device, preprocess=None) -> None:

        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.data_module = SubcellularLocalizationBinaryDataModule(f'{dump_dir}/downstream/subcellular_localization_2', preprocess=preprocess)
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
