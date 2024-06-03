import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metal_ion_binding_components import MetalIonBindingDataModule


class MetalIonBinding(object):
    labels = ["no", "yes"]
    def __init__(self, dump_dir, model, tokenizer, device, preprocess=None) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.data_module = MetalIonBindingDataModule(f'{dump_dir}/downstream/metal_ion_binding/', preprocess=preprocess)
        self.device = device

    def run(self, instruction):
        dataloader = self.data_module.dataloader('test')
        result = []
        encoded_label = [self.tokenizer.encode(" " + label, add_special_tokens=False) for label in self.labels]
    
        for idx, item in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
            sequence, label = item
            encoded_input = self.tokenizer.encode(instruction.format(sequence[0]), return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = F.log_softmax(self.model(encoded_input).logits, dim=-1)

            logits = logits[0][-1]
            logits = torch.gather(logits, 0, torch.tensor([token[0] for token in encoded_label], dtype=torch.long, device=self.device))
            result.append((torch.argmax(logits).item() == label[0].item(), torch.argmax(logits).item(), label, logits))
        print(f'Acc: {sum([item[0] for item in result]) / len(result)}')
