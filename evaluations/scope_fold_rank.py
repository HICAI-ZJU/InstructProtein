import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .scope_fold_rank_components import ScopeFoldRankDataModule


class ScopeFoldRank(object):
    def __init__(self, root_dir, model, tokenizer, device, preprocess=None) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.data_module = ScopeFoldRankDataModule(f'{root_dir}/data/scope_fold_rank', preprocess=preprocess)
        self.device = device

    def run(self, data_name, instruction):
        dataloader = self.data_module.dataloader(data_name)
        result = []

        for idx, item in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
            sequence, label, negative_labels = item
            labels = [label[0]] + negative_labels[0]
            template = []

            for label in labels:
                labeled_instruction = instruction.format(label)
                whole_enc = self.tokenizer.encode(" ".join([labeled_instruction, sequence[0]]), add_special_tokens=False)
                context_enc = self.tokenizer.encode(labeled_instruction, add_special_tokens=False)
                context_enc_len = len(context_enc)
                sequence_enc = whole_enc[context_enc_len:]
                inp = self.tokenizer((" ".join([labeled_instruction, sequence[0]])), return_tensors="pt").to(self.device)
                inp = {
                    'input_ids': inp['input_ids'][..., :-1],
                    'attention_mask': inp['attention_mask'][..., :-1]
                }
                with torch.no_grad():
                    logits = F.log_softmax(self.model(inp['input_ids']).logits, dim=-1)
                logits = logits[0][-len(sequence_enc):]
                logits = torch.gather(logits, 1, torch.tensor([sequence_enc], dtype=torch.long, device=self.device)[0, :, None])


                mean_logits = logits.mean()
                template.append((float(mean_logits), logits))
            mean_template = [item[0] for item in template]
            result.append(torch.argmax(torch.tensor(mean_template)).item() == 0)
        print(f"Acc: {sum(result) / len(result)}")

    