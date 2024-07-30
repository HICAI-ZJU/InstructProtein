import sys
sys.path.append("..")

import torch
from utils import set_seed
from models import instructprotein
from evaluations import MetalIonBinding

sys.path.append("../..")
from configuration import dump_dir


def main():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, preprocess = instructprotein(dump_dir)
    evaluation = MetalIonBinding(dump_dir, model, tokenizer, device, preprocess)
    evaluation.run(instruction="{}Instruction: Does the protein associate with metal ion binding?\n\nOutput: Based on the record, the answer is")


if __name__ == '__main__':
    main()


