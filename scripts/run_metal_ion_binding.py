import sys
sys.path.append("..")

import torch
from utils import set_seed
from models import opt
from evaluations import MetalIonBinding

from configuration import root_dir


def main():
    set_seed(42)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, preprocess = opt(root_dir, '1.3b')
    evaluation = MetalIonBinding(root_dir, model, tokenizer, device, preprocess)
    evaluation.run(instruction="{}Instruction: Does the protein enable metal ion binding?\n\nOutput: Based on the record, the answer is")


if __name__ == '__main__':
    main()


