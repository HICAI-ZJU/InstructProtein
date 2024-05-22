import sys
sys.path.append("..")

import torch
from utils import set_seed
from models import opt
from evaluations import SubcellularLocalization

from configuration import root_dir


def main():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, preprocess = opt(root_dir)
    evaluation = SubcellularLocalization(root_dir, model, tokenizer, device, preprocess)
    evaluation.run(instruction="{}Instruction: What cellular components is the protein located in?\n\nOutput: The protein is located in the")


if __name__ == '__main__':
    main()
