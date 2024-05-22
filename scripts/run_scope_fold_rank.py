import sys
sys.path.append("..")

import torch
from utils import set_seed
from models import opt
from evaluations import ScopeFoldRank

from configuration import root_dir


def main():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, preprocess = opt(root_dir)
    evaluation = ScopeFoldRank(root_dir, model, tokenizer, device, preprocess)
    evaluation.run(data_name='test_family.lm', instruction="Instruction: I would like a protein that is in {}.\n\nOutput: One of the protein that meets the demand is")


if __name__ == '__main__':
    main()
