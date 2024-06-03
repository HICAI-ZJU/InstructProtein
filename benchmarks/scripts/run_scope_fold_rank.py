import sys
sys.path.append("..")

import torch
from utils import set_seed
from models import opt
from evaluations import ScopeFoldRank

sys.path.append("../..")
from configuration import dump_dir


def main():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, preprocess = opt(dump_dir)
    evaluation = ScopeFoldRank(dump_dir, model, tokenizer, device, preprocess)
    evaluation.run(data_name='test_family.lm', instruction="Instruction: I would like a protein that is in {}.\n\nOutput: One of the protein that meets the demand is")


if __name__ == '__main__':
    main()
