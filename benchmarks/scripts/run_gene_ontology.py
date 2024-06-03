import sys
sys.path.append("..")

import torch
from utils import set_seed
from models import opt
from evaluations import GeneOntology

sys.path.append("../..")
from configuration import dump_dir


def main():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, preprocess = opt(dump_dir, '1.3b')
    evaluation = GeneOntology(dump_dir, model, tokenizer, device, preprocess)
    evaluation.run(data_name='test_bp', instruction="{}Instruction: Does the protein associate with {}?\n\nOutput: Based on the record, the answer is")

if __name__ == '__main__':
    main()
