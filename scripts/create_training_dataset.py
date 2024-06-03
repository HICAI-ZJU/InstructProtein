import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional
sys.path.append("..")

import transformers
from transformers import HfArgumentParser, TrainingArguments
from transformers import AutoTokenizer
from transformers.testing_utils import CaptureLogger

import datasets
from datasets import load_dataset
from configuration import dump_dir


logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    cache_dir: Optional[str] = field(
        default=None,
    )
    tokenizer_dir: Optional[str] = field(default=None, metadata={"help": "The tokenizer file."})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    parser = HfArgumentParser((DataTrainingArguments, TrainingArguments))
    data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    data_files, dataset_args = {}, {}
    data_files['train'] = data_args.train_file
    dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
    raw_datasets = load_dataset(
        'text',
        data_files={key: f'{dump_dir}/{file}' for key, file in data_files.items()},
        cache_dir = f'{dump_dir}/{data_args.cache_dir}',
        **dataset_args
    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", cache_dir=f'{dump_dir}/{data_args.tokenizer_dir}')
    additional_tokens = ["<protein>", "</protein>", "ƤA", "ƤC", "ƤD", "ƤE", "ƤF", "ƤG", "ƤH", "ƤI", "ƤK", "ƤL", "ƤM", "ƤN", "ƤP", "ƤQ", "ƤR", "ƤS", "ƤT", "ƤV", "ƤW", "ƤY"]
    tokenizer.add_tokens(additional_tokens)
    
    column_names = list(raw_datasets['train'].features)
    text_column_name = 'text' if 'text' in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            instruction = examples[text_column_name].replace("\\n", "\n")
            instruction = tokenizer(instruction)
            instruction['length'] = len(instruction['input_ids'])
        # clm input could be much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return instruction

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    tokenized_datasets = tokenized_datasets.filter(lambda example: example['length'] < 1024)
    tokenized_datasets = tokenized_datasets.shuffle(seed=42)

    tokenized_datasets.save_to_disk(f'{dump_dir}/{training_args.output_dir}')


if __name__ == '__main__':
    main()
