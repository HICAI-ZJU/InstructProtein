import torch
from pathlib import Path

def opt(dump_dir, model_name='1.3b'):
    from transformers import AutoTokenizer, OPTForCausalLM
    checkpoint_path = Path(dump_dir) / 'checkpoints/opt/'
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/opt-{model_name}", cache_dir=f"{checkpoint_path}")
    model = OPTForCausalLM.from_pretrained(f"facebook/opt-{model_name}", cache_dir=f"{checkpoint_path}", torch_dtype=torch.float16)
    return model, tokenizer, None

def galactica(dump_dir, model_name='1.3b'):
    from transformers import AutoTokenizer, OPTForCausalLM
    checkpoint_path = Path(dump_dir) / 'checkpoints/galactica/'
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/galactica-{model_name}", cache_dir=f"{checkpoint_path}")
    model = OPTForCausalLM.from_pretrained(f"facebook/galactica-{model_name}", cache_dir=f"{checkpoint_path}")
    def preprocess(sequences):
        return ['[START_AMINO]'+s+'[END_AMINO]' for s in sequences]
    return model, tokenizer, preprocess

def instructprotein(dump_dir, model_name=None):
    from transformers import AutoTokenizer, OPTForCausalLM
    checkpoint_path = Path(dump_dir) / 'checkpoints/instructprotein/'
    tokenizer = AutoTokenizer.from_pretrained("hicai-zju/InstructProtein", cache_dir=f"{checkpoint_path}")
    model = OPTForCausalLM.from_pretrained("hicai-zju/InstructProtein", cache_dir=f"{checkpoint_path}")
    def preprocess(sequences):
        return ["<protein>Ƥ" + 'Ƥ'.join(list(s)) + "</protein>" for s in sequences]
    return model, tokenizer, preprocess


