# InstructProtein: Aligning Human and Protein Language via Knowledge Instruction
This repository is the official model and benchmark proposed in a paper: [InstructProtein: Aligning Human and Protein Language via Knowledge Instruction](https://arxiv.org/abs/2310.03269)

## Description

InstructProtein is the first large generative language model exploring the feasibility of bidirectional generation between human and protein language. The idea comes from the research of the [Understudied Proteins Initiative](https://www.nature.com/articles/s41587-022-01316-z): Some proteins, such as the tumor suppressor p53, have been studied extensively. By contrast, thousands of human proteins remain 'understudied'. To align proteins with natural language, InstructProtein is fine-tuned on a high-quality dataset constructed from Knowledge Instruction.

## Installation
```
>> git clone https://github.com/HICAI-ZJU/InstructProtein
>> cd InstructProtein
```

Edit `configuration.py` script with the absolute location of `dump` folder in your environment.

### How to evaluate models
We provide scripts for evaluating the protein understanding and design capabilities of large language models in `./benchmarks`. Let’s take subcellular localization prediction as an example.

```
>> cd benchmarks/scripts
>> python run_subcellular_localization.py
```


### How to train models
We provide a toy instruction dataset in `./dump/pretrain/raw/instructions.txt`.

```
# Build training dataset
>> cd ./scripts
>> bash ./create_training_dataset.sh

# Train model
>> bash ./training.sh
```

## Limitations

The current model, developed through instruction tuning using knowledge instruction dataset, serves as a preliminary example. Despite its initial success in controlled environments, it lacks the robustness to manage complex, real-world, production-level tasks.

## Reference

If you use our repository, please cite the following related paper:

```
@misc{wang2023instructproteinaligninghumanprotein,
      title={Instruct{P}rotein: {A}ligning {H}uman and {P}rotein {L}anguage via {K}nowledge {I}nstruction}, 
      author={Zeyuan Wang and Qiang Zhang and Keyan Ding and Ming Qin and Xiang Zhuang and Xiaotong Li and Huajun Chen},
      year={2023},
      eprint={2310.03269},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2310.03269}, 
}
```
