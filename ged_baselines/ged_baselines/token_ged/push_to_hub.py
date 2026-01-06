import argparse
from huggingface_hub import create_repo, upload_folder, ModelCard
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
def main(args):
    create_repo(repo_id=args.repo_id, exist_ok=True, repo_type='model', private=True, token=os.environ['HF_TOKEN'])
    model = AutoModelForTokenClassification.from_pretrained(args.dir)
    tokenizer = AutoTokenizer.from_pretrained(args.dir)
    model.push_to_hub(args.repo_id, token=os.environ['HF_TOKEN'])
    tokenizer.push_to_hub(args.repo_id, token=os.environ['HF_TOKEN'])
    exit()
    # create_repo(repo_id=args.repo_id, exist_ok=True, repo_type='model')
    upload_folder(
        folder_path=args.dir,
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns='*output/',
        token=os.environ['HF_TOKEN']
    )
    content = f"""
---
language: en
license: cc-by-nc-sa-4.0
tags:
- grammatical error correction
- grammatical error detection
---

Binary and multi-class grammatical error detection models.  
The experiment was performed according to [Yuan+ 21](https://aclanthology.org/2021.emnlp-main.687/).

The code and the performance on GEC benchmarks are avaliable from https://github.com/gotutiyan/ged_baselines.

Trained models are distributed for research and educational purposes only. 
"""

    card = ModelCard(content)
    # print(card.data.to_dict())
    card.push_to_hub(args.repo_id)

    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', required=True)
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)