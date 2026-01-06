# ged_baselines
Confirmed that it works on python3.11.0.

# Installation
```sh
pip install git+https://github.com/gotutiyan/gecommon
pip install git+https://github.com/gotutiyan/ged_baselines
```

# Token-level Grammatical Error Detection
The models can be found in [HERE](https://huggingface.co/collections/gotutiyan/token-level-ged-662bd988259fa63f77ca8997).  
If CUDA is available, the following script automatically uses a GPU.
```python
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer
)
from ged_baselines import predict_token
srcs = ['This are wrong sentece .', 'This is correct .']
restore_dir = 'gotutiyan/token-ged-electra-large-25cls'
model = AutoModelForTokenClassification.from_pretrained(restore_dir)
tokenizer = AutoTokenizer.from_pretrained(restore_dir)
results = predict_token(
    srcs=srcs,
    model=model,
    tokenizer=tokenizer,
    return_id=False,
    batch_size=2
)
print(results)
# [['CORRECT', 'VERB:SVA', 'CORRECT', 'SPELL', 'CORRECT'], ['CORRECT', 'CORRECT', 'CORRECT', 'CORRECT']]
```

If you use the model `-25cls` or `-55cls`, the output represents error types.  
The definition of the error types can be referred to Table 2 in the paper: [Automatic Annotation and Evaluation of Error Types for Grammatical Error Correction](https://aclanthology.org/P17-1074/).