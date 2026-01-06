from predict import predict
from transformers import AutoModelForTokenClassification, AutoTokenizer
restore_dir = 'gotutiyan/token-ged-electra-large-bin'
model = AutoModelForTokenClassification.from_pretrained(restore_dir)
tokenizer = AutoTokenizer.from_pretrained(restore_dir)
srcs = ['This are wrong sentece .', 'This is correct .']

# predict() returns word-level error detection labels
# If return_id=True
results = predict(
    srcs=srcs,
    model=model,
    tokenizer=tokenizer,
    return_id=True,
    batch_size=32
)
print(results)
# An example of outputs: [[0, 1, 0, 1, 0], [0, 0, 0, 0]]

# If return_id=False
results = predict(
    srcs=srcs,
    model=model,
    tokenizer=tokenizer,
    return_id=False,
    batch_size=32
)
print(results)