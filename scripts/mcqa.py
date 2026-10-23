"""Convert labeled CommonsenseQA rows into five-choice model inputs."""

import numpy as np


def encode_example(row, tokenizer, max_length):
    choices = row["choices"]
    labels = choices["label"]
    texts = choices["text"]
    if len(labels) != 5 or set(labels) != {"A", "B", "C", "D", "E"} or len(texts) != 5:
        raise ValueError("A supervised example must contain all five labeled choices.")
    answer = row.get("answerKey")
    if answer not in labels:
        raise ValueError("Unlabeled examples cannot be used for supervised training or accuracy.")
    encoded = tokenizer(
        [row["question"]] * 5, texts,
        padding=False, truncation=True, max_length=max_length,
        return_token_type_ids=False,
    )
    encoded["labels"] = labels.index(answer)
    return dict(encoded)


def compute_accuracy(eval_prediction):
    logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.asarray(logits).argmax(axis=-1)
    return {"accuracy": float(np.mean(predictions == np.asarray(labels)))}
