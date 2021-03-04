# roberta-czech
This repository stores code to work with RoBERTa-BASE model trained on Czech texts.

The checkpoints are stored in `/net/projects/robeczech/checkpoints`, each
checkpoint has the following subdirectories:
- `pytorch`: PyTorch checkpoint for the Huggingface Transformers library;
- `tf`: TensorFlow checkpoint for the Huggingface Transformers library;
- `tokenizer`: Tokenizer using the Huggingface Tokenizers library.

The model is still being trained, so the checkpoints have dates and new
checkpoints can still be added.

There are two versions of the checkpoints:
- `eol`: these variants use two special tokens to mark end of sequences, notably
  `[EOL]` and `[SEP]`. This was a mistake in the original data processing.
  However, they have been training the longest.
- `noeol`: the `[EOL]` token is not used, but the model has been training for
  two months less.

In both cases, the corresponding tokenizers correctly add suitable special
tokens automatically.

See [playground.ipynb](playground.ipynb) to get some basic idea how to use it
and what subwords does the MLM predicts (both for unmasked and masked subwords).

We are planning to write a paper (most probably to TSD) with experiments on
whether the models built upon it are better than when utilizing multilingual
BERT, so if you have any interesting observations (or results), please let us
know:
- Jakub Náplava, Milan Straka, Petra Vysušilová

## Requirements
We have tested the models with the following packages:
```
tensorflow == 2.4.1
torch == 1.6.0, 1.7.1
transformers == 2.9.1, 3.5.0
tokenizers == 0.7.0, 0.9.3
```
