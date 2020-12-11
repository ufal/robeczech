# roberta-czech
This repository stores code to work with RoBERTa-BASE model trained on Czech texts.

I am still training the model, but it should hopefully already work decently well. See [playground.ipynb](playground.ipynb) to get some basic idea how to use it. 
Set ```path=/ha/home/naplava/large_temporary_space/naplava/bert-pretraining/roberta_fairseq/hf_model``` when running it inside UFAL cluster.

Note that the majority of texts the model was trained on ends with newline character. Therefore, tokenizer in [czech_roberta_tokenizer.py](czech_roberta_tokenizer.py) has a special argument of its encode method that inserts newline to end of each text.
This should not cause any trouble, but we are already training another model that does not have this "flaw".

We are planning to write a paper (most probably to TSD) with experiments on whether the models built upon it are better than when utilizing multilingual BERT, so if you have any interesting observations (or results), please let us know:
- Jakub Náplava, Milan Straka, Petra Vysušilová

## Requirements
```
torch == 1.6.0
transformers == 3.5.0
tokenizers == 0.9.3
```
