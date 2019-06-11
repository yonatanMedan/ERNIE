# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.2
#   kernelspec:
#     display_name: Environment (conda_fastai)
#     language: python
#     name: conda_fastai
# ---

from enum import Enum

Activation = Enum('Activation', 'ReLU Swish GeLU')

feed_forward??

import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

import torch.nn as nn

from torch.nn.modules.activation import ReLU

from fastai.text.models import MultiHeadRelativeAttention,DecoderLayer,MultiHeadAttention,feed_forward


MultiHeadRelativeAttention??

MultiHeadAttention??

model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
model.eval()

# +
# Load pre-trained model tokenizer (vocabulary from wikitext 103)
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')

# Tokenized input
text_1 = "Who was Jim Henson ?"
text_2 = "Jim Henson was a puppeteer"
tokenized_text_1 = tokenizer.tokenize(text_1)
tokenized_text_2 = tokenizer.tokenize(text_2)

# Convert token to vocabulary indices
indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)

# Convert inputs to PyTorch tensors
tokens_tensor_1 = torch.tensor([indexed_tokens_1])
tokens_tensor_2 = torch.tensor([indexed_tokens_2])

# +
# If you have a GPU, put everything on cuda
tokens_tensor_1 = tokens_tensor_1.to('cuda')
tokens_tensor_2 = tokens_tensor_2.to('cuda')
model.to('cuda')

with torch.no_grad():
    # Predict all tokens
    predictions_1, mems_1 = model(tokens_tensor_1)
    # We can re-use the memory cells in a subsequent call to attend a longer context
    predictions_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

# get the predicted last token
predicted_index = list(map(lambda x:x.item(),predictions_2[0, :, :].argmax(1)))
predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)

# -

with torch.no_grad():
    hidden_2, mems_2 = model.transformer(tokens_tensor_2, mems=mems_1)

hidden_2

len(tokenizer.counter.items())



