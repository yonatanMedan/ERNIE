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

import torch
from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel

import torch.nn as nn

from fastai.text.models import MultiHeadRelativeAttention,DecoderLayer


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
tokens_tensor_1 = tokens_tensor_1.to('cpu')
tokens_tensor_2 = tokens_tensor_2.to('cpu')
model.to('cpu')

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


# +
class EntEmbedingLayer(torch.nn.module):
    def __init__(vocab_sz_ent:int,ctx_len:it,d_model_ent:int):
        super().__init__()
        self.ent_enc = nn.Embedding(vocab_sz_ent, d_model_ent)

    def forword(ent_ids_tensor):
        return self.ent_enc(ent_ids_tensor)

class MaskedKnowledgeAGGBlock(torch.nn.module):
    def __init__(tkn_conf,ent_conf):
        self.mhra_wrd = MultiHeadRelativeAttention(**tkn_conf)
        self.mhra_ent = MultiHeadRelativeAttention(**ent_conf)
        self.dense = nn.Linear(tkn_conf.d_model, tkn_conf.intermediate_size)
        self.dense_ent = nn.Linear(ent_conf.d_model, tkn_conf.intermediate_size)
        self.act = nn.GeLU()
    def forward(self,hidden,hidden_ent,mask,mask_ent):
        hidden = mhra_wrd(hidden,mask)
        hidden_ent = mhra_ent(hidden_ent,mask_ent)
        intermediate = dense(hidden)+self.dense_ent(hidden_ent)
        intermediate = act(intermediate_size)
    
class KETFeedForward(torch.nn.module):
    def __init__(self, wrd_conf,ent_conf):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, wrd_conf.d_model)
        self.dense_ent = nn.Linear(config.intermediate_size, ent_conf.d_model)
        self.LayerNorm = nn.LayerNorm(wrd_conf.d_model, eps=1e-12)
        self.LayerNorm_ent = nn.LayerNorm(ent_conf.d_model, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states_, input_tensor, input_tensor_ent):
        hidden_states = self.dense(hidden_states_)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        hidden_states_ent = self.dense_ent(hidden_states_)
        hidden_states_ent = self.dropout(hidden_states_ent)
        hidden_states_ent = self.LayerNorm_ent(hidden_states_ent + input_tensor_ent)


class MaskedKnowledgeAGG(torch.nn.module):
    def __init__(self,
                 vocab_sz:int,
                 d_model:int,
                 d_head:int,
                 n_heads:int,
                 d_inner: int,

                 vocab_sz_ent: int,
                 d_model_ent: int,
                 d_head_ent: int,
                 n_heads_ent: int,
                 d_inner_ent: int,

                 ctx_len: int,
                 mem_len:int,
                 n_layers:int,

                 embed_p: float=0.,
                 resid_p: float=0.,
                 attn_p: float=0.,
                 ff_p: float=0.,

                 embed_p_ent: float=0.,
                 resid_p_ent: float=0.,
                 attn_p_ent: float=0.,
                 ff_p_ent: float=0.,

                 mask:bool=True,
                 act: Activation = Activation.ReLU,
                 double_drop: bool = True,
                 attn_cls = MultiHeadRelativeAttention,
                 bias: bool = False,
                 scale:bool = True,
                ):
        super().__init__()
        # get id as inputs and outputs vectors
        # embeding should be done in before this module layers
        # self.token_enc = nn.Embedding(vocab_sz, d_model)

        self.pos_enc_ent = nn.Embedding(ctx_len, d_model_ent)
        self.pos_enc_tokens = nn.Embedding(ctx_len, d_model)
        self.drop_emb_token = nn.Dropout(embed_p)
        self.drop_emb_ent = nn.Dropout(embed_p_ent)

        self.u_tkn = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.u_ent = nn.Parameter(torch.Tensor(n_heads_ent, 1, d_head_ent)) #Remove 1 for einsum implementation of attention

        self.v_tkn = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.v_ent = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention

        self.mem_len,self.n_layers,self.d_model,self.mask = mem_len,n_layers,d_model,mask
        self.init = False
        self.ent_attention =[]
        self.tkn_attention = []
        for layer in range(n_layers):
            self.ent_attention.append(attn_cls(n_heads_ent, d_model_ent, d_head_ent, d_inner_ent, resid_p=resid_p_ent, attn_p=attn_p_ent,ff_p=ff_p_ent, bias=bias, scale=scale, act=act, double_drop=double_drop,attn_cls=attn_cls))
            self.tkn_attention.append(attn_cls(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,ff_p=ff_p, bias=bias, scale=scale, act=act, double_drop=double_drop,attn_cls=attn_cls))

            

# -

len(mems_2)

mems_2[0].shape

predicted_token

list(map(lambda x:x.item(),predictions_2[0, :, :].argmax(1)))

predictions_2[0, -1, :]

tokenizer.convert_ids_to_tokens([predicted_index])


