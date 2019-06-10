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

class KETFeedForward(torch.nn.module):
    def __init__(self, wrd_conf,ent_conf):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(wrd_conf.intermediate_size, wrd_conf.d_model)
        self.dense_ent = nn.Linear(wrd_conf.intermediate_size, ent_conf.d_model)
        self.LayerNorm = nn.LayerNorm(wrd_conf.d_model, eps=1e-12)
        self.LayerNorm_ent = nn.LayerNorm(ent_conf.d_model, eps=1e-12)
        self.dropout = nn.Dropout(wrd_conf.hidden_dropout_prob)

    def forward(self, hidden_states_, input_tensor, input_tensor_ent):
        hidden_states = self.dense(hidden_states_)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        hidden_states_ent = self.dense_ent(hidden_states_)
        hidden_states_ent = self.dropout(hidden_states_ent)
        hidden_states_ent = self.LayerNorm_ent(hidden_states_ent + input_tensor_ent)
        return hidden_states, hidden_states_ent

class MaskedKnowledgeAGGBlock(torch.nn.module):
    def __init__(wrd_conf,ent_conf):
        self.mhra_wrd = MultiHeadRelativeAttention(**wrd_conf)
        self.mhra_ent = MultiHeadRelativeAttention(**ent_conf)
        self.dense = nn.Linear(wrd_conf.d_model, wrd_conf.intermediate_size)
        self.dense_ent = nn.Linear(ent_conf.d_model, wrd_conf.intermediate_size)
        self.act = nn.GeLU()
        self ff = KETFeedForward(wrd_conf,ent_conf)

    def forward(self,hidden,hidden_ent,mask,mask_ent,r, u, v,r_ent,u_ent,v_ent, mem,mem_ent):
        hidden = self.mhra_wrd(hidden,r=r, u=u, v=v,mask=mask,mem=mem)
        hidden_ent = self.mhra_ent(hidden_ent,r=r_ent, u=u_ent, v=v_ent,mask=mask_ent,mem=mem_ent)
        # get knowledge and words to same dim
        intermediate = self.dense(hidden)+self.dense_ent(hidden_ent)
        intermediate = self.act(intermediate_size)
        # mixes the knowledge with the tokens
        hidden,hidden_ent = self.ff(intermediate,hidden,hidden_ent)
    



class MaskedKnowledgeAGG(torch.nn.module):
    def __init__(self,
                 wrd_conf,
                    #  vocab_sz:int,
                    #  d_model:int,
                    #  d_head:int,
                    #  n_heads:int,
                    #  d_inner: int,
                    #  embed_p: float=0.,
                    #  resid_p: float=0.,
                    #  attn_p: float=0.,
                    #  ff_p: float=0.,
  
                 ent_conf,
                    #  vocab_sz_ent: int,
                    #  d_model_ent: int,
                    #  d_head_ent: int,
                    #  n_heads_ent: int,
                    #  d_inner_ent: int,
                    #  embed_p_ent: float=0.,
                    #  resid_p_ent: float=0.,
                    #  attn_p_ent: float=0.,
                    #  ff_p_ent: float=0.,
                 
                 ctx_len: int,
                 mem_len:int,
                 n_layers:int,

                 mask:bool=True,
                 act: Activation = Activation.ReLU,
                 double_drop: bool = True,
                 bias: bool = False,
                 scale:bool = True,
                ):
        super().__init__()
        # embeding should be done in before this module layers
        # self.token_enc = nn.Embedding(vocab_sz, d_model)

        self.pos_enc_ent = nn.Embedding(ctx_len, d_model_ent)
        self.pos_enc_wrd = nn.Embedding(ctx_len, d_model)
        self.drop_emb_wrd = nn.Dropout(embed_p)
        self.drop_emb_ent = nn.Dropout(embed_p_ent)

        self.u_wrd = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.u_ent = nn.Parameter(torch.Tensor(n_heads_ent, 1, d_head_ent)) #Remove 1 for einsum implementation of attention

        self.v_wrd = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.v_ent = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention

        self.mem_len,self.n_layers,self.d_model,self.mask = mem_len,n_layers,d_model,mask
        self.init = False
        self.layers = []
        for i in range(n_layers):
            layers.append(MaskedKnowledgeAGGBlock(wrd_conf,ent_conf))


    def _update_mems(self, mem,hids):
        if mem is None:
             return None
        assert len(hids) == len(mem), 'len(hids) != len(self.hidden)'
        with torch.no_grad():
            for i in range(len(hids)):
                cat = torch.cat([mem[1], hids[i]], dim=1)
            return  cat[:,-self.mem_len:].detach()

    def forward(self, x,x_ent,mem,mem_ent):
        #The hidden state has to be initiliazed in the forward pass for nn.DataParallel
        bs,x_len = x.size()
        inp = self.drop_emb_wrd(x) #.mul_(self.d_model ** 0.5)
        inp_ent = self.drop_emb_ent(x_ent)

        m_len = mem[0].size(1) if len(self.mem[0].size()) > 1 else 0
        
        seq_len = m_len + x_len
        
        mask = torch.triu(x.new_ones(x_len, seq_len), diagonal=1+m_len).byte()[None,None] if self.mask else None
         # need to addapt this map acording to traning task (
         #  for next token prediction dont mask next entity
         #  this will train the network to predict next token with signals from next entity
         #  for next entity prediction mask next entity to evoid cheating
         # )
        mask_ent = torch.triu(x.new_ones(x_len, seq_len), diagonal=1+m_len).byte()[None,None] if self.mask_ent else None     

        #[None,:,:None] for einsum implementation of attention
        hids = []
        hids_ent = []
        pos = torch.arange(seq_len-1, -1, -1, device=inp.device, dtype=inp.dtype)

        pos_enc = self.pos_enc_wrd(pos)
        pos_enc_ent = self.pos_enc_ent(pos)

        hids.append(inp)
        hids_ent.append(inp_ent)
        for i, layer in enumerate(self.layers):
            # mem = self.hidden[i] if self.mem_len > 0 else None

            inp,inp_ent = layer(inp,inp_ent, r=pos_enc, u=self.u_wrd, v=self.v_wrd, mask=mask, mem=mem[i],r_net=pos_enc_ent, u_ent=self.u_ent,v_ent=self.v_ent,mask_ent=mask_ent,mem=mem_ent[i])
            hids.append(inp)
            hids_ent.append(inp_ent)
        core_out = inp[:,-x_len:]

        mem =  self._update_mems(mem,hids)
        mem_ent = self._update_mems(mem_ent,hids_ent)
        return mem,mem_ent ,inp,inp_ent

            

# -

len(mems_2)

mems_2[0].shape

predicted_token

list(map(lambda x:x.item(),predictions_2[0, :, :].argmax(1)))

predictions_2[0, -1, :]

tokenizer.convert_ids_to_tokens([predicted_index])


