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

default_wrd_conf = {
    "d_model":1024,
    "embed_p":0.,
    "n_heads":5,
    "d_head":64,
    "resid_p":0.0,
    "attn_p":0.0,
    "bias": False,
    "scale":True,
    "intermediate_size":3072,
    "ff_p":0.
}

default_ent_conf = default_wrd_conf.copy()

default_ent_conf["d_model"] = 100

mhra_keys = ["d_model","n_heads","d_head","resid_p","attn_p","bias","scale"]


# +
class EntEmbedingLayer(torch.nn.Module):
    def __init__(vocab_sz_ent:int,ctx_len:int,d_model_ent:int):
        super().__init__()
        self.ent_enc = nn.Embedding(vocab_sz_ent, d_model_ent)

    def forword(ent_ids_tensor):
        return self.ent_enc(ent_ids_tensor)

class KETFeedForward(torch.nn.Module):
    def __init__(self, wrd_conf,ent_conf):
        super().__init__()
        self.dense = nn.Linear(wrd_conf["intermediate_size"], wrd_conf["d_model"])
        self.dense_ent = nn.Linear(wrd_conf["intermediate_size"], ent_conf["d_model"])
        self.LayerNorm = nn.LayerNorm(wrd_conf["d_model"], eps=1e-12)
        self.LayerNorm_ent = nn.LayerNorm(ent_conf["d_model"], eps=1e-12)
        self.dropout = nn.Dropout(wrd_conf["ff_p"])

    def forward(self, hidden_states_, input_tensor, input_tensor_ent):
        hidden_states = self.dense(hidden_states_)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        hidden_states_ent = self.dense_ent(hidden_states_)
        hidden_states_ent = self.dropout(hidden_states_ent)
        hidden_states_ent = self.LayerNorm_ent(hidden_states_ent + input_tensor_ent)
        return hidden_states, hidden_states_ent

class MaskedKnowledgeAGGBlock(torch.nn.Module):
    def __init__(self,wrd_conf,ent_conf):
        super().__init__()
        self.mhra_wrd = MultiHeadRelativeAttention(**{ k: wrd_conf[k] for k in mhra_keys })
        self.mhra_ent = MultiHeadRelativeAttention(**{ k: ent_conf[k] for k in mhra_keys })
        self.dense = nn.Linear(wrd_conf["d_model"], wrd_conf["intermediate_size"])
        self.dense_ent = nn.Linear(ent_conf["d_model"], wrd_conf["intermediate_size"])
        self.act = ReLU()
        self.ff = KETFeedForward(wrd_conf,ent_conf)

    def forward(self,hidden,hidden_ent,mask,mask_ent,r, u, v,r_ent,u_ent,v_ent, mem,mem_ent):
        hidden = self.mhra_wrd(hidden,r=r, u=u, v=v,mask=mask,mem=mem)
        hidden_ent = self.mhra_ent(hidden_ent,r=r_ent, u=u_ent, v=v_ent,mask=mask_ent,mem=mem_ent)
        # get knowledge and words to same dim
        intermediate = self.dense(hidden)+self.dense_ent(hidden_ent)
        intermediate = self.act(intermediate_size)
        # mixes the knowledge with the tokens
        hidden,hidden_ent = self.ff(intermediate,hidden,hidden_ent)
    

def create_MKAGG_submodules(ctx_len,conf):
    pos_enc = nn.Embedding(ctx_len, conf["d_model"])
    drop_emb = nn.Dropout(conf["embed_p"])
    u = nn.Parameter(torch.Tensor(conf["n_heads"], 1, conf["d_head"]))
    v = nn.Parameter(torch.Tensor(conf["n_heads"], 1, conf["d_head"]))
    return pos_enc,drop_emb,u,v

class MaskedKnowledgeAGG(torch.nn.Module):
    def __init__(self,
                 wrd_conf,
                    #  d_model:int,
                    #  d_head:int,
                    #  n_heads:int,
                    #  d_inner: int,
                    #  embed_p: float=0.,
                    #  resid_p: float=0.,
                    #  attn_p: float=0.,
                    #  ff_p: float=0.,
  
                 ent_conf,
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

                ):
        super().__init__()
        # embeding should be done in before this module layers
        # self.token_enc = nn.Embedding(vocab_sz, d_model)
        self.pos_enc_wrd, self.drop_emb_wrd, self.u_wrd, self.v_wrd = create_MKAGG_submodules(ctx_len,wrd_conf)
        self.pos_enc_ent, self.drop_emb_ent, self.u_ent, self.v_ent = create_MKAGG_submodules(ctx_len,ent_conf)
        self.mem_len,self.n_layers,self.mask = mem_len,n_layers,mask
        
        self.layers = []
        for i in range(n_layers):
            self.layers.append(MaskedKnowledgeAGGBlock(wrd_conf,ent_conf))
            
        self.layers =  nn.ModuleList(self.layers)



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

            inp,inp_ent = layer(inp,inp_ent, r=pos_enc, u=self.u_wrd, v=self.v_wrd, mask=mask, mem=mem[i],r_net=pos_enc_ent, u_ent=self.u_ent,v_ent=self.v_ent,mask_ent=mask_ent,mem_ent=mem_ent[i])
            hids.append(inp)
            hids_ent.append(inp_ent)
        # core_out = inp[:,-x_len:]

        mem =  self._update_mems(mem,hids)
        mem_ent = self._update_mems(mem_ent,hids_ent)
        return mem,mem_ent ,inp,inp_ent

            

# -
if __name__=="__main__":
    model = MaskedKnowledgeAGG(default_wrd_conf,default_ent_conf,512,512,5)
    model


