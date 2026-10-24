"""BART rewriter with a selective SSM refinement adapter.
ReferenceMamba implements the Mamba-1 recurrence in plain PyTorch for verification.
It is intentionally sequential and is NOT the official optimized CUDA kernel.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput

class ReferenceMamba(nn.Module):
    def __init__(self, d_model, d_state=8, expand=2, d_conv=4):
        super().__init__(); self.inner=expand*d_model; self.d_state=d_state; self.rank=max(1,math.ceil(d_model/16))
        self.in_proj=nn.Linear(d_model,2*self.inner,bias=False)
        self.conv=nn.Conv1d(self.inner,self.inner,d_conv,groups=self.inner,padding=d_conv-1)
        self.x_proj=nn.Linear(self.inner,self.rank+2*d_state,bias=False)
        self.dt_proj=nn.Linear(self.rank,self.inner)
        nn.init.uniform_(self.dt_proj.weight,-self.rank**-.5,self.rank**-.5)
        dt=torch.exp(torch.empty(self.inner).uniform_(math.log(.001),math.log(.1)))
        with torch.no_grad(): self.dt_proj.bias.copy_(dt+torch.log(-torch.expm1(-dt)))
        self.A_log=nn.Parameter(torch.arange(1,d_state+1).float().log().repeat(self.inner,1))
        self.D=nn.Parameter(torch.ones(self.inner)); self.out_proj=nn.Linear(self.inner,d_model,bias=False)
    def forward(self, hidden, mask=None):
        x,z=self.in_proj(hidden).chunk(2,-1)
        x=F.silu(self.conv(x.transpose(1,2))[...,:hidden.shape[1]].transpose(1,2))
        delta,B,C=torch.split(self.x_proj(x),[self.rank,self.d_state,self.d_state],dim=-1)
        dt=F.softplus(self.dt_proj(delta)).float(); A=-self.A_log.float().exp()
        state=torch.zeros(hidden.shape[0],self.inner,self.d_state,device=hidden.device,dtype=torch.float32)
        outputs=[]
        for t in range(hidden.shape[1]):
            decay=torch.exp(dt[:,t,:,None]*A)
            update=dt[:,t,:,None]*B[:,t,None,:].float()*x[:,t,:,None].float()
            next_state=decay*state+update
            if mask is not None:
                valid=mask[:,t,None,None].bool(); state=torch.where(valid,next_state,state)
            else: state=next_state
            y=(state*C[:,t,None,:].float()).sum(-1)+self.D.float()*x[:,t].float()
            if mask is not None: y=y*mask[:,t,None]
            outputs.append(y.to(hidden.dtype))
        y=torch.stack(outputs,dim=1)*F.silu(z)
        return self.out_proj(y)

class HybridRewriter(nn.Module):
    def __init__(self, backbone, fusion='gated', backend='reference', d_state=8):
        super().__init__(); self.backbone=backbone; self.fusion=fusion; self.backend=backend; self.d_state=d_state
        d=backbone.config.d_model; self.norm=nn.LayerNorm(d)
        if backend=='reference': self.mixer=ReferenceMamba(d,d_state=d_state)
        elif backend=='official':
            from mamba_ssm import Mamba
            self.mixer=Mamba(d_model=d,d_state=d_state,d_conv=4,expand=2)
        else: raise ValueError('backend must be reference or official')
        self.gate=nn.Linear(2*d,d) if fusion=='gated' else None
        if fusion not in {'gated','residual','replace'}: raise ValueError('Unknown fusion')
        if self.gate is not None: nn.init.constant_(self.gate.bias,-2)
    def encode(self,input_ids,attention_mask):
        if self.backend=='official' and torch.any(attention_mask[:,1:] > attention_mask[:,:-1]):
            raise ValueError('Official causal mixer requires right padding')
        h=self.backbone.get_encoder()(input_ids=input_ids,attention_mask=attention_mask,return_dict=True).last_hidden_state
        n=self.norm(h); n=n*attention_mask.unsqueeze(-1)
        m=self.mixer(n,attention_mask) if self.backend=='reference' else self.mixer(n)
        if self.fusion=='gated': fused=h+torch.sigmoid(self.gate(torch.cat([h,m],-1)))*m
        elif self.fusion=='residual': fused=h+m
        else: fused=m
        return BaseModelOutput(last_hidden_state=fused*attention_mask.unsqueeze(-1))
    def forward(self,input_ids,attention_mask,labels=None,**kwargs):
        return self.backbone(encoder_outputs=self.encode(input_ids,attention_mask),attention_mask=attention_mask,labels=labels,**kwargs)
    @torch.no_grad()
    def generate(self,input_ids,attention_mask,**kwargs):
        return self.backbone.generate(encoder_outputs=self.encode(input_ids,attention_mask),attention_mask=attention_mask,**kwargs)
    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad=False
