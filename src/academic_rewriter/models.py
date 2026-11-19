"""Selective state-space mixer used by the six-layer rewriter.
ReferenceMamba implements the Mamba-1 recurrence in plain PyTorch for verification.
It is intentionally sequential and is NOT the official optimized CUDA kernel.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

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
