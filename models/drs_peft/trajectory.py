"""One fixed anchor-preserving cycle-query head for every TSFM/PEFT arm."""
from __future__ import annotations

import torch
from torch import nn

from .bridges import NativeBridge,CycleEmbeddingBridge


class CycleQueryHead(nn.Module):
    def __init__(self, state_dim=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(state_dim+1,64),nn.GELU(),
                               nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,state,anchor,query):
        if query.ndim==1:
            query=query[None].expand(len(state),-1)
        if query.ndim!=2 or query.shape[0]!=len(state) or not bool(torch.isfinite(query).all() and (query>=100).all()):
            raise ValueError('Queries must be finite physical/supplied cycle coordinates >=100')
        u=(query-100)/1000
        context=state[:,None].expand(-1,query.shape[1],-1)
        value=self.net(torch.cat((context,u[...,None]),-1)).squeeze(-1)
        reference=self.net(torch.cat((state,torch.zeros_like(anchor[:,None])),-1))
        return anchor[:,None]+.2*(value-reference)


class BatteryTSFM(nn.Module):
    def __init__(self,checkpoint_dir,policy_dim,bridge='native_multivariate',pretrained=True):
        super().__init__()
        constructors={'native_multivariate':NativeBridge,'cycle_embedding':CycleEmbeddingBridge}
        if bridge not in constructors:
            raise ValueError('Unregistered bridge')
        self.bridge=constructors[bridge](checkpoint_dir,policy_dim,pretrained=pretrained)
        self.head=CycleQueryHead()
        self.bridge_kind=bridge

    @property
    def backbone(self):
        return self.bridge.backbone.module

    def encode(self,early):
        return self.bridge.encode(**{k:early[k] for k in ('x','point_mask','cycle_ids','policy','policy_mask','anchor')})

    def forward(self,early,query):
        return self.head(self.encode(early),early['anchor'],query)

    def parameter_report(self):
        report=self.bridge.parameter_budget()
        report['head']=sum(v.numel() for v in self.head.parameters())
        report['head_trainable']=sum(v.numel() for v in self.head.parameters() if v.requires_grad)
        report['total']=sum(v.numel() for v in self.parameters())
        report['trainable']=sum(v.numel() for v in self.parameters() if v.requires_grad)
        return report
