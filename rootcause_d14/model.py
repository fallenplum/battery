"""Lightweight explicit-physics branches around the unchanged N012 encoder."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from n012.model import TrajectoryNet


def masked_pool(v, mask, dim):
    safe = torch.where(mask,v,0.)
    return safe.sum(dim)/mask.sum(dim).clamp_min(1)


class PhysicsBranch(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if 'G' in kind:
            self.global_mlp = nn.Sequential(nn.Linear(40,32),nn.GELU(),nn.Linear(32,32))
        if 'L' in kind:
            self.residual_mlp = nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,32))
            self.local_mlp = nn.Sequential(nn.Linear(52,32),nn.GELU(),nn.Linear(32,32))
        count = int('G' in kind)+2*int('L' in kind)
        self.project = nn.Sequential(nn.Linear(count*64,64),nn.GELU(),nn.LayerNorm(64))

    def temporal(self, token, valid):
        m = valid[...,None].expand_as(token)
        mean = masked_pool(token,m,1)
        positions = torch.arange(token.shape[1],device=token.device)[None].expand_as(valid)
        last_idx = torch.where(valid,positions,0).max(1).values
        last = token[torch.arange(len(token),device=token.device),last_idx]
        last = torch.where(valid.any(1)[:,None],last,0.)
        return torch.cat((mean,last),-1)

    def forward(self, arrays):
        parts = []
        for value,mask,network in (
            ('global_features','global_mask','global_mlp'),
            ('local_residual','local_mask','residual_mlp'),
            ('local_features','local_feature_mask','local_mlp')):
            if not hasattr(self,network):
                continue
            m = arrays[mask]
            token = getattr(self,network)(torch.cat((torch.where(m,arrays[value],0.),m.float()),-1))
            valid = m.any(-1)
            if value == 'local_features':
                token = masked_pool(token,valid[...,None].expand_as(token),2)
                valid = valid.any(-1)
            parts.append(self.temporal(token,valid))
        return self.project(torch.cat(parts,-1))


class RootCauseNet(nn.Module):
    def __init__(self, encoder, decoder, policy_dim, features='none'):
        super().__init__()
        self.base = TrajectoryNet(encoder,decoder,policy_dim=policy_dim,width=64)
        self.features = features
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(420014)
            if features == 'capacity':
                target = sum(p.numel() for p in PhysicsBranch('GL').parameters())
                # Two active layers from the existing z, rounded to nearest width.
                hidden = round((target-64)/129)
                self.branch = nn.Sequential(nn.Linear(64,hidden),nn.GELU(),nn.Linear(hidden,64),nn.LayerNorm(64))
            elif features != 'none':
                self.branch = PhysicsBranch(features)

    def encode(self, x,point_mask,cycle_ids,policy,policy_mask,physics):
        z = self.base.encoder(x,point_mask,cycle_ids,policy,policy_mask)
        if self.features == 'capacity':
            z = z+self.branch(z)
        elif self.features != 'none':
            z = z+self.branch(physics)
        return z

    def forward(self,x,point_mask,cycle_ids,policy,policy_mask,anchor,query,physics,return_z=False):
        z = self.encode(x,point_mask,cycle_ids,policy,policy_mask,physics)
        if query.ndim == 1:
            query = query[None].expand(len(x),-1)
        pred = self.base.decoder(z,anchor,query)
        return (pred,z) if return_z else pred
