"""SR-111-independent N1/N2 battery trajectory model families.

Input is a fixed 100-cycle, 64-position VIQ+delta-Q tensor. No modules
are imported from models.unified_matr. Inference takes only early input and
known future query coordinates, NEVER future SOH.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

ENCODERS = ("resmlp", "voltage_cnn", "temporal_tcn", "cnn_tcn", "cnn_transformer")
DECODERS = ("query_mlp", "legacy_flow", "free_rate", "monotone_spline", "physics_basis")


class ConvCycle(nn.Module):
    def __init__(self, cin=4, width=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 48, 5, padding=2), nn.GELU())
        self.reduce = nn.Sequential(nn.Linear(48 * 8, width), nn.LayerNorm(width))

    def forward(self, x, point_mask):
        b, c, v, f = x.shape
        z = self.conv(x.reshape(b*c, v, f).transpose(1,2))
        z = F.avg_pool1d(z, kernel_size=8).flatten(1)
        return self.reduce(z).reshape(b, c, -1)


class FlatCycle(nn.Module):
    def __init__(self, cin=4, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64*cin, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, x, point_mask):
        return self.net(x.flatten(-2))


class ResidualFlatCycle(FlatCycle):
    def __init__(self, cin=4, width=64):
        super().__init__(cin, width)
        self.residual = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, x, point_mask):
        projected = super().forward(x, point_mask)
        return self.norm(projected + self.residual(projected))


class ResidualTemporalConv(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, dilation=dilation, padding=dilation, groups=width),
            nn.Conv1d(width, width, 1), nn.GELU(),
            nn.Conv1d(width, width, 1))
        self.norm = nn.LayerNorm(width)

    def forward(self, x, valid):
        # Only valid positions contribute features. A 100-cycle prefix has no future input.
        x = x * valid[..., None]
        z = self.net(x.transpose(1,2)).transpose(1,2)
        return self.norm(x + z) * valid[..., None]


class EarlyEncoder(nn.Module):
    def __init__(self, kind, in_channels=4, policy_dim=15, width=64):
        super().__init__()
        if kind not in ENCODERS:
            raise ValueError(f"unknown encoder {kind}")
        self.kind = kind
        if kind == 'resmlp':
            self.cycle = ResidualFlatCycle(in_channels, width)
        elif kind == 'temporal_tcn':
            self.cycle = FlatCycle(in_channels, width)
        else:
            self.cycle = ConvCycle(in_channels, width)
        self.cycle_gap = nn.Linear(3, width)
        if kind in ("temporal_tcn", "cnn_tcn"):
            self.temporal = nn.ModuleList(ResidualTemporalConv(width,d) for d in (1,2,4,8,16,32))
        elif kind == "cnn_transformer":
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width*2,
                                               batch_first=True, dropout=0.1, activation="gelu")
            self.temporal = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        else:
            self.temporal = nn.Identity()
        self.policy_net = nn.Sequential(nn.Linear(policy_dim*2, 32), nn.GELU(), nn.Linear(32,32))
        self.output = nn.Sequential(nn.Linear(width*2+32, width), nn.GELU(), nn.LayerNorm(width))

    def forward(self, x, point_mask, cycle_ids, policy, policy_mask):
        if x.ndim != 4 or tuple(x.shape[1:3]) != (100,64) or x.shape[-1] != 4:
            raise ValueError("expected [B,100,64,4] input")
        b = x.shape[0]
        if point_mask.shape != x.shape[:3] or cycle_ids.shape != (b,100):
            raise ValueError("input mask/cycle dimensions invalid")
        if policy_mask.shape != policy.shape or policy.shape[0] != b:
            raise ValueError("policy mask invalid")
        valid = point_mask.any(dim=-1)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("no eligible early cycle")
        x = torch.where(point_mask[...,None], x, 0.)
        cycle = self.cycle(x, point_mask)
        safe_cyc = torch.where(valid, cycle_ids.float(), 0)
        slots = torch.arange(100, device=valid.device)[None].expand(b, -1)
        previous_index = torch.cummax(torch.where(valid, slots, -1), dim=1).values
        previous_index = F.pad(previous_index[:, :-1], (1, 0), value=-1)
        previous = torch.where(previous_index >= 0,
                               safe_cyc.gather(1, previous_index.clamp_min(0)), 0)
        gap = (safe_cyc-previous).clamp_min(0)/100
        position = torch.stack((safe_cyc/100, gap, valid.float()), dim=-1)
        cycle = (cycle+self.cycle_gap(position))*valid[...,None]
        if isinstance(self.temporal, nn.ModuleList):
            for block in self.temporal:
                cycle = block(cycle,valid)
        elif self.kind == "cnn_transformer":
            cycle = self.temporal(cycle, src_key_padding_mask=~valid)*valid[...,None]
        pooled = cycle.sum(dim=1)/valid.sum(dim=1).clamp_min(1)[:,None]
        indices = torch.where(valid, torch.arange(100,device=valid.device)[None], -1).max(dim=1).values
        last = cycle[torch.arange(b,device=valid.device), indices]
        safe_policy = torch.where(policy_mask,policy,0.)
        p = self.policy_net(torch.cat((safe_policy,policy_mask.float()),dim=-1))
        return self.output(torch.cat((pooled,last,p),dim=-1))


class QueryMLP(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(width+1,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,z,anchor,query):
        u=(query-100)/1000
        b,t=u.shape
        zt=z[:,None].expand(-1,t,-1)
        correction=self.net(torch.cat((zt,u[...,None]),-1)).squeeze(-1)
        zero=self.net(torch.cat((z,torch.zeros_like(u[:,0:1])),dim=-1)).squeeze(-1)
        return anchor[:,None]+0.2*(correction-zero[:,None])


class LegacyStyleFlow(nn.Module):
    """Independent recreation of analytic latent relaxation; NOT SR-111 import."""
    def __init__(self,width):
        super().__init__()
        self.state=nn.Linear(width,16)
        self.gen=nn.Sequential(nn.Linear(width,32),nn.GELU(),nn.Linear(32,32))
        self.decode=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,z,anchor,query):
        u=(query-100)/1000
        rate,mean=self.gen(z).chunk(2,-1)
        rate=F.softplus(rate)+1e-4
        mean=mean.tanh()
        h0=self.state(z).tanh()
        decay=torch.exp(-rate[:,None,:]*u[...,None])
        h=decay*h0[:,None,:]+(1-decay)*mean[:,None,:]
        delta=self.decode(h).squeeze(-1)-self.decode(h0).squeeze(-1)[:,None]
        return anchor[:,None]+0.2*delta


class FreeRate(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(width+1,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
        self.bias=nn.Parameter(torch.tensor(-2.2))

    def forward(self,z,anchor,query):
        end = max(101, int(torch.ceil(query.max()).detach().item()))
        grid = torch.arange(100, end + 1, dtype=z.dtype, device=z.device)
        u = (grid - 100) / 1000
        context = z[:, None].expand(-1, len(grid), -1)
        coordinates = u[None, :, None].expand(len(z), -1, -1)
        rate = F.softplus(self.net(torch.cat((context, coordinates), -1)).squeeze(-1) + self.bias)
        increments = (rate[:, 1:] + rate[:, :-1]) / 2000
        # CPU cumulative sum keeps the forward and adjoint deterministic on CUDA.
        cumulative = increments.cpu().cumsum(dim=1).to(z.device)
        cumulative = torch.cat((rate.new_zeros(len(z), 1), cumulative), dim=1)
        position = query - 100
        left = position.floor().long().clamp_max(len(grid) - 2)
        fraction = position - left.to(position.dtype)
        integral = cumulative.gather(1, left) + fraction * (
            cumulative.gather(1, left + 1) - cumulative.gather(1, left))
        return anchor[:, None] - integral


class MonotoneSpline(nn.Module):
    """Piecewise-constant positive rate, continuous integrated monotone curve."""
    def __init__(self,width,segments=10,max_u=5.0):
        super().__init__()
        self.segments=segments
        self.max_u=max_u
        self.net=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,segments))
        self.register_buffer("knots",torch.linspace(0,max_u,segments+1))

    def forward(self,z,anchor,query):
        u=(query-100)/1000
        rate=F.softplus(self.net(z))*0.15
        left=self.knots[:-1]
        right=self.knots[1:]
        durations=(u[...,None]-left).clamp_min(0).minimum(right-left)
        # Continue with the last nonnegative slope beyond the last registered knot.
        tail=(u-self.max_u).clamp_min(0)*rate[:,-1,None]
        cumulative=(durations*rate[:,None,:]).sum(-1)+tail
        return anchor[:,None]*torch.exp(-cumulative)


class PhysicsBasis(nn.Module):
    """Nonnegative decreasing-rate, linear-rate and delayed acceleration components."""
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(width,64),nn.GELU(),nn.Linear(64,4))
        self.log_c=nn.Parameter(torch.tensor(-0.8))
        self.log_w=nn.Parameter(torch.tensor(-1.5))

    def forward(self,z,anchor,query):
        u=(query-100)/1000
        raw=self.net(z)
        a=F.softplus(raw[:,0:1])*0.05
        b=F.softplus(raw[:,1:2])*0.05
        c=F.softplus(raw[:,2:3])*0.06
        tau=4.0*torch.sigmoid(raw[:,3:4])
        time_c=F.softplus(self.log_c)+0.05
        smooth=F.softplus(self.log_w)+0.03
        cumulative=(a*torch.log1p(u/time_c)+b*u +
                    c*smooth*(F.softplus((u-tau)/smooth)-F.softplus(-tau/smooth)))
        return anchor[:,None]*torch.exp(-cumulative)


HEADS={"query_mlp": QueryMLP,"legacy_flow": LegacyStyleFlow,"free_rate": FreeRate,
       "monotone_spline":MonotoneSpline,"physics_basis":PhysicsBasis}


class TrajectoryNet(nn.Module):
    def __init__(self,encoder="cnn_tcn",decoder="query_mlp",policy_dim=15,width=64):
        super().__init__()
        if decoder not in HEADS:
            raise ValueError(f"unknown decoder {decoder}")
        self.encoder=EarlyEncoder(encoder,policy_dim=policy_dim,width=width)
        self.decoder=HEADS[decoder](width)
        self.encoder_name=encoder
        self.decoder_name=decoder

    def forward(self,x,point_mask,cycle_ids,policy,policy_mask,anchor,query):
        if query.ndim==1:
            query=query[None].expand(x.shape[0],-1)
        if query.ndim!=2 or query.shape[0]!=x.shape[0] or not query.shape[1] or not bool(torch.isfinite(query).all() and (query>=100).all()):
            raise ValueError("future queries must have shape [B,T] and t >= 100")
        z=self.encoder(x,point_mask,cycle_ids,policy,policy_mask)
        return self.decoder(z,anchor,query)
