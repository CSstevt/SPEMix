import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import BaseModule, force_fp32
from torch import Tensor, LongTensor
from typing import Tuple

class DAMixBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 unsampling_mode='bilinear',
                 lam_concat=False,
                 lam_concat_v=False,
                 lam_mul=0.,
                 lam_mul_k=-1,
                 lam_residual=False,
                 value_neck_cfg=None,
                 x_qk_concat=False,
                 x_v_concat=False,
                 att_norm_cfg=None,
                 att_act_cfg=None,
                 mask_loss_mode="L1",
                 mask_loss_margin=0,
                 frozen=False,
                 init_cfg=None,
                 **kwargs):
        super(DAMixBlock, self).__init__(init_cfg)
        # non-local args
        self.in_channels = int(in_channels)
        self.reduction = int(reduction)
        self.use_scale = bool(use_scale)
        self.inter_channels = max(in_channels // reduction, 1)
        self.unsampling_mode = [unsampling_mode] if isinstance(unsampling_mode, str) \
            else list(unsampling_mode)
        for m in self.unsampling_mode:
            assert m in ['nearest', 'bilinear', 'bicubic', ]
        self.lam_concat = bool(lam_concat)
        self.lam_concat_v = bool(lam_concat_v)
        self.lam_mul = float(lam_mul) if float(lam_mul) > 0 else 0
        self.lam_mul_k = [lam_mul_k] if isinstance(lam_mul_k, (int, float)) else list(lam_mul_k)
        self.lam_residual = bool(lam_residual)
        assert att_norm_cfg is None or isinstance(att_norm_cfg, dict)
        assert att_act_cfg is None or isinstance(att_act_cfg, dict)
        assert value_neck_cfg is None or isinstance(value_neck_cfg, dict)
        self.value_neck_cfg = value_neck_cfg
        self.x_qk_concat = bool(x_qk_concat)
        self.x_v_concat = bool(x_v_concat)
        self.mask_loss_mode = str(mask_loss_mode)
        self.mask_loss_margin = max(mask_loss_margin, 0.)
        self.frozen = bool(frozen)
        self.attn=BiLevelRoutingAttention(128)
        assert 0 <= lam_mul and lam_mul <= 1
        for i in range(len(self.lam_mul_k)):
            self.lam_mul_k[i] = min(self.lam_mul_k[i], 10) if self.lam_mul_k[i] >= 0 else -1
        assert mask_loss_mode in ["L1", "L1+Variance", "L2+Variance", "Sparsity"]
        if self.lam_concat or self.lam_concat_v:
            assert self.lam_concat != self.lam_concat_v, \
                "lam_concat can be adopted on q,k,v or only on v"
        if self.lam_concat or self.lam_mul:
            assert self.lam_concat != self.lam_mul, \
                "both lam_concat and lam_mul change q,k,v in terms of lam"
        if self.lam_concat or self.x_qk_concat:
            assert self.lam_concat != self.x_qk_concat, \
                "x_lam=x_lam_=cat(x,x_) if x_qk_concat=True, it's no use to concat lam"
        # FP16 training: exit after 10 times overflow
        self.overflow = 0

        # concat all as k,q,v
        self.qk_in_channels = int(in_channels + 1) \
            if self.lam_concat else int(in_channels)
        self.v_in_channels = int(in_channels + 1) \
            if self.lam_concat or self.lam_concat_v else int(in_channels)

        if self.x_qk_concat:
            self.qk_in_channels = int(2 * self.in_channels)
        if self.x_v_concat:
            self.v_in_channels = int(2 * self.in_channels)

            # MixBlock, conv value
        if value_neck_cfg is None:
            self.value = nn.Conv2d(
                in_channels=self.v_in_channels,
                out_channels=1,
                kernel_size=1,
                stride=1)  # 提取value的卷积核
        self.key = None
        if self.x_qk_concat:  # sym conv q and k
            # conv key
            self.key = ConvModule(
                in_channels=self.qk_in_channels,
                out_channels=self.inter_channels,
                kernel_size=1, stride=1, padding=0,
                groups=1, bias='auto',
                norm_cfg=att_norm_cfg,
                act_cfg=att_act_cfg,
            )
        # conv query
        self.query = ConvModule(
            in_channels=self.qk_in_channels,
            out_channels=self.inter_channels,
            kernel_size=1, stride=1, padding=0,
            groups=1, bias='auto',
            norm_cfg=att_norm_cfg,
            act_cfg=att_act_cfg,
        )

        self.init_weights()
        if self.frozen:
            self._freeze()

    def init_weights(self, init_linear='normal', std=0.01, bias=0.):
            if self.init_cfg is not None:
                super(DAMixBlock, self).init_weights()
                return
            assert init_linear in ['normal', 'kaiming'], \
                "Undefined init_linear: {}".format(init_linear)
            # init mixblock
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    if init_linear == 'normal':
                        normal_init(m, std=std, bias=bias)
                    else:
                        kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                    constant_init(m, val=1, bias=0)

    def _freeze(self):
        if self.frozen:
                # mixblock
            for param in self.query.parameters():
                param.requires_grad = False
            if self.key is not None:
                for param in self.key.parameters():
                    param.requires_grad = False
            for param in self.value.parameters():
                param.requires_grad = False


    @force_fp32(apply_to=('q_x', 'k_x',))
    def embedded_gaussian(self, q_x, k_x):
            pairwise_weight = torch.matmul(
                q_x.type(torch.float32), k_x.type(torch.float32)
            ).type(torch.float32)
            if torch.any(torch.isnan(pairwise_weight)):
                print("Warming attention map is nan, P: {}. Exit FP16!".format(pairwise_weight))
                raise ValueError
            if torch.any(torch.isinf(pairwise_weight)):
                print("Warming attention map is inf, P: {}, climp!".format(pairwise_weight))
                pairwise_weight = pairwise_weight.type(torch.float32).clamp(min=-1e25, max=1e25)
                self.overflow += 1
                if self.overflow > 10:
                    raise ValueError("Precision overflow in MixBlock, try fp32 training.")
            if self.use_scale:
                # q_x.shape[-1] is `self.inter_channels`
                pairwise_weight /= q_x.shape[-1] ** 0.5
            # force fp32 in exp
            pairwise_weight = pairwise_weight.type(torch.float32).softmax(dim=-1)
            return pairwise_weight

    def rescale_lam_mult(self, lam, k=1):
            """ adjust lam against y=x in terms of k """
            assert k >= 0
            k += 1
            lam = float(lam)
            return 1 / (k - 2 / 3) * (4 / 3 * math.pow(lam, 3) - 2 * lam ** 2 + k * lam)

    def forward(self, x, lam, index, scale_factor, debug=True, unsampling_override=None):
            x=self.attn(x)
            results = dict()  # 储存返回的结果
            # pre-step 0: input 2d feature map x, [N, C, H, W]
            if isinstance(x, list) and index is None:
                assert len(x) == 2  # only for SSL mixup
                x = torch.cat(x)
            n, _, h, w = x.size()

            if index is None:  # only for SSL mixup, [2N, C, H, W]
                n = n // 2
                x_lam = x[:n, ...]
                x_lam_ = x[n:, ...]
            else:  # supervised cls
                x_lam = x
                x_lam_ = x[index, :]  # 将N划分成两部分         shuffle within a gpu
            results = dict(x_lam=x_lam, x_lam_=x_lam_)

            # pre-step 1: lambda encoding
            if self.lam_mul > 0:  # multiply lam to x_lam
                assert self.lam_concat == False
                # rescale lam
                _lam_mul_k = random.choices(self.lam_mul_k, k=1)[0]
                if _lam_mul_k >= 0:
                    lam_rescale = self.rescale_lam_mult(lam, _lam_mul_k)
                else:
                    lam_rescale = lam
                # using residual
                if self.lam_residual:
                    x_lam = x_lam * (1 + lam_rescale * self.lam_mul)
                    x_lam_ = x_lam_ * (1 + (1 - lam_rescale) * self.lam_mul)
                else:
                    x_lam = x_lam * lam_rescale
                    x_lam_ = x_lam_ * (1 - lam_rescale)
            if self.lam_concat:  # concat lam as a new channel
                # assert self.lam_mul > 0 and self.x_qk_concat == False
                lam_block = torch.zeros(n, 1, h, w).to(x_lam)  # 将混合率编码创建一个n,h,w和输入特>征图相同，其通道数为1 的零张量
                lam_block[:] = lam  # 将所有的值赋成混合率
                x_lam = torch.cat([x_lam, lam_block], dim=1)  # concat Zi and lamda ,(n,c,h,w  -> n,c+1,h,w)
                x_lam_ = torch.cat([x_lam_, 1 - lam_block], dim=1)  # concat Zj and lamda

                # **** step 1: conpute 1x1 conv value, v: [N, HxW, 1] ****
            v_ = x_lam_
            if self.x_v_concat:
                v_ = torch.cat([x_lam, x_lam_], dim=1)
            if self.lam_concat_v:
                lam_block = torch.zeros(n, 1, h, w).to(x_lam)
                lam_block[:] = lam
                v_ = torch.cat([x_lam_, 1 - lam_block], dim=1)
            # compute v_i
            if self.value_neck_cfg is None:
                #            print("v_ size before value:", v_.size())
                v_ = self.value(v_)
                #           print("v_ size after value:", v_.size())
                v_ = v_.view(n, 1, -1)
            else:
                v_ = self.value([v_])[0].view(n, 1, -1)  # [N, 1, HxW]
            v_ = v_.permute(0, 2, 1)  # v_ for 1-lam: [N, HxW, 1]
            if debug:
                debug_plot = dict(value=v_.view(n, h, -1).clone().detach())
            if self.x_qk_concat:
                x_lam = torch.cat([x_lam, x_lam_], dim=1)
                x_lam_ = x_lam
                # query
            q_x = self.query(x_lam).view(n, self.inter_channels, -1).permute(0, 2, 1)  # q for lam: [N, HxW, C/r]
            # key
            if self.key is not None:
                k_x = self.key(x_lam_).view(n, self.inter_channels, -1)  # [N, C/r, HxW]
            else:
                k_x = self.query(x_lam_).view(n, self.inter_channels, -1)  # [N, C/r, HxW]

            # **** step 3: 2d pairwise_weight: [N, HxW, HxW] ****
            pairwise_weight = self.embedded_gaussian(q_x, k_x)  # x_lam [N, HxW, C/r] x [N, C/r, HxW] x_lam_

            # debug mode
            if debug:
                debug_plot["pairwise_weight"] = pairwise_weight.clone().detach()
                results["debug_plot"] = debug_plot

            # choose upsampling mode
            if unsampling_override is not None:
                if isinstance(unsampling_override, str):
                    up_mode = unsampling_override
                elif isinstance(unsampling_override, list):
                    up_mode = random.choices(unsampling_override, k=1)[0]
                else:
                    print("Warming upsampling_mode: {}, override to 'nearest'!".format(unsampling_override))
                    up_mode = "nearest"
            else:
                up_mode = random.choices(self.unsampling_mode, k=1)[0]
            mask_lam_ = torch.matmul(
                pairwise_weight.type(torch.float32), v_.type(torch.float32)
            ).view(n, 1, h, w)  # mask for 1-lam
            if torch.any(torch.isnan(mask_lam_)):
                print("Warming mask_lam_ is nan, P: {}, v: {}, remove nan.".format(pairwise_weight, v_))
                mask_lam_ = torch.matmul(
                    pairwise_weight.type(torch.float64), v_.type(torch.float64)
                ).view(n, 1, h, w)
                mask_lam_ = torch.where(torch.isnan(mask_lam_),
                                        torch.full_like(mask_lam_, 1e-4), mask_lam_)

            mask_lam_ = F.interpolate(mask_lam_, scale_factor=4, mode=up_mode)
            # mask for 1-lam in [0, 1], force fp32 in exp (causing NAN in fp16)
            mask_lam_ = torch.sigmoid(mask_lam_.type(torch.float32))
            mask = torch.cat([1 - mask_lam_, mask_lam_], dim=1)
            results["mask"] = mask
            return results

    def mask_loss(self, mask, lam):
            """ loss for mixup masks """
            losses = dict()
            assert mask.dim() == 4
            n, k, h, w = mask.size()  # mixup mask [N, 2, H, W]
            if k > 1:  # the second mask has no grad!
                mask = mask[:, 1, :, :].unsqueeze(1)
            m_mean = mask.sum() / (n * h * w)  # mask mean in [0, 1]

            if self.mask_loss_mode == "L1":  # [0, 1-m]
                losses['loss'] = torch.clamp(
                    torch.abs(1 - m_mean - lam) - self.mask_loss_margin, min=0.).mean()
            elif self.mask_loss_mode == "Sparsity":  # [0, 0.25-m]
                losses['loss'] = torch.clamp(
                    torch.abs(mask * (mask - 1)).sum() / (n * h * w) - self.mask_loss_margin, min=0.)
            elif self.mask_loss_mode == "L1+Variance":  # [0, 1-m] + [0, 1]
                losses['loss'] = torch.clamp(
                    torch.abs(1 - m_mean - lam) - self.mask_loss_margin, min=0.).mean() - \
                                 2 * torch.clamp((torch.sum((mask - m_mean) ** 2) / (n * h * w)), min=0.)
            elif self.mask_loss_mode == "L2+Variance":  # [0, 1-m^2] + [0, 1]
                losses['loss'] = torch.clamp(
                    (1 - m_mean - lam) ** 2 - self.mask_loss_margin ** 2, min=0.).mean() - \
                                 2 * torch.clamp((torch.sum((mask - m_mean) ** 2) / (n * h * w)), min=0.)
            else:
                raise NotImplementedError
            if torch.isnan(losses['loss']):
                print("Warming mask loss nan, mask sum: {}, skip.".format(mask))
                losses['loss'] = None
                self.overflow += 1
                if self.overflow > 10:
                    raise ValueError("Precision overflow in MixBlock, try fp32 training.")
            return losses

class TopkRouting(nn.Module):
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index

    class KVGather(nn.Module):
        def __init__(self, mul_weight='soft'):
            super().__init__()
            assert mul_weight in ['none', 'soft', 'hard']
            self.mul_weight = mul_weight

        def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
            n, p2, w2, c_kv = kv.size()
            topk = r_idx.size(-1)
            topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                                   # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                   dim=2,
                                   index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                                   # (n, p^2, k, w^2, c_kv)
                                   )

            if self.mul_weight == 'soft':
                topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
            elif self.mul_weight == 'hard':
                raise NotImplementedError('differentiable hard routing TBA')
            return topk_kv
class KVGather(nn.Module):
    def __init__(self, mul_weight='soft'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        return topk_kv
class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv
def _grid2seq(x: Tensor, region_size: Tuple[int], num_heads: int):
    B, C, H, W = x.size()
    region_h, region_w = H // region_size[0], W // region_size[1]
    x = x.view(B, num_heads, C // num_heads, region_h, region_size[0], region_w, region_size[1])
    x = torch.einsum('bmdhpwq->bmhwpqd', x).flatten(2, 3).flatten(-3, -2)  # (bs, nhead, nregion, reg_size, head_dim)
    return x, region_h, region_w
def _seq2grid(x: Tensor, region_h: int, region_w: int, region_size: Tuple[int]):
    bs, nhead, nregion, reg_size_square, head_dim = x.size()
    x = x.view(bs, nhead, region_h, region_w, region_size[0], region_size[1], head_dim)
    x = torch.einsum('bmhwpqd->bmdhpwq', x).reshape(bs, nhead * head_dim,
                                                    region_h * region_size[0], region_w * region_size[1])
    return x
from typing import Optional, Tuple
def regional_routing_attention_torch(
        query: Tensor, key: Tensor, value: Tensor, scale: float,
        region_graph: LongTensor, region_size: Tuple[int],
        kv_region_size: Optional[Tuple[int]] = None,
        auto_pad=True) -> Tensor:
    kv_region_size = kv_region_size or region_size
    bs, nhead, q_nregion, topk = region_graph.size()

    # Auto pad to deal with any input size
    q_pad_b, q_pad_r, kv_pad_b, kv_pad_r = 0, 0, 0, 0
    if auto_pad:
        _, _, Hq, Wq = query.size()
        q_pad_b = (region_size[0] - Hq % region_size[0]) % region_size[0]
        q_pad_r = (region_size[1] - Wq % region_size[1]) % region_size[1]
        if (q_pad_b > 0 or q_pad_r > 0):
            query = F.pad(query, (0, q_pad_r, 0, q_pad_b))  # zero padding

        _, _, Hk, Wk = key.size()
        kv_pad_b = (kv_region_size[0] - Hk % kv_region_size[0]) % kv_region_size[0]
        kv_pad_r = (kv_region_size[1] - Wk % kv_region_size[1]) % kv_region_size[1]
        if (kv_pad_r > 0 or kv_pad_b > 0):
            key = F.pad(key, (0, kv_pad_r, 0, kv_pad_b))  # zero padding
            value = F.pad(value, (0, kv_pad_r, 0, kv_pad_b))  # zero padding

    # to sequence format, i.e. (bs, nhead, nregion, reg_size, head_dim)
    query, q_region_h, q_region_w = _grid2seq(query, region_size=region_size, num_heads=nhead)
    key, _, _ = _grid2seq(key, region_size=kv_region_size, num_heads=nhead)
    value, _, _ = _grid2seq(value, region_size=kv_region_size, num_heads=nhead)
    bs, nhead, kv_nregion, kv_region_size, head_dim = key.size()
    broadcasted_region_graph = region_graph.view(bs, nhead, q_nregion, topk, 1, 1). \
        expand(-1, -1, -1, -1, kv_region_size, head_dim)
    key_g = torch.gather(key.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
                         expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                         index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    value_g = torch.gather(value.view(bs, nhead, 1, kv_nregion, kv_region_size, head_dim). \
                           expand(-1, -1, query.size(2), -1, -1, -1), dim=3,
                           index=broadcasted_region_graph)  # (bs, nhead, q_nregion, topk, kv_region_size, head_dim)
    attn = (query * scale) @ key_g.flatten(-3, -2).transpose(-1, -2)
    attn = torch.softmax(attn, dim=-1)
    output = attn @ value_g.flatten(-3, -2)
    output = _seq2grid(output, region_h=q_region_h, region_w=q_region_w, region_size=region_size)

    if auto_pad and (q_pad_b > 0 or q_pad_r > 0):
        output = output[:, :, :Hq, :Wq]

    return output, attn
class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_scale=None, topk=8, side_dwconv=3, auto_pad=False,
                 attn_backend='torch'):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, 'dim must be divisible by num_heads!'
        self.head_dim = self.dim // self.num_heads
        self.scale = qk_scale or self.dim ** -0.5  # NOTE: to be consistent with old models.
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)
        self.topk = topk
        self.n_win = n_win  # number of windows per row/col
        self.qkv_linear = nn.Conv2d(self.dim, 3 * self.dim, kernel_size=1)
        self.output_linear = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        if attn_backend == 'torch':
            self.attn_fn = regional_routing_attention_torch
        else:
            raise ValueError('CUDA implementation is not available yet. Please stay tuned.')

    def forward(self, x: Tensor, ret_attn_mask=False):
        N, C, H, W = x.size()
        region_size = (H // self.n_win, W // self.n_win)
        qkv = self.qkv_linear.forward(x)  # ncHW
        q, k, v = qkv.chunk(3, dim=1)  # ncHW
        q_r = F.avg_pool2d(q.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        k_r = F.avg_pool2d(k.detach(), kernel_size=region_size, ceil_mode=True, count_include_pad=False)  # nchw
        q_r: Tensor = q_r.permute(0, 2, 3, 1).flatten(1, 2)  # n(hw)c
        k_r: Tensor = k_r.flatten(2, 3)  # nc(hw)
        a_r = q_r @ k_r  # n(hw)(hw), adj matrix of regional graph
        _, idx_r = torch.topk(a_r, k=self.topk, dim=-1)  # n(hw)k long tensor
        idx_r: LongTensor = idx_r.unsqueeze_(1).expand(-1, self.num_heads, -1, -1)
        output, attn_mat = self.attn_fn(query=q, key=k, value=v, scale=self.scale,
                                        region_graph=idx_r, region_size=region_size
                                        )
        output = output + self.lepe(v)  # ncHW
        output = self.output_linear(output)  # ncHW

        if ret_attn_mask:
            return output, attn_mat

        return output


