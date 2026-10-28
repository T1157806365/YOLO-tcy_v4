"""AGDH-DRG v2: decoupled, reliability-guided detection adaptation.

This module never modifies backbone, Neck, ADD, or P4/P5.  Hooks are
installed only around a single invocation of the YOLO26 Detect module:
  * P3 cv3 / one2one_cv3 outputs: additive UAV classification-logit correction.
  * P3 cv2 / one2one_cv2 inputs: bounded multiplicative regression guidance.

It supports the standard Ultralytics Detect(cv2/cv3, one2one_cv2/cv3) API,
including its one-to-one detached-feature behavior, without copying or
re-implementing Ultralytics forward, decode, or loss operations.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import torch
from torch import nn
from torch.nn import functional as F


class AGDHDRGV2(nn.Module):
    def __init__(self, nc: int = 1, hidden_channels: int = 16,
                 cls_gain: float = 0.5, reg_gain: float = 0.05,
                 use_reliability: bool = True, detach_alignment_prior: bool = True,
                 prior_dropout: float = 0.0):
        super().__init__()
        self.nc = int(nc)
        self.hidden_channels = int(hidden_channels)
        self.cls_gain = float(cls_gain)
        self.reg_gain = float(reg_gain)
        self.use_reliability = bool(use_reliability)
        self.detach_alignment_prior = bool(detach_alignment_prior)
        self.prior_dropout = float(prior_dropout)
        if self.nc <= 0 or self.hidden_channels <= 0:
            raise ValueError('nc and hidden_channels must be positive')
        if self.cls_gain < 0 or self.reg_gain < 0:
            raise ValueError('cls_gain and reg_gain must be nonnegative')
        if not 0 <= self.prior_dropout < 1:
            raise ValueError('prior_dropout must be in [0,1)')

        # Mean(P3) + [RGB, aligned-TIR, joint, union, disagreement] = 6 channels.
        # No assumption about P3's channel count, allowing strict baseline reuse.
        self.stem = nn.Sequential(
            nn.Conv2d(6, self.hidden_channels, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3,
                      padding=1, groups=self.hidden_channels, bias=True),
            nn.SiLU(),
        )
        self.reliability = nn.Conv2d(self.hidden_channels, 1, 1, bias=True)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)  # initial Q=0.5; not an accuracy probability

        # Separate one2many / one2one heads.  All start as strict identity.
        self.cls_many = nn.Conv2d(self.hidden_channels + self.nc, self.nc, 1)
        self.cls_one = nn.Conv2d(self.hidden_channels + self.nc, self.nc, 1)
        self.reg_many = nn.Conv2d(self.hidden_channels, 1, 1)
        self.reg_one = nn.Conv2d(self.hidden_channels, 1, 1)
        for layer in (self.cls_many, self.cls_one, self.reg_many, self.reg_one):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def make_prior(self, guidance_3ch: torch.Tensor, target_hw) -> Dict[str, torch.Tensor]:
        if guidance_3ch is None or guidance_3ch.ndim != 4 or guidance_3ch.shape[1] != 3:
            raise ValueError('alignment_info.multi_target_guidance must be [B,3,H,W]')
        x = guidance_3ch.detach() if self.detach_alignment_prior else guidance_3ch
        if x.shape[-2:] != tuple(target_hw):
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        x = x.clamp(0., 1.)
        sr, st, joint = x.split(1, dim=1)
        union = (1. - (1. - sr) * (1. - st)).clamp(0., 1.)
        disagreement = (sr - st).abs()
        # Sample-level prior dropout, all prior channels consistently disabled.
        if self.training and self.prior_dropout:
            keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.prior_dropout).to(x.dtype)
            sr, st, joint, union, disagreement = [t * keep for t in
                                                 (sr, st, joint, union, disagreement)]
        return {'packet': torch.cat((sr, st, joint, union, disagreement), dim=1),
                'union': union, 'joint': joint}

    def _encoded(self, p3: torch.Tensor, prior: Dict[str, torch.Tensor]):
        # Using the actual per-branch module input is important: the native
        # one-to-one feature detach is preserved by Detect.forward().
        signal = p3.mean(dim=1, keepdim=True)
        signal = signal.to(dtype=prior['packet'].dtype)
        latent = self.stem(torch.cat((signal, prior['packet']), dim=1))
        quality = (torch.sigmoid(self.reliability(latent))
                   if self.use_reliability else torch.ones_like(prior['union']))
        return latent, quality, prior['union']

    def _reg_input(self, x: torch.Tensor, prior, branch: str, debug: Optional[dict]):
        if self.reg_gain == 0:
            if debug is not None:
                debug['reg_' + branch] = torch.zeros_like(x[:, :1])
            return x
        latent, quality, union = self._encoded(x, prior)
        head = self.reg_many if branch == 'many' else self.reg_one
        delta = self.reg_gain * quality * union * torch.tanh(head(latent))
        if debug is not None:
            debug['reg_' + branch] = delta.detach()
            debug['quality_' + branch] = quality.detach()
        return x * (1. + delta.to(dtype=x.dtype))

    def _cls_output(self, logits: torch.Tensor, p3: torch.Tensor, prior,
                    branch: str, debug: Optional[dict]):
        if self.cls_gain == 0:
            if debug is not None:
                debug['cls_' + branch] = torch.zeros_like(logits)
            return logits
        if logits.ndim != 4 or logits.shape[1] != self.nc:
            raise RuntimeError(f'Expected [B,{self.nc},H,W] raw classification logits, '
                               f'got {tuple(logits.shape)}')
        latent, quality, union = self._encoded(p3, prior)
        head = self.cls_many if branch == 'many' else self.cls_one
        correction = head(torch.cat((latent, logits), dim=1))
        delta = self.cls_gain * quality * union * torch.tanh(correction)
        if debug is not None:
            debug['cls_' + branch] = delta.detach()
            debug['quality_' + branch] = quality.detach()
        return logits + delta.to(dtype=logits.dtype)

    @staticmethod
    def check_detect(detect):
        if getattr(detect, 'nl', None) != 3 or not bool(getattr(detect, 'end2end', False)):
            raise RuntimeError('AGDH-DRG v2 requires YOLO26 end2end Detect with 3 levels')
        names = ('cv2', 'cv3', 'one2one_cv2', 'one2one_cv3')
        for name in names:
            obj = getattr(detect, name, None)
            if obj is None or len(obj) != 3 or not isinstance(obj[0], nn.Module):
                raise RuntimeError(f'Detect must expose {name}[0] and 3 levels')
        if detect.cv2[0] is detect.one2one_cv2[0] or detect.cv3[0] is detect.one2one_cv3[0]:
            raise RuntimeError('Classification/regression branches must be independent')

    def forward_detect(self, detect: nn.Module, detect_inputs, guidance_3ch,
                       return_debug: bool = False):
        """Run the native Detect forward once with temporary P3-only hooks."""
        self.check_detect(detect)
        if not isinstance(detect_inputs, (list, tuple)) or len(detect_inputs) != 3:
            raise RuntimeError('Expected Detect input [P3,P4,P5]')
        p3 = detect_inputs[0]
        prior = self.make_prior(guidance_3ch, p3.shape[-2:])
        debug = {'prior': prior['packet'].detach(),
                 'union': prior['union'].detach()} if return_debug else None
        # Per-branch inputs are saved by the pre-hooks; the one-to-one input
        # is already detached inside Detect, so no backbone gradient leakage.
        seen = dict(reg_many=0, reg_one=0, cls_many=0, cls_one=0)
        last_inputs = {}
        handles = []

        def attach_reg(which, block):
            def pre_hook(_module, inputs):
                seen['reg_' + which] += 1
                if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                    raise RuntimeError('Unexpected P3 box head input')
                original = inputs[0]
                last_inputs[which] = original
                modulated = self._reg_input(original, prior, which, debug)
                return (modulated,)
            handles.append(block.register_forward_pre_hook(pre_hook))

        def attach_cls(which, block):
            def post_hook(_module, inputs, output):
                seen['cls_' + which] += 1
                if which not in last_inputs:
                    raise RuntimeError('Detect invoked P3 cls head before P3 box head; unsupported')
                if not torch.is_tensor(output):
                    raise RuntimeError('Unexpected P3 cls head output')
                return self._cls_output(output, last_inputs[which], prior, which, debug)
            handles.append(block.register_forward_hook(post_hook))

        # Installed hooks are removed even if native Detect raises an error.
        attach_reg('many', detect.cv2[0])
        attach_reg('one', detect.one2one_cv2[0])
        attach_cls('many', detect.cv3[0])
        attach_cls('one', detect.one2one_cv3[0])
        try:
            prediction = detect(list(detect_inputs))
        finally:
            for handle in handles:
                handle.remove()
        if any(n != 1 for n in seen.values()):
            raise RuntimeError(f'AGDH-DRG expected one call per P3 branch: {seen}')
        if debug is not None:
            debug['calls'] = seen.copy()
        return (prediction, debug) if return_debug else prediction


__all__ = ['AGDHDRGV2']
