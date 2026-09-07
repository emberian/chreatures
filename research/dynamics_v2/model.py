"""Research-only operating-point physiology on the authenticated full MaleCNS.

This functional NumPy/SciPy equation reference is for calibration, never a
resident tick implementation. Production recurrence belongs in Rust/GPU kernels.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from scipy import sparse
from scipy.special import expit
# Frozen v1 anatomy/afferent import only, isolated from the current V2 runtime.
MAGIC = b"CHCNS1\0\0"
ARRAY_SPECS = (('graph.crow', '<u4', (165123,)), ('graph.col', '<u4', (25563197,)), ('graph.weight', '<f4', (25563197,)), ('atlas.receptor_rows', '<u4', (4107,)), ('atlas.receptor_type', '<u4', (4107,)), ('atlas.receptor_ptr', '<u4', (4108,)), ('atlas.site_indices', '<u4', (4669,)), ('atlas.site_weight', '<f4', (4669,)), ('atlas.body_rows', '<u4', (11233,)), ('atlas.neuron_type', '<u4', (165122,)), ('optic.spectral_logits', '<f4', (10, 3)), ('optic.gain_raw', '<f4', (10,)), ('optic.bias', '<f4', (10,)), ('body.mean', '<f4', (43,)), ('body.scale', '<f4', (43,)), ('body.input.weight', '<f4', (128, 43)), ('body.input.bias', '<f4', (128,)), ('body.output.weight', '<f4', (11233, 128)), ('body.output.bias', '<f4', (11233,)), ('dynamics.bias_raw', '<f4', (11752,)), ('dynamics.tau_raw', '<f4', (11752,)), ('dynamics.source_raw', '<f4', (11752,)), ('dynamics.target_raw', '<f4', (11752,)), ('dynamics.excitability_raw', '<f4', (11752,)), ('readout.weight', '<f4', (512, 165122)), ('readout.bias', '<f4', (512,)))

N = 165122


def load_service(path: Path):
    """Memory-map and authenticate every original artifact array, including W."""
    with path.open('rb') as f:
        if f.read(8) != MAGIC:
            raise ValueError('requires archived CNS v1 service for research import')
        n = struct.unpack('<I', f.read(4))[0]
        meta = json.loads(f.read(n))
    offset = 12 + n
    arrays = {}
    for name, dtype, shape in ARRAY_SPECS:
        a = np.memmap(path, mode='r', offset=offset, dtype=dtype, shape=shape)
        if hashlib.sha256(a).hexdigest() != meta['array_sha256'][name]:
            raise ValueError('array checksum mismatch: '+name)
        arrays[name] = a
        offset += a.nbytes
    if path.stat().st_size != offset:
        raise ValueError('trailing bytes')
    return arrays, meta


@dataclass(frozen=True)
class Regime:
    gain: float = .9
    tau: float = .08
    baseline: float = .2
    adaptation_gain: float = .15
    adaptation_tau: float = 1.5
    dt: float = .05


class FullGraph:
    def __init__(self, arrays, identity=None):
        self.identity = dict(identity or {})
        self.arrays = arrays
        a = arrays
        self.w = sparse.csr_matrix((a['graph.weight'], a['graph.col'], a['graph.crow']), shape=(N,N))
        self.optic = sparse.csr_matrix((a['atlas.site_weight'],a['atlas.site_indices'],a['atlas.receptor_ptr']),shape=(4107,1771))
        self.receptors = np.asarray(a['atlas.receptor_rows'], dtype=np.int64)
        self.body = np.asarray(a['atlas.body_rows'],dtype=np.int64)
        self.mask = np.ones(N, dtype=bool)
        self.mask[self.receptors] = False
        self.mask[self.body] = False
        self.supported = np.diff(a['atlas.receptor_ptr']) > 0
        self.neutral = self.drive(np.full((1,1771,3),.5,np.float32),np.asarray(a['body.mean'])[None,:])

    def drive(self, optic_rgb, body):
        a = self.arrays
        batch = body.shape[0]
        rgb = (self.optic @ optic_rgb.transpose(1,0,2).reshape(1771,-1)).reshape(4107,batch,3)
        typ = a['atlas.receptor_type']
        spectral = np.exp(a['optic.spectral_logits']-a['optic.spectral_logits'].max(-1,keepdims=True))
        spectral /= spectral.sum(-1,keepdims=True)
        mix = (rgb*spectral[typ,None,:]).sum(-1)
        gain = np.logaddexp(0,a['optic.gain_raw'])[typ]
        receptor = expit(a['optic.bias'][typ,None]+gain[:,None]*(2*mix-1))*self.supported[:,None]
        b = np.clip((body-a['body.mean'])/a['body.scale'],-8,8)
        hidden = np.tanh(b @ a['body.input.weight'].T+a['body.input.bias'])
        current = expit(hidden @ a['body.output.weight'].T+a['body.output.bias'])
        drive = np.zeros((N,batch),np.float32)
        drive[self.receptors] = receptor
        drive[self.body] = current.T
        return drive

    def step_v2(self, drive, state, regime):
        """State x is signed deviation from immutable baseline; a is signed too."""
        x, adaptation, support = state
        h = min(regime.baseline, 1-regime.baseline)
        alpha = -np.expm1(-regime.dt/(2*regime.tau))
        for _ in range(2):
            u = drive-self.neutral+regime.gain*(self.w @ x)-regime.adaptation_gain*adaptation
            target = support*h*np.tanh(u/h)
            x = x+alpha*(target-x)
        adaptation = adaptation+(-np.expm1(-regime.dt/regime.adaptation_tau))*(x-adaptation)
        support = np.clip(support+regime.dt*(.024*(1-support)-.003*np.abs(x)/h),.65,1)
        return x.astype(np.float32), adaptation.astype(np.float32), support.astype(np.float32)

    def step_v1(self, drive, state, dt=.05):
        r, adaptation, support = state
        a = self.arrays
        typ = a['atlas.neuron_type']
        bias = (.5*np.tanh(a['dynamics.bias_raw']))[typ,None]
        tau = (.025+.475*expit(a['dynamics.tau_raw']))[typ,None]
        source = (.5+expit(a['dynamics.source_raw']))[typ,None]
        target = (.5+expit(a['dynamics.target_raw']))[typ,None]
        exc = (.5+expit(a['dynamics.excitability_raw']))[typ,None]
        alpha = np.minimum(1,dt/(2*tau))
        for _ in range(2):
            q = np.maximum(0,np.tanh(bias+exc*(drive+.92*target*(self.w @ (source*r)))-.1*adaptation))
            r = r+alpha*(q*support-r)
        adaptation = adaptation+dt/5*(r-adaptation)
        support = np.clip(support+dt*(.024*(1-support)-.003*r),.65,1)
        return r,adaptation,support
