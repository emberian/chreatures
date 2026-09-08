"""Differentiable full-graph implementation of embodiment-driven CNS V4."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

N = 165122
E = 25563197
SITES = 1771
RECEPTORS = 4107
BODY = 807
BODY_ROWS = 11798
CONTEXT = 12
CONTEXT_ROWS = 1314
MOTOR = 92
MOTOR_ROWS = 815
TYPES = 11752
LATENT = 512
DT = 0.01


def _inv_sigmoid(x):
    return math.log(x / (1 - x))


def _inv_softplus(x):
    return math.log(math.expm1(x))


def _csr(crow, col, val, device):
    return torch.sparse_csr_tensor(
        torch.as_tensor(crow.astype(np.int32), device=device),
        torch.as_tensor(col.astype(np.int32), device=device),
        torch.as_tensor(val.astype(np.float32), device=device),
        size=(N, N),
        device=device,
    )


class _FixedSparseMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, transpose, dense):
        ctx.transpose = transpose
        return torch.sparse.mm(matrix, dense)

    @staticmethod
    def backward(ctx, gradient):
        return None, None, torch.sparse.mm(ctx.transpose, gradient)


def fixed_sparse_mm(matrix, transpose, dense):
    return _FixedSparseMM.apply(matrix, transpose, dense)


class _MaskedBodyProjection(torch.autograd.Function):
    """Apply the fixed anatomical BODY807 support without a dense masked GEMM."""

    @staticmethod
    def forward(ctx, weight, standardized, crow, row, column):
        supported_weight = weight[row, column]
        supported_matrix = torch.sparse_csr_tensor(
            crow,
            column,
            supported_weight,
            size=(BODY_ROWS, BODY),
            device=weight.device,
        )
        current = torch.sparse.mm(supported_matrix, standardized.T)
        ctx.save_for_backward(supported_weight, standardized, row, column)
        return current

    @staticmethod
    def backward(ctx, gradient):
        supported_weight, standardized, row, column = ctx.saved_tensors
        supported_gradient = gradient[row]

        weight_gradient = gradient.new_zeros((BODY_ROWS, BODY))
        weight_gradient[row, column] = (
            supported_gradient * standardized[:, column].T
        ).sum(dim=1)

        standardized_gradient = standardized.new_zeros(standardized.shape)
        standardized_gradient.index_add_(
            1,
            column,
            supported_gradient.T * supported_weight[None],
        )
        return weight_gradient, standardized_gradient, None, None, None


def masked_body_projection(weight, standardized, crow, row, column):
    return _MaskedBodyProjection.apply(weight, standardized, crow, row, column)


@dataclass(frozen=True)
class CNSState:
    rates: torch.Tensor
    adaptation: torch.Tensor
    support: torch.Tensor
    release: torch.Tensor
    mod_da: torch.Tensor
    mod_oa: torch.Tensor
    mod_ht: torch.Tensor

    def detach(self):
        return CNSState(*(x.detach() for x in self.fields()))

    def fields(self):
        return (
            self.rates,
            self.adaptation,
            self.support,
            self.release,
            self.mod_da,
            self.mod_oa,
            self.mod_ht,
        )


class AnatomicalCNS(nn.Module):
    """Raw retina, body and delivered context enter only structurally masked currents."""

    def __init__(self, arrays, *, device=torch.device("cpu")):
        super().__init__()
        from chreatures.cns_adapter_contract import validate_arrays

        validate_arrays(arrays)
        self.device_ = torch.device(device)

        def p(name):
            return nn.Parameter(
                torch.as_tensor(np.array(arrays[name], copy=True), device=device)
            )

        self.optic_spectral_logits = p("optic.spectral_logits")
        self.optic_gain_raw = p("optic.gain_raw")
        self.optic_bias = p("optic.bias")
        self.register_buffer(
            "body_mean",
            torch.as_tensor(np.array(arrays["body.mean"], copy=True), device=device),
        )
        self.register_buffer(
            "body_scale_value",
            torch.as_tensor(np.array(arrays["body.scale"], copy=True), device=device),
        )
        self.body_weight = p("body.weight")
        self.body_bias = p("body.bias")
        self.context_weight = p("context.weight")
        self.context_bias = p("context.bias")
        for key in (
            "baseline_raw",
            "recurrent_gain_raw",
            "tau_raw",
            "adaptation_gain_raw",
            "adaptation_tau_raw",
            "release_tau_raw",
            "release_use_raw",
            "mod_gain_raw",
            "mod_adaptation_raw",
            "modulation_tau_raw",
        ):
            setattr(self, "dynamics_" + key, p("dynamics." + key))
        self.readout_projection = nn.Parameter(
            torch.as_tensor(
                np.array(arrays["readout.projection.weight"], copy=True), device=device
            )
        )
        self.readout_output = nn.Linear(64, LATENT, device=device)
        with torch.no_grad():
            self.readout_output.weight.copy_(p("readout.output.weight"))
            self.readout_output.bias.copy_(p("readout.output.bias"))
        self.register_buffer(
            "motor_reference_rate",
            torch.as_tensor(
                np.array(arrays["motor.reference_rate"], copy=True), device=device
            ),
        )
        self.register_buffer(
            "motor_rate_scale",
            torch.as_tensor(
                np.array(arrays["motor.rate_scale"], copy=True), device=device
            ),
        )
        self.motor_weight = p("motor.weight")
        self.motor_intercept = p("motor.intercept")

        def buf(name, value):
            tensor = (
                value
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(np.array(value, copy=True), device=device)
            )
            self.register_buffer(name, tensor, persistent=False)

        crow = np.asarray(arrays["graph.crow"])
        col = np.asarray(arrays["graph.col"])
        weight = np.asarray(arrays["graph.weight_bits"]).view("<f2").astype("<f4")
        channels = np.asarray(arrays["graph.channel"])
        source_channel = channels[col]
        from scipy import sparse

        for k, name in ((1, "fast"), (2, "da"), (3, "oa"), (4, "ht")):
            keep = source_channel == k
            edge_prefix = np.empty(E + 1, dtype=np.int64)
            edge_prefix[0] = 0
            np.cumsum(keep, out=edge_prefix[1:])
            compact = sparse.csr_matrix(
                (weight[keep], col[keep], edge_prefix[crow]), shape=(N, N)
            )
            transpose = compact.transpose().tocsr()
            buf(
                name + "_graph",
                _csr(compact.indptr, compact.indices, compact.data, device),
            )
            buf(
                name + "_graph_transpose",
                _csr(transpose.indptr, transpose.indices, transpose.data, device),
            )
        for name, key in (
            ("receptor_rows", "atlas.receptor_rows"),
            ("receptor_type", "atlas.receptor_type"),
            ("receptor_ptr", "atlas.receptor_ptr"),
            ("site_indices", "atlas.site_indices"),
            ("site_weight", "atlas.site_weight"),
            ("body_rows", "atlas.body_rows"),
            ("body_mask", "atlas.body_mask"),
            ("context_rows", "atlas.context_rows"),
            ("motor_rows", "atlas.motor_rows"),
            ("motor_mask", "atlas.motor_mask"),
            ("neuron_type", "atlas.neuron_type"),
        ):
            buf(name, arrays[key])
        body_support_row, body_support_column = np.nonzero(arrays["atlas.body_mask"])
        body_support_crow = np.zeros(BODY_ROWS + 1, dtype=np.int64)
        np.cumsum(
            np.bincount(body_support_row, minlength=BODY_ROWS),
            out=body_support_crow[1:],
        )
        buf("body_support_crow", body_support_crow)
        buf("body_support_row", body_support_row.astype(np.int64, copy=False))
        buf("body_support_column", body_support_column.astype(np.int64, copy=False))
        buf("optic_supported", np.diff(np.asarray(arrays["atlas.receptor_ptr"])) > 0)
        mask = np.ones(N, np.float32)
        mask[
            np.concatenate(
                (
                    arrays["atlas.receptor_rows"],
                    arrays["atlas.body_rows"],
                    arrays["atlas.context_rows"],
                )
            )
        ] = 0
        buf("readout_mask", mask)
        # receptor-by-site mapping
        buf(
            "optic_matrix",
            torch.sparse_csr_tensor(
                torch.as_tensor(
                    np.array(arrays["atlas.receptor_ptr"], dtype=np.int32, copy=True),
                    device=device,
                ),
                torch.as_tensor(
                    np.array(arrays["atlas.site_indices"], dtype=np.int32, copy=True),
                    device=device,
                ),
                torch.as_tensor(
                    np.array(arrays["atlas.site_weight"], dtype=np.float32, copy=True),
                    device=device,
                ),
                size=(RECEPTORS, SITES),
                device=device,
            ),
        )

    @property
    def device(self):
        return self.optic_bias.device

    @property
    def body_scale(self):
        return self.body_scale_value

    def effective(self):
        ti = self.neuron_type.long()
        raw = lambda n: getattr(self, "dynamics_" + n)
        vals = (
            0.05 + 0.4 * torch.sigmoid(raw("baseline_raw")),
            0.5 + 1.5 * torch.sigmoid(raw("recurrent_gain_raw")),
            0.02 + 0.23 * torch.sigmoid(raw("tau_raw")),
            0.5 * torch.sigmoid(raw("adaptation_gain_raw")),
            0.25 + 4.75 * torch.sigmoid(raw("adaptation_tau_raw")),
            0.05 + 1.95 * torch.sigmoid(raw("release_tau_raw")),
            0.01 + 0.49 * torch.sigmoid(raw("release_use_raw")),
        )
        return tuple(x[ti, None] for x in vals)

    def initial_state(self, batch):
        r0 = self.effective()[0].expand(N, batch).clone()
        z = torch.zeros_like(r0)
        return CNSState(
            r0,
            z,
            torch.ones_like(r0),
            torch.ones_like(r0),
            z.clone(),
            z.clone(),
            z.clone(),
        )

    def set_motor_normalization(self, reference_rate, rate_scale):
        """Install train-world motor-neuron moments before fitting the decoder."""
        if (
            reference_rate.shape != (MOTOR_ROWS,)
            or rate_scale.shape != (MOTOR_ROWS,)
            or reference_rate.dtype != torch.float32
            or rate_scale.dtype != torch.float32
            or reference_rate.device != self.device
            or rate_scale.device != self.device
            or not torch.all(torch.isfinite(reference_rate))
            or not torch.all(torch.isfinite(rate_scale))
            or torch.any(rate_scale <= 0)
        ):
            raise ValueError(
                "motor normalization must be finite float32 [815] with positive scale"
            )
        with torch.no_grad():
            self.motor_reference_rate.copy_(reference_rate)
            self.motor_rate_scale.copy_(rate_scale)

    def set_body_normalization(self, mean, scale):
        """Install BODY807 train-world moments before fitting afferent tuning."""
        if (
            mean.shape != (BODY,)
            or scale.shape != (BODY,)
            or mean.dtype != torch.float32
            or scale.dtype != torch.float32
            or mean.device != self.device
            or scale.device != self.device
            or not torch.all(torch.isfinite(mean))
            or not torch.all(torch.isfinite(scale))
            or torch.any(scale <= 0)
        ):
            raise ValueError(
                "body normalization must be finite float32 [807] with positive scale"
            )
        with torch.no_grad():
            self.body_mean.copy_(mean)
            self.body_scale_value.copy_(scale)

    def selected_activity(self, state):
        """CNS-derived activity exposed to training diagnostics, never raw senses."""
        return {
            "motor": state.rates[self.motor_rows.long()].T,
            "descending": state.rates[self.context_rows.long()].T,
        }

    def afferent_current(self, optic, body, context):
        if (
            optic.ndim != 3
            or optic.shape[1:] != (SITES, 3)
            or body.shape != (optic.shape[0], BODY)
            or context.shape != (optic.shape[0], CONTEXT)
            or optic.dtype != torch.float32
            or body.dtype != torch.float32
            or context.dtype != torch.float32
            or optic.device != self.device
            or body.device != self.device
            or context.device != self.device
        ):
            raise ValueError("invalid V4 optic, body, or delivered context tensor")
        if (
            not torch.all(torch.isfinite(optic))
            or not torch.all(torch.isfinite(body))
            or not torch.all(torch.isfinite(context))
        ):
            raise ValueError("V4 inputs must be finite")
        if torch.any((optic < 0) | (optic > 1)) or torch.any(
            (context < -1) | (context > 1)
        ):
            raise ValueError("V4 optic/context input outside contract bounds")
        b = optic.shape[0]
        rgb = (
            torch.sparse.mm(
                self.optic_matrix, optic.permute(1, 0, 2).reshape(SITES, b * 3)
            )
            .reshape(RECEPTORS, b, 3)
            .permute(1, 0, 2)
        )
        spectral = torch.softmax(self.optic_spectral_logits, -1)[
            self.receptor_type.long()
        ]
        mix = (rgb * spectral[None]).sum(-1)
        oc = (
            torch.sigmoid(
                self.optic_bias[self.receptor_type.long()][None]
                + F.softplus(self.optic_gain_raw)[self.receptor_type.long()][None]
                * (2 * mix - 1)
            )
            * self.optic_supported[None]
        )
        standardized = ((body - self.body_mean) / self.body_scale).clamp(-8, 8)
        bc = torch.sigmoid(
            masked_body_projection(
                self.body_weight,
                standardized,
                self.body_support_crow,
                self.body_support_row,
                self.body_support_column,
            )
            + self.body_bias[:, None]
        )
        cc = 0.15 * (
            torch.tanh(self.context_weight @ context.T + self.context_bias[:, None])
            - torch.tanh(self.context_bias[:, None])
        )
        drive = optic.new_zeros((N, b))
        drive.index_copy_(0, self.receptor_rows.long(), oc.T)
        drive.index_copy_(0, self.body_rows.long(), bc)
        drive.index_copy_(0, self.context_rows.long(), cc)
        return drive

    def neutral_current(self, batch):
        return self.afferent_current(
            torch.full((batch, SITES, 3), 0.5, device=self.device),
            self.body_mean[None].expand(batch, -1),
            torch.zeros((batch, CONTEXT), device=self.device),
        )

    def step_from_current(self, current, state, dt=DT, neutral=None):
        r0, g, tau, k, ta, tr, use = self.effective()
        h = torch.minimum(r0, 1 - r0)
        r = state.rates
        m = [state.mod_da, state.mod_oa, state.mod_ht]
        delta = dt / 2
        tm = 0.1 + 4.9 * torch.sigmoid(self.dynamics_modulation_tau_raw)
        neutral = self.neutral_current(current.shape[1]) if neutral is None else neutral
        for _ in range(2):
            x = r - r0
            fast = fixed_sparse_mm(
                self.fast_graph, self.fast_graph_transpose, x * state.release
            )
            mn = []
            for graph, transpose, old, t in zip(
                (self.da_graph, self.oa_graph, self.ht_graph),
                (
                    self.da_graph_transpose,
                    self.oa_graph_transpose,
                    self.ht_graph_transpose,
                ),
                m,
                tm,
            ):
                mn.append(
                    old
                    + (-torch.expm1(current.new_tensor(-delta) / t))
                    * (fixed_sparse_mm(graph, transpose, x) - old)
                )
            type_index = self.neuron_type.long()
            mg = sum(
                0.5
                * torch.tanh(self.dynamics_mod_gain_raw[:, i])[type_index, None]
                * mn[i]
                / h
                for i in range(3)
            )
            ma = sum(
                0.5
                * torch.tanh(self.dynamics_mod_adaptation_raw[:, i])[type_index, None]
                * mn[i]
                / h
                for i in range(3)
            )
            ge = g * torch.exp(0.5 * torch.tanh(mg))
            ke = k * (1 + 0.5 * torch.tanh(ma))
            u = current - neutral + ge * fast - ke * state.adaptation
            target = r0 + state.support * h * torch.tanh(u / h)
            r = r + (-torch.expm1(current.new_tensor(-delta) / tau)) * (target - r)
            m = mn
        x = r - r0
        a = state.adaptation + (-torch.expm1(current.new_tensor(-dt) / ta)) * (
            x - state.adaptation
        )
        s = torch.clamp(
            state.support + dt * (0.024 * (1 - state.support) - 0.003 * x.abs() / h),
            0.65,
            1,
        )
        q = torch.clamp(
            state.release
            + dt * ((1 - state.release) / tr - use * x.abs() / h * state.release),
            0.2,
            1,
        )
        return CNSState(r, a, s, q, *m)

    def outputs(self, state):
        r0 = self.effective()[0]
        dev = state.rates - r0
        latent = torch.tanh(
            self.readout_output(
                F.linear((dev * self.readout_mask[:, None]).T, self.readout_projection)
            )
        )
        motor_rates = state.rates[self.motor_rows.long()]
        centered = (
            motor_rates - self.motor_reference_rate[:, None]
        ) / self.motor_rate_scale[:, None]
        pre = (self.motor_weight * self.motor_mask) @ centered + self.motor_intercept[
            :, None
        ]
        motor = torch.cat((torch.tanh(pre[:84]), torch.sigmoid(pre[84:])), dim=0).T
        return latent, motor

    def forward(self, optic, body, context, state=None, dt=DT):
        if (
            optic.ndim != 3
            or optic.shape[1:] != (SITES, 3)
            or body.shape != (optic.shape[0], BODY)
            or context.shape != (optic.shape[0], CONTEXT)
        ):
            raise ValueError("expected optic[B,1771,3], body[B,807], context[B,12]")
        state = self.initial_state(optic.shape[0]) if state is None else state
        state = self.step_from_current(
            self.afferent_current(optic, body, context), state, dt
        )
        latent, motor = self.outputs(state)
        return latent, motor, state


def initialized_arrays(static, seed=20260907):
    """Create explicit untrained V4 interfaces around supplied immutable anatomy."""
    from chreatures.cns_adapter_contract import ARRAY_SPECS, neutral_afferent_drive

    rng = np.random.default_rng(seed)
    a = {
        k: np.ascontiguousarray(static[k], dtype=d)
        for k, d, _ in ARRAY_SPECS
        if k in static
    }
    for k, d, s in ARRAY_SPECS:
        if k in a or k == "afferent.neutral_drive":
            continue
        a[k] = np.zeros(s, dtype=d)
    a["body.scale"].fill(1)
    a["body.bias"].fill(math.log(0.05 / 0.95))
    a["body.weight"][:] = rng.normal(0, 0.02, a["body.weight"].shape)
    a["context.weight"][:] = rng.normal(0, 0.01, a["context.weight"].shape)
    for k, v in (
        ("baseline_raw", _inv_sigmoid(0.375)),
        ("recurrent_gain_raw", _inv_sigmoid(0.4 / 1.5)),
        ("tau_raw", _inv_sigmoid(0.06 / 0.23)),
        ("adaptation_gain_raw", _inv_sigmoid(0.3)),
        ("adaptation_tau_raw", _inv_sigmoid(1.25 / 4.75)),
        ("release_tau_raw", _inv_sigmoid((0.5 - 0.05) / 1.95)),
        ("release_use_raw", _inv_sigmoid((0.1 - 0.01) / 0.49)),
    ):
        a["dynamics." + k].fill(v)
    a["dynamics.mod_gain_raw"].fill(0.2)
    a["dynamics.mod_adaptation_raw"].fill(0.2)
    a["dynamics.modulation_tau_raw"][:] = [
        _inv_sigmoid((x - 0.1) / 4.9) for x in (0.5, 1, 2)
    ]
    a["readout.projection.weight"][:] = rng.normal(
        0, 0.002, a["readout.projection.weight"].shape
    )
    a["readout.output.weight"][:] = rng.normal(
        0, 0.02, a["readout.output.weight"].shape
    )
    a["motor.reference_rate"][:] = 0.05 + 0.4 / (
        1
        + np.exp(
            -a["dynamics.baseline_raw"][a["atlas.neuron_type"][a["atlas.motor_rows"]]]
        )
    )
    a["motor.rate_scale"].fill(0.05)
    a["motor.weight"][:] = (
        rng.normal(0, 0.002, a["motor.weight"].shape) * a["atlas.motor_mask"]
    )
    a["motor.intercept"][:84] = 0
    a["motor.intercept"][84:] = math.log(0.02 / 0.98)
    a["afferent.neutral_drive"] = neutral_afferent_drive(a)
    return a


def export_arrays(model, static):
    """Return a complete service tensor mapping from a trained V4 module."""
    result = {k: np.ascontiguousarray(v) for k, v in static.items()}
    mapping = {
        "optic.spectral_logits": model.optic_spectral_logits,
        "optic.gain_raw": model.optic_gain_raw,
        "optic.bias": model.optic_bias,
        "body.mean": model.body_mean,
        "body.scale": model.body_scale,
        "body.weight": model.body_weight,
        "body.bias": model.body_bias,
        "context.weight": model.context_weight,
        "context.bias": model.context_bias,
        "readout.projection.weight": model.readout_projection,
        "readout.output.weight": model.readout_output.weight,
        "readout.output.bias": model.readout_output.bias,
        "motor.reference_rate": model.motor_reference_rate,
        "motor.rate_scale": model.motor_rate_scale,
        "motor.weight": model.motor_weight,
        "motor.intercept": model.motor_intercept,
    }
    for key in (
        "baseline_raw",
        "recurrent_gain_raw",
        "tau_raw",
        "adaptation_gain_raw",
        "adaptation_tau_raw",
        "release_tau_raw",
        "release_use_raw",
        "mod_gain_raw",
        "mod_adaptation_raw",
        "modulation_tau_raw",
    ):
        mapping["dynamics." + key] = getattr(model, "dynamics_" + key)
    for key, value in mapping.items():
        result[key] = np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
    from chreatures.cns_adapter_contract import neutral_afferent_drive

    result["afferent.neutral_drive"] = neutral_afferent_drive(result)
    return result
