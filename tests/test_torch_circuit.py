from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

torch = pytest.importorskip("torch")

from chreatures.brain import Genome
from chreatures.torch_circuit import TorchCircuit


def fixture_graph():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [0.0, 0.20, 0.0, 0.0],
                [0.10, 0.0, -0.15, 0.0],
                [0.0, 0.25, 0.0, 0.05],
                [-0.05, 0.0, 0.30, 0.0],
            ],
            dtype=np.float32,
        )
    )
    input_map = np.zeros((4, 16), dtype=np.float32)
    input_map[0, 0] = 0.8
    input_map[1, 3] = 0.8
    input_map[2, 11] = 0.55
    decoder = np.zeros((4, 16), dtype=np.float32)
    decoder[:, :4] = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype=np.float32,
    )
    return SimpleNamespace(
        n=4,
        matrix=matrix,
        input_map=input_map,
        output_cells=np.arange(4),
        baseline=np.zeros(4, dtype=np.float32),
        decoder=decoder,
    )


def numpy_step(state, graph, encoded, reward, dt, genome, learning=True, silenced=False):
    drive = encoded @ graph.input_map.T
    for _ in range(2):
        recurrent = np.zeros_like(state["rates"]) if silenced else (graph.matrix @ state["rates"].T).T
        target = np.maximum(
            0,
            np.tanh(0.005 + drive + genome.neural_gain * recurrent - 0.10 * state["adaptation"]),
        )
        state["rates"] += min(1.0, dt / 2 / genome.neural_tau) * (
            target * state["support"] - state["rates"]
        )
    state["adaptation"] += dt / 5 * (state["rates"] - state["adaptation"])
    state["support"] += dt * (
        genome.support_recovery * (1 - state["support"]) - 0.003 * state["rates"]
    )
    np.clip(state["support"], 0.65, 1, out=state["support"])
    decoded = (state["rates"] - graph.baseline) @ graph.decoder
    decoded = np.clip(decoded, 0, 1)
    state["context"] += dt / 3 * (decoded - state["context"])
    odor_mean = decoded[:, :6].reshape(len(decoded), 2, 3).mean(axis=1)
    state["eligibility"] = state["eligibility"] * np.exp(-dt / 4) + odor_mean * dt / 4
    state["sound_trace"] = state["sound_trace"] * np.exp(-dt / 3) + decoded[:, 11:14] * dt / 3
    prediction = (state["values"] * odor_mean).sum(axis=1)
    positive = np.maximum(reward, 0)
    if learning:
        positive_mask = positive > 0
        state["values"][positive_mask] += (
            genome.learning_rate
            * positive[positive_mask, None]
            * 120
            * state["eligibility"][positive_mask]
            * (1 - state["values"][positive_mask])
        )
        no_reward = ~positive_mask
        state["values"][no_reward] -= (
            dt
            * 0.0008
            * state["eligibility"][no_reward]
            * np.maximum(0, state["values"][no_reward] - 0.12)
        )
        np.clip(state["values"], 0.05, 1, out=state["values"])
    return decoded, prediction


def test_cpu_sparse_fixture_matches_numpy_recurrence_and_plasticity():
    graph = fixture_graph()
    genome = Genome()
    batch = 3
    circuit = TorchCircuit(graph, batch, dtype=torch.float32)
    state = {
        "rates": np.zeros((batch, graph.n), dtype=np.float32),
        "adaptation": np.zeros((batch, graph.n), dtype=np.float32),
        "support": np.ones((batch, graph.n), dtype=np.float32),
        "context": np.zeros((batch, 16), dtype=np.float32),
        "eligibility": np.zeros((batch, 3), dtype=np.float32),
        "values": np.tile(np.array([0.22, 0.22, 0.12], dtype=np.float32), (batch, 1)),
        "sound_trace": np.zeros((batch, 3), dtype=np.float32),
    }
    rng = np.random.default_rng(41)
    for step in range(12):
        encoded = rng.random((batch, 16), dtype=np.float32)
        reward = np.array([0.002 if step == 6 else 0, 0, 0.001 if step == 9 else 0], dtype=np.float32)
        expected_decoded, expected_prediction = numpy_step(
            state, graph, encoded, reward, 0.05, genome
        )
        actual = circuit.step(encoded, 0.05, reward)
        np.testing.assert_allclose(actual.decoded.numpy(), expected_decoded, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(actual.prediction.numpy(), expected_prediction, rtol=2e-5, atol=2e-6)

    for name in ("rates", "adaptation", "support", "context", "eligibility", "values", "sound_trace"):
        np.testing.assert_allclose(getattr(circuit, name).numpy(), state[name], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-5), (torch.float16, 8e-3)])
def test_supported_precisions_are_finite_and_close(dtype, tolerance):
    graph = fixture_graph()
    encoded = np.full((2, 16), 0.3, dtype=np.float32)
    reference = TorchCircuit(graph, 2, dtype=torch.float32)
    candidate = TorchCircuit(graph, 2, dtype=dtype)
    for _ in range(8):
        expected = reference.step(encoded, 0.05).decoded
        actual = candidate.step(encoded, 0.05).decoded
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, rtol=tolerance, atol=tolerance)


def test_silencing_removes_recurrent_edge_effects():
    graph = fixture_graph()
    encoded = np.zeros((1, 16), dtype=np.float32)
    encoded[0, 0] = 1
    recurrent = TorchCircuit(graph, 1)
    silenced = TorchCircuit(graph, 1)
    for _ in range(20):
        recurrent.step(encoded, 0.05)
        silenced.step(encoded, 0.05, silenced=True)
    assert not torch.allclose(recurrent.rates, silenced.rates)
