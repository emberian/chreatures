import copy

from chreatures.neural_ports import encoding_sha256, load_port_spec


def test_physical_encoding_identity_excludes_graph_and_routing() -> None:
    spec = load_port_spec()
    identity = encoding_sha256(spec)

    derived = copy.deepcopy(spec)
    derived["graph"]["dataset_hash"] = "0" * 64
    derived["routing"]["input_gain"] = 0.25
    derived["readouts"]["count"] = 1
    assert encoding_sha256(derived) == identity

    changed = copy.deepcopy(spec)
    changed["physical_inputs"]["scaling"]["maximum_contacts"] += 1
    assert encoding_sha256(changed) != identity
