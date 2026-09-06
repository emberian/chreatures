"""Current body-local organism interface; old formats belong to frozen releases."""

FORMAT = "chreatures-organism-interface-v4"
ACTION_NAMES = (
    "thrust", "yaw", "gaze_pitch", "posture", "grip",
    "signal_low", "signal_mid", "signal_high", "eat", "release", "secrete", "allocate",
)
SIGNED_AXES = (0, 1, 2, 3)
RECTIFIED_AXES = (4, 5, 6, 7, 8, 9, 10, 11)
PHYSIOLOGY_NAMES = (
    "energy", "gut", "fatigue", "speed", "turn", "neural_support",
    "structural_integrity", "development_fraction", "gland_fill", "brood_fill",
    "reproductive_maturity", "exchange_load",
)
RICH_DIM = 4096
CANONICAL_DIM = 351
PHYSIOLOGY_DIM = len(PHYSIOLOGY_NAMES)
BODY_DIM = CANONICAL_DIM + PHYSIOLOGY_DIM
OBSERVATION_DIM = RICH_DIM + BODY_DIM
ACTION_DIM = len(ACTION_NAMES)
PREVIOUS_DIM = ACTION_DIM
NEURAL_DIM = 384
MAX_RESIDENTS = 32
OBSERVATION_ORDER = ("rich_body_v1_4096", "canonical_channels_351", "physiology_12")


def identity():
    """Portable contract, embedded in artifacts rather than inferred from shape."""
    return {
        "format": FORMAT,
        "actions": list(ACTION_NAMES),
        "physiology": list(PHYSIOLOGY_NAMES),
        "observation_order": list(OBSERVATION_ORDER),
        "observation_dim": OBSERVATION_DIM,
        "previous_dim": PREVIOUS_DIM,
        "neural_dim": NEURAL_DIM,
    }
