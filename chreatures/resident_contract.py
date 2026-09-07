"""Torch-free identities for the one current CNS-only resident lineage."""

CONTROLLER_INPUT_FORMAT = "chreatures-cns-only-controller-v1"
NATIVE_POPULATION_FORMAT = "chreatures-native-cns-only-resident-population-v1"
NATIVE_EXECUTION = "native-cns-only-resident-v1"
NATIVE_POPULATION_VERSION = 1
NATIVE_SNAPSHOT_FORMAT = "chreatures-native-cns-only-resident-snapshot-v1"
NATIVE_SNAPSHOT_VERSION = 1

# Archived Torch-v5 research corpus tools may identify their own source data
# with these values.  The current native resident wrapper never accepts them.
LEGACY_BOOTSTRAP_FORMAT = "chreatures-rich-sensorimotor-bootstrap-v5"
LEGACY_DEVELOPMENT_FORMAT = "chreatures-rich-online-sensorimotor-development-v5"

__all__ = [
    "CONTROLLER_INPUT_FORMAT",
    "LEGACY_BOOTSTRAP_FORMAT",
    "LEGACY_DEVELOPMENT_FORMAT",
    "NATIVE_EXECUTION",
    "NATIVE_POPULATION_FORMAT",
    "NATIVE_POPULATION_VERSION",
    "NATIVE_SNAPSHOT_FORMAT",
    "NATIVE_SNAPSHOT_VERSION",
]
