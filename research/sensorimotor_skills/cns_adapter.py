"""Current CNS model entrypoint; implementation moved to Anatomical CNS V3."""
from research.anatomical_cns.model import (
    AnatomicalCNS,
    CNSState,
    initialized_arrays,
    export_arrays,
)

TrainableCNSAdapter = AnatomicalCNS
__all__ = [
    "AnatomicalCNS",
    "TrainableCNSAdapter",
    "CNSState",
    "initialized_arrays",
    "export_arrays",
]
