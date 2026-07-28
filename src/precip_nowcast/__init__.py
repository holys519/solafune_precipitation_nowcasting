"""Research-grade precipitation nowcasting components.

The package intentionally contains no experiment-specific paths.  Experiments are
fully described by validated YAML configuration and use the public factories here.
"""

from .config import ExperimentConfig, load_config
from .losses import CompositePrecipitationLoss
from .metrics import MetricAccumulator
from .model import PrecipitationOutput, TotalShapeNowcaster

__all__ = [
    "CompositePrecipitationLoss",
    "ExperimentConfig",
    "MetricAccumulator",
    "PrecipitationOutput",
    "TotalShapeNowcaster",
    "load_config",
]
