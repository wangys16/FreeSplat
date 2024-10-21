from dataclasses import dataclass

# This is in a separate file to avoid circular imports.


@dataclass
class EncoderVisualizerEpipolarCfg:
    num_samples: int
    min_resolution: int
    export_ply: bool
