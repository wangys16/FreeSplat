from typing import Optional

from .encoder import Encoder
from .encoder_freesplat import EncoderFreeSplat, EncoderFreeSplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_epipolar import EncoderVisualizerEpipolar

ENCODERS = {
    "freesplat": (EncoderFreeSplat, EncoderVisualizerEpipolar),
}

EncoderCfg = EncoderFreeSplatCfg


def get_encoder(cfg: EncoderCfg, depth_range=[0.5,15.0]) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg, depth_range=depth_range)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
