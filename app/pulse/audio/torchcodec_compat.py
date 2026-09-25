"""
File: torchcodec_compat.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: TorchCodec compatibility imports for the PULSE Gradio application.
License: MIT License
"""

from typing import TYPE_CHECKING

import torchcodec.decoders as torchcodec_decoders

if TYPE_CHECKING:
    from torchcodec.decoders._audio_decoder import AudioDecoder as AudioDecoder
else:
    AudioDecoder = torchcodec_decoders.AudioDecoder

__all__ = ["AudioDecoder"]
