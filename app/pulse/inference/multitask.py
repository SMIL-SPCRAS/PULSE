"""
File: multitask.py
Authors: Dmitry Ryumin, Timur Abdulkadirov, Elena Ryumina, and Alexey Karpov
Description: Shared multitask inference interface for the PULSE application.
License: MIT License
"""

from dataclasses import dataclass
from math import log

from pulse.audio.preprocessing import PreprocessedAudio


@dataclass(frozen=True, slots=True)
class MultitaskRawOutputs:
    """Raw outputs produced by the shared multitask PULSE model."""

    mosei_logits: list[float]
    resd_logits: list[float]
    bah_logits: list[float]
    personality_scores: list[float]


def probabilities_to_logits(probabilities: list[float]) -> list[float]:
    """Convert probabilities to log-space placeholder logits."""

    return [log(max(probability, 1e-8)) for probability in probabilities]


def run_placeholder_multitask_model(
    preprocessed_audio: PreprocessedAudio,
) -> MultitaskRawOutputs:
    """Run placeholder shared multitask inference before real models are integrated."""

    _ = preprocessed_audio

    return MultitaskRawOutputs(
        # neutral, anger, disgust, fear, happiness, sadness, surprise
        mosei_logits=probabilities_to_logits(
            [0.18, 0.04, 0.01, 0.03, 0.64, 0.07, 0.03],
        ),
        # anger, disgust, fear, happiness, neutral, sadness, enthusiasm
        resd_logits=probabilities_to_logits(
            [0.03, 0.01, 0.02, 0.72, 0.12, 0.04, 0.06],
        ),
        # absence, presence
        bah_logits=probabilities_to_logits(
            [0.37, 0.63],
        ),
        # openness, conscientiousness, extraversion, agreeableness, non-neuroticism
        # Adapter converts non-neuroticism to neuroticism as 1 - score.
        personality_scores=[
            0.74,
            0.81,
            0.76,
            0.69,
            0.72,
        ],
    )
