import argparse

import librosa
import numpy as np


def key_tone_recognition(audio_array):
    """
    请大家实现这一部分代码
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", type=str, help="test file name", required=True)
    args = parser.parse_args()
    input_audio_array = librosa.load(
        args.audio_file, sr=48000, dtype=np.float32
    )  # audio file is numpy float array
    key_tone_recognition(input_audio_array)
