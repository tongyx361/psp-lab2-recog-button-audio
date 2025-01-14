import argparse
from pathlib import Path

import librosa
import numpy as np

# Constants for DTMF frequencies and corresponding keys
DTMF_FREQS = {
    (697, 1209): "1",
    (697, 1336): "2",
    (697, 1477): "3",
    (770, 1209): "4",
    (770, 1336): "5",
    (770, 1477): "6",
    (852, 1209): "7",
    (852, 1336): "8",
    (852, 1477): "9",
    (941, 1209): "*",
    (941, 1336): "0",
    (941, 1477): "#",
}


def calculate_rms_energy(frame):
    """Calculate the Root Mean Square energy of an audio frame."""
    return np.sqrt(np.mean(frame**2))


def find_peak_frequencies(frequencies, magnitudes):
    """Find the two frequencies with highest magnitudes."""
    peak_indices = np.argpartition(magnitudes, -2)[-2:]
    peak_freqs = frequencies[peak_indices]
    return sorted(peak_freqs)


def process_audio_frame(
    frame, sample_rate, fft_size=750, energy_threshold=0.1, freq_tolerance=20
):
    """Process a single audio frame to detect DTMF tones."""
    if calculate_rms_energy(frame) < energy_threshold:
        return "-1"

    # Perform Short-Time Fourier Transform
    spectrum = np.abs(librosa.stft(frame, n_fft=fft_size))
    magnitudes = np.mean(spectrum, axis=1)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=fft_size)

    # Find peak frequencies
    peak_freqs = find_peak_frequencies(frequencies, magnitudes)

    # Match frequencies to DTMF keys
    for freq_pair, key in DTMF_FREQS.items():
        if np.allclose(peak_freqs, sorted(freq_pair), atol=freq_tolerance):
            return key

    return "-1"


def key_tone_recognition(
    audio_array,
    frame_size=64,  # Number of frames per second
    fft_size=750,
    energy_threshold=0.1,
    freq_tolerance=20,
):
    """
    Recognize DTMF tones in audio and return the sequence of detected keys.

    Args:
        audio_array: Tuple of (audio_samples, sample_rate)
        frame_size: Number of frames per second (default: 64)
        fft_size: Size of FFT window (default: 750)
        energy_threshold: Threshold for audio energy detection (default: 0.1)
        freq_tolerance: Hz tolerance for frequency matching (default: 20)

    Returns:
        str: Space-separated sequence of detected keys
    """
    samples, sample_rate = audio_array
    frames_per_second = sample_rate / frame_size
    frames = np.array_split(samples, len(samples) / frames_per_second)

    detected_keys = [
        process_audio_frame(
            frame,
            sample_rate,
            fft_size=fft_size,
            energy_threshold=energy_threshold,
            freq_tolerance=freq_tolerance,
        )
        for frame in frames
    ]

    return " ".join(detected_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_file",
        type=str,
        help="Test file name",
        required=True,
        default="./test.wav",
    )
    parser.add_argument(
        "--ref_file", type=str, help="Reference file path", default=None
    )
    args = parser.parse_args()
    input_audio_array = librosa.load(
        args.audio_file, sr=48000, dtype=np.float32
    )  # audio file is numpy float array
    result = key_tone_recognition(input_audio_array)

    # Print the recognition result
    print(result)

    # Compare with reference file
    if args.ref_file is not None and Path(args.ref_file).exists():
        with open(args.ref_file, "r") as f:
            reference = f.read().strip()
            print("Reference sequence:", reference)
            print("Matches reference:", result == reference)
