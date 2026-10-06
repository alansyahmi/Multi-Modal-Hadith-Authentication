"""
transcribe_voicenotes.py
Transcribes the 4 Mufti Tahsin Alam voice note WAV files using Whisper on CUDA.
Saves transcripts to output/transcripts/
"""

import os
import sys
import types
import wave
import numpy as np
import scipy.signal
import torch

# Bypass transformers dependency versions check
dvc = types.ModuleType("transformers.dependency_versions_check")
dvc.dep_version_check = lambda *a, **k: None
sys.modules["transformers.dependency_versions_check"] = dvc

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

OUTPUT_DIR = "output/transcripts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading Whisper model on {DEVICE}...")
processor = AutoProcessor.from_pretrained("openai/whisper-base")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-base",
    torch_dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True
).to(DEVICE)

def load_wav_as_16k_mono(file_path):
    with wave.open(file_path, "rb") as wf:
        orig_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # Convert bytes to numpy array
    if sampwidth == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16,000 Hz if needed
    TARGET_RATE = 16000
    if orig_rate != TARGET_RATE:
        num_samples = int(len(audio) * float(TARGET_RATE) / orig_rate)
        audio = scipy.signal.resample(audio, num_samples)

    return audio

def transcribe_audio(file_path):
    audio = load_wav_as_16k_mono(file_path)
    chunk_len = 30 * 16000  # 30-second chunks
    total_len = len(audio)

    full_transcript = []

    for start in range(0, total_len, chunk_len):
        end = min(start + chunk_len, total_len)
        chunk = audio[start:end]
        if len(chunk) < 1600:  # less than 0.1s
            continue

        input_features = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE).to(TORCH_DTYPE)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_new_tokens=400,
                return_timestamps=False
            )

        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        full_transcript.append(transcription.strip())

    return " ".join(full_transcript)

def main():
    wav_files = [
        "mufti-tahsin-alam-recommendation-1.wav",
        "mufti-tahsin-alam-recommendation-2.wav",
        "mufti-tahsin-alam-recommendation-3.wav",
        "mufti-tahsin-alam-recommendation-4.wav"
    ]

    all_transcripts = {}

    for fname in wav_files:
        if not os.path.exists(fname):
            print(f"Skipping {fname} (not found)")
            continue

        print(f"\nTranscribing {fname}...")
        text = transcribe_audio(fname)
        all_transcripts[fname] = text
        print(f"Result preview: {text[:150]}...")

        # Save individual txt
        out_txt = os.path.join(OUTPUT_DIR, fname.replace(".wav", ".txt"))
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {out_txt}")

    # Save combined json
    out_json = os.path.join(OUTPUT_DIR, "mufti_tahsin_all_transcripts.json")
    import json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_transcripts, f, indent=2, ensure_ascii=False)
    print(f"\nAll transcripts saved to {out_json}")

if __name__ == "__main__":
    main()
