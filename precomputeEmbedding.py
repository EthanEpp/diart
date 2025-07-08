import os
import json
import numpy as np
import torch
import soundfile as sf

from diart.blocks.embedding import SpeakerEmbedding

# 1) instantiate exactly the same embedding model DIART will use:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spk_emb = SpeakerEmbedding.from_pretrained("pyannote/embedding", device=device)

# 2) map each speaker to one or more WAV files on disk:
#    e.g. {"alice": ["alice1.wav","alice2.wav"], "bob":["bob.wav"], …}
speaker_files = {
    "matt": ["biometricsCalibration_04_01_trimmed/64inches/medNum/Matt/mattMedNum_03_241_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/Matt/mattMedNum_03_242_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/Matt/mattMedNum_03_243_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/Matt/mattMedNum_03_244_raw.wav"],
    "harrison":   ["biometricsCalibration_04_01_trimmed/64inches/medNum/harrison/harrisonMedNum1_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/harrison/harrisonMedNum2_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/harrison/harrisonMedNum3_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/harrison/harrisonMedNum4_raw.wav"],
    "epp": ["biometricsCalibration_04_01_trimmed/64inches/medNum/epp/ethanMedNum_03_241_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/epp/ethanMedNum_03_242_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/epp/ethanMedNum_03_243_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/epp/ethanMedNum_03_244_raw.wav"],
    "bruce": ["biometricsCalibration_04_01_trimmed/64inches/medNum/bruce/bruceMedNum1_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/bruce/bruceMedNum2_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/bruce/bruceMedNum3_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/bruce/bruceMedNum4_raw.wav"],
    "ben": ["biometricsCalibration_04_01_trimmed/64inches/medNum/ben/benMedNum_04_011_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/ben/benMedNum_04_012_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/ben/benMedNum_04_013_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/ben/benMedNum_04_014_raw.wav"],
    "ali": ["biometricsCalibration_04_01_trimmed/64inches/medNum/ali/aliMedNum_03_241_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/ali/aliMedNum_03_242_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/ali/aliMedNum_03_243_raw.wav", "biometricsCalibration_04_01_trimmed/64inches/medNum/ali/aliMedNum_03_244_raw.wav"],
    "alfonso": ["biometricsCalibration_04_01_trimmed/64inches/medNum/alfonso/alfonsoMedNum_03_251_raw.wav","biometricsCalibration_04_01_trimmed/64inches/medNum/alfonso/alfonsoMedNum_03_252_raw.wav","biometricsCalibration_04_01_trimmed/64inches/medNum/alfonso/alfonsoMedNum_03_253_raw.wav","biometricsCalibration_04_01_trimmed/64inches/medNum/alfonso/alfonsoMedNum_03_254_raw.wav"]
}

centroids = []
speaker_ids = []

for spk, files in speaker_files.items():
    embs = []
    for fn in files:
        # load at the correct sample rate (16 kHz by default)
        waveform, sr = sf.read(fn, always_2d=True)  # shape (samples, channels)
        if sr != 16_000:
            # resample if needed; e.g. using librosa or torchaudio here
            raise RuntimeError(f"{fn} is {sr} Hz, DIART expects 16 kHz")
        # compute one vector per file
        emb = spk_emb(waveform)                      # torch.Tensor (dim,)
        embs.append(emb.numpy())
    # average across that speaker’s files to get a single centroid
    centroid = np.mean(embs, axis=0)                # shape (embedding_dim,)
    centroids.append(centroid)
    speaker_ids.append(spk)

centroids = np.stack(centroids)                     # shape (n_known, embedding_dim)

# 3) save into your desired folder
output_dir = "/Users/SAI/Documents/Code/diart/embeddings"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "initial_centroids.npy"), centroids)
with open(os.path.join(output_dir, "speaker_ids.json"), "w") as f:
    json.dump(speaker_ids, f)

print(f"Saved {len(speaker_ids)} centroids to {output_dir}")