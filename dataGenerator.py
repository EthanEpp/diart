import os
import random
import json
from pathlib import Path
from pydub import AudioSegment

# ───── CONFIGURATION ──────────────────────────────────────────────────────────
UTTERANCE_ROOT      = Path("/Users/SAI/Documents/Code/diart/biometricsCalibration_04_01_trimmed/64inches")
SILENCE_ROOT        = Path("/Users/SAI/Documents/Code/diart/biometricsCalibration_04_01_trimmed/silence")
OUTPUT_ROOT         = Path("/Users/SAI/Documents/Code/diart/dataLabels")
NUM_SAMPLES         = 50
SEED                = 42
SILENCE_TARGET_DBFS = -65.0
# ───────────────────────────────────────────────────────────────────────────────

TEST_CASES = ["with_silence_between", "no_silence_between"]

# Create structure:
# OUTPUT_ROOT/
#   with_silence_between/
#     wav/, json/, rttm/
#   no_silence_between/
#     wav/, json/, rttm/
for case in TEST_CASES:
    for sub in ("wav", "json", "rttm"):
        (OUTPUT_ROOT / case / sub).mkdir(parents=True, exist_ok=True)

def load_utterances(root_path: Path):
    utterances = {}
    for sentence_dir in sorted(root_path.iterdir()):
        if not sentence_dir.is_dir(): continue
        stype = sentence_dir.name
        utterances[stype] = {}
        for user_dir in sorted(sentence_dir.iterdir()):
            if not user_dir.is_dir(): continue
            user = user_dir.name
            wavs = list(user_dir.glob(f"{user}*.wav"))
            if wavs:
                utterances[stype][user] = wavs
    return utterances

def load_silences(root_path: Path):
    silences = {}
    for length_dir in sorted(root_path.iterdir()):
        if not length_dir.is_dir(): continue
        try:
            sec = int(length_dir.name.replace('sec', ''))
        except ValueError:
            continue
        files = list(length_dir.glob("*.wav"))
        if files:
            silences[sec] = files
    return silences

def match_target_amplitude(seg: AudioSegment, target_dBFS: float) -> AudioSegment:
    change = target_dBFS - seg.dBFS
    return seg.apply_gain(change)

def pick_silence(silences: dict, sec: int) -> AudioSegment:
    if sec <= 0:
        return AudioSegment.silent(duration=0)
    options = silences.get(sec)
    if not options:
        return AudioSegment.silent(duration=0)
    seg = AudioSegment.from_wav(str(random.choice(options)))
    return match_target_amplitude(seg, SILENCE_TARGET_DBFS)

def generate_samples(utterances: dict, silences: dict, num_samples: int, seed: int):
    random.seed(seed)

    for i in range(num_samples):
        # pick two different users saying the same sentence type
        stype = random.choice(list(utterances.keys()))
        users = list(utterances[stype].keys())
        if len(users) < 2:
            continue
        user1, user2 = random.sample(users, 2)
        utt1 = random.choice(utterances[stype][user1])
        utt2 = random.choice(utterances[stype][user2])

        # decide silences
        begin_sec  = random.choice([4])
        middle_sec = random.choice([0, 1, 2, 3, 4])
        after_sec  = random.choice([0, 1])

        # build combined audio
        seg_begin = pick_silence(silences, begin_sec)
        seg1      = AudioSegment.from_wav(str(utt1))
        seg_mid   = pick_silence(silences, middle_sec)
        seg2      = AudioSegment.from_wav(str(utt2))
        seg_after = pick_silence(silences, after_sec)
        combined  = seg_begin + seg1 + seg_mid + seg2 + seg_after

        # select case folder
        case = "with_silence_between" if middle_sec > 0 else "no_silence_between"
        base = f"sample_{i:04d}_{stype}_{user1}_{user2}_b{begin_sec}_m{middle_sec}_a{after_sec}"

        # output paths under OUTPUT_ROOT/case/{wav,json,rttm}
        wav_out  = OUTPUT_ROOT / case / "wav"  / f"{base}.wav"
        json_out = OUTPUT_ROOT / case / "json" / f"{base}.json"
        rttm_out = OUTPUT_ROOT / case / "rttm" / f"{base}.rttm"

        # export wav
        combined.export(str(wav_out), format="wav")

        # compute timings
        start1 = begin_sec
        end1   = start1 + seg1.duration_seconds
        start2 = end1 + middle_sec
        end2   = start2 + seg2.duration_seconds

        # write json
        labels = {
           "file": wav_out.name,
           "segments": [
              {"speaker": user1, "start": round(start1, 3), "end": round(end1, 3)},
              {"speaker": user2, "start": round(start2, 3), "end": round(end2, 3)}
           ]
        }
        with open(json_out, 'w') as jf:
            json.dump(labels, jf, indent=2)

        # write rttm
        dur1 = end1 - start1
        dur2 = end2 - start2
        with open(rttm_out, 'w') as rf:
            rf.write(f"SPEAKER {base} 1 {start1:.3f} {dur1:.3f} <NA> <NA> speaker0 <NA> <NA>\n")
            rf.write(f"SPEAKER {base} 1 {start2:.3f} {dur2:.3f} <NA> <NA> speaker1 <NA> <NA>\n")

    print(
        f"Generated {num_samples} samples:\n"
        f" - OUTPUT_ROOT/<case>/wav\n"
        f" - OUTPUT_ROOT/<case>/json\n"
        f" - OUTPUT_ROOT/<case>/rttm"
    )

if __name__ == "__main__":
    utterances = load_utterances(UTTERANCE_ROOT)
    silences   = load_silences(SILENCE_ROOT)
    generate_samples(utterances, silences, NUM_SAMPLES, SEED)
