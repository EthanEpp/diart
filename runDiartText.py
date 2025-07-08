import os
import torch
import warnings
import matplotlib.pyplot as plt
import wave
import numpy as np
import json
import statistics
import matplotlib.patches as mpatches
from pathlib import Path

from diart import models as m
from diart import sources as src
from diart import utils
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter

# ───── CONFIGURATION ──────────────────────────────────────────────────────────
INPUT_ROOT        = Path("/Users/SAI/Documents/Code/diart/dataLabels")
OUTPUT_ROOT       = Path("/Users/SAI/Documents/Code/diart/dataLabelsResults")
LABEL_ROOT        = INPUT_ROOT

PIPELINE_NAME     = "SpeakerDiarization"
SEGMENTATION_NAME = "pyannote/segmentation"
SEGMENTATION_NAME_ONNX = "/Users/SAI/Documents/Code/diart/models/pyanote/segmentation_simplified.onnx"
EMBEDDING_NAME    = "pyannote/embedding"
EMBEDDING_NAME_ONNX = "/Users/SAI/Documents/Code/diart/models/pyanote/embedding_dynamic.onnx"
HF_TOKEN          = "true"

DURATION     = 5.0
STEP         = 0.5
LATENCY      = 0.5
TAU_ACTIVE   = 0.527
RHO_UPDATE   = 0.1
DELTA_NEW    = 0.8
GAMMA        = 3
BETA         = 10
MAX_SPEAKERS = 10
USE_CPU      = False
speaker_ids_global = json.load(open("/Users/SAI/Documents/Code/diart/embeddings/speaker_ids.json"))
# best_perf=27.1, best_tau_active=0.536, best_rho_update=0.0307, best_delta_new=0.73
# best_perf=27.1, best_tau_active=0.536, best_rho_update=0.0307, best_delta_new=0.731
# best_rho_update=0.261, best_delta_new=1.48
#  best_perf=27, best_rho_update=0.195, best_delta_new=1.15
# best_perf=27, best_rho_update=0.173, best_delta_new=1.34
# best_perf=27, best_rho_update=0.155, best_delta_new=1.34
# warnings.filterwarnings("ignore", message="Mismatch between frames*", category=UserWarning)

# ───── DEVICE & PIPELINE SETUP ─────────────────────────────────────────────────

def setup_pipeline():
    if USE_CPU:
        device = torch.device("cpu")
    else:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print("Using device:", device)

    hf_token    = utils.parse_hf_token_arg(HF_TOKEN)
    # segmentation = m.SegmentationModel.from_pretrained(SEGMENTATION_NAME, hf_token).to(device)
    segmentation = (m.SegmentationModel.from_onnx(SEGMENTATION_NAME_ONNX).to(device))

    # embedding    = m.EmbeddingModel.from_pretrained(EMBEDDING_NAME, hf_token).to(device)
    embedding = (m.EmbeddingModel.from_onnx(EMBEDDING_NAME_ONNX, input_names=["waveform","weights"], output_name="embedding").to(device))
    centroids   = np.load("/Users/SAI/Documents/Code/diart/embeddings/initial_centroids.npy")
    speaker_ids = json.load(open("/Users/SAI/Documents/Code/diart/embeddings/speaker_ids.json"))

    pipeline_class = utils.get_pipeline_class(PIPELINE_NAME)
    config = pipeline_class.get_config_class()(
        segmentation=segmentation,
        embedding=embedding,
        duration=DURATION,
        step=STEP,
        latency=LATENCY,
        tau_active=TAU_ACTIVE,
        rho_update=RHO_UPDATE,
        delta_new=DELTA_NEW,
        gamma=GAMMA,
        beta=BETA,
        max_speakers=MAX_SPEAKERS,
        device=device,
        no_plot=True,
        initial_embeddings=centroids,
        initial_speaker_names=speaker_ids,
    )
    return pipeline_class(config)

# ───── PARSING & EVALUATION ────────────────────────────────────────────────────

def parse_rttm(rttm_path: Path):
    segments = {}
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            label = parts[7]
            segments.setdefault(label, []).append((start, end))
    return segments


def evaluate_diarization(true_segs, pred_segs):
    """
    Compute precision, recall, f1 for each true speaker vs predicted clusters.
    true_segs: list of dicts {speaker, start, end}
    pred_segs: dict cluster->[(start,end),...]
    """
    # total true duration per speaker
    true_durations = {seg['speaker']: 0.0 for seg in true_segs}
    for seg in true_segs:
        true_durations[seg['speaker']] += seg['end'] - seg['start']

    # compute overlap per (true speaker, cluster)
    overlaps = {s: {} for s in true_durations}
    for seg in true_segs:
        s, t0, t1 = seg['speaker'], seg['start'], seg['end']
        for cluster, cl_segs in pred_segs.items():
            ov = 0.0
            for p0, p1 in cl_segs:
                o0 = max(t0, p0)
                o1 = min(t1, p1)
                if o1 > o0:
                    ov += (o1 - o0)
            overlaps[s][cluster] = overlaps[s].get(cluster, 0.0) + ov

    metrics = {}
    for s, total in true_durations.items():
        # best-matching cluster
        best_cluster, ov = max(overlaps[s].items(), key=lambda kv: kv[1])
        # precision & recall
        pred_total = sum(e - s for s, e in pred_segs.get(best_cluster, []))
        prec = ov / pred_total if pred_total > 0 else 0.0
        rec  = ov / total if total > 0 else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        metrics[s] = {
            'cluster': best_cluster,
            'precision': round(prec, 3),
            'recall':    round(rec, 3),
            'f1':        round(f1, 3)
        }
    return metrics, overlaps, true_durations

# ───── PLOTTING ────────────────────────────────────────────────────────────────

def plot_diarization(wav_path: Path, rttm_path: Path, plot_out: Path):
    # 1) load audio
    with wave.open(str(wav_path), 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / sr
        audio = wf.readframes(n_frames)
        samples = np.frombuffer(audio, dtype=np.int16)
        times = np.linspace(0, duration, len(samples))

    # 2) parse predicted RTTM
    pred_segments = parse_rttm(rttm_path)
    pred_speakers = sorted(pred_segments.keys())  # speaker0, speaker1, …

    # 3) load true segments from JSON
    case = wav_path.parent.parent.name   # …/dataLabels/<case>/wav/…
    true_json = LABEL_ROOT / case / 'json' / f"{wav_path.stem}.json"
    with open(true_json) as jf:
        true_data = json.load(jf)
    true_speakers = sorted({seg["speaker"] for seg in true_data["segments"]})

    # 4) make figure
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 4), sharex=True,
        gridspec_kw={'height_ratios': [1, 2]}
    )

    # 5) shade true regions FULL-HEIGHT & build legend patches
    patches = []
    for idx, spk in enumerate(true_speakers):
        color = f"C{idx}"
        patches.append(mpatches.Patch(facecolor=color, alpha=0.2, label=spk))
        for seg in true_data["segments"]:
            if seg["speaker"] == spk:
                ax0.axvspan(seg["start"], seg["end"],
                            facecolor=color, alpha=0.2,
                            label="_nolegend_")

    # 6) plot predicted lines
    for idx, spk in enumerate(pred_speakers):
        for (start, end) in pred_segments.get(spk, []):
            ax0.hlines(idx, start, end, linewidth=6)

    ax0.set_yticks(range(len(pred_speakers)))
    ax0.set_yticklabels(pred_speakers)
    ax0.set_xlim(0, duration)
    ax0.set_ylabel('Predicted speakers')

    # 7) true-speaker legend outside
    ax0.legend(
        handles=patches,
        title="True speakers",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0
    )

    # 8) waveform
    ax1.plot(times, samples)
    ax1.set_xlim(0, duration)
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Time (s)')

    fig.tight_layout()
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, bbox_inches='tight')
    plt.close(fig)


def plot_diarization_ids(
    wav_path: Path,
    rttm_path: Path,
    plot_out: Path,
    speaker_ids: list[str]  # ← pass in your loaded names here
):
    # 1) load audio
    with wave.open(str(wav_path), 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / sr
        audio = wf.readframes(n_frames)
        samples = np.frombuffer(audio, dtype=np.int16)
        times = np.linspace(0, duration, len(samples))

    # 2) parse predicted RTTM
    pred_segments = parse_rttm(rttm_path)
    # labels in the RTTM are strings like "speaker0", "speaker1", …
    pred_labels = sorted(pred_segments.keys())

    # map each "speaker{idx}" → your actual name (or fallback label)
    mapped_labels = []
    for lbl in pred_labels:
        idx = int(lbl.replace("speaker", ""))
        if idx < len(speaker_ids):
            mapped_labels.append(speaker_ids[idx])
        else:
            mapped_labels.append(f"new_speaker_{idx}")

    # 3) load true segments (unchanged)
    case = wav_path.parent.parent.name
    true_json = LABEL_ROOT / case / 'json' / f"{wav_path.stem}.json"
    with open(true_json) as jf:
        true_data = json.load(jf)
    true_speakers = sorted({seg["speaker"] for seg in true_data["segments"]})

    # 4) make figure
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 4), sharex=True,
        gridspec_kw={'height_ratios': [1, 2]}
    )

    # 5) shade true regions (unchanged)
    patches = []
    for idx, spk in enumerate(true_speakers):
        color = f"C{idx}"
        patches.append(mpatches.Patch(facecolor=color, alpha=0.2, label=spk))
        for seg in true_data["segments"]:
            if seg["speaker"] == spk:
                ax0.axvspan(seg["start"], seg["end"],
                            facecolor=color, alpha=0.2,
                            label="_nolegend_")

    # 6) plot predicted lines
    for idx, lbl in enumerate(pred_labels):
        for (start, end) in pred_segments[lbl]:
            ax0.hlines(idx, start, end, linewidth=6)

    # 7) replace y-tick labels with your mapped names
    ax0.set_yticks(range(len(pred_labels)))
    ax0.set_yticklabels(mapped_labels)
    ax0.set_xlim(0, duration)
    ax0.set_ylabel('Predicted speakers')

    # 8) true-speaker legend (unchanged)
    ax0.legend(
        handles=patches,
        title="True speakers",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0
    )

    # 9) waveform (unchanged)
    ax1.plot(times, samples)
    ax1.set_xlim(0, duration)
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Time (s)')

    fig.tight_layout()
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, bbox_inches='tight')
    plt.close(fig)



def diarize_file(pipeline, wav_path: Path, rttm_out: Path, plot_out: Path):
    padding = pipeline.config.get_file_padding(wav_path)
    source = src.FileAudioSource(str(wav_path), pipeline.config.sample_rate, padding, pipeline.config.step)
    pipeline.set_timestamp_shift(-padding[0])

    rttm_out.parent.mkdir(parents=True, exist_ok=True)
    inference = StreamingInference(
        pipeline,
        source,
        batch_size=1,
        do_profile=False,
        do_plot=False,
        show_progress=False,
    )
    inference.attach_observers(RTTMWriter(source.uri, str(rttm_out)))
    inference()
    # After writing RTTM, generate custom plot
    # plot_diarization(wav_path, rttm_out, plot_out)
    plot_diarization_ids(wav_path, rttm_out, plot_out, speaker_ids=speaker_ids_global)


# ───── MAIN LOOP + AGGREGATION ─────────────────────────────────────────────────

def main():
    pipeline = setup_pipeline()
    all_results = []

    for case in ['with_silence_between','no_silence_between']:
        wav_dir  = INPUT_ROOT  / case / 'wav'
        rttm_dir = OUTPUT_ROOT / case / 'rttm'
        plot_dir = OUTPUT_ROOT / case / 'plots'
        rttm_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        if not wav_dir.exists():
            continue

        for wav_path in sorted(wav_dir.glob('*.wav')):
            stem = wav_path.stem
            rttm_path = rttm_dir / f"{stem}.rttm"
            plot_path = plot_dir / f"{stem}.png"

            diarize_file(pipeline, wav_path, rttm_path, plot_path)

            # load ground truth & predictions
            true_data = json.load(open(LABEL_ROOT/case/'json'/f"{stem}.json"))['segments']
            pred_data = parse_rttm(rttm_path)

            # compute metrics + overlap tables
            metrics, overlaps, true_durs = evaluate_diarization(true_data, pred_data)

            # speakers in original JSON order
            spk1 = true_data[0]['speaker']
            spk2 = true_data[1]['speaker']

            # cluster assignments
            c1 = metrics[spk1]['cluster']
            c2 = metrics[spk2]['cluster']
            distinct = (c2 != c1)

            # confusion: how much of spk2's speech fell in c1
            total2 = true_durs[spk2]
            conf_time = overlaps[spk2].get(c1, 0.0)
            conf_rate = conf_time/total2 if total2>0 else 0.0

            all_results.append({
                'file': stem,
                'speaker1': spk1,
                'speaker2': spk2,
                'metrics': metrics,
                'distinct_clusters': distinct,
                'speaker2_confusion_rate': round(conf_rate, 3)
            })

    # write detailed per-file
    with open(OUTPUT_ROOT/'results.json','w') as f:
        json.dump(all_results, f, indent=2)

    # summary
    n = len(all_results)
    s2_f1 = [r['metrics'][r['speaker2']]['f1'] for r in all_results]
    distinct_count = sum(r['distinct_clusters'] for r in all_results)
    conf_rates = [r['speaker2_confusion_rate'] for r in all_results]

    summary = {
        'num_samples': n,
        'speaker2_f1': {
            'mean':   round(statistics.mean(s2_f1),3),
            'median': round(statistics.median(s2_f1),3),
            'stdev':  round(statistics.stdev(s2_f1),3) if n>1 else 0.0
        },
        'distinct_cluster_rate': round(distinct_count/n,3),
        'speaker2_confusion': {
            'mean':   round(statistics.mean(conf_rates),3),
            'median': round(statistics.median(conf_rates),3),
            'stdev':  round(statistics.stdev(conf_rates),3) if n>1 else 0.0
        }
    }
    # write summary
    with open(OUTPUT_ROOT/'summary.json','w') as f:
        json.dump(summary, f, indent=2)

    print("Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
