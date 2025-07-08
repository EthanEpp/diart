# import argparse
# from pathlib import Path
# import numpy as np
# import json

# import optuna
# import torch
# from optuna.samplers import TPESampler

# from diart import argdoc
# from diart import models as m
# from diart import utils
# from diart.blocks.base import HyperParameter
# from diart.optim import Optimizer


# def run():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "root",
#         type=str,
#         help="Directory with audio files CONVERSATION.(wav|flac|m4a|...)",
#     )
#     parser.add_argument(
#         "--reference",
#         required=True,
#         type=str,
#         help="Directory with RTTM files CONVERSATION.rttm. Names must match audio files",
#     )
#     parser.add_argument(
#         "--pipeline",
#         default="SpeakerDiarization",
#         type=str,
#         help="Class of the pipeline to optimize. Defaults to 'SpeakerDiarization'",
#     )
#     parser.add_argument(
#         "--segmentation",
#         default="pyannote/segmentation",
#         type=str,
#         help=f"{argdoc.SEGMENTATION}. Defaults to pyannote/segmentation",
#     )
#     parser.add_argument(
#         "--embedding",
#         default="pyannote/embedding",
#         type=str,
#         help=f"{argdoc.EMBEDDING}. Defaults to pyannote/embedding",
#     )
#     parser.add_argument(
#         "--duration",
#         type=float,
#         default=5,
#         help=f"{argdoc.DURATION}. Defaults to training segmentation duration",
#     )
#     parser.add_argument(
#         "--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5"
#     )
#     parser.add_argument(
#         "--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5"
#     )
#     parser.add_argument(
#         "--tau-active", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5"
#     )
#     parser.add_argument(
#         "--rho-update", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3"
#     )
#     parser.add_argument(
#         "--delta-new", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1"
#     )
#     parser.add_argument(
#         "--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3"
#     )
#     parser.add_argument(
#         "--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10"
#     )
#     parser.add_argument(
#         "--max-speakers",
#         default=20,
#         type=int,
#         help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20",
#     )
#     parser.add_argument(
#         "--batch-size",
#         default=32,
#         type=int,
#         help=f"{argdoc.BATCH_SIZE}. Defaults to 32",
#     )
#     parser.add_argument(
#         "--cpu",
#         dest="cpu",
#         action="store_true",
#         help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise",
#     )
#     parser.add_argument(
#         "--mps",
#         dest="mps",
#         action="store_true",
#         help="Use Apple MPS backend if available",
#     )

#     parser.add_argument(
#         "--hparams",
#         nargs="+",
#         default=("tau_active", "rho_update", "delta_new"),
#         help="Hyper-parameters to optimize. Must match names in `PipelineConfig`. Defaults to tau_active, rho_update and delta_new",
#     )
#     parser.add_argument(
#         "--num-iter", default=100, type=int, help="Number of optimization trials"
#     )
#     parser.add_argument(
#         "--storage",
#         type=str,
#         help="Optuna storage string. If provided, continue a previous study instead of creating one. The database name must match the study name",
#     )
#     parser.add_argument("--output", type=str, help="Working directory")
#     parser.add_argument(
#         "--hf-token",
#         default="true",
#         type=str,
#         help=f"{argdoc.HF_TOKEN}. Defaults to 'true' (required by pyannote)",
#     )
#     parser.add_argument(
#         "--initial-embeddings",
#         type=str,
#         required=True,
#         help="Path to your initial_centroids.npy",
#     )
#     parser.add_argument(
#         "--initial-speaker-names",
#         type=str,
#         required=True,
#         help="Path to your speaker_ids.json",
#     )

#     parser.add_argument(
#         "--normalize-embedding-weights",
#         action="store_true",
#         help=f"{argdoc.NORMALIZE_EMBEDDING_WEIGHTS}. Defaults to False",
#     )
#     args = parser.parse_args()
#     centroids   = np.load(args.initial_embeddings)
#     speaker_ids = json.load(open(args.initial_speaker_names))
    
#     args.initial_embeddings    = centroids
#     args.initial_speaker_names = speaker_ids
#     # Resolve device
#     if args.cpu:
#         args.device = torch.device("cpu")

#     elif args.mps and torch.backends.mps.is_available():
#         args.device = torch.device("mps")
#     else:
#         # fallback to CUDA if available, otherwise CPU
#         args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     # Resolve models
#     hf_token = utils.parse_hf_token_arg(args.hf_token)
#     args.segmentation = m.SegmentationModel.from_pretrained(args.segmentation, hf_token)
#     args.embedding = m.EmbeddingModel.from_pretrained(args.embedding, hf_token)

#     # Retrieve pipeline class
#     pipeline_class = utils.get_pipeline_class(args.pipeline)

#     # Create the base configuration for each trial
#     base_config = pipeline_class.get_config_class()(**vars(args))

#     # Create hyper-parameters to optimize
#     possible_hparams = pipeline_class.hyper_parameters()
#     hparams = [HyperParameter.from_name(name) for name in args.hparams]
#     hparams = [hp for hp in hparams if hp in possible_hparams]
#     if not hparams:
#         print(
#             f"No hyper-parameters to optimize. "
#             f"Make sure to select one of: {', '.join([hp.name for hp in possible_hparams])}"
#         )
#         exit(1)

#     # Use a custom storage if given
#     if args.output is not None:
#         msg = "Both `output` and `storage` were set, but only one was expected"
#         assert args.storage is None, msg
#         args.output = Path(args.output).expanduser()
#         args.output.mkdir(parents=True, exist_ok=True)
#         study_or_path = args.output
#     elif args.storage is not None:
#         db_name = Path(args.storage).stem
#         study_or_path = optuna.load_study(db_name, args.storage, TPESampler())
#     else:
#         msg = "Please provide either `output` or `storage`"
#         raise ValueError(msg)

#     # Run optimization
#     Optimizer(
#         pipeline_class=pipeline_class,
#         speech_path=args.root,
#         reference_path=args.reference,
#         study_or_path=study_or_path,
#         batch_size=args.batch_size,
#         hparams=hparams,
#         base_config=base_config,
#     )(num_iter=args.num_iter, show_progress=True)


# if __name__ == "__main__":
#     run()

#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler

from diart import models as m, utils
from diart.blocks.base import HyperParameter
from diart.optim import Optimizer

# ───── USER-CONFIGURABLE PARAMETERS ────────────────────────────────────────────

# 1) Data paths
ROOT_DIR            = "/Users/SAI/Documents/Code/diart/dataLabels/with_silence_between/wav"
REFERENCE_DIR       = "/Users/SAI/Documents/Code/diart/dataLabels/with_silence_between/rttm"

# 2) Precomputed embeddings & names
INITIAL_EMBEDDINGS       = "/Users/SAI/Documents/Code/diart/embeddings/initial_centroids.npy"
INITIAL_SPEAKER_NAMES    = "/Users/SAI/Documents/Code/diart/embeddings/speaker_ids.json"

# 3) Pipeline selection
PIPELINE_NAME       = "SpeakerDiarization"
SEGMENTATION_NAME   = "pyannote/segmentation"
EMBEDDING_NAME      = "pyannote/embedding"
HF_TOKEN            = "true"

# 4) Pipeline hyperparams (defaults; will be overridden by Optuna)
DURATION            = 5.0
STEP                = 0.5
LATENCY             = 0.5
TAU_ACTIVE          = 0.5
RHO_UPDATE          = 0.3
DELTA_NEW           = 1.0
GAMMA               = 3.0
BETA                = 10.0
MAX_SPEAKERS        = 20

# 5) Optimization settings
# HYPERPARAM_NAMES    = ["tau_active", "rho_update", "delta_new"]
HYPERPARAM_NAMES    = ["rho_update", "delta_new"]
NUM_ITER            = 40
BATCH_SIZE          = 32

# 6) Where to write your Optuna study (SQLite + optional plots)
OUTPUT_DIR          = "/Users/SAI/Documents/Code/diart/dataLabelTuneSilence"
STORAGE_URL         = None   # e.g. "sqlite:////full/path/to/db.sqlite" or None

# 7) Device flags
USE_CPU             = False
USE_MPS             = False

# ───── END CONFIG ───────────────────────────────────────────────────────────────

def main():
    # — Load precomputed embeddings & speaker names
    centroids   = np.load(INITIAL_EMBEDDINGS)
    speaker_ids = json.load(open(INITIAL_SPEAKER_NAMES))
    assert centroids.ndim == 2, f"expected (n_known, emb_dim), got {centroids.shape}"
    assert len(speaker_ids) == centroids.shape[0]

    # — Select torch device
    if USE_CPU:
        device = torch.device("cpu")
    elif USE_MPS and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[tune] Using device: {device}")

    # — Load segmentation & embedding models
    hf_token = utils.parse_hf_token_arg(HF_TOKEN)
    segmentation = m.SegmentationModel.from_pretrained(SEGMENTATION_NAME, hf_token).to(device)
    embedding    = m.EmbeddingModel.from_pretrained(EMBEDDING_NAME, hf_token).to(device)

    # — Build base config kwargs
    config_kwargs = dict(
        segmentation=segmentation,
        embedding   =embedding,
        duration    =DURATION,
        step        =STEP,
        latency     =LATENCY,
        tau_active  =TAU_ACTIVE,
        rho_update  =RHO_UPDATE,
        delta_new   =DELTA_NEW,
        gamma       =GAMMA,
        beta        =BETA,
        max_speakers=MAX_SPEAKERS,
        device      =device,
        # seed your precomputed embeddings & names here:
        initial_embeddings    =centroids,
        initial_speaker_names =speaker_ids,
    )

    # — Retrieve pipeline & base config
    pipeline_class = utils.get_pipeline_class(PIPELINE_NAME)
    base_config    = pipeline_class.get_config_class()(**config_kwargs)

    # — Determine which hyper-parameters to optimize
    possible = pipeline_class.hyper_parameters()
    hparams  = [HyperParameter.from_name(n) for n in HYPERPARAM_NAMES]
    hparams  = [hp for hp in hparams if hp in possible]
    if not hparams:
        names = ", ".join(h.name for h in possible)
        raise ValueError(f"No valid hparams in {HYPERPARAM_NAMES}; choose from: {names}")

    # — Figure out Optuna storage vs. output dir
    if OUTPUT_DIR:
        out = Path(OUTPUT_DIR).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        study_or_path = out

    elif STORAGE_URL:
        # derive study name from the DB filename
        study_name = Path(STORAGE_URL).stem
        study_or_path = optuna.load_study(
            study_name=study_name,
            storage=STORAGE_URL,
            sampler=TPESampler(),
        )

    else:
        raise ValueError("Set either OUTPUT_DIR or STORAGE_URL")
    
    # — Launch Optuna
    tuner = Optimizer(
        pipeline_class=pipeline_class,
        speech_path=ROOT_DIR,
        reference_path=REFERENCE_DIR,
        study_or_path=study_or_path,
        batch_size=BATCH_SIZE,
        hparams=hparams,
        base_config=base_config,
    )
    tuner(num_iter=NUM_ITER, show_progress=True)


if __name__ == "__main__":
    main()
