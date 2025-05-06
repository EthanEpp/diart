from diart import models as m
import torch, os
from diart import utils
import torchaudio
import numpy as np
import torch.nn as nn

SEGMENTATION_NAME = "pyannote/segmentation"
EMBEDDING_NAME    = "pyannote/embedding"
HF_TOKEN          = "true"
hf_token    = utils.parse_hf_token_arg(HF_TOKEN)

OUTPUT_DIR = "/Users/SAI/Documents/Code/diart/models/pyanote"
# -- load as you already do:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
segmentation = m.SegmentationModel.from_pretrained(SEGMENTATION_NAME, HF_TOKEN).to(device)
embedding    = m.EmbeddingModel.from_pretrained(EMBEDDING_NAME, HF_TOKEN).to(device)

# -- ensure export directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


segmentation = (
    m.SegmentationModel
     .from_pretrained(SEGMENTATION_NAME, hf_token)
     .to(device)
     .eval()
)

embedding = (
    m.EmbeddingModel
     .from_pretrained(EMBEDDING_NAME, hf_token)
     .to(device)
     .eval()
)

# pick a representative length (in samples)
sample_rate    = 16000
duration_secs  = 2
num_samples    = sample_rate * duration_secs

dummy_waveform = torch.randn(1, 1, 80000, device=device)

torch.onnx.export(
    segmentation.model,                   # the underlying nn.Module
    dummy_waveform,                       # example input
    "segmentation.onnx",                  # output file
    export_params    = True,
    opset_version    = 12,
    input_names      = ["waveform"],
    output_names     = ["segmentation"],
    dynamic_axes     = {
        "waveform":      {2: "samples"},   # allow N to vary
        "segmentation":  {1: "frames"}     # allow #frames to vary
    }
)


segmentation_offline = (
  m.SegmentationModel
   .from_onnx("segmentation.onnx")
   .to(device)
)


# 2. wrap the embedding so it’s a pure nn.Module returning torch.Tensor
class WrappedEmbedding(nn.Module):
    def __init__(self, emb_model):
        super().__init__()
        self.emb = emb_model
    def forward(self, waveform: torch.Tensor, weights: torch.Tensor):
        out = self.emb(waveform, weights)
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out)
        return out

wrapped = WrappedEmbedding(embedding.model).to(device).eval()


# 3. pick a representative “long enough” dummy input
sample_rate   = 16000
duration_secs = 2
num_samples   = sample_rate * duration_secs
num_frames    = num_samples // 100  # or whatever your hop-size is

dummy_wav     = torch.randn(1, 1, 80000, device=device)
dummy_w       = torch.ones(1, num_frames, device=device)

# 4. export with dynamic BATCH axis (axis 0), plus your existing dynamic time/frame axes
torch.onnx.export(
    wrapped,
    (dummy_wav, dummy_w),
    "embedding_dynamic.onnx",
    export_params   = True,
    opset_version   = 12,
    input_names     = ["waveform", "weights"],
    output_names    = ["embedding"],
    dynamic_axes    = {
        "waveform":  {0: "batch", 2: "samples"},
        "weights":   {0: "batch", 1: "frames"},
        "embedding": {0: "batch"}  # allow batch>1 on the output
    }
)

offline_emb = (
    m.EmbeddingModel
      .from_onnx("embedding_dynamic.onnx",
                 input_names=["waveform","weights"],
                 output_name="embedding")
      .to(device)
)

# # run a forward pass on both
# emb_out = embedding_offline(dummy_waveform, dummy_weights)
# print(emb_out.shape)   # should be (1, embedding_dim)


# from onnxsim import simplify
# from diart import models as m
# import torch, os
# from diart import utils
# import torchaudio
# import numpy as np
# import torch.nn as nn
# import onnx

# model = onnx.load("/Users/SAI/Documents/Code/diart/models/pyanote/segmentation.onnx")   # your original dynamic model
# model_simp, check = simplify(
#     model,
#     dynamic_input_shape=False,
#     input_shapes={"waveform":[1,1,80000]}
# )
# assert check, "Simplifier failed to validate"
# onnx.save(model_simp, "/Users/SAI/Documents/Code/diart/models/pyanote/segmentation_simplified.onnx")