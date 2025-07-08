from pathlib import Path
from typing import Union, Text, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pyannote.core import Annotation, Segment, SlidingWindowFeature, notebook
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from rx.core import Observer
from typing_extensions import Literal


class WindowClosedException(Exception):
    pass


def _extract_prediction(value: Union[Tuple, Annotation]) -> Annotation:
    if isinstance(value, tuple):
        return value[0]
    if isinstance(value, Annotation):
        return value
    msg = f"Expected tuple or Annotation, but got {type(value)}"
    raise ValueError(msg)


class RTTMWriter(Observer):
    def __init__(self, uri: Text, path: Union[Path, Text], patch_collar: float = 0.05):
        super().__init__()
        self.uri = uri
        self.patch_collar = patch_collar
        self.path = Path(path).expanduser()
        if self.path.exists():
            self.path.unlink()

    def patch(self):
        """Stitch same-speaker turns that are close to each other"""
        if not self.path.exists():
            return
        annotations = list(load_rttm(self.path).values())
        if annotations:
            annotation = annotations[0]
            annotation.uri = self.uri
            with open(self.path, "w") as file:
                annotation.support(self.patch_collar).write_rttm(file)

    def on_next(self, value: Union[Tuple, Annotation]):
        prediction = _extract_prediction(value)
        # Write prediction in RTTM format
        prediction.uri = self.uri
        with open(self.path, "a") as file:
            prediction.write_rttm(file)

    def on_error(self, error: Exception):
        self.patch()

    def on_completed(self):
        self.patch()



class ConsoleWriterNoId(Observer):
    """Print each detected speaker segment to the console as it arrives."""
    def __init__(self, uri: Optional[Text] = None):
        super().__init__()
        self.uri = uri

    def on_next(self, value):
        # value can be (Annotation, …) or an Annotation directly
        annotation = _extract_prediction(value)
        annotation.uri = self.uri or annotation.uri

        # Iterate over each (segment, track, label)
        for segment, _, label in annotation.itertracks(yield_label=True):
            print(f"[{annotation.uri}] {segment.start:.3f}–{segment.end:.3f}: {label}")

    def on_error(self, error: Exception):
        print(f"[{self.uri}] ERROR: {error}")

    def on_completed(self):
        print(f"[{self.uri}] Completed.")

class ConsoleWriter(Observer):
    """Print each detected segment or silence to the console with human names."""
    def __init__(
        self,
        uri: Optional[str] = None,
        speaker_names: Optional[list[str]] = None,
    ):
        super().__init__()
        self.uri = uri or ""
        self.speaker_names = speaker_names or []

    def _map_label(self, label: str|int) -> str:
        s = str(label)
        if s.startswith("speaker"):
            idx = int(s[len("speaker"):])
        else:
            idx = int(s)
        return self.speaker_names[idx] if 0 <= idx < len(self.speaker_names) else s

    def on_next(self, value):
        # value is either (Annotation, SlidingWindowFeature) or Annotation
        if isinstance(value, tuple) and len(value) == 2:
            annotation, waveform = value
            # get the time‐window for this chunk:
            sw = waveform.sliding_window
            start, end = sw.start, sw.start + sw.duration
        else:
            annotation = _extract_prediction(value)
            start = end = None

        # pull out just the *new* segments in this callback
        segments = list(annotation.itertracks(yield_label=True))

        if not segments:
            # no speaker detected in this window
            if start is not None:
                print(f"[{self.uri}] {start:.3f}–{end:.3f}: <silence>")
            else:
                print(f"[{self.uri}] <silence>")
        else:
            for segment, _, label in segments:
                name = self._map_label(label)
                print(f"[{self.uri}] {segment.start:.3f}–{segment.end:.3f}: {name}")

    def on_error(self, error: Exception):
        print(f"[{self.uri}] ERROR: {error}")

    def on_completed(self):
        print(f"[{self.uri}] Completed.")

class SilenceWindowObserver(Observer):
    """
    Buffers audio and closes the window when silence is detected after an initial warm-up.

    - Waits `warmup_duration` seconds before starting silence detection.
    - Detects `silence_threshold` seconds of continuous silence (no active speaker segments).
    - Writes each closed window to a WAV file in `output_dir`.
    - If `verbose` is True, prints debug logs.
    """

    def __init__(
        self,
        sample_rate: int,
        silence_threshold: float,
        output_dir: str,
        base_filename: str = "window",
        warmup_duration: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.warmup_duration = warmup_duration
        self.output_dir = output_dir
        self.base = base_filename
        self.verbose = verbose

        os.makedirs(self.output_dir, exist_ok=True)
        self.window_count = 0
        self.reset()

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def reset(self):
        self.log("[DEBUG] Resetting buffer, silence counter, elapsed time")
        self.buffer = np.array([], dtype=np.float32)
        self.silence_duration = 0.0
        self.elapsed_time = 0.0

    def on_next(self, value):
        # value: (Annotation, SlidingWindowFeature) or (Annotation, SlidingWindowFeature, real_time)
        if isinstance(value, tuple):
            if len(value) == 3:
                annotation, waveform, _ = value
            elif len(value) == 2:
                annotation, waveform = value
            else:
                self.log(f"[DEBUG] Unsupported tuple length: {len(value)}")
                return
        else:
            self.log(f"[DEBUG] Unsupported value type: {type(value)}")
            return

        audio = waveform.data.flatten()
        frame_duration = len(audio) / self.sample_rate

        # Update elapsed time
        self.elapsed_time += frame_duration
        self.log(f"[DEBUG] Elapsed time: {self.elapsed_time:.3f}s (warm-up: {self.warmup_duration:.3f}s)")

        # Always accumulate audio
        self.buffer = np.concatenate([self.buffer, audio])
        self.log(f"[DEBUG] Buffer length (samples): {len(self.buffer)}")

        # Skip silence detection until warm-up is done
        if self.elapsed_time < self.warmup_duration:
            self.log("[DEBUG] Warm-up phase; skipping silence detection")
            return

        # Count active segments (speech)
        tracks_count = len(list(annotation.itertracks(yield_label=True)))
        self.log(f"[DEBUG] frame_dur={frame_duration:.3f}s, tracks={tracks_count}")

        if tracks_count > 0:
            self.log("[DEBUG] Speech detected; resetting silence counter")
            self.silence_duration = 0.0
        else:
            self.silence_duration += frame_duration
            self.log(f"[DEBUG] Silence running total: {self.silence_duration:.3f}s")

        # Close window on silence threshold
        if self.silence_duration >= self.silence_threshold:
            self.window_count += 1
            fname = f"{self.base}_{self.window_count:03d}.wav"
            path = os.path.join(self.output_dir, fname)
            self.log(f"[DEBUG] Silence threshold reached; writing window #{self.window_count} to {path}")
            self._write_wav(path, self.buffer)
            self.reset()
            raise WindowClosedException(
                f"Wrote silence window #{self.window_count} to {path}"
            )

    def on_error(self, error: Exception):
        self.log(f"[DEBUG] Observer encountered error: {error}")
        if len(self.buffer) > 0:
            err_path = os.path.join(self.output_dir, f"{self.base}_err.wav")
            self.log(f"[DEBUG] Writing remaining buffer to {err_path} before error")
            self._write_wav(err_path, self.buffer)

    def on_completed(self):
        self.log(f"[DEBUG] Stream completed; flushing buffer if any")
        if len(self.buffer) > 0:
            self.window_count += 1
            fname = f"{self.base}_{self.window_count:03d}.wav"
            path = os.path.join(self.output_dir, fname)
            self.log(f"[DEBUG] Writing final buffer to {path}")
            self._write_wav(path, self.buffer)

    def _write_wav(self, path: str, buffer: np.ndarray):
        samples = (buffer * 32767).astype(np.int16)
        self.log(f"[DEBUG] _write_wav: writing {len(samples)} samples to {path}")
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(samples.tobytes())
        self.log(f"[DEBUG] Wrote WAV file: {path}")



class PredictionAccumulator(Observer):
    def __init__(self, uri: Optional[Text] = None, patch_collar: float = 0.05):
        super().__init__()
        self.uri = uri
        self.patch_collar = patch_collar
        self._prediction: Optional[Annotation] = None

    def patch(self):
        """Stitch same-speaker turns that are close to each other"""
        if self._prediction is not None:
            self._prediction = self._prediction.support(self.patch_collar)

    def get_prediction(self) -> Annotation:
        # Patch again in case this is called before on_completed
        self.patch()
        return self._prediction

    def on_next(self, value: Union[Tuple, Annotation]):
        prediction = _extract_prediction(value)
        prediction.uri = self.uri
        if self._prediction is None:
            self._prediction = prediction
        else:
            self._prediction.update(prediction)

    def on_error(self, error: Exception):
        self.patch()

    def on_completed(self):
        self.patch()


class StreamingPlot(Observer):
    def __init__(
        self,
        duration: float,
        latency: float,
        visualization: Literal["slide", "accumulate"] = "slide",
        reference: Optional[Union[Path, Text]] = None,
    ):
        super().__init__()
        assert visualization in ["slide", "accumulate"]
        self.visualization = visualization
        self.reference = reference
        if self.reference is not None:
            self.reference = list(load_rttm(reference).values())[0]
        self.window_duration = duration
        self.latency = latency
        self.figure, self.axs, self.num_axs = None, None, -1
        # This flag allows to catch the matplotlib window closed event and make the next call stop iterating
        self.window_closed = False

    def _on_window_closed(self, event):
        self.window_closed = True

    def _init_num_axs(self):
        if self.num_axs == -1:
            self.num_axs = 2
            if self.reference is not None:
                self.num_axs += 1

    def _init_figure(self):
        self._init_num_axs()
        self.figure, self.axs = plt.subplots(
            self.num_axs, 1, figsize=(10, 2 * self.num_axs)
        )
        if self.num_axs == 1:
            self.axs = [self.axs]
        self.figure.canvas.mpl_connect("close_event", self._on_window_closed)

    def _clear_axs(self):
        for i in range(self.num_axs):
            self.axs[i].clear()

    def get_plot_bounds(self, real_time: float) -> Segment:
        start_time = 0
        end_time = real_time - self.latency
        if self.visualization == "slide":
            start_time = max(0.0, end_time - self.window_duration)
        return Segment(start_time, end_time)

    def on_next(self, values: Tuple[Annotation, SlidingWindowFeature, float]):
        if self.window_closed:
            raise WindowClosedException

        prediction, waveform, real_time = values

        # Initialize figure if first call
        if self.figure is None:
            self._init_figure()
        # Clear previous plots
        self._clear_axs()
        # Set plot bounds
        notebook.crop = self.get_plot_bounds(real_time)

        # Align prediction and reference if possible
        if self.reference is not None:
            metric = DiarizationErrorRate()
            mapping = metric.optimal_mapping(self.reference, prediction)
            prediction.rename_labels(mapping=mapping, copy=False)

        # Plot prediction
        notebook.plot_annotation(prediction, self.axs[0])
        self.axs[0].set_title("Output")

        # Plot waveform
        notebook.plot_feature(waveform, self.axs[1])
        self.axs[1].set_title("Audio")

        # Plot reference if available
        if self.num_axs == 3:
            notebook.plot_annotation(self.reference, self.axs[2])
            self.axs[2].set_title("Reference")

        # Draw
        plt.tight_layout()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(0.05)
