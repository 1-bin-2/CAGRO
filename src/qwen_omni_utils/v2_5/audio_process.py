import base64
from io import BytesIO

import audioread
import av
import numpy as np
try:
    import librosa as _librosa
except Exception:
    _librosa = None


SAMPLE_RATE=16000
def _check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations: list[dict] | list[list[dict]], use_audio_in_video: bool):
    """
    Read and process audio info

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(
                                path[int(SAMPLE_RATE * audio_start) : None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                            )
                            continue
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        elif path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        if path.startswith("http://") or path.startswith("https://"):
                            data = audioread.ffdec.FFmpegAudioFile(path)
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
                # Robust audio loading: try librosa (if available), otherwise fall back to PyAV/audioread
                def _robust_load_audio(data_obj, sr=SAMPLE_RATE, offset=0.0, duration=None):
                    # Try librosa first when available
                    if _librosa is not None:
                        try:
                            return _librosa.load(
                                data_obj, sr=sr, offset=offset, duration=duration
                            )[0]
                        except Exception:
                            pass

                    # Try PyAV for container formats (mp4, mkv, mov, etc.) and local files
                    try:
                        container = av.open(data_obj)
                        audio_streams = [s for s in container.streams if s.type == "audio"]
                        if not audio_streams:
                            raise RuntimeError("No audio stream found")
                        audio_stream = audio_streams[0]
                        frames = []
                        for packet in container.demux(audio_stream):
                            for frame in packet.decode():
                                try:
                                    arr = frame.to_ndarray()
                                except Exception:
                                    # fallback, try converting via frame.planes
                                    arr = np.asarray(frame.planes[0].to_bytes())
                                arr = np.asarray(arr)
                                if arr.ndim == 2:
                                    # average channels -> mono
                                    arr = arr.mean(axis=0)
                                # convert integer PCM to float32 in [-1, 1]
                                if np.issubdtype(arr.dtype, np.integer):
                                    maxv = np.iinfo(arr.dtype).max
                                    arr = arr.astype(np.float32) / float(maxv)
                                else:
                                    arr = arr.astype(np.float32)
                                frames.append(arr)
                        if len(frames) == 0:
                            raise RuntimeError("No audio frames decoded")
                        y = np.concatenate(frames)
                        native_sr = getattr(audio_stream, "rate", None) or getattr(getattr(audio_stream, 'codec_context', None), 'sample_rate', None)
                        if native_sr is None:
                            native_sr = sr
                        if native_sr != sr:
                            if _librosa is not None:
                                y = _librosa.resample(y, native_sr, sr)
                            else:
                                try:
                                    from scipy.signal import resample

                                    num = int(len(y) * sr / native_sr)
                                    y = resample(y, num)
                                except Exception:
                                    pass
                        start_sample = int(offset * sr) if offset else 0
                        end_sample = start_sample + int(duration * sr) if duration is not None else None
                        return y[start_sample:end_sample]
                    except Exception:
                        # Last resort: try audioread (ffmpeg backend) which yields raw PCM blocks
                        try:
                            with audioread.audio_open(data_obj) as fh:
                                sr_native = fh.samplerate
                                blocks = []
                                for block in fh:
                                    arr = np.frombuffer(block, dtype=np.int16).astype(np.float32) / 32768.0
                                    blocks.append(arr)
                                if len(blocks) == 0:
                                    raise RuntimeError("No audio data from audioread")
                                y = np.concatenate(blocks)
                                if sr_native != sr:
                                    if _librosa is not None:
                                        y = _librosa.resample(y, sr_native, sr)
                                    else:
                                        try:
                                            from scipy.signal import resample

                                            num = int(len(y) * sr / sr_native)
                                            y = resample(y, num)
                                        except Exception:
                                            pass
                                start_sample = int(offset * sr) if offset else 0
                                end_sample = start_sample + int(duration * sr) if duration is not None else None
                                return y[start_sample:end_sample]
                        except Exception as e:
                            raise e

                try:
                    loaded_audio = _robust_load_audio(
                        data, sr=SAMPLE_RATE, offset=audio_start, duration=(audio_end - audio_start) if audio_end is not None else None
                    )
                    audios.append(loaded_audio)
                except Exception as e:
                    print(f"[audio_process] Failed to load audio {path if 'path' in locals() else data}: {e}")
                    audios.append(None)
    if len(audios) == 0:
        audios = None
    return audios
