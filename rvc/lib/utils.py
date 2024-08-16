import os, sys
import librosa
import soundfile as sf
import numpy as np
import re
import unicodedata
from fairseq import checkpoint_utils
import wget
import subprocess
from pydub import AudioSegment
import tempfile

import logging

logging.getLogger("fairseq").setLevel(logging.WARNING)

now_dir = os.getcwd()
sys.path.append(now_dir)

platform_stft_mapping = {
    "linux": "stftpitchshift",
    "darwin": "stftpitchshift",
    "win32": "stftpitchshift.exe",
}

stft = platform_stft_mapping.get(sys.platform)


def load_audio(file, sample_rate):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")

    return audio.flatten()


def load_audio_infer(
    file, sample_rate, formant_shifting, formant_qfrency, formant_timbre
):
    try:
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        audio, sr = sf.read(file)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.T)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        if formant_shifting:
            audio = (audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1,
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file_path = temp_file.name
                audio_segment.export(temp_file_path, format="wav")

            command = (
                f'{stft} -i "{temp_file_path}" -q "{formant_qfrency}" '
                f'-t "{formant_timbre}" -o "{temp_file_path}_formatted.wav"'
            )
            subprocess.run(command, shell=True)
            formatted_audio_path = f"{temp_file_path}_formatted.wav"
            audio, sr = sf.read(formatted_audio_path)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio.T)
            if sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    except Exception as error:
        raise RuntimeError(f"An error occurred loading the audio: {error}")


def format_title(title):
    formatted_title = (
        unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("utf-8")
    )
    formatted_title = re.sub(r"[\u2500-\u257F]+", "", formatted_title)
    formatted_title = re.sub(r"[^\w\s.-]", "", formatted_title)
    formatted_title = re.sub(r"\s+", "_", formatted_title)
    return formatted_title


def load_embedding(embedder_model, custom_embedder=None):
    embedder_root = os.path.join(now_dir, "rvc", "models", "embedders")
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec_base.pt"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese-hubert-base.pt"),
        "chinese-hubert-large": os.path.join(embedder_root, "chinese-hubert-large.pt"),
    }

    online_embedders = {
        "japanese-hubert-base": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt",
        "chinese-hubert-large": "https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt",
    }

    if embedder_model == "custom":
        model_path = custom_embedder
        if not custom_embedder and os.path.exists(custom_embedder):
            model_path = embedding_list["contentvec"]
    else:
        model_path = embedding_list[embedder_model]
        if embedder_model in online_embedders:
            if not os.path.exists(model_path):
                url = online_embedders[embedder_model]
                print(f"\nDownloading {url} to {model_path}...")
                wget.download(url, out=model_path)
        else:
            model_path = embedding_list["contentvec"]

    models = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )

    # print(f"Embedding model {embedder_model} loaded successfully.")
    return models
