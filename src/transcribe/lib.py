import datetime as dt
import os

import torch
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
from whisper import Whisper

TEMP_FILE = "temp.wav"


def load_models(token: str, device: str) -> tuple[Pipeline, Whisper]:
    pyannote_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    ).to(torch.device(device))
    whisper_model = whisper.load_model("medium", device=device)
    return pyannote_model, whisper_model


def _seconds_to_timestamp(s):
    return (dt.datetime(2000, 1, 1) + dt.timedelta(seconds=s)).strftime("%H:%M:%S")


def _iter_tracks(diarization):
    current_speaker = None
    current_start = None
    current_end = None
    speaker = ""
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start, end = segment.start, segment.end

        if current_speaker is None:
            current_speaker = speaker
            current_start = start
            current_end = end

        if speaker != current_speaker:
            yield current_start, current_end, speaker
            current_speaker = speaker
            current_start = start

        current_end = end

    yield current_start, current_end, speaker


def diarized_transcription(
        pyannote_model: Pipeline, whisper_model: Whisper, audio_path: str, language: str = None
) -> str:
    diarization = pyannote_model(audio_path)
    audio = AudioSegment.from_mp3(audio_path)

    paragraphs = []
    for start, end, speaker in _iter_tracks(diarization):
        timestamp = _seconds_to_timestamp(start)
        speaker_str = f"Speaker {int(speaker.split('_')[1]) + 1}"

        start = int(start * 1000)
        end = int(end * 1000)
        audio[start:end].export(TEMP_FILE, format="wav")

        transcript = whisper_model.transcribe(TEMP_FILE, language=language)
        paragraphs.append(f"{speaker_str} ({timestamp}):\n{transcript['text'].strip()}")

    os.remove(TEMP_FILE)

    return "\n\n".join(paragraphs)
