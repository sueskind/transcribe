# transcribe

Transcribe recordings including speaker recognition (diarization) and timestamps.

Powered by [OpenAI Whisper](https://github.com/openai/whisper)
and [pyannote-audio](https://github.com/pyannote/pyannote-audio).

## Setup

### Prerequisites

- Python 3.11+ (Tested with 3.11.5)
- [Hugging Face](https://huggingface.co) account

### Steps

1. Install via pip:
    ```shell
    pip install git+https://github.com/sueskind/transcribe
    ```
2. Accept [pyannote/segmentation](https://huggingface.co/pyannote/segmentation-3.0)'s user conditions
3. Accept [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)'s user conditions
4. Create a [Hugging Face access token](https://huggingface.co/settings/tokens)

## Usage

Example:

```shell
$ transcribe pulpfiction.mp3 -t <token>
Speaker 1 (00:00:00):
They don't call it a quarter pounder with cheese?

Speaker 2 (00:00:06):
No, they got the metric system there, they wouldn't know what a quarter pounder is.

Speaker 1 (00:00:13):
What do they call it?

Speaker 2 (00:00:17):
Royale with cheese.
```

Use `--help` to show all options:

```shell
$  transcribe --help
Usage: transcribe [OPTIONS] AUDIO

  Transcribe and diarize (recognize speakers) recorded audio.

Arguments:
  AUDIO  Path to the input audio file.  [required]

Options:
  -t, --token TEXT  Hugging face access token.  [required]
  -o, --out TEXT    Path to the output file. Print to stdout if not set.
  --device TEXT     Device to run the models on.  [default: cuda]
  --language TEXT   Spoken language in the recording.  [default: en]
  --help            Show this message and exit.
```
