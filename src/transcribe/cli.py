import typer

app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@app.command()
def main(
        audio: str = typer.Argument(..., help="Path to the input audio file."),
        token: str = typer.Option(..., "-t", "--token", help="Hugging face access token."),
        out: str = typer.Option(None, "-o", "--out", help="Path to the output file. Print to stdout if not set."),
        device: str = typer.Option("cuda", help="Device to run the models on."),
        language: str = typer.Option("en", help="Spoken language in the recording.")
):
    """
    Transcribe and diarize (recognize speakers) recorded audio.
    """
    from transcribe import lib

    pyannote_model, whisper_model = lib.load_models(token, device)
    text = lib.diarized_transcription(pyannote_model, whisper_model, audio_path=audio, language=language)

    if out is None:
        print(text)
    else:
        with open(out, "w") as f:
            f.write(text)
