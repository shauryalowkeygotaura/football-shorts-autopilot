# football-shorts-autopilot

Free autonomous faceless YouTube factory for the FIFA World Cup 2026 window. One cycle picks a topic, writes a documentary script with Groq, builds a 16:9 master doc with edge-tts narration, FFmpeg and Wikimedia b-roll, slices it into 4 vertical Shorts, writes the metadata and uploads everything.

## Quick start

```bash
git clone https://github.com/shauryalowkeygotaura/football-shorts-autopilot && cd football-shorts-autopilot && pip install -r requirements.txt && YT_DRY_RUN=1 python pipeline.py --now
```

`YT_DRY_RUN=1` renders only: no upload, no YouTube quota. Copy `.env.example` to `.env` (or use Doppler) and add a Groq key first.

## How it works

```
pick topic -> write doc script -> build 16:9 master doc
          -> slice 4 vertical shorts -> generate metadata
          -> upload doc + shorts to YouTube -> record state
```

- `pipeline.py`: one full cycle per run
- `modules/`: topics, scriptwriter, composer, shorts, metadata, thumbnail, uploader, run_metrics
- `.github/workflows/autopilot.yml`: the cron that sets the cadence
- `auth_youtube.py`: one-time YouTube OAuth

## License

MIT
