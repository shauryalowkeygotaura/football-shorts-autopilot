#!/usr/bin/env python
"""One-time YouTube OAuth bootstrapper for football-shorts-autopilot.

The ONLY step that can't be done headless. Run this ONCE on your own machine:
it opens a browser, you pick the channel + grant upload permission, and it
prints a long-lived refresh token (plus the exact Doppler commands to store
it). After that the pipeline uploads forever with no browser, locally or in CI.

──────────────────────────────────────────────────────────────────────────────
PREREQUISITE (Google Cloud, ~3 min, free):
  1. https://console.cloud.google.com/  -> create/pick a project
  2. APIs & Services -> Library -> enable "YouTube Data API v3"
  3. APIs & Services -> OAuth consent screen -> External -> add yourself as a
     Test user (so the app needn't be verified). App can stay in "Testing".
  4. Credentials -> Create credentials -> OAuth client ID -> type "Desktop app"
     -> Download JSON. Save it next to this file as  client_secret.json
     (or pass --client-secret <path>).

RUN:
    python auth_youtube.py
    # or, if you already have the id/secret as env/Doppler:
    YT_CLIENT_ID=... YT_CLIENT_SECRET=... python auth_youtube.py

The script tries a local-server browser flow first (best UX). If you're on a
headless box, pass --console to use copy-paste device-less flow instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
HERE = Path(__file__).parent.resolve()


def _client_config_from_env() -> dict | None:
    cid = os.environ.get("YT_CLIENT_ID", "").strip()
    secret = os.environ.get("YT_CLIENT_SECRET", "").strip()
    if cid and secret:
        return {
            "installed": {
                "client_id": cid,
                "client_secret": secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }
    return None


def _find_client_secret(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    # auto-discover a downloaded client_secret*.json next to this file
    for name in ("client_secret.json", "client_secrets.json"):
        if (HERE / name).exists():
            return HERE / name
    hits = sorted(HERE.glob("client_secret*.json"))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--client-secret", help="path to the GCP OAuth client JSON")
    ap.add_argument("--console", action="store_true",
                    help="use console (copy-paste) flow for headless machines")
    ap.add_argument("--port", type=int, default=0,
                    help="local server port (0 = auto-pick a free one)")
    args = ap.parse_args()

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        sys.exit("Missing dep. Install with:\n    pip install google-auth-oauthlib")

    env_cfg = _client_config_from_env()
    if env_cfg:
        flow = InstalledAppFlow.from_client_config(env_cfg, SCOPES)
        src = "env (YT_CLIENT_ID/YT_CLIENT_SECRET)"
    else:
        cs = _find_client_secret(args.client_secret)
        if not cs:
            sys.exit(
                "No OAuth client found.\n"
                "  Either set YT_CLIENT_ID + YT_CLIENT_SECRET in the env,\n"
                "  or download the Desktop-app OAuth JSON from Google Cloud and\n"
                "  save it as client_secret.json next to this script\n"
                "  (see the header of this file for the 4 GCP steps).")
        flow = InstalledAppFlow.from_client_secrets_file(str(cs), SCOPES)
        src = str(cs)

    print(f"Using OAuth client from: {src}")
    print("Scope: youtube.upload  (upload-only; cannot read/delete your videos)\n")

    if args.console:
        # Manual flow: prints a URL, you paste back the code. No local server.
        flow.run_console()
    else:
        print("Opening a browser. Pick the channel you want to publish to and "
              "click Allow...\n")
        flow.run_local_server(port=args.port, prompt="consent",
                              authorization_prompt_message="")

    creds = flow.credentials
    if not creds.refresh_token:
        sys.exit("No refresh token returned. Re-run; if it persists, revoke the "
                 "app at https://myaccount.google.com/permissions and try again "
                 "(consent must be fresh to mint a refresh token).")

    cid = creds.client_id
    secret = creds.client_secret
    refresh = creds.refresh_token

    print("\n" + "=" * 70)
    print("SUCCESS — refresh token generated. Store these 3 secrets in Doppler:")
    print("=" * 70)
    proj = "youtube-title-autoresearch"   # shared Doppler home for this pipeline
    cfg = "dev"
    base = f"doppler secrets set --project {proj} --config {cfg}"
    print(f'\n  {base} YT_CLIENT_ID="{cid}"')
    print(f'  {base} YT_CLIENT_SECRET="{secret}"')
    print(f'  {base} YT_REFRESH_TOKEN="{refresh}"')
    print(f'\n  # then flip live (was YT_DRY_RUN=1):')
    print(f'  {base} YT_DRY_RUN="0"')
    print("\nFor the GitHub Actions cron, also add a DOPPLER_TOKEN repo secret "
          "(a service token for this project/config).")

    # Also drop a local .secrets.json so you can copy/inspect, gitignored.
    out = HERE / ".yt_oauth.json"
    out.write_text(json.dumps(
        {"YT_CLIENT_ID": cid, "YT_CLIENT_SECRET": secret,
         "YT_REFRESH_TOKEN": refresh}, indent=2))
    print(f"\n(Also written to {out.name} — gitignored; delete after storing.)")


if __name__ == "__main__":
    main()
