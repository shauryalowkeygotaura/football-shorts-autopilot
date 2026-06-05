"""YouTube Data API v3 uploader (free quota: ~6 uploads/day on the default
10,000-unit budget, since each upload costs 1,600 units). Enough for 1 doc +
4 shorts is over budget on ONE channel/day, so the pipeline uploads the doc
daily and staggers shorts; tune SHORTS_PER_DOC or request a quota bump.

Auth uses an OAuth refresh token stored in Doppler (headless-friendly: no
browser needed at run time once you've generated the token once locally).

Doppler keys expected:
  YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN

Set YT_DRY_RUN=1 to render + log without actually uploading (used by tests
and first smoke runs so we never burn quota by accident).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402

_TOKEN_URI = "https://oauth2.googleapis.com/token"
_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def is_dry() -> bool:
    """Dry when YT_DRY_RUN=1, or when the OAuth creds are absent: a scheduled
    run without creds must degrade to render-only, never crash the cycle."""
    if os.environ.get("YT_DRY_RUN", "").strip() == "1":
        return True
    if not all(os.environ.get(k, "").strip()
               for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN")):
        print("[uploader] YT OAuth creds missing; forcing dry run (no quota burn)",
              file=sys.stderr)
        return True
    return False


def _service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    cid = os.environ.get("YT_CLIENT_ID", "").strip()
    secret = os.environ.get("YT_CLIENT_SECRET", "").strip()
    refresh = os.environ.get("YT_REFRESH_TOKEN", "").strip()
    if not (cid and secret and refresh):
        raise ValueError("YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN "
                         "must be set (provide via Doppler).")
    creds = Credentials(
        token=None, refresh_token=refresh, token_uri=_TOKEN_URI,
        client_id=cid, client_secret=secret, scopes=_SCOPES)
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def upload(video_path: str | Path, title: str, description: str,
           tags: list[str], privacy: str | None = None) -> str:
    """Upload one video. Returns the YouTube video id (or 'DRYRUN-...')."""
    video_path = Path(video_path)
    if is_dry():
        print(f"[DRY RUN] would upload {video_path.name}: {title!r} "
              f"({len(tags)} tags, privacy={privacy or config.YT_PRIVACY})")
        return f"DRYRUN-{video_path.stem}"

    from googleapiclient.http import MediaFileUpload
    youtube = _service()
    body = {
        "snippet": {
            "title": title, "description": description, "tags": tags,
            "categoryId": config.YT_CATEGORY_ID,
        },
        "status": {
            "privacyStatus": privacy or config.YT_PRIVACY,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True,
                            mimetype="video/mp4")
    req = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _status, resp = req.next_chunk()
    return resp["id"]


def set_thumbnail(video_id: str, thumb_path: str | Path) -> bool:
    """Attach a custom thumbnail to an already-uploaded video. No-op in dry run
    and non-fatal: a thumbnail failure (e.g. channel not phone-verified) must
    never sink the run. Returns True on success."""
    thumb_path = Path(thumb_path)
    if is_dry():
        print(f"[DRY RUN] would set thumbnail {thumb_path.name} on {video_id}")
        return True
    if not thumb_path.exists():
        return False
    try:
        from googleapiclient.http import MediaFileUpload
        youtube = _service()
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(thumb_path), mimetype="image/jpeg"),
        ).execute()
        return True
    except Exception as e:  # custom thumbnails need a verified channel
        print(f"[thumbnail] upload failed for {video_id}: {e}")
        return False
