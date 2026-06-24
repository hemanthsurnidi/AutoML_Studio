import json
import os
import uuid
from config import Config


def new_session() -> str:
    """Create a new session ID and initialize an empty state file."""
    sid = str(uuid.uuid4())
    os.makedirs(Config.SESSION_FOLDER, exist_ok=True)
    _write(sid, {})
    return sid


def save_state(sid: str, state: dict):
    """Persist session state to disk as JSON."""
    _write(sid, state)


def load_state(sid: str) -> dict:
    """Load session state from disk. Returns empty dict if not found."""
    path = _path(sid)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def update_state(sid: str, updates: dict):
    """Merge updates into existing session state."""
    state = load_state(sid)
    state.update(updates)
    save_state(sid, state)


def clear_session(sid: str):
    """Delete session state file."""
    path = _path(sid)
    if os.path.exists(path):
        os.remove(path)


def _path(sid: str) -> str:
    return os.path.join(Config.SESSION_FOLDER, f"{sid}.json")


def _write(sid: str, state: dict):
    os.makedirs(Config.SESSION_FOLDER, exist_ok=True)
    with open(_path(sid), "w", encoding="utf-8") as f:
        json.dump(state, f, default=str)
