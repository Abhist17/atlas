"""Authentication — user store + password hashing + signed session cookies.

Users are persisted to data_store/users.json. Passwords are hashed with
PBKDF2-HMAC-SHA256 (stdlib, no plaintext ever stored). Sessions are stateless
signed cookies via itsdangerous.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from config.settings import DATA_STORE
from utils.logger import get_logger

log = get_logger("web.auth")

USERS_FILE = DATA_STORE / "users.json"
_SECRET_FILE = DATA_STORE / ".session_secret"
COOKIE_NAME = "atlas_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7   # 7 days
_PBKDF2_ITERS = 200_000


def _secret_key() -> str:
    """Persistent secret for signing sessions (env override, else generated)."""
    env = os.getenv("ATLAS_SECRET_KEY")
    if env:
        return env
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_text().strip()
    key = secrets.token_hex(32)
    _SECRET_FILE.write_text(key)
    return key


_serializer = URLSafeTimedSerializer(_secret_key(), salt="atlas-session")


# ---------------------------------------------------------------- storage
def _load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}


def _save_users(users: dict) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2))


# ---------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERS)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERS)
    return secrets.compare_digest(dk.hex(), digest)


# ---------------------------------------------------------------- users
def create_user(email: str, password: str, name: str = "") -> tuple[bool, str]:
    email = email.strip().lower()
    if not email or "@" not in email:
        return False, "Enter a valid email."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    users = _load_users()
    if email in users:
        return False, "An account with this email already exists."
    users[email] = {
        "name": name.strip() or email.split("@")[0],
        "password": hash_password(password),
        "created": datetime.now().isoformat(),
    }
    _save_users(users)
    log.info("New user: %s", email)
    return True, "Account created."


def authenticate(email: str, password: str) -> bool:
    users = _load_users()
    user = users.get(email.strip().lower())
    return bool(user and verify_password(password, user["password"]))


def get_user(email: str) -> dict | None:
    return _load_users().get(email.strip().lower())


# ---------------------------------------------------------------- sessions
def make_session(email: str) -> str:
    return _serializer.dumps(email.strip().lower())


def read_session(token: str | None) -> str | None:
    if not token:
        return None
    try:
        return _serializer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
