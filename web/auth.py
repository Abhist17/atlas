"""Authentication — user store + password hashing + signed session cookies.

Users are persisted to data_store/users.json. Passwords are hashed with
PBKDF2-HMAC-SHA256 (stdlib, no plaintext ever stored). Sessions are stateless
signed cookies via itsdangerous.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import secrets
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage

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


# ---------------------------------------------------------------- OTP login
OTP_TTL = 600            # 10 minutes
OTP_MAX_ATTEMPTS = 5
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_otps: dict[str, dict] = {}   # email -> {hash, plain, expires, attempts}


def valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match((email or "").strip()))


def smtp_configured() -> bool:
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_USER") and os.getenv("SMTP_PASS"))


def generate_otp(email: str) -> str:
    """Create + store a 6-digit code for this email. Returns the code."""
    email = email.strip().lower()
    code = f"{random.randint(0, 999999):06d}"
    _otps[email] = {"hash": hashlib.sha256(code.encode()).hexdigest(),
                    "plain": code, "expires": time.time() + OTP_TTL, "attempts": 0}
    log.info("Login OTP for %s: %s", email, code)   # printed to server console (dev)
    return code


def dev_code(email: str) -> str | None:
    """The plain code, exposed only when SMTP isn't configured (local dev)."""
    if smtp_configured():
        return None
    rec = _otps.get(email.strip().lower())
    return rec["plain"] if rec and time.time() <= rec["expires"] else None


def verify_otp(email: str, code: str) -> bool:
    email = email.strip().lower()
    rec = _otps.get(email)
    if not rec or time.time() > rec["expires"] or rec["attempts"] >= OTP_MAX_ATTEMPTS:
        _otps.pop(email, None)
        return False
    rec["attempts"] += 1
    if secrets.compare_digest(rec["hash"], hashlib.sha256(code.strip().encode()).hexdigest()):
        _otps.pop(email, None)
        _ensure_user(email)
        return True
    return False


def _ensure_user(email: str) -> dict:
    """Passwordless: create the user record on first successful login."""
    users = _load_users()
    email = email.strip().lower()
    if email not in users:
        users[email] = {"name": email.split("@")[0], "created": datetime.now().isoformat()}
        _save_users(users)
        log.info("New user via OTP: %s", email)
    return users[email]


def send_otp_email(email: str, code: str) -> tuple[bool, str]:
    """Email the code via SMTP if configured; otherwise dev mode (False)."""
    if not smtp_configured():
        return False, "dev"
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user, pw = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    msg = EmailMessage()
    msg["Subject"] = "Your Atlas login code"
    msg["From"] = os.getenv("SMTP_FROM", user)
    msg["To"] = email
    msg.set_content(f"Your Atlas login code is: {code}\n\nIt expires in 10 minutes.\n"
                    f"If you didn't request this, ignore this email.")
    try:
        with smtplib.SMTP(host, port, timeout=15) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        return True, "sent"
    except Exception as e:
        log.error("OTP email send failed: %s", e)
        return False, str(e)
