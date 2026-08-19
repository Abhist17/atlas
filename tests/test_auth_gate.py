"""Who is allowed to log in, and when the OTP may be shown on screen.

Atlas prints the login code on the verify page when SMTP isn't set up. That is
convenient on localhost and an authentication bypass anywhere else: the code is
the only credential, so showing it to whoever asks lets anyone sign in as anyone.
These tests pin the conditions under which it is withheld.
"""
from __future__ import annotations

import pytest

from web import auth


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS",
              "ATLAS_ALLOWED_EMAILS", "ATLAS_BIND_HOST"):
        monkeypatch.delenv(k, raising=False)


def _smtp(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USER", "u")
    monkeypatch.setenv("SMTP_PASS", "p")


# ---------------------------------------------------------------- bind host
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", ""])
def test_loopback_is_not_public(monkeypatch, host):
    monkeypatch.setenv("ATLAS_BIND_HOST", host)
    assert auth.bound_publicly() is False


def test_the_default_bind_is_loopback():
    """Unset means localhost — the safe end of the default."""
    assert auth.bound_publicly() is False


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
def test_any_other_bind_is_public(monkeypatch, host):
    monkeypatch.setenv("ATLAS_BIND_HOST", host)
    assert auth.bound_publicly() is True


# ---------------------------------------------------------------- dev code
def test_code_is_shown_on_localhost(monkeypatch):
    auth.generate_otp("a@b.com")
    assert auth.dev_code("a@b.com") is not None


def test_code_is_withheld_on_a_public_bind(monkeypatch):
    """The whole point: a LAN-reachable server must not print the credential."""
    monkeypatch.setenv("ATLAS_BIND_HOST", "0.0.0.0")
    auth.generate_otp("a@b.com")
    assert auth.dev_code("a@b.com") is None


def test_code_is_withheld_once_smtp_can_deliver_it(monkeypatch):
    _smtp(monkeypatch)
    auth.generate_otp("a@b.com")
    assert auth.dev_code("a@b.com") is None


# ---------------------------------------------------------------- allowlist
def test_no_allowlist_permits_everyone():
    assert auth.email_allowed("anyone@example.com") is True


def test_allowlist_restricts_and_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("ATLAS_ALLOWED_EMAILS", "me@x.com, You@Y.com")
    assert auth.email_allowed("ME@x.com") is True
    assert auth.email_allowed("you@y.com") is True
    assert auth.email_allowed("stranger@z.com") is False


def test_a_disallowed_email_cannot_verify_even_with_the_right_code(monkeypatch):
    """Belt and braces: the code check itself refuses, not just the login route."""
    code = auth.generate_otp("stranger@z.com")
    monkeypatch.setenv("ATLAS_ALLOWED_EMAILS", "me@x.com")
    assert auth.verify_otp("stranger@z.com", code) is False


# ---------------------------------------------------------------- open login
def test_login_is_open_only_with_no_smtp_and_no_allowlist(monkeypatch):
    assert auth.login_is_open() is True
    monkeypatch.setenv("ATLAS_ALLOWED_EMAILS", "me@x.com")
    assert auth.login_is_open() is False


def test_smtp_alone_closes_the_open_login(monkeypatch):
    _smtp(monkeypatch)
    assert auth.login_is_open() is False
