"""
src.auth.totp
=============
Authenticator-app codes (TOTP, RFC 6238) for administrator two-factor sign-in.

Works with Google Authenticator, Microsoft Authenticator, Authy and any app that
reads an otpauth:// QR code: HMAC-SHA1, 6 digits, 30-second steps. One step of
clock drift either way is accepted, and a code that has been used once is
refused afterwards, so a code seen over someone's shoulder cannot be replayed.

Recovery codes are random, shown once, stored only as hashes, and each works once.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote, urlencode

from werkzeug.security import check_password_hash, generate_password_hash

STEP_SEC = 30
DIGITS = 6
DRIFT_STEPS = 1
ISSUER = "Nairobi Flood Twin"
RECOVERY_CODES = 8


def new_secret() -> str:
    """A random 160-bit secret, base32 as authenticator apps expect."""
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def _code_at(secret: str, step: int) -> str:
    key = base64.b32decode(secret + "=" * (-len(secret) % 8), casefold=True)
    digest = hmac.new(key, struct.pack(">Q", step), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(value % 10 ** DIGITS).zfill(DIGITS)


def code_now(secret: str, at: float | None = None) -> str:
    """The current code (used by tests and for diagnostics)."""
    return _code_at(secret, int((time.time() if at is None else at) // STEP_SEC))


def matching_step(secret: str, code: str, last_step: int | None, at: float | None = None) -> int | None:
    """
    The time step the code belongs to, or None if it is wrong or already used.

    `last_step` is the step of the last accepted code; anything at or before it
    is refused, which blocks replay within the drift window.
    """
    code = "".join(ch for ch in (code or "") if ch.isdigit())
    if len(code) != DIGITS:
        return None
    now = int((time.time() if at is None else at) // STEP_SEC)
    for step in range(now - DRIFT_STEPS, now + DRIFT_STEPS + 1):
        if last_step is not None and step <= last_step:
            continue
        if hmac.compare_digest(_code_at(secret, step), code):
            return step
    return None


def provisioning_uri(secret: str, account: str) -> str:
    """The otpauth:// address encoded in the setup QR code."""
    label = quote(f"{ISSUER}:{account}")
    query = urlencode({"secret": secret, "issuer": ISSUER, "algorithm": "SHA1",
                       "digits": DIGITS, "period": STEP_SEC})
    return f"otpauth://totp/{label}?{query}"


def qr_svg_data_uri(uri: str) -> str:
    import segno
    return segno.make(uri, error="m").svg_data_uri(scale=5, border=2, dark="#000000", light="#ffffff")


def new_recovery_codes() -> list[str]:
    """Codes like 'k7m2-q9xd': 8 characters, no look-alike letters."""
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"
    out = []
    for _ in range(RECOVERY_CODES):
        raw = "".join(secrets.choice(alphabet) for _ in range(8))
        out.append(f"{raw[:4]}-{raw[4:]}")
    return out


def hash_recovery_code(code: str) -> str:
    return generate_password_hash(normalise_recovery_code(code))


def normalise_recovery_code(code: str) -> str:
    return "".join(ch for ch in (code or "").lower() if ch.isalnum())


def recovery_code_matches(code: str, hashed: str) -> bool:
    return check_password_hash(hashed, normalise_recovery_code(code))
