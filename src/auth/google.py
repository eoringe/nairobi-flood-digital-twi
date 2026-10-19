"""
src.auth.google
===============
Firebase Authentication sign-in: "Continue with Google" and passwordless
email links.

The browser completes the Firebase sign-in (Google popup, or the link Firebase
emails) and receives a Firebase ID token, a JWT signed by Google. The server
checks it against Google's public keys with google-auth, using only the
Firebase project ID as the expected audience. No service-account key is held on
the server.

Both methods prove the person controls the email address, so the server
accepts a token only when its email is verified, then opens the account with
that email. Roles always come from this system's database, never from Firebase.

Configuration (public values from Firebase console -> Project settings -> Your
apps -> Web app):

    FIREBASE_API_KEY, FIREBASE_AUTH_DOMAIN, FIREBASE_PROJECT_ID, FIREBASE_APP_ID

In the console, enable Authentication -> Sign-in method -> Google, and
Email/Password with "Email link (passwordless sign-in)". Without
FIREBASE_PROJECT_ID only email + password sign-in is offered.

Role assignment: a first-time user becomes a citizen. Emails listed in
TWIN_ADMIN_EMAILS (comma separated) become administrators.
"""

from __future__ import annotations

import os

from loguru import logger

from src.auth.store import AuthError, AuthStore

#: Firebase sign_in_provider claim -> how this system records the method.
#: Email-link sign-ins are issued by Firebase's email provider ("password").
FIREBASE_METHODS = {"google.com": "google", "password": "email_link", "emailLink": "email_link"}
METHOD_LABELS = {"google": "Google", "email_link": "email link"}


def firebase_config() -> dict | None:
    """Public web config for the Firebase JS SDK, or None when not configured."""
    project = os.environ.get("FIREBASE_PROJECT_ID")
    api_key = os.environ.get("FIREBASE_API_KEY")
    if not (project and api_key):
        return None
    return {
        "apiKey": api_key,
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", f"{project}.firebaseapp.com"),
        "projectId": project,
        "appId": os.environ.get("FIREBASE_APP_ID", ""),
    }


def verify_token(id_token: str) -> tuple[dict, str]:
    """
    (claims, method) for a valid Firebase ID token; method is "google" or
    "email_link". Raises AuthError otherwise.

    google-auth checks the signature, expiry, issuer
    (https://securetoken.google.com/<project>) and audience (<project>).
    """
    project = os.environ.get("FIREBASE_PROJECT_ID")
    if not project:
        raise AuthError("Google and email-link sign-in are not configured on this server.")
    try:
        from google.auth.transport import requests as g_requests
        from google.oauth2 import id_token as g_id_token
        claims = g_id_token.verify_firebase_token(id_token, g_requests.Request(), audience=project)
    except Exception as exc:                                # noqa: BLE001
        logger.warning(f"Firebase token rejected: {exc}")
        raise AuthError("Sign-in could not be verified. Please try again.") from None
    if not claims:
        raise AuthError("Sign-in could not be verified. Please try again.")
    method = FIREBASE_METHODS.get((claims.get("firebase") or {}).get("sign_in_provider"))
    if method is None:
        raise AuthError("That sign-in method is not accepted here.")
    if not claims.get("email") or not claims.get("email_verified"):
        raise AuthError("Your email address has not been verified.")
    return claims, method


def _admin_emails() -> set[str]:
    raw = os.environ.get("TWIN_ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def user_from_claims(store: AuthStore, claims: dict, method: str = "google") -> tuple[dict, bool]:
    """
    The account for this verified email, creating a citizen account on first
    sign-in. An existing account with the same email - including one an
    administrator created with a staff role - is the one opened.
    Returns (user, created).
    """
    email = claims["email"].strip().lower()
    user = store.get_by_email(email)
    if user:
        if not user["active"]:
            raise AuthError("This account has been disabled. Contact the administrator.")
        if email in _admin_emails() and user["role"] != "admin":
            user = store.update_user(user["user_id"], role="admin")
        store.set_display_name(user["user_id"], claims.get("name"))
        return user, False
    role = "admin" if email in _admin_emails() else "citizen"
    user = store.create_user(email, None, role, provider=method, display_name=claims.get("name"))
    return user, True
