"""
src.auth.cli
============
Account administration from the command line, for the first administrator and
for recovering access.

    python -m src.auth.cli create-user --email alice@example.com --role admin
    python -m src.auth.cli create-user --email jane@gmail.com --role planner --passwordless
    python -m src.auth.cli set-role --email bob@example.com --role planner
    python -m src.auth.cli reset-password --email bob@example.com
    python -m src.auth.cli reset-2fa --email alice@example.com
    python -m src.auth.cli list

--passwordless accounts sign in with Google or an emailed link only. Passwords
are read with a hidden prompt, never from the command line, so they do not end
up in shell history. reset-2fa is for an administrator who has lost both the
phone and the recovery codes: they set up the authenticator again at the next
sign-in. In Docker, prefix with `docker compose exec dashboard`.
"""

from __future__ import annotations

import argparse
import getpass
import sys

from src.auth.store import ROLES, AuthError, get_store


def _ask_password() -> str:
    pw = getpass.getpass("Password: ")
    if pw != getpass.getpass("Confirm password: "):
        raise AuthError("The passwords do not match.")
    return pw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.auth.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-user")
    c.add_argument("--email", required=True)
    c.add_argument("--role", choices=ROLES, default="citizen")
    c.add_argument("--passwordless", action="store_true",
                   help="no password: sign in with Google or an emailed link")
    r = sub.add_parser("set-role")
    r.add_argument("--email", required=True)
    r.add_argument("--role", choices=ROLES, required=True)
    for name in ("reset-password", "reset-2fa"):
        sub.add_parser(name).add_argument("--email", required=True)
    sub.add_parser("list")
    args = ap.parse_args(argv)

    store = get_store()
    try:
        if args.cmd == "create-user":
            if args.passwordless:
                user = store.create_user(args.email, None, args.role, provider="google")
            else:
                user = store.create_user(args.email, _ask_password(), args.role)
            store.audit("user_created", None, f"{user['email']} as {user['role']} (command line)")
            print(f"Created {user['email']} ({user['role']}).")
            if user["role"] == "admin":
                print("They will set up an authenticator app at their first sign-in.")
            return 0
        if args.cmd == "list":
            for u in store.list_users():
                state = "active" if u["active"] else "disabled"
                second = "2FA on" if u["totp_enabled"] else ("2FA pending" if u["role"] == "admin" else "")
                login = "password+firebase" if u["has_password"] else "firebase only"
                print(f"{u['email']:<36} {u['role']:<10} {login:<18} {state:<9} {second}")
            return 0

        user = store.find(args.email)
        if not user:
            raise AuthError(f"No account for {args.email}.")
        if args.cmd == "set-role":
            store.update_user(user["user_id"], role=args.role)
            store.audit("user_updated", None, f"{user['email']}: role -> {args.role} (command line)")
            print(f"{user['email']} is now {args.role}.")
        elif args.cmd == "reset-password":
            store.set_password(user["user_id"], _ask_password())
            store.audit("password_reset", None, f"{user['email']} (command line)")
            print(f"Password set for {user['email']}.")
        else:
            store.disable_totp(user["user_id"])
            store.audit("2fa_reset", None, f"{user['email']} (command line)")
            print(f"Authenticator removed for {user['email']}; they set it up again at the next sign-in.")
    except AuthError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
