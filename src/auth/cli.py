"""
src.auth.cli
============
Account administration from the command line, for the first administrator and
for recovering access.

    python -m src.auth.cli create-user --email alice@example.com --role admin
    python -m src.auth.cli set-role --email bob@example.com --role planner
    python -m src.auth.cli reset-2fa --email alice@example.com
    python -m src.auth.cli list

Accounts have no passwords: people sign in with Google or an emailed link for
their address. reset-2fa is for an administrator who has lost both the phone
and the recovery codes: they set up the authenticator again at the next
sign-in. In Docker, prefix with `docker compose exec dashboard`.
"""

from __future__ import annotations

import argparse
import sys

from src.auth.store import ROLES, AuthError, get_store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.auth.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-user")
    c.add_argument("--email", required=True)
    c.add_argument("--role", choices=ROLES, default="citizen")
    r = sub.add_parser("set-role")
    r.add_argument("--email", required=True)
    r.add_argument("--role", choices=ROLES, required=True)
    sub.add_parser("reset-2fa").add_argument("--email", required=True)
    sub.add_parser("list")
    args = ap.parse_args(argv)

    store = get_store()
    try:
        if args.cmd == "create-user":
            user = store.create_user(args.email, args.role)
            store.audit("user_created", None, f"{user['email']} as {user['role']} (command line)")
            print(f"Created {user['email']} ({user['role']}).")
            if user["role"] == "admin":
                print("They will set up an authenticator app at their first sign-in.")
            return 0
        if args.cmd == "list":
            for u in store.list_users():
                state = "active" if u["active"] else "disabled"
                second = "2FA on" if u["totp_enabled"] else ("2FA pending" if u["role"] == "admin" else "")
                print(f"{u['email']:<36} {u['role']:<10} {state:<9} {second}")
            return 0

        user = store.find(args.email)
        if not user:
            raise AuthError(f"No account for {args.email}.")
        if args.cmd == "set-role":
            store.update_user(user["user_id"], role=args.role)
            store.audit("user_updated", None, f"{user['email']}: role -> {args.role} (command line)")
            print(f"{user['email']} is now {args.role}.")
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
