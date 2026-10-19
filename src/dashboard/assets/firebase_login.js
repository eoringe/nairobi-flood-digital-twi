/*
 * Firebase sign-in in the browser: "Continue with Google" and passwordless
 * email links.
 *
 * Both end the same way: Firebase gives this page an ID token, which is sent to
 * /auth/firebase, where the server verifies it, matches the account by email
 * and starts the session (administrators then enter an authenticator code).
 * The Firebase SDK is loaded only when one of these is used.
 *
 * Email link: the user enters an address, Firebase emails a link back to
 * /login, and opening it completes the sign-in. The address is remembered in
 * this browser so the link can be finished without retyping it; opened on
 * another device, the page asks for the address again (Firebase's guard against
 * a forwarded link signing in the wrong person).
 */
(function () {
    "use strict";
    var SDK = "https://www.gstatic.com/firebasejs/10.12.2/";
    var EMAIL_KEY = "twinEmailForSignIn";
    var busy = false;
    var sdk = null;

    function say(id, msg, ok) {
        var el = document.getElementById(id);
        if (el) {
            el.textContent = msg;
            el.className = (ok ? "twin-auth-ok" : "twin-auth-error") + " mt-2";
        }
    }

    async function firebaseAuth() {
        if (sdk) { return sdk; }
        // Config and both SDK modules download in parallel, not one after another.
        var loaded = await Promise.all([
            fetch("/auth/firebase-config", { credentials: "same-origin" }).then(function (r) { return r.json(); }),
            import(SDK + "firebase-app.js"),
            import(SDK + "firebase-auth.js")
        ]);
        var config = loaded[0], appMod = loaded[1], authMod = loaded[2];
        if (!config.projectId) { throw new Error("Google and email-link sign-in are not configured."); }
        var app = appMod.getApps().length ? appMod.getApp() : appMod.initializeApp(config);
        sdk = { mod: authMod, auth: authMod.getAuth(app) };
        return sdk;
    }

    // Shown the moment a sign-in link is opened, before Dash draws anything, so the
    // sign-in form never flashes up while the link is being completed.
    function showSigningIn() {
        if (document.getElementById("twin-signing-in")) { return; }
        var el = document.createElement("div");
        el.id = "twin-signing-in";
        el.setAttribute("role", "status");
        el.setAttribute("aria-live", "polite");
        el.innerHTML = '<div class="twin-signing-card"><div class="twin-signing-spinner"></div>' +
            '<div class="twin-signing-title">Signing you in…</div>' +
            '<div class="twin-signing-sub">Confirming your email link</div></div>';
        (document.body || document.documentElement).appendChild(el);
    }

    function hideSigningIn() {
        var el = document.getElementById("twin-signing-in");
        if (el) { el.remove(); }
    }

    async function finish(result) {
        var idToken = await result.user.getIdToken();
        // The server keeps its own session; the Firebase one is not needed.
        await sdk.mod.signOut(sdk.auth);
        var resp = await fetch("/auth/firebase", {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ idToken: idToken })
        });
        var data = await resp.json();
        if (!resp.ok || !data.ok) { throw new Error(data.error || "Sign-in failed."); }
        window.location.assign(data.next || "/");
    }

    async function google(button) {
        if (busy) { return; }
        busy = true; button.disabled = true; say("google-error", "", false);
        try {
            var f = await firebaseAuth();
            var provider = new f.mod.GoogleAuthProvider();
            provider.setCustomParameters({ prompt: "select_account" });
            await finish(await f.mod.signInWithPopup(f.auth, provider));
        } catch (err) {
            say("google-error", (err && err.code === "auth/popup-closed-by-user")
                ? "The Google window was closed before sign-in finished."
                : (err && err.message) || "Google sign-in failed.", false);
        } finally {
            busy = false; button.disabled = false;
        }
    }

    async function sendLink(button) {
        if (busy) { return; }
        var input = document.getElementById("f-link-email");
        var email = (input && input.value || "").trim();
        if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
            say("link-status", "Enter a valid email address.", false);
            return;
        }
        busy = true; button.disabled = true; say("link-status", "Sending…", true);
        try {
            var f = await firebaseAuth();
            await f.mod.sendSignInLinkToEmail(f.auth, email, {
                url: window.location.origin + "/login",
                handleCodeInApp: true
            });
            try { window.localStorage.setItem(EMAIL_KEY, email); } catch (e) { /* private mode */ }
            say("link-status", "Check your inbox at " + email + " and open the link in this browser.", true);
        } catch (err) {
            say("link-status", (err && err.message) || "The sign-in link could not be sent.", false);
        } finally {
            busy = false; button.disabled = false;
        }
    }

    // Plain words for what went wrong. Firebase errors carry a code; our server's
    // refusals (for example a disabled account) arrive as a readable message.
    function linkErrorMessage(err) {
        var code = (err && err.code) || "";
        if (code === "auth/invalid-action-code" || code === "auth/expired-action-code") {
            return "This sign-in link has expired or has already been used. Request a new one below.";
        }
        if (code === "auth/invalid-email") {
            return "That email address does not match the one the link was sent to. Request a new link below.";
        }
        if (code === "auth/network-request-failed") {
            return "Could not reach Google to finish signing in. Check your connection, then request a new link.";
        }
        if (code.indexOf("auth/") === 0) {
            return "The sign-in link could not be completed. Request a new link below.";
        }
        return (err && err.message) || "The sign-in link could not be completed. Request a new link below.";
    }

    async function completeLinkIfPresent() {
        var href = window.location.href;
        if (href.indexOf("oobCode=") < 0 || href.indexOf("mode=signIn") < 0) { return; }
        showSigningIn();
        try {
            var f = await firebaseAuth();
            if (!f.mod.isSignInWithEmailLink(f.auth, href)) { return; }
            var email = null;
            try { email = window.localStorage.getItem(EMAIL_KEY); } catch (e) { /* private mode */ }
            if (!email) {
                email = window.prompt("Confirm the email address this sign-in link was sent to:");
            }
            if (!email) { hideSigningIn(); window.location.replace("/login"); return; }
            var result = await f.mod.signInWithEmailLink(f.auth, email.trim(), href);
            try { window.localStorage.removeItem(EMAIL_KEY); } catch (e) { /* private mode */ }
            await finish(result);
        } catch (err) {
            hideSigningIn();
            var msg = linkErrorMessage(err);
            // The page may not be rendered yet; wait briefly for the status box.
            var tries = 0;
            (function show() {
                if (document.getElementById("link-status") || tries++ > 20) {
                    var spinner = document.querySelector(".twin-signing-inline");
                    if (spinner) { spinner.style.display = "none"; }
                    say("link-status", msg, false);
                    if (!document.getElementById("link-status")) { window.alert(msg); }
                } else { setTimeout(show, 250); }
            })();
        }
    }

    // Dash renders the login page after this script loads, so listen at the document.
    document.addEventListener("click", function (ev) {
        var g = ev.target.closest && ev.target.closest("#btn-google");
        if (g) { ev.preventDefault(); google(g); return; }
        var l = ev.target.closest && ev.target.closest("#btn-email-link");
        if (l) { ev.preventDefault(); sendLink(l); }
    });
    completeLinkIfPresent();
})();
