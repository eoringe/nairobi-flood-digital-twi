/*
 * Firebase sign-in in the browser: "Continue with Google" and passwordless
 * email links.
 *
 * Both end the same way: Firebase gives this page an ID token, which is sent to
 * /auth/firebase, where the server verifies it, matches the account by email
 * and starts the session (administrators then enter an authenticator code).
 *
 * Google popup: browsers allow a popup only when it opens within the click
 * itself. So the SDK is preloaded as soon as the sign-in page appears, and the
 * click opens the popup straight away, with nothing awaited first. If a popup is
 * still blocked (strict settings, some in-app and mobile browsers), sign-in falls
 * back to a full-page redirect to Google and back, completed on return.
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
    var REDIRECT_KEY = "twinGoogleRedirect";
    var busy = false;
    var sdk = null;
    var loading = null;

    function say(id, msg, ok) {
        var el = document.getElementById(id);
        if (el) {
            el.textContent = msg;
            el.className = (ok ? "twin-auth-ok" : "twin-auth-error") + " mt-2";
        }
    }

    function firebaseAuth() {
        if (sdk) { return Promise.resolve(sdk); }
        if (loading) { return loading; }
        // Config and both SDK modules download in parallel, not one after another.
        loading = Promise.all([
            fetch("/auth/firebase-config", { credentials: "same-origin" }).then(function (r) { return r.json(); }),
            import(SDK + "firebase-app.js"),
            import(SDK + "firebase-auth.js")
        ]).then(function (loaded) {
            var config = loaded[0], appMod = loaded[1], authMod = loaded[2];
            if (!config.projectId) { throw new Error("Google and email-link sign-in are not configured."); }
            var app = appMod.getApps().length ? appMod.getApp() : appMod.initializeApp(config);
            sdk = { mod: authMod, auth: authMod.getAuth(app) };
            return sdk;
        }).catch(function (err) { loading = null; throw err; });
        return loading;
    }

    // Covers the page while a sign-in completes, so the form never flashes up.
    function showSigningIn(subtitle) {
        var el = document.getElementById("twin-signing-in");
        if (!el) {
            el = document.createElement("div");
            el.id = "twin-signing-in";
            el.setAttribute("role", "status");
            el.setAttribute("aria-live", "polite");
            el.innerHTML = '<div class="twin-signing-card"><div class="twin-signing-spinner"></div>' +
                '<div class="twin-signing-title">Signing you in…</div>' +
                '<div class="twin-signing-sub"></div></div>';
            (document.body || document.documentElement).appendChild(el);
        }
        var sub = el.querySelector(".twin-signing-sub");
        if (sub && subtitle) { sub.textContent = subtitle; }
    }

    // Shows a spinner and a status label on the clicked button until it is reset.
    function setLoading(button, label) {
        if (!button.dataset.idleHtml) { button.dataset.idleHtml = button.innerHTML; }
        button.disabled = true;
        button.setAttribute("aria-busy", "true");
        button.innerHTML = '<span class="twin-btn-spinner" aria-hidden="true"></span>';
        button.appendChild(document.createTextNode(label));
    }

    function resetButton(button) {
        if (button.dataset.idleHtml) { button.innerHTML = button.dataset.idleHtml; delete button.dataset.idleHtml; }
        button.disabled = false;
        button.removeAttribute("aria-busy");
    }

    function hideSigningIn() {
        var el = document.getElementById("twin-signing-in");
        if (el) { el.remove(); }
    }

    // Shows a message once the status box exists (the page may still be rendering).
    function sayWhenReady(id, msg) {
        var tries = 0;
        (function show() {
            if (document.getElementById(id) || tries++ > 20) {
                var spinner = document.querySelector(".twin-signing-inline");
                if (spinner) { spinner.style.display = "none"; }
                say(id, msg, false);
                if (!document.getElementById(id)) { window.alert(msg); }
            } else { setTimeout(show, 250); }
        })();
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

    function googleErrorMessage(err) {
        var code = (err && err.code) || "";
        if (code === "auth/popup-closed-by-user" || code === "auth/cancelled-popup-request") {
            return "The Google window was closed before sign-in finished.";
        }
        if (code === "auth/unauthorized-domain") {
            return "This site (" + window.location.hostname + ") is not yet allowed to use Google sign-in. " +
                "The administrator must add it in Firebase under Authentication → Settings → Authorized domains.";
        }
        if (code === "auth/network-request-failed") {
            return "Could not reach Google. Check your connection and try again.";
        }
        if (code.indexOf("auth/") === 0) { return "Google sign-in failed. Please try again."; }
        return (err && err.message) || "Google sign-in failed.";
    }

    function provider(f) {
        var p = new f.mod.GoogleAuthProvider();
        p.setCustomParameters({ prompt: "select_account" });
        return p;
    }

    function redirectToGoogle(f) {
        try { window.sessionStorage.setItem(REDIRECT_KEY, "1"); } catch (e) { /* storage blocked */ }
        showSigningIn("Opening Google…");
        return f.mod.signInWithRedirect(f.auth, provider(f));
    }

    async function google(button) {
        if (busy) { return; }
        busy = true; say("google-error", "", false);
        var leaving = false;   // the page is about to change, so keep the spinner showing
        try {
            var f, result;
            if (sdk) {
                // Ready: open the popup now, inside the click, before anything is awaited.
                f = sdk;
                var popup = f.mod.signInWithPopup(f.auth, provider(f));
                setLoading(button, "Waiting for Google…");
                result = await popup;
            } else {
                // Still loading (very slow connection): the click's permission to open a
                // popup is lost while we wait, so go to Google by redirect instead.
                setLoading(button, "Opening Google…");
                f = await firebaseAuth();
                leaving = true;
                await redirectToGoogle(f);
                return;
            }
            setLoading(button, "Signing you in…");
            await finish(result);
            leaving = true;
        } catch (err) {
            if (err && err.code === "auth/popup-blocked" && sdk) {
                leaving = true;
                await redirectToGoogle(sdk);
                return;
            }
            leaving = false;
            hideSigningIn();
            say("google-error", googleErrorMessage(err), false);
        } finally {
            busy = false;
            if (!leaving) { resetButton(button); }
        }
    }

    async function completeRedirectIfPending() {
        var pending = false;
        try { pending = window.sessionStorage.getItem(REDIRECT_KEY) === "1"; } catch (e) { /* storage blocked */ }
        if (!pending) { return; }
        showSigningIn("Finishing Google sign-in");
        try {
            var f = await firebaseAuth();
            var result = await f.mod.getRedirectResult(f.auth);
            try { window.sessionStorage.removeItem(REDIRECT_KEY); } catch (e) { /* storage blocked */ }
            if (!result) { hideSigningIn(); return; }          // came back without signing in
            await finish(result);
        } catch (err) {
            try { window.sessionStorage.removeItem(REDIRECT_KEY); } catch (e) { /* storage blocked */ }
            hideSigningIn();
            sayWhenReady("google-error", googleErrorMessage(err));
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
        busy = true; setLoading(button, "Sending…"); say("link-status", "", true);
        try {
            var f = await firebaseAuth();
            await f.mod.sendSignInLinkToEmail(f.auth, email, {
                url: window.location.origin + "/login",
                handleCodeInApp: true
            });
            try { window.localStorage.setItem(EMAIL_KEY, email); } catch (e) { /* private mode */ }
            say("link-status", "Check your inbox at " + email + " and open the link in this browser.", true);
        } catch (err) {
            say("link-status", (err && err.code === "auth/unauthorized-continue-uri")
                ? googleErrorMessage({ code: "auth/unauthorized-domain" })
                : (err && err.message) || "The sign-in link could not be sent.", false);
        } finally {
            busy = false; resetButton(button);
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
        showSigningIn("Confirming your email link");
        try {
            var f = await firebaseAuth();
            if (!f.mod.isSignInWithEmailLink(f.auth, href)) { hideSigningIn(); return; }
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
            sayWhenReady("link-status", linkErrorMessage(err));
        }
    }

    // Preload the SDK the moment the sign-in page appears (Dash renders it after
    // this script runs, and in-app navigation shows it without a page load).
    function preloadWhenSignInShows() {
        function check() {
            if (document.getElementById("btn-google")) {
                firebaseAuth().catch(function () { /* reported when the button is used */ });
                return true;
            }
            return false;
        }
        if (check()) { return; }
        var obs = new MutationObserver(function () { if (check()) { obs.disconnect(); } });
        obs.observe(document.documentElement, { childList: true, subtree: true });
    }

    // Dash renders the login page after this script loads, so listen at the document.
    document.addEventListener("click", function (ev) {
        var g = ev.target.closest && ev.target.closest("#btn-google");
        if (g) { ev.preventDefault(); google(g); return; }
        var l = ev.target.closest && ev.target.closest("#btn-email-link");
        if (l) { ev.preventDefault(); sendLink(l); }
    });
    preloadWhenSignInShows();
    completeLinkIfPresent();
    completeRedirectIfPending();
})();
