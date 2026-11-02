/*
 * Map-first layout: collapsible floating panels over a full-window map.
 *
 * Buttons carry data-rail-close / data-rail-open ("left" | "right") or
 * data-rail-toggle="all". The state is kept in localStorage and applied as
 * .left-closed / .right-closed on #twin-shell. After every change, and every
 * time the map document reloads, the map is told how much of its width the
 * open panels cover, so its legends move into the clear part ("insets").
 * Phones (900 px and narrower) keep the stacked layout and ignore all of this.
 */
(function () {
  var KEY = "twin-rails-v2";
  var DESKTOP = "(min-width: 901px)";

  function load() {
    try {
      var s = JSON.parse(localStorage.getItem(KEY) || "null");
      if (s && typeof s.left === "boolean" && typeof s.right === "boolean") return s;
    } catch (e) { /* storage blocked: defaults */ }
    // First visit: the map alone; the panels open from their edge buttons.
    return { left: false, right: false };
  }
  function save(s) { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {} }

  var state = load();

  function insets() {
    if (!window.matchMedia(DESKTOP).matches) return { left: 0, right: 0 };
    var l = document.getElementById("rail-left"), r = document.getElementById("rail-right");
    return {
      left: state.left && l ? l.offsetWidth + 12 : 0,
      right: state.right && r ? r.offsetWidth + 12 : 0
    };
  }

  function postInsets() {
    var f = document.getElementById("3d-map-frame");
    if (!f || !f.contentWindow) return;
    var i = insets();
    try { f.contentWindow.postMessage({ type: "insets", left: i.left, right: i.right }, "*"); } catch (e) {}
  }

  function apply() {
    var el = document.getElementById("twin-shell");
    if (!el) return false;
    el.classList.toggle("left-closed", !state.left);
    el.classList.toggle("right-closed", !state.right);
    var focus = document.getElementById("btn-focus-map");
    if (focus) focus.title = state.left || state.right ? "Hide the panels to see the whole map" : "Show the panels";
    setTimeout(postInsets, 320);              // after the 0.28 s slide
    return true;
  }

  document.addEventListener("click", function (ev) {
    var b = ev.target.closest && ev.target.closest("[data-rail-close],[data-rail-open],[data-rail-toggle]");
    if (!b) return;
    var close = b.getAttribute("data-rail-close"), open = b.getAttribute("data-rail-open");
    if (close) state[close] = false;
    else if (open) state[open] = true;
    else {
      var anyOpen = state.left || state.right;
      state = { left: !anyOpen, right: !anyOpen };
    }
    save(state);
    apply();
  });

  // Dash builds the page after this script runs: wait for the shell, then keep
  // the insets current whenever the map document is replaced.
  var tries = 0;
  var timer = setInterval(function () {
    var f = document.getElementById("3d-map-frame");
    if ((apply() && f) || ++tries > 300) {
      clearInterval(timer);
      if (f) f.addEventListener("load", function () { setTimeout(postInsets, 500); });
    }
  }, 100);
  window.addEventListener("resize", function () {
    clearTimeout(window.__twinInsetT);
    window.__twinInsetT = setTimeout(postInsets, 200);
  });
})();

/*
 * 3D quality: "auto" (default), "off" (full effects) or "on" (lighter). The
 * choice is kept in localStorage and sent to each new map document when it
 * reports ready; the map answers with what it applied and, on auto, the frame
 * rate it measured, shown under the buttons.
 */
(function () {
  var KEY = "twin-perf";
  function mode() { try { return localStorage.getItem(KEY) || "auto"; } catch (e) { return "auto"; } }
  function send() {
    var f = document.getElementById("3d-map-frame");
    if (f && f.contentWindow) { try { f.contentWindow.postMessage({ type: "perf", mode: mode() }, "*"); } catch (e) {} }
  }
  function mark() {
    var m = mode();
    document.querySelectorAll("[data-perf]").forEach(function (b) {
      b.classList.toggle("is-active", b.getAttribute("data-perf") === m);
      b.setAttribute("aria-pressed", b.getAttribute("data-perf") === m ? "true" : "false");
    });
  }
  document.addEventListener("click", function (ev) {
    var b = ev.target.closest && ev.target.closest("[data-perf]");
    if (!b) return;
    try { localStorage.setItem(KEY, b.getAttribute("data-perf")); } catch (e) {}
    mark();
    send();
  });
  window.addEventListener("message", function (ev) {
    var msg = ev.data;
    if (!msg) return;
    if (msg.type === "perfReady") { mark(); send(); return; }
    if (msg.type !== "perfState") return;
    var note = document.getElementById("perf-note");
    if (!note) return;
    var fps = msg.fps ? " (this device drew " + msg.fps + " frames a second with full effects)" : "";
    note.textContent = msg.mode === "auto"
      ? (msg.lite
          ? (msg.reason === "moving"
              ? "Auto switched to Lighter: zooming over flooding ran at about " + msg.fps + " frames a second."
              : "Auto chose Lighter" + fps + ".")
          : "Auto kept Full 3D" + fps + ".")
      : (msg.lite ? "Lighter: still water and flat building footprints." : "Full 3D: every effect on.");
  });
  var tries = 0, t = setInterval(function () { if (document.querySelector("[data-perf]") || ++tries > 300) { clearInterval(t); mark(); } }, 100);
})();
