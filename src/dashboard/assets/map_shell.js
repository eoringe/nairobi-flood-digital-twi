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
  var KEY = "twin-rails";
  var DESKTOP = "(min-width: 901px)";

  function load() {
    try {
      var s = JSON.parse(localStorage.getItem(KEY) || "null");
      if (s && typeof s.left === "boolean" && typeof s.right === "boolean") return s;
    } catch (e) { /* storage blocked: defaults */ }
    // First visit: both panels open on wide screens, insights folded on narrow ones.
    return { left: true, right: window.innerWidth > 1240 };
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
