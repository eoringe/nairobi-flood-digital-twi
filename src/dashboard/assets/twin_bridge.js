/*
 * Parent-page half of the map bridge (the iframe half is injected by
 * src/dashboard/components/map_3d.get_deck_html_with_embedded_legend).
 *
 * Dash loads every .js file in /assets automatically. This listens for
 * messages from the map iframe and turns them into Dash property updates:
 *
 *   mapClick        - when the user armed "Pick on map", drops a start or
 *                     destination pin (map-pick-store).
 *   layerUpdateAck  - if an in-place layer patch failed, requests a full
 *                     document rebuild so stale geometry is not left on screen.
 */
(function () {
  window.twinPick = null;

  window.addEventListener("message", function (ev) {
    var m = ev.data;
    if (!m || !m.type) return;
    var dc = window.dash_clientside;
    if (!dc || typeof dc.set_props !== "function") return;

    if (m.type === "mapClick" && window.twinPick) {
      dc.set_props("map-pick-store", {
        data: { target: window.twinPick, lat: m.lat, lon: m.lon, t: Date.now() }
      });
      window.twinPick = null;
    } else if (m.type === "layerUpdateAck" && m.ok === false) {
      console.warn("Map live update failed; rebuilding the map:", m.error);
      dc.set_props("map-rebuild-request", { data: Date.now() });
    }
  });
})();
