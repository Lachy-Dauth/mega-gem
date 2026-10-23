// MegaGem static client.
//
// All game logic runs in engine-worker.js (Pyodide + the canonical
// research/megagem package). This file only renders the views that
// megagem.browser.BrowserSession returns and sends back bids/reveals.

const COLORS = ["Blue", "Green", "Pink", "Purple", "Yellow"];
const COLOR_HEX = { Blue: "#4f9bff", Green: "#33d68b", Pink: "#ff5fae", Purple: "#a875ff", Yellow: "#ffcc3d" };
const TOTAL_AUCTIONS = 25;

const CHARTS = {
  A: { values: [0, 4, 8, 12, 16, 20], label: "Rising" },
  B: { values: [20, 16, 12, 8, 4, 0], label: "Falling" },
  C: { values: [0, 2, 5, 9, 14, 20], label: "Slow climb" },
  D: { values: [20, 18, 15, 11, 6, 0], label: "Slow fall" },
  E: { values: [0, 4, 10, 18, 6, 0], label: "Peaks at 3" },
};

// Strongest first. Keys match megagem.browser.AI_FACTORIES.
const BOTS = [
  { kind: "search", label: "Search", tier: "Champion" },
  { kind: "evo4", label: "Evo4", tier: "Expert" },
  { kind: "evo3", label: "Evo3", tier: "Very hard" },
  { kind: "evo2", label: "Evo2", tier: "Hard" },
  { kind: "evolved", label: "Evo1", tier: "Tough" },
  { kind: "heuristic", label: "Heuristic", tier: "Medium" },
  { kind: "random", label: "Random", tier: "Very easy" },
];
const BOT_BY_KIND = Object.fromEntries(BOTS.map((b) => [b.kind, b]));
const BOT_NAMES = ["Ruby", "Onyx", "Jade", "Opal", "Topaz", "Garnet"];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --------------------------------------------------------------------------
// Gem art: one cut per colour so colours stay distinguishable without hue.
// --------------------------------------------------------------------------

const ngon = (n, r, rot = -Math.PI / 2, cx = 16, cy = 16) =>
  Array.from({ length: n }, (_, i) => [cx + r * Math.cos(rot + (2 * Math.PI * i) / n), cy + r * Math.sin(rot + (2 * Math.PI * i) / n)]);

const GEM_SHAPES = {
  Blue: ngon(5, 14.5),
  Green: [[10, 3], [22, 3], [28, 9], [28, 23], [22, 29], [10, 29], [4, 23], [4, 9]],
  Pink: ngon(10, 14, -Math.PI / 2 + Math.PI / 10),
  Purple: ngon(6, 14.5),
  Yellow: [[16, 3], [29.5, 27], [2.5, 27]],
};

const gemCache = {};
function gem(color, size = "") {
  const key = color + size;
  if (gemCache[key]) return gemCache[key];
  const pts = GEM_SHAPES[color];
  const hex = COLOR_HEX[color];
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const inner = pts.map(([x, y]) => [cx + (x - cx) * 0.5, cy + (y - cy) * 0.5]);
  const f = (arr) => arr.map((p) => p.map((v) => v.toFixed(2)).join(",")).join(" ");
  let facets = "";
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    // Light from the top-left: shade each crown facet by how much its
    // outward direction (centre → edge midpoint) faces the light.
    const nx = (a[0] + b[0]) / 2 - cx, ny = (a[1] + b[1]) / 2 - cy;
    const light = (nx * -0.6 + ny * -0.8) / (Math.hypot(nx, ny) || 1);
    const fill = light > 0 ? "#fff" : "#000";
    const op = Math.min(0.42, Math.abs(light) * 0.4).toFixed(2);
    facets += `<polygon points="${f([a, b, inner[(i + 1) % pts.length], inner[i]])}" fill="${fill}" opacity="${op}"/>`;
  }
  const svg =
    `<svg class="gem ${size}" viewBox="0 0 32 32" role="img" aria-label="${color}"><title>${color}</title>` +
    `<polygon points="${f(pts)}" fill="${hex}"/>${facets}` +
    `<polygon points="${f(inner)}" fill="#fff" opacity="0.18"/>` +
    `<polygon points="${f(pts)}" fill="none" stroke="#fff" stroke-opacity="0.45" stroke-width="0.8" stroke-linejoin="round"/></svg>`;
  return (gemCache[key] = svg);
}

const coin = (n) => `<span class="coin">${n}</span>`;

function gemCounts(counts, size = "sm") {
  const parts = COLORS.filter((c) => counts[c] > 0).map((c) =>
    counts[c] > 1 ? `<span class="gem-stack">${gem(c, size)}×${counts[c]}</span>` : gem(c, size));
  return parts.length ? parts.join("") : '<span class="empty">none</span>';
}

function gemList(colors, size = "") {
  return colors.length ? colors.map((c) => gem(c, size)).join("") : '<span class="empty">none</span>';
}

const countColors = (colors) => colors.reduce((m, c) => ((m[c] = (m[c] || 0) + 1), m), {});

// --------------------------------------------------------------------------
// Engine worker wrapper
// --------------------------------------------------------------------------

class Engine {
  constructor() {
    const pageParams = new URLSearchParams(location.search);
    const workerParams = new URLSearchParams({ v: window.MEGAGEM_BUILD || "dev" });
    if (pageParams.get("pyodide")) workerParams.set("pyodide", pageParams.get("pyodide"));
    this.worker = new Worker(`engine-worker.js?${workerParams}`);
    this.pending = new Map();
    this.nextId = 1;
    this.status = "loading";
    this.onStatus = () => {};
    this.worker.onmessage = (e) => this._onMessage(e.data);
    this.worker.onerror = (e) => this._setStatus("error", e.message || "Worker failed to start.");
  }
  _setStatus(status, detail) {
    this.status = status;
    this.detail = detail;
    this.onStatus(status, detail);
  }
  _onMessage(msg) {
    if (msg.type === "progress") return this._setStatus("loading", msg.stage);
    if (msg.type === "ready") return this._setStatus("ready", "Python engine ready");
    if (msg.type === "fatal") return this._setStatus("error", msg.error);
    const p = this.pending.get(msg.id);
    if (!p) return;
    this.pending.delete(msg.id);
    msg.ok ? p.resolve(msg.result) : p.reject(new Error(msg.error));
  }
  call(cmd, args = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, cmd, args });
    });
  }
}

// --------------------------------------------------------------------------
// Preferences (per-browser convenience only)
// --------------------------------------------------------------------------

const PREFS_KEY = "megagem.prefs.v2";
const defaultPrefs = { mode: "play", players: 4, name: "", chart: "A", explain: false, bots: ["evo4", "evo4", "evo4", "evo4", "evo4"] };

function loadPrefs() {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (raw) return { ...defaultPrefs, ...JSON.parse(raw) };
  } catch (_) { /* storage unavailable */ }
  return { ...defaultPrefs };
}
function savePrefs() {
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch (_) { /* ignore */ }
}

// --------------------------------------------------------------------------
// App state
// --------------------------------------------------------------------------

const engine = new Engine();
const prefs = loadPrefs();
let game = null; // { config, view, token, busy, auto, autoTimer }
let gameToken = 0;

function showScreen(name) {
  for (const s of ["setup", "game", "end"]) $(`screen-${s}`).hidden = s !== name;
  $("nav-quit").hidden = name !== "game";
  window.scrollTo({ top: 0 });
}

let toastTimer = null;
function hideToast() {
  clearTimeout(toastTimer);
  $("toast").hidden = true;
}
function toast(html, { error = false, ms = 3200 } = {}) {
  const el = $("toast");
  el.innerHTML = html;
  el.classList.toggle("error", error);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), ms);
}

// --------------------------------------------------------------------------
// Setup screen
// --------------------------------------------------------------------------

function seg(id, value, onChange) {
  const el = $(id);
  for (const b of el.querySelectorAll("button")) {
    b.setAttribute("aria-pressed", String(b.dataset.value === String(value)));
    b.onclick = () => onChange(b.dataset.value);
  }
}

function botOptions(selected) {
  return BOTS.map((b) => `<option value="${b.kind}"${b.kind === selected ? " selected" : ""}>${b.label} · ${b.tier}</option>`).join("");
}

function renderSetup() {
  seg("mode-seg", prefs.mode, (v) => { prefs.mode = v; savePrefs(); renderSetup(); });
  seg("players-seg", prefs.players, (v) => { prefs.players = Number(v); savePrefs(); renderSetup(); });
  $("name-field").hidden = prefs.mode !== "play";
  $("setup-name").value = prefs.name;
  $("setup-explain").checked = prefs.explain;

  const rows = [];
  let botIdx = 0;
  for (let i = 0; i < prefs.players; i++) {
    if (i === 0 && prefs.mode === "play") {
      rows.push(`<div class="seat you"><span class="seat-no">${i + 1}</span>
        <span class="seat-name">${esc(prefs.name || "You")}<small>Human: that's you</small></span><span></span></div>`);
      continue;
    }
    const kind = prefs.bots[botIdx] || "evo4";
    rows.push(`<div class="seat"><span class="seat-no">${i + 1}</span>
      <span class="seat-name">${BOT_NAMES[botIdx]}<small>Bot seat</small></span>
      <select data-bot="${botIdx}" aria-label="Bot for seat ${i + 1}">${botOptions(kind)}</select></div>`);
    botIdx++;
  }
  $("seat-list").innerHTML = rows.join("");
  for (const sel of $("seat-list").querySelectorAll("select")) {
    sel.onchange = () => { prefs.bots[Number(sel.dataset.bot)] = sel.value; savePrefs(); };
  }

  $("setup-all").innerHTML = `<option value="">Set all bots to…</option>` + BOTS.map((b) => `<option value="${b.kind}">${b.label}</option>`).join("");

  $("chart-picker").innerHTML = Object.entries(CHARTS).map(([key, c]) => {
    const bars = c.values.map((v) => `<i style="height:${Math.max(6, (v / 20) * 100)}%"></i>`).join("");
    return `<button type="button" class="chart-opt" data-chart="${key}" aria-pressed="${key === prefs.chart}" title="Chart ${key}: ${c.values.join(", ")}">
      <b>${key}</b><span class="mini-bars">${bars}</span><small>${c.label}</small></button>`;
  }).join("");
  for (const b of $("chart-picker").querySelectorAll("button")) {
    b.onclick = () => { prefs.chart = b.dataset.chart; savePrefs(); renderSetup(); };
  }
}

function setupConfig() {
  const seats = [];
  let botIdx = 0;
  for (let i = 0; i < prefs.players; i++) {
    if (i === 0 && prefs.mode === "play") {
      seats.push({ kind: "human", name: (prefs.name || "You").slice(0, 20) });
    } else {
      const kind = prefs.bots[botIdx] || "evo4";
      seats.push({ kind, name: BOT_NAMES[botIdx] });
      botIdx++;
    }
  }
  const seedRaw = $("setup-seed").value.trim();
  return { seats, chart: prefs.chart, explain: prefs.explain, seed: seedRaw === "" ? null : Math.abs(parseInt(seedRaw, 10)) || 0 };
}

function wireSetup() {
  $("setup-name").addEventListener("input", (e) => {
    prefs.name = e.target.value;
    savePrefs();
    const you = document.querySelector(".seat.you .seat-name");
    if (you) you.firstChild.textContent = prefs.name || "You";
  });
  $("setup-explain").addEventListener("change", (e) => { prefs.explain = e.target.checked; savePrefs(); });
  $("setup-all").addEventListener("change", (e) => {
    if (!e.target.value) return;
    prefs.bots = prefs.bots.map(() => e.target.value);
    savePrefs();
    renderSetup();
  });
  $("setup-form").addEventListener("submit", (e) => {
    e.preventDefault();
    startGame(setupConfig());
  });
  $("hero-gems").innerHTML = COLORS.map((c) => gem(c, "lg")).join("");
}

function renderEngineStatus(status, detail) {
  const el = $("engine-status");
  el.className = `engine-status ${status}`;
  el.querySelector("span").textContent = detail || "";
  $("setup-start").textContent = status === "ready" ? "Deal the cards" : status === "error" ? "Engine failed to load" : "Deal the cards (loading…)";
  $("setup-start").disabled = status === "error";
}

// --------------------------------------------------------------------------
// Game flow
// --------------------------------------------------------------------------

async function startGame(config) {
  stopAuto();
  const token = ++gameToken;
  game = { config, view: null, token, busy: true, auto: false, autoTimer: null, log: [] };
  hideToast();
  $("g-log").innerHTML = "";
  showScreen("game");
  renderLoadingTable();
  try {
    const view = await engine.call("new", config);
    if (token !== gameToken) return;
    game.config.seed = view.seed; // pin the seed so Rematch replays this exact deal
    handleView(view);
  } catch (err) {
    if (token !== gameToken) return;
    toast(`Couldn't start the game: ${esc(err.message)}`, { error: true, ms: 8000 });
    showScreen("setup");
  }
}

async function act(cmd, args) {
  if (!game || game.busy) return;
  const token = game.token;
  game.busy = true;
  renderAction();
  try {
    const view = await engine.call(cmd, args);
    if (token !== gameToken) return;
    handleView(view);
  } catch (err) {
    if (token !== gameToken) return;
    game.busy = false;
    stopAuto();
    renderAction();
    toast(`Engine error: ${esc(err.message)}`, { error: true, ms: 8000 });
  }
}

function handleView(view) {
  game.view = view;
  game.busy = false;
  if (view.last_round) {
    game.log.push(view.last_round);
    appendLog(view.last_round, view);
    if (view.human_idx !== null) toast(roundSummary(view.last_round, view), { ms: 4200 });
  }
  renderGame();
  if (view.phase === "game_over") {
    stopAuto();
    setTimeout(() => { if (game && game.view === view) renderEnd(); }, view.human_idx === null ? 600 : 1400);
  } else if (view.phase === "ready" && game.auto) {
    scheduleAuto();
  } else if (view.phase === "bid") {
    requestAnimationFrame(() => { const inp = $("bid-num"); if (inp && window.matchMedia("(pointer:fine)").matches) inp.select(); });
  }
}

function quitGame() {
  if (game && game.view && game.view.phase !== "game_over" && !confirm("Abandon this game?")) return;
  stopAuto();
  gameToken++;
  game = null;
  showScreen("setup");
}

// Spectator autoplay ------------------------------------------------------

function scheduleAuto() {
  clearTimeout(game.autoTimer);
  game.autoTimer = setTimeout(() => act("step"), Number($("sp-speed").value));
}
function stopAuto() {
  if (!game) return;
  game.auto = false;
  clearTimeout(game.autoTimer);
  $("sp-auto").innerHTML = 'Auto-play <span class="kbd">Space</span>';
}
function toggleAuto() {
  if (!game || game.view?.phase !== "ready" && !game.busy) return;
  if (game.auto) return stopAuto();
  game.auto = true;
  $("sp-auto").innerHTML = 'Pause <span class="kbd">Space</span>';
  if (!game.busy) act("step");
}

// --------------------------------------------------------------------------
// Rendering helpers
// --------------------------------------------------------------------------

const chartValue = (chart, n) => CHARTS[chart].values[Math.min(Math.max(n, 0), 5)];

function seatLabel(view, i) {
  const kind = view.seat_kinds[i];
  return kind === "human" ? "You" : (BOT_BY_KIND[kind]?.label || kind);
}

function scoreNow(state, ps) {
  const gems = COLORS.reduce((s, c) => s + (ps.collection[c] || 0) * chartValue(state.value_chart, state.value_display[c] || 0), 0);
  const missions = ps.completed_missions.reduce((s, m) => s + m.coins, 0);
  const loans = ps.loans.reduce((s, x) => s + x, 0);
  const invest = ps.investments.reduce((s, x) => s + x.amount + x.locked, 0);
  return ps.coins + gems + missions - loans + invest;
}

function auctionText(a) {
  if (!a) return "";
  if (a.kind === "treasure") return `Treasure ×${a.gems}`;
  if (a.kind === "loan") return `Loan ${a.amount}`;
  return `Invest ${a.amount}`;
}

function roundSummary(r, view) {
  const name = esc(view.state.players[r.winner_idx].name);
  const you = r.winner_idx === view.human_idx;
  const gems = r.taken_gems.length ? ` and takes ${r.taken_gems.map((c) => gem(c, "sm")).join("")}` : "";
  const missions = r.completed_missions.map((m) => `<br>${esc(view.state.players[m.player_idx].name)} completes <b>${esc(missionText(m.mission.name))}</b> (+${m.mission.coins})`).join("");
  return `<b>Round ${r.round}:</b> ${you ? "You win" : `${name} wins`} ${auctionText(r.auction)} for ${coin(r.winning_bid)}${gems}.${missions}`;
}

// Mission text + a UI-only "how close am I" hint. The engine decides
// completion; this just mirrors the requirement shapes in missions.py.
const missionText = (name) => name.replace(/^[A-Za-z]+:\s*/, "");

function missionReq(name) {
  const text = missionText(name);
  const specific = [...text.matchAll(/(\d+) (Blue|Green|Pink|Purple|Yellow)/g)];
  if (specific.length) return specific.map(([, n, c]) => `${gem(c, "sm")}${Number(n) > 1 ? `×${n}` : ""}`).join('<span class="muted">+</span>');
  return esc(text);
}

function missionNeed(name, collection) {
  const text = missionText(name);
  const counts = COLORS.map((c) => collection[c] || 0);
  let m;
  const specific = [...text.matchAll(/(\d+) (Blue|Green|Pink|Purple|Yellow)/g)];
  if (specific.length) return specific.reduce((s, [, n, c]) => s + Math.max(0, Number(n) - (collection[c] || 0)), 0);
  if ((m = text.match(/^(\d+) different colors/))) return Math.max(0, Number(m[1]) - counts.filter((n) => n > 0).length);
  if ((m = text.match(/^(\d+) pairs/))) return [...counts].sort((a, b) => b - a).slice(0, Number(m[1])).reduce((s, n) => s + Math.max(0, 2 - n), 0);
  if ((m = text.match(/^(\d+) of the same color/))) return Math.max(0, Number(m[1]) - Math.max(...counts));
  return null;
}

// --------------------------------------------------------------------------
// Game screen
// --------------------------------------------------------------------------

function renderLoadingTable() {
  $("g-round").textContent = "Dealing…";
  $("g-progress").style.width = "0%";
  $("g-meta").innerHTML = "";
  $("g-opponents").innerHTML = "";
  $("g-auction").className = "auction";
  $("g-auction").innerHTML = `<div class="waiting"><span class="spinner"></span><span>${esc(engine.status === "ready" ? "Shuffling…" : engine.detail || "Loading the engine…")}</span></div>`;
  $("g-market").innerHTML = "";
  $("g-chart-track").innerHTML = "";
  $("g-missions").innerHTML = "";
  $("g-you").hidden = true;
  $("g-spectate").hidden = true;
  $("g-log-count").textContent = "";
}

function renderGame() {
  const { view } = game;
  const st = view.state;
  const spectator = view.human_idx === null;
  const current = view.phase === "bid" || view.phase === "ready" ? st.round_number + 1 : st.round_number;
  $("g-round").textContent = view.phase === "game_over" ? "Final round played" : `Round ${current} of ${TOTAL_AUCTIONS}`;
  $("g-progress").style.width = `${(st.round_number / TOTAL_AUCTIONS) * 100}%`;
  $("g-meta").innerHTML = [
    `<span class="chip">Chart ${st.value_chart} · ${CHARTS[st.value_chart].label}</span>`,
    `<span class="chip">${st.auction_deck_count} auctions left</span>`,
    `<span class="chip">${st.gem_deck_count} gems in deck</span>`,
    `<span class="chip" title="Enter this seed on the setup screen to replay the same deal">Seed ${view.seed}</span>`,
  ].join("");

  renderSeats();
  renderAuction();
  renderMarket($("g-market"), st, spectator ? null : st.players[view.human_idx]);
  renderChartTrack();
  renderMissions();
  $("g-you").hidden = spectator;
  $("g-spectate").hidden = !spectator;
  if (!spectator) renderYou();
  renderAction();
  $("g-log-count").textContent = game.log.length ? `${game.log.length} played` : "";
}

function renderSeats() {
  const { view } = game;
  const st = view.state;
  const last = view.last_round || game.log[game.log.length - 1] || null;
  $("g-opponents").innerHTML = st.players.map((ps, i) => {
    const isYou = i === view.human_idx;
    const winner = last && last.winner_idx === i;
    const kind = view.seat_kinds[i];
    const bot = BOT_BY_KIND[kind];
    const hand = ps.hand
      ? `<div class="gem-row">${gemList(ps.hand, "sm")}</div>`
      : `<span class="hand-backs" title="${ps.hand_size} gems in hand">${"<i></i>".repeat(ps.hand_size)}</span>`;
    const extras = [
      ...ps.loans.map((a) => `<span class="chip loan">Loan −${a}</span>`),
      ...ps.investments.map((x) => `<span class="chip invest">Invest ${x.locked}+${x.amount}</span>`),
      ...ps.completed_missions.map((m) => `<span class="chip mission" title="${esc(m.name)}">★ +${m.coins}</span>`),
    ].join("");
    return `<article class="seat-card${winner ? " winner" : ""}${isYou ? " you" : ""}">
      ${last ? `<span class="last-bid">${winner ? "won · " : ""}bid ${last.bids[i]}</span>` : ""}
      <div class="seat-card-head">
        <div class="who"><b>${esc(ps.name)}</b>${isYou ? '<span class="chip">You</span>' : `<span class="chip kind" title="${esc(bot?.tier || "")}">${esc(bot?.label || kind)}</span>`}</div>
        ${coin(ps.coins)}
      </div>
      <div class="stats"><span>Hand</span>${hand}</div>
      <div class="stats"><span>Collected</span><span class="gem-row">${gemCounts(ps.collection)}</span></div>
      ${extras ? `<div class="stats">${extras}</div>` : ""}
      <div class="score-now" title="Coins + collected gems at today's values + missions + investments − loans (hands not counted)">Worth now: <b>${scoreNow(st, ps)}</b></div>
    </article>`;
  }).join("");
}

function renderAuction() {
  const { view } = game;
  const st = view.state;
  const el = $("g-auction");
  const a = view.auction;
  if (!a) {
    el.className = "auction";
    el.innerHTML = `<div class="auction-kind">Auction deck empty</div><div class="auction-title">Scoring…</div>`;
    return;
  }
  el.className = `auction ${a.kind}`;
  const won = view.phase === "reveal" ? `<span class="chip mission">You won this auction</span>` : "";
  let body = "";
  if (a.kind === "treasure") {
    // In the reveal phase the gems have already moved into your collection.
    const offered = view.phase === "reveal" ? [] : st.revealed_gems;
    const slots = offered.map((c, i) => `<span class="slot${i < a.gems ? "" : " off"}" title="${i < a.gems ? "Goes to the winner" : "Stays face-up"}">${gem(c, "xl")}</span>`).join("");
    const value = offered.slice(0, a.gems).reduce((s, c) => s + chartValue(st.value_chart, st.value_display[c] || 0), 0);
    body = `<div class="auction-title">Treasure ×${a.gems}</div>
      ${view.phase === "reveal" ? "" : `<div class="auction-gems">${slots || '<span class="empty">no gems left</span>'}</div>`}
      <div class="auction-sub">${view.phase === "reveal" ? "The gems are in your collection." : `Winner takes the highlighted gem${a.gems > 1 ? "s" : ""}. Worth ${coin(value)} at today's display.`}</div>`;
  } else if (a.kind === "loan") {
    body = `<div class="auction-title">Loan ${coin(a.amount)}</div>
      <div class="auction-sub">Winner pays their bid, receives ${a.amount} now and repays ${a.amount} at game end. You may bid up to your coins + ${a.amount}.</div>`;
  } else {
    body = `<div class="auction-title">Invest ${coin(a.amount)}</div>
      <div class="auction-sub">Winner's bid is locked away and paid back at game end with a +${a.amount} bonus.</div>`;
  }
  el.innerHTML = `<div class="auction-kind">${a.kind}</div>${won}${body}
    <div class="auction-deck">${st.auction_deck_count - (view.phase === "reveal" ? 0 : 1)} more auction cards after this</div>`;
}

function renderMarket(el, st, me, { final = false } = {}) {
  el.innerHTML = COLORS.map((c) => {
    const n = st.value_display[c] || 0;
    const val = chartValue(st.value_chart, n);
    const nxt = chartValue(st.value_chart, n + 1);
    const dir = nxt > val ? "up" : nxt < val ? "down" : "";
    const pips = Array.from({ length: 5 }, (_, i) => `<i class="${i < n ? "on" : ""}"></i>`).join("");
    const owned = st.players.reduce((s, p) => s + (p.collection[c] || 0), 0);
    const mine = me
      ? `hand ${(me.hand || []).filter((x) => x === c).length}<br>owned ${me.collection[c] || 0}`
      : `collected ${owned}`;
    return `<div class="mcol" style="color:${COLOR_HEX[c]}">
      ${gem(c, "lg")}
      <span class="pips" title="${n} in display">${pips}</span>
      <span class="val" style="color:var(--text)">${val}</span>
      ${final ? "" : `<span class="next ${dir}" title="Value if one more ${c} is revealed">${n >= 5 ? "maxed" : `next → ${nxt}`}</span>`}
      <span class="mine">${mine}</span>
    </div>`;
  }).join("");
}

function renderChartTrack() {
  const st = game.view.state;
  const vals = CHARTS[st.value_chart].values;
  $("g-chart-name").textContent = `· chart ${st.value_chart}`;
  $("g-chart-track").innerHTML =
    `<span class="label">in display</span>${vals.map((_, i) => `<span>${i === 5 ? "5+" : i}</span>`).join("")}` +
    `<span class="label">gem worth</span>${vals.map((v) => `<span style="color:var(--text);font-weight:600">${v}</span>`).join("")}`;
}

function renderMissions() {
  const { view } = game;
  const st = view.state;
  const me = view.human_idx === null ? null : st.players[view.human_idx];
  $("g-missions-left").textContent = `${st.active_missions.length} open`;
  if (!st.active_missions.length) {
    $("g-missions").innerHTML = '<span class="empty">All missions claimed.</span>';
    return;
  }
  $("g-missions").innerHTML = st.active_missions.map((m) => {
    const need = me ? missionNeed(m.name, me.collection) : null;
    const close = need === 1;
    const note = need === null ? "" : `<span class="progress-note">${need === 0 ? "Complete!" : `You need ${need} more gem${need > 1 ? "s" : ""}`}</span>`;
    return `<div class="mission${close ? " close" : ""}" title="${esc(m.name)}">
      <div class="mission-head"><span>${esc(m.category)}</span>${coin(m.coins)}</div>
      <div class="mission-req">${missionReq(m.name)}</div>${note}</div>`;
  }).join("");
}

function renderYou() {
  const { view } = game;
  const me = view.state.players[view.human_idx];
  $("g-you-name").textContent = me.name;
  $("g-you-score").innerHTML = `worth now <b style="color:var(--text)">${scoreNow(view.state, me)}</b>`;
  $("g-you-stats").innerHTML = `${coin(me.coins)}<span class="muted small">coins to bid with</span>`;
  $("g-you-hand").innerHTML = gemList(me.hand || [], "lg");
  $("g-you-collection").innerHTML = gemCounts(me.collection, "");
  const extras = [
    ...me.loans.map((a) => `<span class="chip loan">Loan: repay ${a}</span>`),
    ...me.investments.map((x) => `<span class="chip invest">Invest: ${x.locked} + ${x.amount} back</span>`),
    ...me.completed_missions.map((m) => `<span class="chip mission">★ ${esc(missionText(m.name))} +${m.coins}</span>`),
  ];
  $("g-you-extras").innerHTML = extras.length ? `<h4>Loans, investments, missions</h4><div class="gem-row">${extras.join("")}</div>` : "";
}

function renderAction() {
  const el = $("g-action");
  if (!game || !game.view) return;
  const { view } = game;
  el.classList.remove("live");
  if (view.human_idx === null) {
    $("sp-step").disabled = game.busy || view.phase !== "ready";
    return;
  }
  if (game.busy) {
    el.innerHTML = `<div class="waiting"><span class="spinner"></span>Bots are bidding…</div>`;
    return;
  }
  if (view.phase === "bid") return renderBid(el);
  if (view.phase === "reveal") return renderReveal(el);
  el.innerHTML = `<div class="waiting">Game over: tallying scores…</div>`;
}

function renderBid(el) {
  const { view } = game;
  const st = view.state;
  const a = view.auction;
  const max = view.max_bid;
  const me = st.players[view.human_idx];
  let hint = "";
  if (a.kind === "treasure") {
    const worth = st.revealed_gems.slice(0, a.gems).reduce((s, c) => s + chartValue(st.value_chart, st.value_display[c] || 0), 0);
    hint = `The gem${a.gems > 1 ? "s are" : " is"} worth ${coin(worth)} at today's display. Final value depends on what gets revealed.`;
  } else if (a.kind === "invest") {
    hint = `Whatever you pay comes back at the end, plus ${a.amount}. It just ties up coins.`;
  } else {
    hint = `Your bid is a cost: the ${a.amount} you receive is repaid at the end. Worth it only if you need cash now.`;
  }
  const quick = [...new Set([0, 1, Math.round(max / 4), Math.round(max / 2), max])].filter((v) => v <= max);
  el.classList.add("live");
  el.innerHTML = `<h3>Your sealed bid</h3>
    <p class="muted small" style="margin:0">${hint}</p>
    <div class="bid-amount">
      <input id="bid-num" type="number" min="0" max="${max}" value="0" inputmode="numeric" aria-label="Bid amount">
      <span class="of">of ${max} max${a.kind === "loan" ? ` (${me.coins} + loan)` : ""}</span>
    </div>
    <input id="bid-range" type="range" min="0" max="${max}" value="0" aria-label="Bid slider">
    <div class="quick">${quick.map((v) => `<button type="button" class="btn sm ghost" data-q="${v}">${v === max ? `All in (${v})` : v}</button>`).join("")}</div>
    <button id="bid-go" type="button" class="btn primary lg block">Bid 0 <span class="kbd">Enter</span></button>`;
  const num = $("bid-num"), range = $("bid-range"), go = $("bid-go");
  const set = (v) => {
    v = Math.max(0, Math.min(max, Math.round(Number(v) || 0)));
    num.value = v; range.value = v;
    go.innerHTML = `Bid ${v} <span class="kbd">Enter</span>`;
  };
  num.oninput = () => set(num.value);
  range.oninput = () => set(range.value);
  for (const b of el.querySelectorAll("[data-q]")) b.onclick = () => set(b.dataset.q);
  const submit = () => { set(num.value); act("bid", { amount: Number(num.value) }); };
  go.onclick = submit;
  num.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); submit(); } };
}

function renderReveal(el) {
  const { view } = game;
  const st = view.state;
  const me = st.players[view.human_idx];
  const options = [...new Set(me.hand)];
  el.classList.add("live");
  el.innerHTML = `<h3>You won! Reveal a gem</h3>
    <p class="muted small" style="margin:0">Pick one gem from your hand to add to the Value Display. It changes what every gem of that colour is worth, for you and for everyone else.</p>
    <div class="reveal-options">${options.map((c) => {
      const n = st.value_display[c] || 0;
      const before = chartValue(st.value_chart, n), after = chartValue(st.value_chart, n + 1);
      const owned = me.collection[c] || 0;
      const swing = (after - before) * owned;
      const cls = after > before ? "pos" : after < before ? "neg" : "";
      return `<button type="button" class="reveal-opt" data-color="${c}">${gem(c, "lg")}
        <span><b>${c}</b><br><span class="delta">worth <b>${before}</b> → <b class="${cls}">${after}</b> each · your ${owned} collected: <b class="${cls}">${swing >= 0 ? "+" : ""}${swing}</b></span></span></button>`;
    }).join("")}</div>`;
  for (const b of el.querySelectorAll("[data-color]")) b.onclick = () => act("reveal", { color: b.dataset.color });
}

function appendLog(r, view) {
  const st = view.state;
  const rationale = r.rationale
    ? `<details><summary>Bot reasoning</summary><div class="rationale">${r.rationale.map((x, i) => x
        ? `<div><b>${esc(st.players[i].name)}</b> <span class="muted small">${esc(x.ai_class)} · bid ${r.bids[i]}</span><pre>${esc(x.lines.join("\n") || "(no detail)")}</pre></div>`
        : "").join("")}</div></details>`
    : "";
  const missions = r.completed_missions.map((m) => `<div>★ ${esc(st.players[m.player_idx].name)}: ${esc(missionText(m.mission.name))} <b>+${m.mission.coins}</b></div>`).join("");
  const div = document.createElement("div");
  div.className = "log-entry fresh";
  div.innerHTML = `<div class="log-head"><span>Round ${r.round} · <b>${auctionText(r.auction)}</b></span><span><b>${esc(st.players[r.winner_idx].name)}</b> wins</span></div>
    <div class="log-bids">${r.bids.map((b, i) => `<span class="chip${i === r.winner_idx ? " win" : ""}" title="${esc(seatLabel(view, i))}">${esc(st.players[i].name)} ${b}</span>`).join("")}</div>
    <div class="gem-row">${r.taken_gems.length ? `<span class="muted small">took</span>${gemList(r.taken_gems, "sm")}` : ""}
      ${r.revealed_gem ? `<span class="muted small">revealed</span>${gem(r.revealed_gem, "sm")}` : ""}</div>
    ${missions}${rationale}`;
  const log = $("g-log");
  log.prepend(div);
}

// --------------------------------------------------------------------------
// End screen
// --------------------------------------------------------------------------

function renderEnd() {
  const { view, config } = game;
  const rows = view.scores.map((s, i) => ({ ...s, i })).sort((a, b) => b.total - a.total);
  const top = rows[0].total;
  const winners = rows.filter((r) => r.total === top);
  const youWon = winners.some((r) => r.i === view.human_idx);
  let title;
  if (view.human_idx === null) title = winners.length > 1 ? "A shared victory" : `${esc(winners[0].name)} wins`;
  else if (youWon) title = winners.length > 1 ? "You tied for the win!" : "You win!";
  else title = `${esc(winners[0].name)} wins`;
  $("end-title").innerHTML = title;
  const myRank = view.human_idx === null ? null : rows.findIndex((r) => r.i === view.human_idx) + 1;
  $("end-sub").textContent = [
    myRank && !youWon ? `You finished #${myRank} of ${rows.length}` : null,
    `Chart ${view.state.value_chart}`,
    `${rows.length} players`,
    `seed ${view.seed}`,
  ].filter(Boolean).join(" · ");
  $("end-gems").innerHTML = COLORS.map((c) => gem(c, "lg")).join("");

  let rank = 0, prev = null;
  $("end-scores").innerHTML = rows.map((r, n) => {
    if (r.total !== prev) { rank = n + 1; prev = r.total; }
    const hand = view.final_hands?.[r.i] || [];
    return `<tr class="${r.total === top ? "first" : ""}${r.i === view.human_idx ? " me" : ""}">
      <td><span class="rank">${rank}</span><b>${esc(r.name)}</b> ${r.i === view.human_idx ? "" : `<span class="chip kind">${esc(seatLabel(view, r.i))}</span>`}
        ${hand.length ? `<div class="gem-row" style="margin-top:4px"><span class="muted small">hand</span>${gemList(hand, "sm")}</div>` : ""}</td>
      <td>${r.coins}</td><td>${r.gem_value}</td><td>${r.mission_value}</td>
      <td class="${r.invest_returns ? "pos" : ""}">${r.invest_returns ? "+" + r.invest_returns : 0}</td>
      <td class="${r.loans_total ? "neg" : ""}">${r.loans_total ? "−" + r.loans_total : 0}</td>
      <td class="total">${r.total}</td></tr>`;
  }).join("");
  renderMarket($("end-market"), view.state, null, { final: true });
  hideToast();
  showScreen("end");
  $("end-rematch").onclick = () => startGame({ ...config });
  $("end-again").onclick = () => startGame({ ...config, seed: null });
}

// --------------------------------------------------------------------------
// Boot
// --------------------------------------------------------------------------

engine.onStatus = renderEngineStatus;
renderEngineStatus(engine.status, "Starting the Python engine…");
wireSetup();
renderSetup();

$("nav-quit").onclick = quitGame;
$("end-setup").onclick = () => { game = null; showScreen("setup"); };
$("sp-step").onclick = () => { stopAuto(); act("step"); };
$("sp-auto").onclick = toggleAuto;
$("sp-speed").onchange = () => { if (game?.auto && !game.busy) scheduleAuto(); };

document.addEventListener("keydown", (e) => {
  if (!game || $("screen-game").hidden || e.target.matches("input, select, textarea")) return;
  const phase = game.view?.phase;
  if (phase === "ready" || (game.auto && game.busy)) {
    if (e.key === " ") { e.preventDefault(); toggleAuto(); }
    else if (e.key.toLowerCase() === "n" && phase === "ready") { stopAuto(); act("step"); }
  } else if (phase === "bid" && e.key === "Enter" && !e.target.matches("button")) {
    e.preventDefault();
    $("bid-go")?.click();
  }
});
