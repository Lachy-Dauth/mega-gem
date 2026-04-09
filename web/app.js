// MegaGem multiplayer client.
//
// Thin SPA: REST to create/join rooms, WebSocket for realtime game
// events. No framework — just vanilla DOM manipulation grouped into
// a few small modules.

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
    // Identity (persisted in sessionStorage so a page reload keeps your seat)
    playerId: null,
    slotIndex: null,
    roomCode: null,
    isHost: false,
    // Lobby / game data
    room: null,
    gameState: null,
    currentAuction: null,
    currentMaxBid: null,
    waitingOnBid: false,
    waitingOnReveal: false,
    // WebSocket
    ws: null,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
};

function showScreen(id) {
    for (const s of document.querySelectorAll(".screen")) {
        s.hidden = s.id !== id;
    }
    $("leave").hidden = id === "screen-menu";
}

function toast(message, kind = "info") {
    const t = $("toast");
    t.textContent = message;
    t.className = "toast" + (kind === "error" ? " error" : "");
    t.hidden = false;
    setTimeout(() => (t.hidden = true), kind === "error" ? 5000 : 2500);
}

async function api(method, path, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
        opts.headers["Content-Type"] = "application/json";
        opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        const msg = data.detail || `${method} ${path} failed`;
        throw new Error(msg);
    }
    return data;
}

function saveSession() {
    sessionStorage.setItem(
        "megagem.session",
        JSON.stringify({
            playerId: state.playerId,
            roomCode: state.roomCode,
            slotIndex: state.slotIndex,
            isHost: state.isHost,
        }),
    );
}

function clearSession() {
    sessionStorage.removeItem("megagem.session");
    state.playerId = null;
    state.slotIndex = null;
    state.roomCode = null;
    state.isHost = false;
    state.room = null;
    state.gameState = null;
    state.currentAuction = null;
}

function restoreSession() {
    try {
        const raw = sessionStorage.getItem("megagem.session");
        if (!raw) return false;
        const data = JSON.parse(raw);
        if (!data.playerId || !data.roomCode) return false;
        state.playerId = data.playerId;
        state.roomCode = data.roomCode;
        state.slotIndex = data.slotIndex ?? null;
        state.isHost = !!data.isHost;
        return true;
    } catch {
        return false;
    }
}

// ---------------------------------------------------------------------------
// Menu screen
// ---------------------------------------------------------------------------

async function onCreateRoom() {
    const name = $("menu-name").value.trim();
    if (!name) return toast("Enter your name first", "error");
    const chart = $("menu-chart").value;
    const seedRaw = $("menu-seed").value;
    const seed = seedRaw === "" ? null : Number(seedRaw);
    try {
        const data = await api("POST", "/api/rooms", {
            host_name: name,
            chart,
            seed,
        });
        applyJoinResponse(data);
    } catch (e) {
        toast(e.message, "error");
    }
}

async function onJoinRoom() {
    const name = $("menu-name-join").value.trim();
    const code = $("menu-code").value.trim().toUpperCase();
    if (!name) return toast("Enter your name first", "error");
    if (!code) return toast("Enter a room code", "error");
    try {
        const data = await api("POST", `/api/rooms/${code}/join`, { name });
        applyJoinResponse(data);
    } catch (e) {
        toast(e.message, "error");
    }
}

function applyJoinResponse(data) {
    state.room = data.room;
    state.roomCode = data.room.code;
    state.playerId = data.you.player_id;
    state.slotIndex = data.you.slot_index;
    state.isHost = data.you.is_host;
    saveSession();
    renderLobby();
    showScreen("screen-lobby");
    connectWebSocket();
}

// ---------------------------------------------------------------------------
// Lobby screen
// ---------------------------------------------------------------------------

function renderLobby() {
    if (!state.room) return;
    $("lobby-code").textContent = state.room.code;
    $("lobby-count").textContent = state.room.slots.length;

    const chartSel = $("lobby-chart");
    chartSel.value = state.room.chart;
    chartSel.disabled = !state.isHost;

    const seedInput = $("lobby-seed");
    seedInput.value = state.room.seed ?? "";
    seedInput.disabled = !state.isHost;

    const list = $("lobby-slots");
    list.innerHTML = "";
    for (const slot of state.room.slots) {
        const li = el("li");
        const left = el("div");
        left.appendChild(el("span", "slot-name", slot.name));
        const tag = el("span", "slot-tag " + slot.kind, slot.kind === "ai" ? `AI: ${slot.ai_kind}` : "human");
        left.appendChild(tag);
        if (slot.index === 0 || state.room.host_player_id === slot.player_id) {
            // Only the server knows the host player_id; but the first
            // slot is always the host in the current flow.
        }
        if (slot.index === 0) {
            left.appendChild(el("span", "slot-tag host", "host"));
        }
        li.appendChild(left);
        if (state.isHost && slot.index !== 0) {
            const btn = el("button", "ghost", "Kick");
            btn.onclick = () => onKickSlot(slot);
            li.appendChild(btn);
        }
        list.appendChild(li);
    }

    $("host-controls").hidden = !state.isHost;
    const canStart = state.isHost && state.room.slots.length >= state.room.min_players;
    $("lobby-start").hidden = !state.isHost;
    $("lobby-start").disabled = !canStart;
    $("lobby-status").textContent = state.isHost
        ? canStart
            ? "Ready when you are."
            : `Need ${state.room.min_players - state.room.slots.length} more player(s) or AI seat(s).`
        : "Waiting for the host to start the game.";
}

async function onAddAI() {
    const kind = $("lobby-ai-kind").value;
    try {
        await api("POST", `/api/rooms/${state.roomCode}/add_ai`, {
            player_id: state.playerId,
            ai_kind: kind,
        });
    } catch (e) {
        toast(e.message, "error");
    }
}

async function onKickSlot(slot) {
    try {
        await api("POST", `/api/rooms/${state.roomCode}/remove_slot`, {
            player_id: state.playerId,
            target_player_id: slot.player_id || `ai-slot-${slot.index}`,
        });
    } catch (e) {
        toast(e.message, "error");
    }
}

async function onConfigureChart(value) {
    try {
        await api("POST", `/api/rooms/${state.roomCode}/configure`, {
            player_id: state.playerId,
            chart: value,
        });
    } catch (e) {
        toast(e.message, "error");
    }
}

async function onConfigureSeed(value) {
    const seed = value === "" ? null : Number(value);
    try {
        await api("POST", `/api/rooms/${state.roomCode}/configure`, {
            player_id: state.playerId,
            seed,
        });
    } catch (e) {
        toast(e.message, "error");
    }
}

async function onStartGame() {
    try {
        await api("POST", `/api/rooms/${state.roomCode}/start`, {
            player_id: state.playerId,
        });
    } catch (e) {
        toast(e.message, "error");
    }
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function connectWebSocket() {
    if (state.ws) {
        try { state.ws.close(); } catch {}
    }
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/api/ws/${state.roomCode}?player_id=${encodeURIComponent(state.playerId)}`;
    const ws = new WebSocket(url);
    state.ws = ws;

    ws.addEventListener("message", (evt) => {
        try {
            const msg = JSON.parse(evt.data);
            handleServerMessage(msg);
        } catch (e) {
            console.error("bad ws message", e);
        }
    });
    ws.addEventListener("close", () => {
        if (state.ws === ws) state.ws = null;
    });
    ws.addEventListener("error", () => {
        toast("WebSocket error", "error");
    });
}

function handleServerMessage(msg) {
    switch (msg.type) {
        case "welcome":
            state.room = msg.room;
            state.slotIndex = msg.slot_index;
            renderLobby();
            break;
        case "lobby_update":
            state.room = msg.room;
            renderLobby();
            break;
        case "game_start":
            showScreen("screen-game");
            logEntry(`Game started — chart ${msg.chart}, ${msg.num_players} players`);
            break;
        case "state":
            state.gameState = msg.state;
            renderGame();
            break;
        case "round_start":
            state.currentAuction = msg.auction;
            logEntry(`— Round ${msg.round}: ${describeAuction(msg.auction)} —`);
            $("g-auction").textContent = describeAuction(msg.auction);
            $("g-revealed").replaceChildren(...msg.revealed_gems.map(gemEl));
            setWaiting("Waiting for bids…");
            break;
        case "request_bid":
            state.waitingOnBid = true;
            state.currentMaxBid = msg.max_bid;
            $("g-bid-panel").hidden = false;
            $("g-bid-hint").textContent = `You have ${msg.my_coins} coins. Max legal bid: ${msg.max_bid}.`;
            $("g-bid-input").max = msg.max_bid;
            $("g-bid-input").value = Math.min(state.currentMaxBid, Number($("g-bid-input").value) || 0);
            $("g-bid-input").focus();
            $("g-bid-input").select();
            setWaiting("");
            break;
        case "round_end":
            state.waitingOnBid = false;
            $("g-bid-panel").hidden = true;
            renderRoundResult(msg.result);
            setWaiting("Waiting for next round…");
            break;
        case "request_reveal":
            state.waitingOnReveal = true;
            $("g-reveal-panel").hidden = false;
            setWaiting("You won — pick a gem from your hand to reveal.");
            renderGame();  // re-render so the hand is clickable
            break;
        case "game_end":
            renderScores(msg.scores);
            showScreen("screen-end");
            break;
        case "session_cancelled":
            toast("Session cancelled", "error");
            break;
        case "chat":
            logEntry(`${msg.from}: ${msg.text}`);
            break;
        case "error":
            toast(msg.message || "Server error", "error");
            break;
        case "pong":
            break;
        default:
            console.log("unhandled", msg);
    }
}

// ---------------------------------------------------------------------------
// Game rendering
// ---------------------------------------------------------------------------

function describeAuction(auction) {
    if (!auction) return "—";
    if (auction.kind === "treasure") return `Treasure (${auction.gems} gem${auction.gems !== 1 ? "s" : ""})`;
    if (auction.kind === "loan") return `Loan (+${auction.amount} coins)`;
    if (auction.kind === "invest") return `Invest (${auction.amount} coins)`;
    return "?";
}

function gemEl(gem) {
    return el("span", `gem ${gem.color}`, gem.color[0]);
}

function gemPillCount(color, count) {
    const pill = el("span", `gem ${color}`, String(count));
    return pill;
}

function renderGame() {
    const s = state.gameState;
    if (!s) return;
    $("g-round").textContent = s.round_number;
    $("g-chart").textContent = s.value_chart;
    $("g-auction-left").textContent = s.auction_deck_count;
    $("g-gem-left").textContent = s.gem_deck_count;
    $("g-revealed").replaceChildren(...s.revealed_gems.map(gemEl));

    // Value display (colors → counts)
    $("g-display").replaceChildren(
        ...Object.entries(s.value_display).map(([c, n]) => gemPillCount(c, n)),
    );

    // Missions
    const missions = $("g-missions");
    missions.innerHTML = "";
    for (const m of s.active_missions) {
        missions.appendChild(el("li", "", `${m.name} (${m.coins})`));
    }

    // Players
    const players = $("g-players");
    players.innerHTML = "";
    s.players.forEach((p, idx) => {
        const li = el("li");
        if (idx === s.last_winner_idx) li.classList.add("winner");
        if (idx === state.slotIndex) li.classList.add("self");
        const top = el("div", "player-top");
        top.appendChild(el("span", "", `${p.name}${p.is_human ? "" : " (AI)"}`));
        top.appendChild(el("span", "", `${p.coins}c`));
        li.appendChild(top);
        const sub = el("div", "player-sub");
        sub.appendChild(el("span", "", `hand: ${p.hand_size}`));
        sub.appendChild(el("span", "", `missions: ${p.completed_missions.length}`));
        const collection = el("span", "");
        const pieces = Object.entries(p.collection).map(([c, n]) => `${c[0]}×${n}`);
        collection.textContent = pieces.length ? "gems: " + pieces.join(" ") : "gems: none";
        sub.appendChild(collection);
        li.appendChild(sub);
        players.appendChild(li);
    });

    // Your seat
    const you = s.players[state.slotIndex];
    if (you) {
        $("g-you-name").textContent = you.name;
        $("g-you-coins").textContent = you.coins;
        $("g-you-hand-count").textContent = you.hand ? you.hand.length : you.hand_size;
        const handDiv = $("g-you-hand");
        handDiv.innerHTML = "";
        if (you.hand) {
            you.hand.forEach((gem) => {
                const node = gemEl(gem);
                if (state.waitingOnReveal) {
                    node.classList.add("clickable");
                    node.onclick = () => submitReveal(gem.color);
                }
                handDiv.appendChild(node);
            });
        }
        const collectionDiv = $("g-you-collection");
        collectionDiv.replaceChildren(
            ...Object.entries(you.collection).map(([c, n]) => gemPillCount(c, n)),
        );
    }
}

function renderRoundResult(result) {
    const winner = state.gameState?.players[result.winner_idx];
    const winnerName = winner ? winner.name : `seat ${result.winner_idx}`;
    const bids = result.bids.join(", ");
    const entry = `R${result.round}: ${describeAuction(result.auction)} — bids [${bids}] — winner: ${winnerName} (${result.winning_bid})`;
    logEntry(entry, "winner");
    if (result.taken_gems.length) {
        logEntry(`  → ${winnerName} took ${result.taken_gems.map((g) => g.color).join(", ")}`);
    }
    if (result.revealed_gem) {
        logEntry(`  → ${winnerName} revealed a ${result.revealed_gem.color}`);
    }
    for (const comp of result.completed_missions) {
        const p = state.gameState?.players[comp.player_idx];
        logEntry(`  ★ ${p ? p.name : "seat " + comp.player_idx} completed: ${comp.mission.name}`);
    }
}

function renderScores(scores) {
    const list = $("end-scores");
    const sorted = scores
        .map((s, i) => ({ ...s, idx: i }))
        .sort((a, b) => b.total - a.total);
    list.innerHTML = "";
    for (const row of sorted) {
        const li = el(
            "li",
            "",
            `${row.name} — ${row.total} (${row.coins}c + ${row.gem_value}gems + ${row.mission_value}miss + ${row.invest_returns}inv − ${row.loans_total}loan)`,
        );
        list.appendChild(li);
    }
}

function logEntry(text, cls) {
    const log = $("g-log");
    const entry = el("div", "entry" + (cls ? " " + cls : ""), text);
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

function setWaiting(text) {
    $("g-waiting").textContent = text;
}

// ---------------------------------------------------------------------------
// Player actions
// ---------------------------------------------------------------------------

function submitBid() {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    if (!state.waitingOnBid) return;
    let amount = Number($("g-bid-input").value);
    if (!Number.isFinite(amount) || amount < 0) amount = 0;
    if (state.currentMaxBid != null && amount > state.currentMaxBid) amount = state.currentMaxBid;
    state.ws.send(JSON.stringify({ type: "bid", amount }));
    state.waitingOnBid = false;
    $("g-bid-panel").hidden = true;
    setWaiting("Bid submitted. Waiting for other players…");
}

function submitReveal(color) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    if (!state.waitingOnReveal) return;
    state.ws.send(JSON.stringify({ type: "reveal", color }));
    state.waitingOnReveal = false;
    $("g-reveal-panel").hidden = true;
    setWaiting("Revealed. Resolving round…");
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function wireUp() {
    $("menu-create").onclick = onCreateRoom;
    $("menu-join").onclick = onJoinRoom;
    $("lobby-add-ai").onclick = onAddAI;
    $("lobby-start").onclick = onStartGame;
    $("lobby-chart").onchange = (e) => onConfigureChart(e.target.value);
    $("lobby-seed").onchange = (e) => onConfigureSeed(e.target.value);
    $("g-bid-submit").onclick = submitBid;
    $("g-bid-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter") submitBid();
    });
    $("leave").onclick = () => {
        if (state.ws) try { state.ws.close(); } catch {}
        clearSession();
        showScreen("screen-menu");
    };
    $("end-back").onclick = () => {
        if (state.ws) try { state.ws.close(); } catch {}
        clearSession();
        showScreen("screen-menu");
    };

    // Auto-rejoin if a previous session is still valid.
    if (restoreSession()) {
        api("GET", `/api/rooms/${state.roomCode}`)
            .then((data) => {
                state.room = data.room;
                renderLobby();
                showScreen("screen-lobby");
                connectWebSocket();
            })
            .catch(() => {
                clearSession();
                showScreen("screen-menu");
            });
    } else {
        showScreen("screen-menu");
    }
}

document.addEventListener("DOMContentLoaded", wireUp);
