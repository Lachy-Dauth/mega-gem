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
    // WebSocket + reconnection
    ws: null,
    wsClosedByUser: false,
    reconnectAttempts: 0,
    reconnectTimer: null,
    // Chat history (kept across lobby → game → reconnects)
    chatMessages: [],
};

// Max exponential backoff cap, in ms.
const RECONNECT_MAX_DELAY_MS = 16000;

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

// Mission category SVG icons (use currentColor so CSS controls hue)
const MISSION_ICONS = {
    pendant: `<svg viewBox="0 0 20 20" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M7 2Q10 0 13 2"/><polygon points="10,18 4,8 8,5 12,5 16,8"/><line x1="4" y1="8" x2="16" y2="8"/><line x1="8" y1="5" x2="10" y2="18"/><line x1="12" y1="5" x2="10" y2="18"/></svg>`,
    crown: `<svg viewBox="0 0 20 20" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M3 15L3 7L7 10L10 4L13 10L17 7L17 15Z"/><line x1="3" y1="17" x2="17" y2="17"/></svg>`,
    shield: `<svg viewBox="0 0 20 20" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M10 18Q4 14 3 8L3 3L10 5L17 3L17 8Q16 14 10 18Z"/><path d="M10 5L10 14" opacity="0.5"/></svg>`,
};

function missionDescription(name) {
    const idx = name.indexOf(": ");
    return idx >= 0 ? name.slice(idx + 2) : name;
}

function showScreen(id) {
    for (const s of document.querySelectorAll(".screen")) {
        s.hidden = s.id !== id;
    }
    // Leave button only shows when you're actually in a room flow.
    const inRoom = id === "screen-lobby" || id === "screen-game" || id === "screen-end";
    $("leave").hidden = !inRoom;
    // Chat is visible whenever the player is in a room (lobby + game).
    // The end screen also shows it so post-game trash talk still works.
    $("chat-panel").hidden = !inRoom;
    // Only show the connection pill while there's an active WS.
    $("conn-status").hidden = !inRoom;

    // On game screen, move chat into the 3-column game layout;
    // on other screens, move it back to <main> so it appears below content.
    const chatPanel = $("chat-panel");
    const gameLayout = document.querySelector(".game-layout");
    if (id === "screen-game" && gameLayout) {
        gameLayout.appendChild(chatPanel);
    } else if (chatPanel.parentElement !== $("app")) {
        $("app").appendChild(chatPanel);
    }
    // Toggle wider max-width on <main> for the 3-column desktop layout.
    $("app").classList.toggle("game-active", id === "screen-game");
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
    state.chatMessages = [];
    renderChat();
    if (state.reconnectTimer) {
        clearTimeout(state.reconnectTimer);
        state.reconnectTimer = null;
    }
    state.reconnectAttempts = 0;
    setConnStatus("offline", "offline");
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

async function onQuickPlay() {
    const name = $("menu-name-quick").value.trim();
    if (!name) return toast("Enter your name first", "error");
    const numPlayers = Number($("menu-num-players").value);
    const aiKind = $("menu-ai-kind").value;
    const chart = $("menu-chart-quick").value;
    const seedRaw = $("menu-seed-quick").value;
    const seed = seedRaw === "" ? null : Number(seedRaw);
    try {
        const data = await api("POST", "/api/rooms/quick_play", {
            host_name: name,
            num_players: numPlayers,
            ai_kind: aiKind,
            chart,
            seed,
        });
        state.room = data.room;
        state.roomCode = data.room.code;
        state.playerId = data.you.player_id;
        state.slotIndex = data.you.slot_index;
        state.isHost = data.you.is_host;
        saveSession();
        showScreen("screen-game");
        connectWebSocket();
    } catch (e) {
        toast(e.message, "error");
    }
}

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
        if (slot.index === state.room.host_slot_index) {
            left.appendChild(el("span", "slot-tag host", "host"));
        }
        li.appendChild(left);
        if (state.isHost && slot.index !== state.room.host_slot_index) {
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
            target_slot_index: slot.index,
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
    if (!state.roomCode || !state.playerId) return;
    if (state.reconnectTimer) {
        clearTimeout(state.reconnectTimer);
        state.reconnectTimer = null;
    }
    if (state.ws) {
        try { state.ws.close(); } catch {}
    }
    state.wsClosedByUser = false;

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/api/ws/${state.roomCode}?player_id=${encodeURIComponent(state.playerId)}`;
    setConnStatus(
        state.reconnectAttempts > 0 ? "reconnecting" : "connecting",
        state.reconnectAttempts > 0 ? "reconnecting…" : "connecting…",
    );
    let ws;
    try {
        ws = new WebSocket(url);
    } catch (e) {
        console.error("ws construct failed", e);
        scheduleReconnect();
        return;
    }
    state.ws = ws;

    ws.addEventListener("open", () => {
        const wasReconnect = state.reconnectAttempts > 0;
        state.reconnectAttempts = 0;
        setConnStatus("connected", "connected");
        if (wasReconnect) {
            toast("Reconnected");
            appendChatSystem("Reconnected to room.");
        }
    });
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
        if (state.wsClosedByUser) {
            setConnStatus("offline", "offline");
            return;
        }
        if (!state.roomCode) {
            setConnStatus("offline", "offline");
            return;
        }
        scheduleReconnect();
    });
    ws.addEventListener("error", () => {
        // "error" always precedes "close"; let the close handler drive
        // the reconnect. Just surface the first failure as a toast so
        // the user sees *something*.
        if (state.reconnectAttempts === 0) {
            toast("Connection lost", "error");
        }
    });
}

function scheduleReconnect() {
    if (!state.roomCode || !state.playerId) return;
    if (state.wsClosedByUser) return;
    state.reconnectAttempts += 1;
    const delay = Math.min(
        1000 * Math.pow(2, state.reconnectAttempts - 1),
        RECONNECT_MAX_DELAY_MS,
    );
    const secs = Math.round(delay / 1000);
    setConnStatus("reconnecting", `reconnecting in ${secs}s…`);
    if (state.reconnectAttempts === 1) {
        appendChatSystem("Connection lost — reconnecting…");
    }
    state.reconnectTimer = setTimeout(() => {
        state.reconnectTimer = null;
        connectWebSocket();
    }, delay);
}

function setConnStatus(cls, text) {
    const el = $("conn-status");
    if (!el) return;
    el.className = "conn-status " + cls;
    el.textContent = text;
}

function handleServerMessage(msg) {
    switch (msg.type) {
        case "welcome":
            state.room = msg.room;
            state.slotIndex = msg.slot_index;
            renderLobby();
            // On a mid-game reconnect the server won't re-send game_start,
            // so detect the playing/done states and flip the screen now.
            if (msg.room.status === "playing") {
                showScreen("screen-game");
            } else if (msg.room.status === "lobby") {
                showScreen("screen-lobby");
            }
            break;
        case "lobby_update":
            state.room = msg.room;
            renderLobby();
            break;
        case "game_start":
            showScreen("screen-game");
            $("g-log").innerHTML = "";
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
            appendChatMessage(msg);
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
        const li = el("li", `mission-item mission-${m.category}`);
        const iconSpan = el("span", "mission-icon");
        iconSpan.innerHTML = MISSION_ICONS[m.category] || "";
        li.appendChild(iconSpan);
        const textSpan = el("span", "mission-text");
        textSpan.appendChild(el("span", "mission-label", m.category));
        textSpan.appendChild(el("span", "mission-desc", missionDescription(m.name)));
        li.appendChild(textSpan);
        li.appendChild(el("span", "mission-coins", String(m.coins)));
        missions.appendChild(li);
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
        const missionsSpan = el("span", "player-missions");
        missionsSpan.appendChild(document.createTextNode(`missions: ${p.completed_missions.length} `));
        for (const cm of p.completed_missions) {
            const mIcon = el("span", `mission-icon-sm mission-${cm.category}`);
            mIcon.innerHTML = MISSION_ICONS[cm.category] || "";
            missionsSpan.appendChild(mIcon);
        }
        sub.appendChild(missionsSpan);
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
        logEntry(`  ★ ${p ? p.name : "seat " + comp.player_idx} completed: ${comp.mission.name}`, `mission-log-${comp.mission.category}`);
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
// Chat
// ---------------------------------------------------------------------------

function onChatSubmit(event) {
    event.preventDefault();
    const input = $("chat-input");
    const text = input.value.trim().slice(0, 200);
    if (!text) return;
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
        toast("Not connected — can't send chat", "error");
        return;
    }
    state.ws.send(JSON.stringify({ type: "chat", text }));
    input.value = "";
}

function appendChatMessage(msg) {
    state.chatMessages.push({
        kind: "user",
        from: msg.from,
        fromIdx: msg.from_idx,
        text: msg.text,
        ts: Date.now(),
    });
    // Cap history to avoid unbounded memory growth on long sessions.
    if (state.chatMessages.length > 200) state.chatMessages.shift();
    renderChat();
}

function appendChatSystem(text) {
    state.chatMessages.push({ kind: "system", text, ts: Date.now() });
    if (state.chatMessages.length > 200) state.chatMessages.shift();
    renderChat();
}

function renderChat() {
    const container = $("chat-messages");
    if (!container) return;
    container.innerHTML = "";
    if (state.chatMessages.length === 0) {
        container.appendChild(el("div", "chat-empty", "No messages yet."));
        return;
    }
    for (const m of state.chatMessages) {
        if (m.kind === "system") {
            container.appendChild(el("div", "chat-entry system", m.text));
            continue;
        }
        const isSelf = m.fromIdx === state.slotIndex;
        const entry = el("div", "chat-entry" + (isSelf ? " self" : ""));
        entry.appendChild(el("span", "chat-from", `${m.from}:`));
        entry.appendChild(document.createTextNode(m.text));
        container.appendChild(entry);
    }
    container.scrollTop = container.scrollHeight;
}

// ---------------------------------------------------------------------------
// Leaderboards
// ---------------------------------------------------------------------------

const AI_KIND_LABELS = {
    random: "Random",
    heuristic: "Heuristic",
    evolved: "EvolvedSplit",
    evo2: "Evo2",
    evo3: "Evo3",
};

function aiLabel(kind) {
    return AI_KIND_LABELS[kind] || kind;
}

async function openLeaderboard() {
    showScreen("screen-leaderboard");
    await refreshLeaderboard();
}

async function refreshLeaderboard() {
    const tablesEl = $("leaderboard-tables");
    const statsEl = $("leaderboard-stats");
    tablesEl.innerHTML = "";
    tablesEl.appendChild(el("p", "muted", "Loading…"));
    try {
        const data = await api("GET", "/api/leaderboard");
        renderLeaderboards(data);
        const s = data.stats || {};
        statsEl.textContent = `Total games recorded: ${s.total_games ?? 0} (with humans: ${s.games_with_humans ?? 0})`;
    } catch (e) {
        tablesEl.innerHTML = "";
        tablesEl.appendChild(el("p", "muted", `Failed to load leaderboards: ${e.message}`));
        statsEl.textContent = "";
    }
}

function renderLeaderboards(data) {
    const tablesEl = $("leaderboard-tables");
    tablesEl.innerHTML = "";
    const boards = data.leaderboards || {};
    for (const n of ["3", "4", "5"]) {
        const section = el("div", "leaderboard-section");
        section.appendChild(el("h3", "", `${n}-player games`));
        const rows = boards[n] || [];
        if (rows.length === 0) {
            section.appendChild(el("div", "leaderboard-empty", "No games recorded yet."));
        } else {
            section.appendChild(buildLeaderboardTable(rows));
        }
        tablesEl.appendChild(section);
    }
}

function buildLeaderboardTable(rows) {
    const table = el("table", "leaderboard-table");
    const thead = el("thead");
    const headRow = el("tr");
    for (const h of ["#", "AI", "Win rate", "W/G"]) {
        headRow.appendChild(el("th", "", h));
    }
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = el("tbody");
    rows.forEach((row, i) => {
        const tr = el("tr", `rank-${i + 1}`);
        tr.appendChild(el("td", "", String(i + 1)));
        tr.appendChild(el("td", "ai-name", aiLabel(row.ai_kind)));
        const pct = (row.win_rate * 100).toFixed(1) + "%";
        tr.appendChild(el("td", "win-rate", pct));
        tr.appendChild(el("td", "games", `${row.wins}/${row.games_played}`));
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
}

function backFromLeaderboard() {
    // Return to wherever the player came from. If they're in a room
    // with an active WS, go back to lobby/game; otherwise the menu.
    if (state.ws && state.roomCode) {
        if (state.gameState) {
            showScreen("screen-game");
        } else {
            showScreen("screen-lobby");
        }
    } else {
        showScreen("screen-menu");
    }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function leaveRoom() {
    state.wsClosedByUser = true;
    if (state.reconnectTimer) {
        clearTimeout(state.reconnectTimer);
        state.reconnectTimer = null;
    }
    if (state.ws) {
        try { state.ws.close(); } catch {}
    }
    clearSession();
    showScreen("screen-menu");
}

function wireUp() {
    $("menu-quick-play").onclick = onQuickPlay;
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
    $("chat-form").addEventListener("submit", onChatSubmit);
    $("topbar-leaderboard").onclick = openLeaderboard;
    $("leaderboard-refresh").onclick = refreshLeaderboard;
    $("leaderboard-back").onclick = backFromLeaderboard;
    $("leave").onclick = leaveRoom;
    $("end-back").onclick = leaveRoom;
    window.addEventListener("beforeunload", () => {
        // Don't trigger reconnect loops on a real navigation away.
        state.wsClosedByUser = true;
    });

    // Auto-rejoin if a previous session is still valid.
    if (restoreSession()) {
        api("GET", `/api/rooms/${state.roomCode}`)
            .then((data) => {
                state.room = data.room;
                renderLobby();
                // If the game is already playing, welcome handler will
                // re-send a state snapshot and flip us to screen-game;
                // start on lobby for the lobby case.
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
