/* ======================================================================
 * MegaGem UI controller — drives the menu, the game loop, and the score
 * screen. Talks to the engine through the global `MegaGem` object set up
 * by megagem.js.
 *
 * State machine:
 *   menu  → awaiting-bid → resolving → awaiting-reveal? → round-summary
 *           → (next round) or → game-over
 * ====================================================================== */

const M = window.MegaGem;

const $ = (id) => document.getElementById(id);

const screens = {
    menu:   $("menu-screen"),
    game:   $("game-screen"),
    scores: $("scores-screen"),
};

function showScreen(name) {
    for (const [k, el] of Object.entries(screens)) {
        if (k === name) el.removeAttribute("hidden");
        else el.setAttribute("hidden", "");
    }
    // Quit button only makes sense while a game is on-screen.
    $("quit-game").hidden = (name !== "game");
}

// ---------- Game session state ------------------------------------------

let session = null;
/* {
 *   state: GameState,
 *   humanIdx: 0,
 *   pendingBids: number[] | null,
 *   pendingAuction: AuctionCard | null,
 *   pendingWinnerIdx: number | null,
 *   phase: "bidding" | "revealing" | "summary",
 *   roundLog: { message, klass }[],   // mirrors the DOM log for replay
 * }
 */

// ---------- Persistence (localStorage) ----------------------------------
//
// The whole session is JSON-stringified into localStorage at every render
// so navigating away (e.g. clicking the Rules link) and coming back —
// or even reloading the tab — drops you straight back into the game in
// the same phase. The save survives until the player either finishes the
// game or hits the Quit button.
//
// What's tricky:
//   * Mission cards carry a `check` function — we save them by name and
//     rehydrate from a fresh `makeMissionDeck()` lookup table.
//   * The seedable rng captures its current `s` via `getState()`; the
//     deserializer feeds it back to `makeRng(seed, s)`.
//   * Each AI carries its own rng for noise / random bids — we save the
//     AI kind plus the rng state per player and rebuild on resume.

const SAVE_KEY = "mega-gem:save:v1";

function buildMissionLookup() {
    const lookup = new Map();
    for (const m of M.makeMissionDeck()) lookup.set(m.name, m);
    return lookup;
}

function aiKindOf(ai) {
    // Order matters: Evo3 and Evo2 must be checked before
    // HyperAdaptiveSplitAI would have been (they're independent classes,
    // but defensive in case someone refactors). HeuristicAI is checked
    // last among the heuristic family because AdaptiveHeuristic could
    // subclass it.
    if (ai instanceof M.Evo3AI) return "evo3";
    if (ai instanceof M.Evo2AI) return "evo2";
    if (ai instanceof M.HyperAdaptiveSplitAI) return "evolved";
    if (ai instanceof M.HeuristicAI) return "heuristic";
    if (ai instanceof M.RandomAI) return "random";
    return "random";
}

function serializeAI(ai) {
    const payload = {
        kind: aiKindOf(ai),
        rngSeed: ai.rng.seed,
        rngState: ai.rng.getState(),
    };
    // Evo3 carries per-instance learning state — the opponent-delta
    // history is the whole point of the bot. Persist it so a resumed
    // game keeps whatever Evo3 learned before the reload.
    if (ai instanceof M.Evo3AI) {
        payload.oppHistory = ai.getOppHistory();
    }
    return payload;
}

function buildAI(name, savedAI, evolvedWeights, evo2Weights, evo3Weights) {
    const rng = M.makeRng(savedAI.rngSeed, savedAI.rngState);
    if (savedAI.kind === "random")    return new M.RandomAI(name, rng);
    if (savedAI.kind === "heuristic") return new M.HeuristicAI(name, rng, { noise: true });
    if (savedAI.kind === "evo2")      return new M.Evo2AI(name, rng, evo2Weights);
    if (savedAI.kind === "evo3") {
        const ai = new M.Evo3AI(name, rng, evo3Weights);
        if (Array.isArray(savedAI.oppHistory)) ai.setOppHistory(savedAI.oppHistory);
        return ai;
    }
    return new M.HyperAdaptiveSplitAI(name, rng, evolvedWeights);
}

function serializePlayer(ps) {
    return {
        name: ps.name,
        isHuman: ps.isHuman,
        coins: ps.coins,
        hand: ps.hand.map(g => g.color),
        collection: { ...ps.collection },
        completedMissions: ps.completedMissions.map(m => m.name),
        loans: ps.loans.map(l => ({ kind: "loan", amount: l.amount })),
        investments: ps.investments.map(inv => ({
            card: { kind: "invest", amount: inv.card.amount },
            locked: inv.locked,
        })),
        ai: ps.isHuman ? null : serializeAI(ps.ai),
    };
}

function serializeSession(sess) {
    if (!sess) return null;
    const { state } = sess;
    return {
        version: 1,
        humanIdx: sess.humanIdx,
        phase: sess.phase,
        pendingAuction: sess.pendingAuction || null,
        pendingWinnerIdx: sess.pendingWinnerIdx ?? null,
        roundLog: sess.roundLog.slice(-150),
        state: {
            chart: state.chart,
            roundNumber: state.roundNumber,
            lastWinnerIdx: state.lastWinnerIdx,
            valueDisplay: { ...state.valueDisplay },
            rngSeed: state.rng.seed,
            rngState: state.rng.getState(),
            gemDeck: state.gemDeck.map(g => g.color),
            revealedGems: state.revealedGems.map(g => g.color),
            auctionDeck: state.auctionDeck.map(c => ({ ...c })),
            activeMissions: state.activeMissions.map(m => m.name),
            playerStates: state.playerStates.map(serializePlayer),
        },
    };
}

function deserializeSession(saved) {
    if (!saved || saved.version !== 1) return null;
    const ss = saved.state;
    if (!ss || !Array.isArray(ss.playerStates)) return null;

    const missionLookup = buildMissionLookup();
    const numPlayers = ss.playerStates.length;
    const evolvedWeights = M.evolvedWeightsFor(numPlayers);
    const evo2Weights = M.evo2WeightsFor(numPlayers);
    const evo3Weights = M.evo3WeightsFor(numPlayers);

    const playerStates = ss.playerStates.map((psSaved) => {
        let ai;
        if (psSaved.isHuman) {
            ai = {
                name: psSaved.name,
                isHuman: true,
                chooseBid: () => 0,
                chooseGemToReveal: () => null,
            };
        } else {
            ai = buildAI(psSaved.name, psSaved.ai, evolvedWeights, evo2Weights, evo3Weights);
        }
        return {
            name: psSaved.name,
            isHuman: !!psSaved.isHuman,
            ai,
            hand: psSaved.hand.map((color) => ({ color })),
            coins: psSaved.coins,
            collection: { ...psSaved.collection },
            completedMissions: psSaved.completedMissions
                .map((n) => missionLookup.get(n))
                .filter(Boolean),
            loans: psSaved.loans.map((l) => ({ ...l })),
            investments: psSaved.investments.map((inv) => ({
                card: { ...inv.card },
                locked: inv.locked,
            })),
        };
    });

    const state = {
        rng: M.makeRng(ss.rngSeed, ss.rngState),
        chart: ss.chart,
        playerStates,
        gemDeck: ss.gemDeck.map((color) => ({ color })),
        revealedGems: ss.revealedGems.map((color) => ({ color })),
        auctionDeck: ss.auctionDeck.map((c) => ({ ...c })),
        activeMissions: ss.activeMissions
            .map((n) => missionLookup.get(n))
            .filter(Boolean),
        valueDisplay: { ...ss.valueDisplay },
        lastWinnerIdx: ss.lastWinnerIdx,
        roundNumber: ss.roundNumber,
    };

    return {
        state,
        humanIdx: saved.humanIdx,
        pendingBids: null,
        pendingAuction: saved.pendingAuction || null,
        pendingWinnerIdx: saved.pendingWinnerIdx,
        phase: saved.phase,
        roundLog: Array.isArray(saved.roundLog) ? saved.roundLog : [],
    };
}

function persistSession() {
    if (!session) return;
    try {
        const data = serializeSession(session);
        if (data) localStorage.setItem(SAVE_KEY, JSON.stringify(data));
    } catch (e) {
        // localStorage may be unavailable (private mode, quota, file://).
        // Persistence is best-effort — never break play because of it.
    }
}

function clearPersistedSession() {
    try { localStorage.removeItem(SAVE_KEY); } catch (e) {}
}

function loadPersistedSession() {
    try {
        const raw = localStorage.getItem(SAVE_KEY);
        if (!raw) return null;
        return deserializeSession(JSON.parse(raw));
    } catch (e) {
        return null;
    }
}

// ---------- Menu --------------------------------------------------------

$("menu-start").addEventListener("click", startGame);
$("play-again").addEventListener("click", () => showScreen("menu"));
$("quit-game").addEventListener("click", () => {
    if (confirm("Quit this game and discard your saved progress?")) {
        clearPersistedSession();
        session = null;
        showScreen("menu");
    }
});

function startGame() {
    // Any prior save belongs to the game we're about to discard. Clear
    // it now so a half-finished game can't sneak back via a reload that
    // races the first render.
    clearPersistedSession();

    const name = ($("menu-name").value || "You").trim().slice(0, 14) || "You";
    const numPlayers = parseInt($("menu-players").value, 10);
    const aiKind = $("menu-ai").value;
    const chart = $("menu-chart").value;
    const seedRaw = $("menu-seed").value;
    const seed = seedRaw === "" ? null : parseInt(seedRaw, 10);

    const aiNames = ["Avery", "Blair", "Casey", "Dylan", "Elliot"];
    const masterRng = M.makeRng(seed);

    const human = {
        name,
        isHuman: true,
        chooseBid: () => 0,         // never invoked — UI handles it
        chooseGemToReveal: () => null,
    };

    const players = [human];
    const evolvedWeights = M.evolvedWeightsFor(numPlayers);
    const evo2Weights = M.evo2WeightsFor(numPlayers);
    const evo3Weights = M.evo3WeightsFor(numPlayers);
    for (let i = 0; i < numPlayers - 1; i++) {
        const aiSeed = masterRng.int(2 ** 31);
        const aiRng = M.makeRng(aiSeed);
        if (aiKind === "random") {
            players.push(new M.RandomAI(aiNames[i], aiRng));
        } else if (aiKind === "heuristic") {
            // Medium: deterministic heuristic with uniform bid noise so it
            // is occasionally beatable / surprising.
            players.push(new M.HeuristicAI(aiNames[i], aiRng, { noise: true }));
        } else if (aiKind === "evo2") {
            // Hard: clean-slate evolved AI tuned against a frozen mix of
            // previous bots. See megagem/players/evo2.py for the design notes.
            players.push(new M.Evo2AI(aiNames[i], aiRng, evo2Weights));
        } else if (aiKind === "evo3") {
            // Hardest: Evo2 + opponent-pricing awareness. Learns an
            // opponent-delta history live during the game and feeds
            // weighted (mean, std) back into every head. See
            // megagem/players/evo3.py for the design notes.
            players.push(new M.Evo3AI(aiNames[i], aiRng, evo3Weights));
        } else {
            // "evolved" — GA-tuned HyperAdaptiveSplitAI with the weight set
            // matched to this player count.
            players.push(new M.HyperAdaptiveSplitAI(aiNames[i], aiRng, evolvedWeights));
        }
    }

    const state = M.setupGame({ players, chart, seed });

    session = {
        state,
        humanIdx: 0,
        pendingBids: null,
        pendingAuction: null,
        phase: "bidding",
        roundLog: [],
    };

    clearLog();
    const aiLabel = aiKind === "random"
        ? "random"
        : aiKind === "heuristic"
            ? "heuristic+noise"
            : aiKind === "evo2"
                ? "evo2"
                : aiKind === "evo3"
                    ? "evo3"
                    : "evolved";
    log(`Game started: ${numPlayers} players, chart ${chart}, AI = ${aiLabel}.`,
        "log-round");
    showScreen("game");
    advanceToNextRound();
}

// ---------- Round flow --------------------------------------------------

function advanceToNextRound() {
    const { state } = session;
    if (M.isGameOver(state)) {
        endGame();
        return;
    }
    state.roundNumber += 1;
    const auction = state.auctionDeck.pop();
    session.pendingAuction = auction;
    session.phase = "bidding";

    renderAll();
    promptHumanBid();
}

function promptHumanBid() {
    const { state, humanIdx, pendingAuction } = session;
    const me = state.playerStates[humanIdx];
    const cap = M.maxLegalBid(me, pendingAuction);

    const desc = M.describeAuction(pendingAuction);
    const note = pendingAuction.kind === "loan"
        ? ` (you may bid up to ${cap} — borrowing into the loan is legal)`
        : "";
    setBidPrompt(`Bid on ${desc}${note}. You have ${me.coins} coins.`, true);

    const input = $("bid-input");
    input.disabled = false;
    input.min = 0;
    input.max = cap;
    input.value = 0;
    input.focus();
    input.select();

    $("bid-submit").disabled = false;
    $("bid-zero").disabled = false;
}

$("bid-submit").addEventListener("click", submitHumanBid);
$("bid-zero").addEventListener("click", () => {
    $("bid-input").value = 0;
    submitHumanBid();
});
$("bid-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") submitHumanBid();
});

function submitHumanBid() {
    if (!session || session.phase !== "bidding") return;
    const { state, humanIdx, pendingAuction } = session;
    const me = state.playerStates[humanIdx];
    const raw = $("bid-input").value;
    const bid = M.clampBid(raw, me, pendingAuction);

    // Lock out controls.
    $("bid-input").disabled = true;
    $("bid-submit").disabled = true;
    $("bid-zero").disabled = true;
    setBidPrompt(`You bid ${bid}. Resolving…`);

    // Collect AI bids deterministically.
    const allBids = state.playerStates.map((ps, i) => {
        if (i === humanIdx) return bid;
        const raw = ps.ai.chooseBid(state, ps, pendingAuction);
        return M.clampBid(raw, ps, pendingAuction);
    });

    resolveAuction(allBids);
}

function resolveAuction(allBids) {
    const { state, pendingAuction } = session;
    const winnerIdx = M.resolveWinner(
        allBids, state.lastWinnerIdx, state.playerStates.length, state.rng
    );
    const winningBid = allBids[winnerIdx];
    const winnerState = state.playerStates[winnerIdx];

    // Log the bid result.
    const bidsLine = state.playerStates
        .map((ps, i) => `${ps.name}=${allBids[i]}`)
        .join("  ");
    log(`Round ${state.roundNumber}: ${M.describeAuction(pendingAuction)}`,
        "log-round");
    log(`Bids: ${bidsLine}`, "log-detail");
    log(`${winnerState.name} wins for ${winningBid}.`);

    let taken = [];
    if (pendingAuction.kind === "treasure") {
        winnerState.coins -= winningBid;
        taken = M.applyTreasure(state, winnerState, pendingAuction);
        if (taken.length > 0) {
            log(`${winnerState.name} took ${taken.map(g => g.color).join(", ")}.`,
                "log-detail");
        }
    } else if (pendingAuction.kind === "loan") {
        M.applyLoan(winnerState, winningBid, pendingAuction);
        log(`${winnerState.name} draws ${pendingAuction.amount} coins (must repay at scoring).`,
            "log-detail");
    } else if (pendingAuction.kind === "invest") {
        M.applyInvest(winnerState, winningBid, pendingAuction);
        log(`${winnerState.name} invests ${winningBid}; payout ${pendingAuction.amount} + locked.`,
            "log-detail");
    }

    // Post-auction observation hook. Evo3 uses this to build up its
    // opponent-delta history; other AIs don't implement it. The engine
    // uses this same { auction, bids } result shape on the Python side,
    // so the JS call site mirrors that exactly.
    const observation = { auction: pendingAuction, bids: allBids };
    state.playerStates.forEach((ps, i) => {
        if (ps.isHuman || !ps.ai || typeof ps.ai.observeRound !== "function") return;
        ps.ai.observeRound(state, i, observation);
    });

    state.lastWinnerIdx = winnerIdx;
    session.pendingWinnerIdx = winnerIdx;
    // The auction is over — clear it so a save taken between resolution
    // and the next phase entry doesn't try to re-render an already-
    // resolved card with stale gem highlights.
    session.pendingAuction = null;

    // Reveal phase.
    if (winnerState.hand.length === 0) {
        finishRound(null, winnerIdx);
    } else if (winnerIdx === session.humanIdx) {
        startHumanReveal();
    } else {
        const choice = winnerState.ai.chooseGemToReveal(state, winnerState);
        finishRound(choice, winnerIdx);
    }
}

function startHumanReveal() {
    session.phase = "revealing";
    setBidPrompt("You won! Click a gem from your hand to reveal it into the Value Display.", true);
    renderAll(); // re-render so click handlers attach with phase=revealing
}

function handleHumanRevealClick(card) {
    if (!session || session.phase !== "revealing") return;
    finishRound(card, session.humanIdx);
}

function finishRound(revealedCard, winnerIdx) {
    const { state } = session;
    const winnerState = state.playerStates[winnerIdx];

    if (revealedCard) {
        // Make sure it's actually in hand (defensive).
        const idx = winnerState.hand.indexOf(revealedCard);
        const actual = idx >= 0
            ? winnerState.hand.splice(idx, 1)[0]
            : winnerState.hand.splice(0, 1)[0];
        state.valueDisplay[actual.color] += 1;
        log(`${winnerState.name} reveals a ${actual.color} into the Value Display.`,
            "log-detail");
    }

    M.replenishRevealed(state);
    const completed = M.checkMissions(state);
    for (const { playerIdx, mission } of completed) {
        const ps = state.playerStates[playerIdx];
        log(`★ ${ps.name} completes "${mission.name}" (+${mission.coins} coins).`,
            "log-mission");
    }

    session.phase = "summary";
    session.pendingAuction = null;
    renderAll();

    // Show advance button.
    setBidPrompt("Round complete. Click Continue to draw the next auction.");
    $("advance-button").hidden = false;
    $("advance-button").focus();
}

$("advance-button").addEventListener("click", () => {
    $("advance-button").hidden = true;
    advanceToNextRound();
});

// ---------- Game over ----------------------------------------------------

function endGame() {
    const { state } = session;
    const scores = M.scoreGame(state);
    renderScores(scores);
    // The game is over — drop the save so a reload doesn't try to
    // resume into an already-finished session.
    clearPersistedSession();
    session = null;
    showScreen("scores");
}

function renderScores(scores) {
    const sorted = scores
        .map((s, i) => ({ ...s, originalIdx: i }))
        .sort((a, b) => b.total - a.total);
    const tbody = document.querySelector("#scores-table tbody");
    tbody.innerHTML = "";
    sorted.forEach((s, rank) => {
        const tr = document.createElement("tr");
        if (rank === 0) tr.classList.add("winner");
        if (s.isHuman) tr.classList.add("you");
        tr.innerHTML = `
            <td>${rank + 1}</td>
            <td>${escapeHtml(s.name)}${s.isHuman ? " (you)" : ""}</td>
            <td>${s.coins}</td>
            <td>${s.gemValue}</td>
            <td>${s.missionValue}</td>
            <td>-${s.loansTotal}</td>
            <td>${s.investReturns}</td>
            <td class="total">${s.total}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ---------- Rendering helpers --------------------------------------------

function renderAll() {
    if (!session) return;
    const { state, humanIdx, pendingAuction } = session;
    const me = state.playerStates[humanIdx];

    $("round-number").textContent = state.roundNumber;
    $("chart-letter").textContent = state.chart;
    $("auction-deck-count").textContent = state.auctionDeck.length;
    $("gem-deck-count").textContent = state.gemDeck.length;

    renderAuctionCard(pendingAuction);
    renderRevealedGems(state, pendingAuction);
    renderValueDisplay(state);
    renderActiveMissions(state);
    renderOpponents(state, humanIdx);
    renderYou(me);

    // Snapshot the session at every stable render so the player can
    // navigate away (Rules link, accidental tab close, …) and come back.
    persistSession();
}

function renderAuctionCard(card) {
    const el = $("auction-card");
    el.className = "auction-card";
    if (!card) {
        el.classList.add("placeholder");
        el.textContent = "—";
        return;
    }
    el.classList.add(card.kind);
    let title = "";
    let sub = "";
    if (card.kind === "treasure") {
        title = "Treasure";
        sub = `${card.gems} gem${card.gems === 1 ? "" : "s"} from the display`;
    } else if (card.kind === "loan") {
        title = "Loan";
        sub = `+${card.amount} coins now, -${card.amount} at scoring`;
    } else if (card.kind === "invest") {
        title = "Invest";
        sub = `Locks bid; pays back +${card.amount} on top at scoring`;
    }
    el.innerHTML = `
        <div class="title">${title}</div>
        <div class="sub">${sub}</div>
    `;
}

function renderRevealedGems(state, auction) {
    const el = $("revealed-gems");
    el.innerHTML = "";
    if (state.revealedGems.length === 0) {
        el.innerHTML = `<span class="muted">(no gems on offer)</span>`;
        return;
    }
    // Engine takes from the FRONT of the revealed list (revealed_gems.shift),
    // so the first N gems are the ones the current treasure auction will
    // actually award. For loan/invest cards no gems are awarded, so nothing
    // is highlighted.
    const isTreasure = auction && auction.kind === "treasure";
    const takeCount = isTreasure
        ? Math.min(auction.gems, state.revealedGems.length)
        : 0;

    state.revealedGems.forEach((gem, i) => {
        const wrap = document.createElement("div");
        wrap.className = "revealed-slot";
        if (i < takeCount) wrap.classList.add("for-auction");
        else if (isTreasure) wrap.classList.add("idle");

        wrap.appendChild(makeGemEl(gem.color));

        const label = document.createElement("span");
        label.className = "slot-label";
        if (i < takeCount) label.textContent = "↑ this round";
        else if (isTreasure) label.textContent = "next up";
        else label.textContent = "";
        wrap.appendChild(label);

        el.appendChild(wrap);
    });
}

function renderValueDisplay(state) {
    const el = $("value-display");
    el.innerHTML = "";
    for (const color of M.COLORS) {
        const count = state.valueDisplay[color] || 0;
        const price = M.valueFor(state.chart, count);
        const cell = document.createElement("div");
        cell.className = `vd-cell color-${color.toLowerCase()}`;
        cell.innerHTML = `
            <div class="vd-label">${color}</div>
            <div class="vd-count">${count}</div>
            <div class="vd-price">${price}/gem</div>
        `;
        el.appendChild(cell);
    }
}

function renderActiveMissions(state) {
    const el = $("active-missions");
    el.innerHTML = "";
    if (state.activeMissions.length === 0) {
        el.innerHTML = `<li class="muted">(no active missions)</li>`;
        return;
    }
    for (const m of state.activeMissions) {
        const li = document.createElement("li");
        li.className = `mission-${m.category}`;
        li.innerHTML = `<span>${escapeHtml(m.name)}</span><span class="coins">${m.coins}</span>`;
        el.appendChild(li);
    }
}

function renderOpponents(state, humanIdx) {
    const el = $("opponents");
    el.innerHTML = "";
    state.playerStates.forEach((ps, i) => {
        if (i === humanIdx) return;
        const card = document.createElement("div");
        card.className = "opponent-card";
        if (state.lastWinnerIdx === i) card.classList.add("last-winner");
        const handCount = ps.hand.length;
        const collTotal = M.COLORS.reduce((s, c) => s + (ps.collection[c] || 0), 0);
        const missionsDone = ps.completedMissions.length;
        card.innerHTML = `
            <div class="opponent-name">
                ${escapeHtml(ps.name)}
                ${state.lastWinnerIdx === i ? '<span class="muted">last winner</span>' : ""}
            </div>
            <div class="opponent-stats">
                <span class="stat">💰 <strong>${ps.coins}</strong></span>
                <span class="stat">✋ <strong>${handCount}</strong></span>
                <span class="stat">💎 <strong>${collTotal}</strong></span>
                <span class="stat">★ <strong>${missionsDone}</strong></span>
            </div>
            <div class="finance-chips"></div>
            <div class="opponent-collection"></div>
        `;
        const finEl = card.querySelector(".finance-chips");
        renderFinanceChips(finEl, ps);
        const collEl = card.querySelector(".opponent-collection");
        for (const c of M.COLORS) {
            const n = ps.collection[c] || 0;
            for (let k = 0; k < n; k++) collEl.appendChild(makeGemEl(c));
        }
        el.appendChild(card);
    });
}

// Render compact loan/invest chips for an opponent card. Loans show the
// debt that will be deducted at scoring; investments show the locked bid
// + the bonus they pay back.
function renderFinanceChips(el, ps) {
    el.innerHTML = "";
    if (ps.loans.length === 0 && ps.investments.length === 0) {
        const empty = document.createElement("span");
        empty.className = "chip chip--empty";
        empty.textContent = "no loans / invests";
        el.appendChild(empty);
        return;
    }
    for (const loan of ps.loans) {
        const chip = document.createElement("span");
        chip.className = "chip chip--loan";
        chip.title = `Loan: must repay ${loan.amount} coins at scoring`;
        chip.textContent = `−${loan.amount}`;
        el.appendChild(chip);
    }
    for (const inv of ps.investments) {
        const chip = document.createElement("span");
        chip.className = "chip chip--invest";
        chip.title =
            `Invest: ${inv.locked} locked, pays back ${inv.locked} + ${inv.card.amount} bonus at scoring`;
        chip.textContent = `${inv.locked}↻+${inv.card.amount}`;
        el.appendChild(chip);
    }
}

function renderYou(me) {
    $("you-name").textContent = me.name;
    $("you-coins").textContent = me.coins;

    // Hand
    const handEl = $("your-hand");
    handEl.innerHTML = "";
    const isReveal = session && session.phase === "revealing";
    me.hand.forEach((card) => {
        const el = makeGemEl(card.color);
        if (isReveal) {
            el.classList.add("selectable");
            el.addEventListener("click", () => handleHumanRevealClick(card));
        }
        handEl.appendChild(el);
    });

    // Collection
    const collEl = $("your-collection");
    collEl.innerHTML = "";
    for (const c of M.COLORS) {
        const n = me.collection[c] || 0;
        if (n === 0) continue;
        const cell = document.createElement("div");
        cell.className = "tally-cell";
        cell.appendChild(makeGemEl(c));
        cell.append(` ×${n}`);
        collEl.appendChild(cell);
    }
    if (collEl.children.length === 0) {
        collEl.innerHTML = `<span class="muted">(no gems yet)</span>`;
    }

    // Missions
    const missionsEl = $("your-missions");
    missionsEl.innerHTML = "";
    if (me.completedMissions.length === 0) {
        missionsEl.innerHTML = `<li class="muted">(none yet)</li>`;
    } else {
        for (const m of me.completedMissions) {
            const li = document.createElement("li");
            li.className = `mission-${m.category}`;
            li.innerHTML = `<span>${escapeHtml(m.name)}</span><span class="coins">${m.coins}</span>`;
            missionsEl.appendChild(li);
        }
    }

    // Loans + investments — full per-row detail with running scoring impact.
    const finEl = $("your-finances");
    finEl.innerHTML = "";
    if (me.loans.length === 0 && me.investments.length === 0) {
        finEl.innerHTML = `<li class="muted">(none yet)</li>`;
    } else {
        for (const loan of me.loans) {
            const li = document.createElement("li");
            li.className = "finance-row finance-row--loan";
            li.innerHTML = `
                <span class="fin-icon">💸</span>
                <span class="fin-label">Loan</span>
                <span class="fin-detail">+${loan.amount} now / −${loan.amount} at scoring</span>
            `;
            finEl.appendChild(li);
        }
        for (const inv of me.investments) {
            const li = document.createElement("li");
            li.className = "finance-row finance-row--invest";
            li.innerHTML = `
                <span class="fin-icon">📈</span>
                <span class="fin-label">Invest</span>
                <span class="fin-detail">${inv.locked} locked → +${inv.locked + inv.card.amount} at scoring</span>
            `;
            finEl.appendChild(li);
        }
        const totalLoans = me.loans.reduce((s, l) => s + l.amount, 0);
        const totalInvestReturns = me.investments.reduce(
            (s, i) => s + i.locked + i.card.amount, 0
        );
        const net = totalInvestReturns - totalLoans;
        const sign = net >= 0 ? "+" : "";
        const li = document.createElement("li");
        li.className = "finance-row finance-row--total";
        li.innerHTML = `
            <span class="fin-icon">Σ</span>
            <span class="fin-label">Net at scoring</span>
            <span class="fin-detail ${net >= 0 ? "fin-positive" : "fin-negative"}">${sign}${net}</span>
        `;
        finEl.appendChild(li);
    }
}

function makeGemEl(color) {
    const el = document.createElement("div");
    el.className = `gem color-${color.toLowerCase()}`;
    el.textContent = color[0];
    return el;
}

function setBidPrompt(text, urgent = false) {
    const el = $("bid-prompt");
    el.textContent = text;
    el.classList.toggle("urgent", urgent);
}

function clearLog() {
    $("game-log").innerHTML = "";
    if (session) session.roundLog = [];
}

function log(message, klass = "") {
    const el = $("game-log");
    const li = document.createElement("li");
    if (klass) li.className = klass;
    li.textContent = message;
    el.appendChild(li);
    el.scrollTop = el.scrollHeight;
    // Mirror into the session so a resumed game can replay the log.
    if (session) {
        session.roundLog.push({ message, klass });
        // Hard cap so localStorage stays small even on long sessions.
        if (session.roundLog.length > 200) session.roundLog.shift();
    }
}

function replayLog(entries) {
    const el = $("game-log");
    el.innerHTML = "";
    for (const entry of entries || []) {
        const li = document.createElement("li");
        if (entry.klass) li.className = entry.klass;
        li.textContent = entry.message;
        el.appendChild(li);
    }
    el.scrollTop = el.scrollHeight;
}

function escapeHtml(s) {
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

// ---------- bootstrap ----------------------------------------------------

// If a save exists, drop straight back into the same phase the player was
// in. Otherwise show the menu.
const restored = loadPersistedSession();
if (restored) {
    session = restored;
    showScreen("game");
    replayLog(session.roundLog);
    renderAll();
    resumeUI();
} else {
    showScreen("menu");
}

function resumeUI() {
    if (!session) return;
    if (session.phase === "bidding" && session.pendingAuction) {
        promptHumanBid();
    } else if (session.phase === "revealing") {
        setBidPrompt("You won! Click a gem from your hand to reveal it into the Value Display.", true);
        $("bid-input").disabled = true;
        $("bid-submit").disabled = true;
        $("bid-zero").disabled = true;
        $("advance-button").hidden = true;
    } else if (session.phase === "summary") {
        setBidPrompt("Round complete. Click Continue to draw the next auction.");
        $("bid-input").disabled = true;
        $("bid-submit").disabled = true;
        $("bid-zero").disabled = true;
        $("advance-button").hidden = false;
    } else {
        // Unknown phase — safest is to drop back to menu without losing
        // the save (player can still quit explicitly).
        setBidPrompt("Resumed game.");
    }
}
