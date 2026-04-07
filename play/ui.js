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
    $("back-to-menu").hidden = (name === "menu");
}

// ---------- Game session state ------------------------------------------

let session = null;
/* {
 *   state: GameState,
 *   humanIdx: 0,
 *   pendingBids: number[] | null,
 *   pendingAuction: AuctionCard | null,
 *   phase: "bidding" | "revealing" | "summary",
 *   roundLog: string[],
 * }
 */

// ---------- Menu --------------------------------------------------------

$("menu-start").addEventListener("click", startGame);
$("play-again").addEventListener("click", () => showScreen("menu"));
$("back-to-menu").addEventListener("click", () => {
    if (confirm("Abandon this game and return to the menu?")) {
        session = null;
        showScreen("menu");
    }
});

function startGame() {
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
    for (let i = 0; i < numPlayers - 1; i++) {
        const aiSeed = masterRng.int(2 ** 31);
        const aiRng = M.makeRng(aiSeed);
        if (aiKind === "random") {
            players.push(new M.RandomAI(aiNames[i], aiRng));
        } else if (aiKind === "heuristic") {
            // Medium: deterministic heuristic with uniform bid noise so it
            // is occasionally beatable / surprising.
            players.push(new M.HeuristicAI(aiNames[i], aiRng, { noise: true }));
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
        : (aiKind === "heuristic" ? "heuristic+noise" : "evolved");
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

    state.lastWinnerIdx = winnerIdx;
    session.pendingWinnerIdx = winnerIdx;

    renderAll();

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
}

function log(message, klass = "") {
    const el = $("game-log");
    const li = document.createElement("li");
    if (klass) li.className = klass;
    li.textContent = message;
    el.appendChild(li);
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

showScreen("menu");
