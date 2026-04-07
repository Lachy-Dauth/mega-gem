/* ======================================================================
 * MegaGem — vanilla JS port of megagem/{engine,cards,missions,players}.py
 *
 * Faithful to the Python rules: 30 gem deck, 25 auction deck, 30 mission
 * deck, five value charts. The HeuristicAI mirrors players.HeuristicAI
 * (greedy treasure value + reserve + missions); the bid resolution logic
 * mirrors engine._resolve_winner.
 * ====================================================================== */

// ---------- Color enum & gem-per-color ------------------------------------

const COLORS = ["Blue", "Green", "Pink", "Purple", "Yellow"];
const GEMS_PER_COLOR = 6;

// ---------- Value charts (verbatim from value_charts.py) ------------------

const VALUE_CHARTS = {
    A: [0, 4, 8, 12, 16, 20],
    B: [20, 16, 12, 8, 4, 0],
    C: [0, 2, 5, 9, 14, 20],
    D: [20, 18, 15, 11, 6, 0],
    E: [0, 4, 10, 18, 6, 0],
};

function valueFor(chart, count) {
    const idx = Math.max(0, Math.min(5, count));
    return VALUE_CHARTS[chart][idx];
}

// _chart_value: round float, clamp, lookup
function chartValue(chart, countFloat) {
    return valueFor(chart, Math.round(countFloat));
}

// ---------- Cards ---------------------------------------------------------

function makeGemDeck() {
    const deck = [];
    for (const color of COLORS) {
        for (let i = 0; i < GEMS_PER_COLOR; i++) {
            deck.push({ color });
        }
    }
    return deck;
}

function makeAuctionDeck() {
    const deck = [];
    for (let i = 0; i < 12; i++) deck.push({ kind: "treasure", gems: 1 });
    for (let i = 0; i < 5; i++)  deck.push({ kind: "treasure", gems: 2 });
    for (let i = 0; i < 2; i++)  deck.push({ kind: "loan", amount: 10 });
    for (let i = 0; i < 2; i++)  deck.push({ kind: "loan", amount: 20 });
    for (let i = 0; i < 2; i++)  deck.push({ kind: "invest", amount: 5 });
    for (let i = 0; i < 2; i++)  deck.push({ kind: "invest", amount: 10 });
    return deck;
}

function describeAuction(card) {
    if (card.kind === "treasure") {
        return `Treasure (${card.gems} gem${card.gems === 1 ? "" : "s"})`;
    }
    if (card.kind === "loan")   return `Loan (${card.amount} coins)`;
    if (card.kind === "invest") return `Invest (${card.amount} coins)`;
    return "?";
}

// ---------- Missions (mirrors missions.make_mission_deck) -----------------

function combinations(arr, k) {
    const out = [];
    const rec = (start, combo) => {
        if (combo.length === k) { out.push(combo.slice()); return; }
        for (let i = start; i < arr.length; i++) {
            combo.push(arr[i]);
            rec(i + 1, combo);
            combo.pop();
        }
    };
    rec(0, []);
    return out;
}

// `gems` is a {color: count} object.
function distinctColorCount(gems) {
    let n = 0;
    for (const c of COLORS) if ((gems[c] || 0) > 0) n++;
    return n;
}

function pairsCount(gems) {
    let n = 0;
    for (const c of COLORS) if ((gems[c] || 0) >= 2) n++;
    return n;
}

function maxOfAnyColor(gems) {
    let m = 0;
    for (const c of COLORS) if ((gems[c] || 0) > m) m = gems[c];
    return m;
}

function colorCountsAtLeast(req) {
    return (gems) => {
        for (const [c, n] of Object.entries(req)) {
            if ((gems[c] || 0) < n) return false;
        }
        return true;
    };
}

function makeMissionDeck() {
    const deck = [];

    // Shields (2)
    deck.push({
        name: "Shield: 4 different colors",
        coins: 10,
        category: "shield",
        check: (gems) => distinctColorCount(gems) >= 4,
    });
    deck.push({
        name: "Shield: 2 pairs",
        coins: 15,
        category: "shield",
        check: (gems) => pairsCount(gems) >= 2,
    });

    // Pendants (16, 5 coins)
    deck.push({
        name: "Pendant: 2 of the same color",
        coins: 5,
        category: "pendant",
        check: (gems) => maxOfAnyColor(gems) >= 2,
    });
    for (const c of COLORS) {
        deck.push({
            name: `Pendant: 2 ${c}`,
            coins: 5,
            category: "pendant",
            check: colorCountsAtLeast({ [c]: 2 }),
        });
    }
    for (const [c1, c2] of combinations(COLORS, 2)) {
        deck.push({
            name: `Pendant: 1 ${c1} + 1 ${c2}`,
            coins: 5,
            category: "pendant",
            check: colorCountsAtLeast({ [c1]: 1, [c2]: 1 }),
        });
    }

    // Crowns (12, 10 coins)
    deck.push({
        name: "Crown: 3 of the same color",
        coins: 10,
        category: "crown",
        check: (gems) => maxOfAnyColor(gems) >= 3,
    });
    deck.push({
        name: "Crown: 3 different colors",
        coins: 10,
        category: "crown",
        check: (gems) => distinctColorCount(gems) >= 3,
    });
    for (const [c1, c2, c3] of combinations(COLORS, 3)) {
        deck.push({
            name: `Crown: 1 ${c1} + 1 ${c2} + 1 ${c3}`,
            coins: 10,
            category: "crown",
            check: colorCountsAtLeast({ [c1]: 1, [c2]: 1, [c3]: 1 }),
        });
    }

    return deck;
}

// ---------- Seedable RNG (Mulberry32) -------------------------------------

// `savedState` is the internal Mulberry32 counter `s` returned by a prior
// rng.getState() call. When provided, the new rng resumes from exactly
// that point — used by the localStorage save/restore path so a resumed
// game produces the same downstream bids/shuffles as the original would
// have. `seed` is still required (kept around for round-tripping).
function makeRng(seed, savedState) {
    if (seed == null || seed === "" || isNaN(seed)) {
        seed = Math.floor(Math.random() * 0x7fffffff);
    }
    let s = (savedState != null) ? (savedState >>> 0) : (seed >>> 0);
    return {
        seed,
        next() {
            s |= 0;
            s = (s + 0x6D2B79F5) | 0;
            let t = Math.imul(s ^ (s >>> 15), 1 | s);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        },
        // 0..n-1
        int(n) { return Math.floor(this.next() * n); },
        choice(arr) { return arr[this.int(arr.length)]; },
        shuffle(arr) {
            // Fisher–Yates
            for (let i = arr.length - 1; i > 0; i--) {
                const j = this.int(i + 1);
                [arr[i], arr[j]] = [arr[j], arr[i]];
            }
            return arr;
        },
        // Normalised to an unsigned 32-bit value so round-tripping through
        // JSON / localStorage produces the exact same integer back.
        getState() { return s >>> 0; },
    };
}

// ---------- Game state (mirrors state.GameState/PlayerState) --------------

const STARTING_COINS = { 3: 35, 4: 25, 5: 20 };
const STARTING_HAND_SIZE = { 3: 5, 4: 4, 5: 3 };

function emptyGemCounter() {
    const c = {};
    for (const col of COLORS) c[col] = 0;
    return c;
}

function setupGame({ players, chart, seed }) {
    const n = players.length;
    if (!STARTING_COINS[n]) throw new Error(`MegaGem supports 3-5 players, got ${n}`);
    if (!VALUE_CHARTS[chart]) throw new Error(`Unknown chart: ${chart}`);

    const rng = makeRng(seed);
    const gemDeck = rng.shuffle(makeGemDeck());
    const auctionDeck = rng.shuffle(makeAuctionDeck());
    const missionDeck = rng.shuffle(makeMissionDeck());

    const handSize = STARTING_HAND_SIZE[n];
    const coins = STARTING_COINS[n];

    const playerStates = players.map((player) => ({
        name: player.name,
        isHuman: !!player.isHuman,
        ai: player,
        hand: gemDeck.splice(gemDeck.length - handSize, handSize),
        coins,
        collection: emptyGemCounter(),
        completedMissions: [],
        loans: [],
        investments: [], // each: {card, locked}
    }));

    const revealedGems = gemDeck.splice(gemDeck.length - 2, 2);
    const activeMissions = missionDeck.splice(missionDeck.length - 4, 4);

    return {
        rng,
        chart,
        playerStates,
        gemDeck,
        revealedGems,
        auctionDeck,
        activeMissions,
        valueDisplay: emptyGemCounter(),
        lastWinnerIdx: null,
        roundNumber: 0,
    };
}

function isGameOver(state) {
    return state.auctionDeck.length === 0 ||
           (state.gemDeck.length === 0 && state.revealedGems.length === 0);
}

function maxLegalBid(playerState, auction) {
    if (auction.kind === "loan") {
        return Math.max(0, playerState.coins + auction.amount);
    }
    return Math.max(0, playerState.coins);
}

function clampBid(bid, playerState, auction) {
    let b = parseInt(bid, 10);
    if (isNaN(b)) b = 0;
    if (b < 0) b = 0;
    const cap = maxLegalBid(playerState, auction);
    if (b > cap) b = cap;
    return b;
}

// Mirrors engine._resolve_winner
function resolveWinner(bids, lastWinnerIdx, numPlayers, rng) {
    const top = Math.max(...bids);
    const candidates = [];
    bids.forEach((b, i) => { if (b === top) candidates.push(i); });
    if (candidates.length === 1) return candidates[0];
    if (lastWinnerIdx == null) {
        return rng.choice(candidates);
    }
    // closest player to the left of the previous winner
    for (let offset = 1; offset <= numPlayers; offset++) {
        const idx = (lastWinnerIdx + offset) % numPlayers;
        if (candidates.includes(idx)) return idx;
    }
    return candidates[0];
}

function applyTreasure(state, winnerState, card) {
    const gemsToTake = Math.min(card.gems, state.revealedGems.length);
    const taken = [];
    for (let i = 0; i < gemsToTake; i++) {
        const gem = state.revealedGems.shift();
        winnerState.collection[gem.color] += 1;
        taken.push(gem);
    }
    return taken;
}

function applyLoan(winnerState, bid, card) {
    winnerState.coins -= bid;
    winnerState.coins += card.amount;
    winnerState.loans.push(card);
}

function applyInvest(winnerState, bid, card) {
    winnerState.coins -= bid;
    winnerState.investments.push({ card, locked: bid });
}

function replenishRevealed(state) {
    while (state.revealedGems.length < 2 && state.gemDeck.length > 0) {
        state.revealedGems.push(state.gemDeck.pop());
    }
}

function checkMissions(state) {
    const completed = [];
    // Iterate over a snapshot since we mutate active_missions.
    const snapshot = state.activeMissions.slice();
    for (const mission of snapshot) {
        for (let idx = 0; idx < state.playerStates.length; idx++) {
            const ps = state.playerStates[idx];
            if (mission.check(ps.collection)) {
                ps.completedMissions.push(mission);
                state.activeMissions = state.activeMissions.filter((m) => m !== mission);
                completed.push({ playerIdx: idx, mission });
                break;
            }
        }
    }
    return completed;
}

function scoreGame(state) {
    // Reveal leftover hands.
    for (const ps of state.playerStates) {
        for (const card of ps.hand) {
            state.valueDisplay[card.color] += 1;
        }
        ps.hand = [];
    }
    const chart = state.chart;
    return state.playerStates.map((ps) => {
        let gemValue = 0;
        for (const color of COLORS) {
            const dispCount = state.valueDisplay[color] || 0;
            gemValue += (ps.collection[color] || 0) * valueFor(chart, dispCount);
        }
        const missionValue = ps.completedMissions.reduce((s, m) => s + m.coins, 0);
        const loansTotal = ps.loans.reduce((s, l) => s + l.amount, 0);
        const investReturns = ps.investments.reduce(
            (s, inv) => s + inv.card.amount + inv.locked, 0
        );
        const total = ps.coins + gemValue + missionValue - loansTotal + investReturns;
        return {
            name: ps.name,
            isHuman: ps.isHuman,
            coins: ps.coins,
            gemValue,
            missionValue,
            loansTotal,
            investReturns,
            total,
        };
    });
}

// ======================================================================
// AIs
// ======================================================================

// ---- helpers shared by HeuristicAI ---------------------------------------

function expectedFinalDisplay(state, myState) {
    const display = state.valueDisplay;
    const myColl = myState.collection;
    const revealedPool = emptyGemCounter();
    for (const g of state.revealedGems) revealedPool[g.color] += 1;
    const myHandCounter = emptyGemCounter();
    for (const g of myState.hand) myHandCounter[g.color] += 1;

    const otherColl = emptyGemCounter();
    let oppHandSize = 0;
    for (const ps of state.playerStates) {
        if (ps === myState) continue;
        for (const c of COLORS) otherColl[c] += ps.collection[c] || 0;
        oppHandSize += ps.hand.length;
    }
    const deckSize = state.gemDeck.length;
    const hiddenSlots = oppHandSize + deckSize;
    const oppShare = hiddenSlots > 0 ? oppHandSize / hiddenSlots : 0.0;

    const expected = {};
    for (const color of COLORS) {
        const seen = (display[color] || 0)
                   + (myColl[color] || 0)
                   + (myHandCounter[color] || 0)
                   + (revealedPool[color] || 0)
                   + (otherColl[color] || 0);
        const hidden = Math.max(0, GEMS_PER_COLOR - seen);
        const expectedInOpp = hidden * oppShare;
        expected[color] = (display[color] || 0)
                        + (myHandCounter[color] || 0)
                        + expectedInOpp;
    }
    return expected;
}

function missionCompletionBonus(myState, missions, extra) {
    if (!missions.length) return 0;
    const hypothetical = { ...myState.collection };
    for (const c of COLORS) hypothetical[c] += (extra[c] || 0);
    let bonus = 0;
    for (const m of missions) {
        if (m.check(myState.collection)) continue;
        if (m.check(hypothetical)) bonus += m.coins;
    }
    return bonus;
}

function missionProgressBonus(myState, missions, extra) {
    if (!missions.length) return 0;
    const hypothetical = { ...myState.collection };
    for (const c of COLORS) hypothetical[c] += (extra[c] || 0);
    let soft = 0;
    for (const m of missions) {
        if (m.check(myState.collection)) continue;
        if (m.check(hypothetical)) continue;
        for (const color of Object.keys(extra)) {
            if ((extra[color] || 0) === 0) continue;
            const stretched = { ...hypothetical };
            stretched[color] += 2;
            if (m.check(stretched) && !m.check(hypothetical)) {
                soft += Math.floor(m.coins / 3);
                break;
            }
        }
    }
    return soft;
}

function treasureValue(card, state, myState) {
    const expected = expectedFinalDisplay(state, myState);
    const gemsForSale = state.revealedGems.slice(
        0, Math.min(card.gems, state.revealedGems.length)
    );
    if (gemsForSale.length === 0) return 0;
    const extra = emptyGemCounter();
    let gemValue = 0;
    for (const gem of gemsForSale) {
        gemValue += chartValue(state.chart, expected[gem.color]);
        extra[gem.color] += 1;
    }
    const hard = missionCompletionBonus(myState, state.activeMissions, extra);
    const soft = missionProgressBonus(myState, state.activeMissions, extra);
    return gemValue + hard + soft;
}

function remainingSupply(state) {
    return state.gemDeck.length + state.revealedGems.length;
}

function expectedAvgTreasureValue(state, myState) {
    const expected = expectedFinalDisplay(state, myState);
    let total = 0;
    for (const c of COLORS) total += chartValue(state.chart, expected[c]);
    return total / COLORS.length;
}

// ---- AI classes ---------------------------------------------------------

class RandomAI {
    constructor(name, rng) {
        this.name = name;
        this.isHuman = false;
        this.rng = rng;
    }
    chooseBid(state, myState, auction) {
        const cap = maxLegalBid(myState, auction);
        if (cap === 0) return 0;
        return this.rng.int(cap + 1);
    }
    chooseGemToReveal(state, myState) {
        return this.rng.choice(myState.hand);
    }
}

class HeuristicAI {
    // `noise` adds uniform jitter of ±max(2, 40% of bid) to the chosen bid
    // before clamping to [0, cap]. The medium-difficulty menu option passes
    // noise=true so the otherwise-deterministic heuristic feels human and
    // beatable; the unit tests / Python parity benchmarks pass noise=false.
    constructor(name, rng, { noise = false } = {}) {
        this.name = name;
        this.isHuman = false;
        this.rng = rng;
        this.DISCOUNT = 0.75;
        this.noise = noise;
    }

    reserveForFuture(state) {
        const gemsLeft = remainingSupply(state);
        const futureTreasures = Math.max(0, Math.floor(gemsLeft / 2));
        const avg = expectedAvgTreasureValue(state, state.playerStates[0]);
        return Math.floor(futureTreasures * avg * 0.2);
    }

    // Apply uniform ±max(2, 40% of bid) noise then clamp to [0, cap].
    // Cap is the legal bid ceiling — never exceeded. This is what makes
    // the medium difficulty wobble: a heuristic bid of 12 might come out
    // as anything in [8, 16], a bid of 0 might come out 0..2.
    _jitter(bid, cap) {
        if (!this.noise) return bid;
        const span = Math.max(2, Math.floor(Math.abs(bid) * 0.4));
        // rng.next() is in [0, 1); shift to [-1, 1) and scale.
        const offset = Math.floor((this.rng.next() * 2 - 1) * (span + 1));
        let noisy = bid + offset;
        if (noisy < 0) noisy = 0;
        if (noisy > cap) noisy = cap;
        return noisy;
    }

    chooseBid(state, myState, auction) {
        const cap = maxLegalBid(myState, auction);
        if (cap === 0) return 0;

        if (auction.kind === "treasure") {
            const value = treasureValue(auction, state, myState);
            const target = Math.floor(value * this.DISCOUNT);
            const reserve = this.reserveForFuture(state);
            const spendable = Math.max(0, myState.coins - reserve);
            const bid = Math.max(0, Math.min(target, spendable, cap));
            return this._jitter(bid, cap);
        }

        if (auction.kind === "invest") {
            const reserve = this.reserveForFuture(state);
            const surplus = Math.max(0, myState.coins - reserve);
            let bid = Math.min(surplus, cap);
            if (bid === 0 && cap > 0) bid = 1;
            return this._jitter(bid, cap);
        }

        if (auction.kind === "loan") {
            if (myState.coins >= 5) return 0;
            if (remainingSupply(state) < 3) return 0;
            const bid = Math.min(auction.amount, cap);
            return this._jitter(bid, cap);
        }
        return 0;
    }

    chooseGemToReveal(state, myState) {
        const chart = state.chart;
        const display = state.valueDisplay;
        const myHolding = myState.collection;
        const oppHolding = emptyGemCounter();
        for (const ps of state.playerStates) {
            if (ps === myState) continue;
            for (const c of COLORS) oppHolding[c] += ps.collection[c] || 0;
        }

        let bestScore = null;
        let bestCard = null;
        for (const card of myState.hand) {
            const color = card.color;
            const current = display[color] || 0;
            const delta = valueFor(chart, current + 1) - valueFor(chart, current);
            const relative = (myHolding[color] || 0) - (oppHolding[color] || 0);
            const netBenefit = delta * relative;
            const tiebreak = -(myHolding[color] || 0);
            const score = [netBenefit, tiebreak];
            if (bestScore === null
                || score[0] > bestScore[0]
                || (score[0] === bestScore[0] && score[1] > bestScore[1])) {
                bestScore = score;
                bestCard = card;
            }
        }
        return bestCard || myState.hand[0];
    }
}

// ======================================================================
// HyperAdaptiveSplitAI — JS port of megagem/players.py:HyperAdaptiveSplitAI
//
// Uses the full hypergeometric distribution over the hidden card pool to
// compute E[chart_value(final_display[c])] per color, then bids via three
// independent linear models (one per auction type) over the same five
// game-state features the Python AI uses. Weight tables below were
// produced by `scripts/evolve_hyper_adaptive.py` (the GA driver) for each
// player count.
// ======================================================================

const _TOTAL_AUCTIONS = 25;

function mathComb(n, k) {
    if (k < 0 || k > n) return 0;
    if (k === 0 || k === n) return 1;
    if (k > n - k) k = n - k;
    let result = 1;
    for (let i = 1; i <= k; i++) {
        result = (result * (n - k + i)) / i;
    }
    return result;
}

function _hyperHiddenDistribution(state, myState) {
    const display = state.valueDisplay;
    const myCollection = myState.collection;
    const revealedPool = emptyGemCounter();
    for (const g of state.revealedGems) revealedPool[g.color] += 1;
    const myHand = emptyGemCounter();
    for (const g of myState.hand) myHand[g.color] += 1;

    const otherCollection = emptyGemCounter();
    let oppHandTotal = 0;
    for (const ps of state.playerStates) {
        if (ps === myState) continue;
        for (const c of COLORS) otherCollection[c] += ps.collection[c] || 0;
        oppHandTotal += ps.hand.length;
    }
    const deckSize = state.gemDeck.length;
    const hiddenTotal = oppHandTotal + deckSize;

    const distributions = {};
    for (const color of COLORS) {
        const seen = (display[color] || 0)
                   + (myCollection[color] || 0)
                   + (myHand[color] || 0)
                   + (revealedPool[color] || 0)
                   + (otherCollection[color] || 0);
        const hiddenOfColor = Math.max(0, GEMS_PER_COLOR - seen);
        const knownOffset = (display[color] || 0) + (myHand[color] || 0);

        const dist = new Map();
        if (hiddenTotal === 0 || oppHandTotal === 0 || hiddenOfColor === 0) {
            dist.set(knownOffset, 1.0);
        } else {
            const denom = mathComb(hiddenTotal, oppHandTotal);
            const kMin = Math.max(0, oppHandTotal - (hiddenTotal - hiddenOfColor));
            const kMax = Math.min(hiddenOfColor, oppHandTotal);
            for (let k = kMin; k <= kMax; k++) {
                const num = mathComb(hiddenOfColor, k)
                          * mathComb(hiddenTotal - hiddenOfColor, oppHandTotal - k);
                dist.set(knownOffset + k, num / denom);
            }
        }
        distributions[color] = dist;
    }
    return distributions;
}

function _hyperExpectedPerGemValue(state, myState, chart) {
    const distributions = _hyperHiddenDistribution(state, myState);
    const perGem = {};
    for (const color of COLORS) {
        let ev = 0;
        for (const [count, p] of distributions[color]) {
            ev += p * valueFor(chart, Math.min(count, 5));
        }
        perGem[color] = ev;
    }
    return perGem;
}

function _hyperTreasureValue(card, state, myState) {
    const gemsForSale = state.revealedGems.slice(
        0, Math.min(card.gems, state.revealedGems.length)
    );
    if (gemsForSale.length === 0) return 0;
    const perGem = _hyperExpectedPerGemValue(state, myState, state.chart);
    let gemV = 0;
    const extra = emptyGemCounter();
    for (const gem of gemsForSale) {
        gemV += perGem[gem.color];
        extra[gem.color] += 1;
    }
    const hard = missionCompletionBonus(myState, state.activeMissions, extra);
    const soft = missionProgressBonus(myState, state.activeMissions, extra);
    return gemV + hard + soft;
}

function _hyperAvgTreasureValue(state, myState) {
    const perGem = _hyperExpectedPerGemValue(state, myState, state.chart);
    let sum = 0;
    for (const c of COLORS) sum += perGem[c];
    return sum / COLORS.length;
}

function _hyperEvRemainingAuctions(state, myState) {
    const avgPerGem = _hyperAvgTreasureValue(state, myState);
    const sellableGems = state.revealedGems.length + state.gemDeck.length;
    return avgPerGem * sellableGems;
}

function _hyperComputeDiscountFeatures(state, myState) {
    const auctionsLeft = state.auctionDeck.length;
    const progress = Math.max(0, Math.min(1, 1 - auctionsLeft / _TOTAL_AUCTIONS));

    const evRemaining = Math.max(1, _hyperEvRemainingAuctions(state, myState));

    const oppCoins = [];
    for (const ps of state.playerStates) {
        if (ps === myState) continue;
        oppCoins.push(ps.coins);
    }
    const avgOpp = oppCoins.length > 0
        ? oppCoins.reduce((a, b) => a + b, 0) / oppCoins.length
        : 0;
    const topOpp = oppCoins.length > 0 ? Math.max(...oppCoins) : 0;

    const myCashRatio = myState.coins / evRemaining;
    const avgCashRatio = avgOpp / evRemaining;
    const topCashRatio = topOpp / evRemaining;

    let hidden = 0;
    for (const ps of state.playerStates) {
        if (ps === myState) continue;
        hidden += ps.hand.length;
    }
    hidden += state.gemDeck.length;
    const chartTable = VALUE_CHARTS[state.chart];
    const chartSwing = (Math.max(...chartTable) - Math.min(...chartTable)) / 20;
    const variance = (hidden / 30) * chartSwing;

    return {
        progress,
        myCashRatio,
        avgCashRatio,
        topCashRatio,
        variance,
    };
}

class BidModel {
    // weights6 is [bias, w_progress, w_my_cash, w_avg_cash, w_top_cash, w_variance]
    constructor(weights6) {
        this.bias       = weights6[0];
        this.wProgress  = weights6[1];
        this.wMyCash    = weights6[2];
        this.wAvgCash   = weights6[3];
        this.wTopCash   = weights6[4];
        this.wVariance  = weights6[5];
    }
    discount(features) {
        const raw = this.bias
                  + this.wProgress * features.progress
                  + this.wMyCash   * features.myCashRatio
                  + this.wAvgCash  * features.avgCashRatio
                  + this.wTopCash  * features.topCashRatio
                  + this.wVariance * features.variance;
        return Math.max(0, Math.min(1, raw));
    }
}

// Evolved by scripts/evolve_hyper_adaptive.py — see artifacts/best_weights_*.json
// Order per 6-block: [bias, w_progress, w_my_cash, w_avg_cash, w_top_cash, w_variance]
// 18-vector order: treasure(6) + invest(6) + loan(6).
const EVOLVED_WEIGHTS_3P = [
    0.5556969710305875, -0.7134105470669515, 0.3798342211945455,
    -0.18296903854598895, 0.69986264640917, -0.2141443800932632,
    -1.1775428111973723, -0.6646192073698377, -0.23285669604606718,
    -0.14481152573219974, 0.8260221064757964, 0.9862430166182075,
    0.2139523538589837, 0.46562220363642354, -0.4314767599827297,
    -0.24545653341399754, 0.3186393873091611, -0.2750860629213795,
];
const EVOLVED_WEIGHTS_4P = [
    0.23062963150464755, -0.2084932164075678, 0.09306115359487846,
    0.30964085613031145, 0.048365061949802925, 0.44901685083222875,
    -1.4307535907748454, -0.45717001965918075, 0.011527903836427128,
    -0.5741938687178983, 0.009271480041767394, -0.40095473614094446,
    0.21954122083356076, 0.10037047137666993, -0.319559369897936,
    -0.10851100914421585, 0.6938757749119884, -0.1635468934150679,
];
const EVOLVED_WEIGHTS_5P = [
    0.20245698353097244, 0.24215369123733455, 0.30476888020845005,
    -0.3062695813624682, 0.7553131658021903, -0.02700736995241911,
    -0.755481356072717, -0.27974519260012143, -0.38995476013913627,
    -0.060750875751926, 0.05638249440972204, -0.6320384741818099,
    0.21954122083356076, 0.13630284182038377, -0.319559369897936,
    -0.2070666067824972, 0.7980160760620152, -0.15211597499091944,
];

function evolvedWeightsFor(numPlayers) {
    if (numPlayers === 3) return EVOLVED_WEIGHTS_3P;
    if (numPlayers === 4) return EVOLVED_WEIGHTS_4P;
    if (numPlayers === 5) return EVOLVED_WEIGHTS_5P;
    return EVOLVED_WEIGHTS_4P;
}

class HyperAdaptiveSplitAI {
    constructor(name, rng, weights) {
        if (!Array.isArray(weights) || weights.length !== 18) {
            throw new Error("HyperAdaptiveSplitAI requires an 18-element weights array");
        }
        this.name = name;
        this.isHuman = false;
        this.rng = rng;
        this.treasureModel = new BidModel(weights.slice(0, 6));
        this.investModel   = new BidModel(weights.slice(6, 12));
        this.loanModel     = new BidModel(weights.slice(12, 18));
    }

    reserveForFuture(state) {
        // Mirrors HyperAdaptiveAI._reserve_for_future: hyper avg × halved gem
        // supply × 0.2. Note Python passes player_states[0] which is the ref
        // player at table seat 0, not necessarily this AI — we keep that
        // exact behaviour so the JS bid matches the Python bid.
        const gemsLeft = remainingSupply(state);
        const futureTreasures = Math.max(0, Math.floor(gemsLeft / 2));
        const avg = _hyperAvgTreasureValue(state, state.playerStates[0]);
        return Math.floor(futureTreasures * avg * 0.2);
    }

    chooseBid(state, myState, auction) {
        const cap = maxLegalBid(myState, auction);
        if (cap === 0) return 0;

        const features = _hyperComputeDiscountFeatures(state, myState);
        const reserve = this.reserveForFuture(state);
        const spendable = Math.max(0, myState.coins - reserve);

        if (auction.kind === "treasure") {
            const d = this.treasureModel.discount(features);
            const value = _hyperTreasureValue(auction, state, myState);
            return Math.max(0, Math.min(Math.floor(value * d), spendable, cap));
        }

        if (auction.kind === "invest") {
            const d = this.investModel.discount(features);
            let bid = Math.min(Math.floor(spendable * d), cap);
            // Token bid floor: invest payouts are strictly positive cash flow.
            if (bid === 0 && cap > 0) bid = 1;
            return bid;
        }

        if (auction.kind === "loan") {
            const d = this.loanModel.discount(features);
            return Math.min(Math.floor(auction.amount * d), cap);
        }

        return 0;
    }

    // Reveal logic mirrors HeuristicAI/HypergeometricAI: pick the gem that
    // maximises (chart-delta × my-vs-opp net holding), tiebreak preferring
    // colors I hold less of.
    chooseGemToReveal(state, myState) {
        const chart = state.chart;
        const display = state.valueDisplay;
        const myHolding = myState.collection;
        const oppHolding = emptyGemCounter();
        for (const ps of state.playerStates) {
            if (ps === myState) continue;
            for (const c of COLORS) oppHolding[c] += ps.collection[c] || 0;
        }
        let bestScore = null;
        let bestCard = null;
        for (const card of myState.hand) {
            const color = card.color;
            const current = display[color] || 0;
            const delta = valueFor(chart, current + 1) - valueFor(chart, current);
            const relative = (myHolding[color] || 0) - (oppHolding[color] || 0);
            const netBenefit = delta * relative;
            const tiebreak = -(myHolding[color] || 0);
            if (bestScore === null
                || netBenefit > bestScore[0]
                || (netBenefit === bestScore[0] && tiebreak > bestScore[1])) {
                bestScore = [netBenefit, tiebreak];
                bestCard = card;
            }
        }
        return bestCard || myState.hand[0];
    }
}

// ---------- exports (attached to window for ui.js) ------------------------

window.MegaGem = {
    COLORS,
    VALUE_CHARTS,
    valueFor,
    describeAuction,
    makeRng,
    makeMissionDeck,
    setupGame,
    isGameOver,
    maxLegalBid,
    clampBid,
    resolveWinner,
    applyTreasure,
    applyLoan,
    applyInvest,
    replenishRevealed,
    checkMissions,
    scoreGame,
    RandomAI,
    HeuristicAI,
    HyperAdaptiveSplitAI,
    evolvedWeightsFor,
    EVOLVED_WEIGHTS_3P,
    EVOLVED_WEIGHTS_4P,
    EVOLVED_WEIGHTS_5P,
};
