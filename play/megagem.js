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

    reserveForFuture(state, myState) {
        const gemsLeft = remainingSupply(state);
        const futureTreasures = Math.max(0, Math.floor(gemsLeft / 2));
        // Use *this* player's view of remaining-treasure value — using a
        // fixed seat would skew non-zero-seat bidders.
        const avg = expectedAvgTreasureValue(state, myState);
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
            const reserve = this.reserveForFuture(state, myState);
            const spendable = Math.max(0, myState.coins - reserve);
            const bid = Math.max(0, Math.min(target, spendable, cap));
            return this._jitter(bid, cap);
        }

        if (auction.kind === "invest") {
            const reserve = this.reserveForFuture(state, myState);
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

    reserveForFuture(state, myState) {
        // Mirrors HyperAdaptiveAI._reserve_for_future: hyper avg × halved gem
        // supply × 0.2, evaluated against *this* player's view of remaining
        // treasure value (not seat 0).
        const gemsLeft = remainingSupply(state);
        const futureTreasures = Math.max(0, Math.floor(gemsLeft / 2));
        const avg = _hyperAvgTreasureValue(state, myState);
        return Math.floor(futureTreasures * avg * 0.2);
    }

    chooseBid(state, myState, auction) {
        const cap = maxLegalBid(myState, auction);
        if (cap === 0) return 0;

        const features = _hyperComputeDiscountFeatures(state, myState);
        const reserve = this.reserveForFuture(state, myState);
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

// ======================================================================
// Evo2AI — JS port of megagem/players_evo2.py:Evo2AI
//
// Clean-slate evolved player. Differences from HyperAdaptiveSplitAI:
//   * Drops the static reserve floor — bids cap at the legal cap directly.
//   * Replaces the auctions_left/25 progress proxy with an EXACT
//     E[rounds remaining] computed by closed-form multivariate
//     hypergeometric over the auction-deck composition + gem supply.
//   * Replaces cash-ratio features with raw integer my_coins / avg_opp /
//     top_opp — the GA learns the right scale.
//   * Treasure head gains two per-card features: ev and std-dev of the
//     prize's coin value, derived from the same hypergeometric used by
//     HyperAdaptiveSplitAI for its gem-value estimator.
//   * Treasure EV also adds a mission probability delta — for every
//     active mission, P(I win mission | I take the gems) − P(I win |
//     highest-coin opponent takes them), scaled by mission.coins.
//   * Heads output the bid in COINS DIRECTLY: bid = bias + Σ wᵢ·featᵢ,
//     clamped to [0, cap] once at chooseBid. No discount fraction × EV.
//   * 19 weights instead of 18: treasure(7) + invest(6) + loan(6).
// ======================================================================

const _IMPOSSIBLE_DISTANCE = 99;

function combinationsWithReplacement(arr, k) {
    const out = [];
    const buf = [];
    const rec = (start) => {
        if (buf.length === k) { out.push(buf.slice()); return; }
        for (let i = start; i < arr.length; i++) {
            buf.push(arr[i]);
            rec(i); // i, not i+1 — replacement
            buf.pop();
        }
    };
    rec(0);
    return out;
}

// _expected_rounds_remaining_impl, JS port. Memoized on the integer
// signature (A, T1, T2, NT, G); within one round all four AIs hit the
// cache for the same arguments.
const _ROUNDS_CACHE = new Map();
const _ROUNDS_CACHE_LIMIT = 4096;

function _expectedRoundsRemainingImpl(A, T1, T2, NT, G) {
    if (A === 0 || G === 0) return 0.0;
    if (T1 === 0 && T2 === 0) return A; // no treasures → only deck-exhaustion ends it

    const key = `${A},${T1},${T2},${NT},${G}`;
    const cached = _ROUNDS_CACHE.get(key);
    if (cached !== undefined) return cached;

    // Precompute binomial tables — math.comb was the dominant cost in
    // the Python profile and the same shape applies here.
    const combT1 = new Array(T1 + 1);
    const combT2 = new Array(T2 + 1);
    const combNT = new Array(NT + 1);
    const combA  = new Array(A + 1);
    for (let k = 0; k <= T1; k++) combT1[k] = mathComb(T1, k);
    for (let k = 0; k <= T2; k++) combT2[k] = mathComb(T2, k);
    for (let k = 0; k <= NT; k++) combNT[k] = mathComb(NT, k);
    for (let k = 0; k <= A;  k++) combA[k]  = mathComb(A, k);

    // E[rounds] = Σ_{k=1..A} P(first k-1 cards consumed < G gems).
    // The j=0 term (no cards drawn yet) is 1.
    let total = 1.0;
    for (let j = 1; j < A; j++) {
        const denom = combA[j];
        let pUnder = 0.0;
        const j1Max = Math.min(T1, j);
        for (let j1 = 0; j1 <= j1Max; j1++) {
            // j2 ≤ (G − j1 − 1) // 2 from gems_consumed < G constraint.
            const j2Cap = Math.min(T2, j - j1, Math.floor((G - j1 - 1) / 2));
            if (j2Cap < 0) continue;
            const ct1 = combT1[j1];
            for (let j2 = 0; j2 <= j2Cap; j2++) {
                const jnt = j - j1 - j2;
                if (jnt < 0 || jnt > NT) continue;
                pUnder += ct1 * combT2[j2] * combNT[jnt] / denom;
            }
        }
        total += pUnder;
    }

    if (_ROUNDS_CACHE.size >= _ROUNDS_CACHE_LIMIT) _ROUNDS_CACHE.clear();
    _ROUNDS_CACHE.set(key, total);
    return total;
}

function _expectedRoundsRemaining(state) {
    const A = state.auctionDeck.length;
    if (A === 0) return 0.0;
    const G = state.gemDeck.length + state.revealedGems.length;
    if (G === 0) return 0.0;
    let T1 = 0, T2 = 0;
    for (const c of state.auctionDeck) {
        if (c.kind !== "treasure") continue;
        if (c.gems === 1) T1++;
        else if (c.gems === 2) T2++;
    }
    const NT = A - T1 - T2;
    return _expectedRoundsRemainingImpl(A, T1, T2, NT, G);
}

// Per-color (E[chart_value(final_display)], Var[...]). Walks the same
// hypergeometric distribution HyperAdaptiveSplitAI uses, accumulating
// first and second moments. Variance clamped at 0 for FP safety.
function _perColorValueStats(state, myState, chart) {
    const distributions = _hyperHiddenDistribution(state, myState);
    const stats = {};
    for (const color of COLORS) {
        let ev = 0.0, ev2 = 0.0;
        for (const [count, p] of distributions[color]) {
            const v = valueFor(chart, Math.min(count, 5));
            ev += p * v;
            ev2 += p * v * v;
        }
        stats[color] = [ev, Math.max(0.0, ev2 - ev * ev)];
    }
    return stats;
}

// (EV, std) for the coin value of winning this treasure auction.
// Same-color contributions are perfectly correlated (exact n²·var);
// across-color treated as independent (an approximation). Mission
// terms are deterministic and feed only the EV.
function _treasureValueStats(card, state, myState) {
    const gemsForSale = state.revealedGems.slice(
        0, Math.min(card.gems, state.revealedGems.length)
    );
    if (gemsForSale.length === 0) return [0.0, 0.0];

    const stats = _perColorValueStats(state, myState, state.chart);
    const colorCounts = emptyGemCounter();
    for (const g of gemsForSale) colorCounts[g.color] += 1;

    let ev = 0.0, varSum = 0.0;
    for (const color of COLORS) {
        const n = colorCounts[color];
        if (n === 0) continue;
        const [mean, v] = stats[color];
        ev += n * mean;
        varSum += (n * n) * v;
    }

    const extra = colorCounts; // alias — same shape as the Python Counter
    ev += missionCompletionBonus(myState, state.activeMissions, extra);
    ev += missionProgressBonus(myState, state.activeMissions, extra);
    ev += _missionProbabilityDelta(card, state, myState);
    return [ev, Math.sqrt(varSum)];
}

// Smallest k such that some k-multiset of colors added to `holdings`
// satisfies the mission. Returns 0 if already satisfied,
// _IMPOSSIBLE_DISTANCE if no multiset of size ≤ max_k works.
const _DISTANCE_CACHE = new Map();
const _DISTANCE_CACHE_LIMIT = 16384;

function _holdingsKey(holdings) {
    // Fixed-order 5-tuple stringified — hashable for the Map.
    return (
        (holdings.Blue || 0) + "," +
        (holdings.Green || 0) + "," +
        (holdings.Pink || 0) + "," +
        (holdings.Purple || 0) + "," +
        (holdings.Yellow || 0)
    );
}

function _minExtraGemsToSatisfy(holdings, mission, maxK = 6) {
    const key = _holdingsKey(holdings) + "|" + mission.name;
    const cached = _DISTANCE_CACHE.get(key);
    if (cached !== undefined) return cached;

    let result;
    if (mission.check(holdings)) {
        result = 0;
    } else {
        // Single mutable working dict; add a combo, test, undo.
        const candidate = { ...holdings };
        result = _IMPOSSIBLE_DISTANCE;
        outer:
        for (let k = 1; k <= maxK; k++) {
            for (const combo of combinationsWithReplacement(COLORS, k)) {
                for (const c of combo) candidate[c] = (candidate[c] || 0) + 1;
                const ok = mission.check(candidate);
                for (const c of combo) candidate[c] -= 1;
                if (ok) { result = k; break outer; }
            }
        }
    }

    if (_DISTANCE_CACHE.size >= _DISTANCE_CACHE_LIMIT) _DISTANCE_CACHE.clear();
    _DISTANCE_CACHE.set(key, result);
    return result;
}

// Heuristic P(player_idx claims `mission` before game end). The shape
// matches the Python: already-satisfied lowest-seat wins (engine
// tie-break); otherwise score by coin_ratio / (1 + distance), zero out
// impossible / supply-blocked, normalize. holdingOverrides lets the
// caller hypothetically add gems to a player's collection without
// touching state.
function _pPlayerWinsMission(playerIdx, state, mission, holdingOverrides) {
    const overrides = holdingOverrides || {};
    const players = state.playerStates;
    const n = players.length;

    const holdingsPerPlayer = players.map((ps, idx) => {
        const h = { ...ps.collection };
        const ovr = overrides[idx];
        if (ovr) {
            for (const c of COLORS) {
                if (ovr[c]) h[c] = (h[c] || 0) + ovr[c];
            }
        }
        return h;
    });

    // Engine tie-break: lowest seat with a satisfying collection wins.
    for (let idx = 0; idx < n; idx++) {
        if (mission.check(holdingsPerPlayer[idx])) {
            return idx === playerIdx ? 1.0 : 0.0;
        }
    }

    // In-play pool: gems not in any collection.
    let inPlayTotal = 0;
    for (const color of COLORS) {
        let held = 0;
        for (const h of holdingsPerPlayer) held += h[color] || 0;
        inPlayTotal += Math.max(0, GEMS_PER_COLOR - held);
    }

    let coinSum = 0;
    for (const ps of players) coinSum += ps.coins;
    const avgCoins = coinSum / Math.max(1, n);

    const scores = new Array(n);
    let totalScore = 0.0;
    for (let idx = 0; idx < n; idx++) {
        const distance = _minExtraGemsToSatisfy(holdingsPerPlayer[idx], mission);
        if (distance >= _IMPOSSIBLE_DISTANCE || inPlayTotal < distance) {
            scores[idx] = 0.0;
            continue;
        }
        const coinRatio = (players[idx].coins + 1) / (avgCoins + 1);
        const s = coinRatio / (1.0 + distance);
        scores[idx] = s;
        totalScore += s;
    }
    if (totalScore === 0.0) return 0.0;
    return scores[playerIdx] / totalScore;
}

// Σ over active missions of (p_win − p_lose) · mission.coins, where the
// "lose" branch assigns the gems to the highest-coin opponent (cheap
// proxy for the most likely auction winner).
function _missionProbabilityDelta(card, state, myState) {
    const gemsForSale = state.revealedGems.slice(
        0, Math.min(card.gems, state.revealedGems.length)
    );
    if (gemsForSale.length === 0 || state.activeMissions.length === 0) {
        return 0.0;
    }
    const extra = emptyGemCounter();
    for (const g of gemsForSale) extra[g.color] += 1;

    const myIdx = state.playerStates.indexOf(myState);
    if (myIdx < 0) return 0.0;

    let likelyOpp = -1;
    let topOppCoins = -Infinity;
    state.playerStates.forEach((ps, idx) => {
        if (idx === myIdx) return;
        if (ps.coins > topOppCoins) { topOppCoins = ps.coins; likelyOpp = idx; }
    });
    if (likelyOpp < 0) return 0.0;

    let delta = 0.0;
    for (const mission of state.activeMissions) {
        const pWin  = _pPlayerWinsMission(myIdx, state, mission, { [myIdx]: extra });
        const pLose = _pPlayerWinsMission(myIdx, state, mission, { [likelyOpp]: extra });
        delta += (pWin - pLose) * mission.coins;
    }
    return delta;
}

function _computeEvo2Features(state, myState) {
    const rounds = _expectedRoundsRemaining(state);
    const opp = [];
    for (const ps of state.playerStates) {
        if (ps !== myState) opp.push(ps.coins);
    }
    const avg = opp.length > 0 ? opp.reduce((a, b) => a + b, 0) / opp.length : 0;
    const top = opp.length > 0 ? Math.max(...opp) : 0;
    return {
        roundsRemaining: rounds,
        myCoins: myState.coins,
        avgOppCoins: avg,
        topOppCoins: top,
    };
}

// --- Three head models. They output the bid in COINS DIRECTLY (a raw
// float — clamping happens once in chooseBid). Each takes the same four
// shared features plus its own per-card feature(s).

class _Evo2TreasureModel {
    // weights7: [bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std]
    constructor(w) {
        this.bias    = w[0];
        this.wRounds = w[1];
        this.wMy     = w[2];
        this.wAvg    = w[3];
        this.wTop    = w[4];
        this.wEv     = w[5];
        this.wStd    = w[6];
    }
    bid(f, ev, std) {
        return this.bias
             + this.wRounds * f.roundsRemaining
             + this.wMy     * f.myCoins
             + this.wAvg    * f.avgOppCoins
             + this.wTop    * f.topOppCoins
             + this.wEv     * ev
             + this.wStd    * std;
    }
}

class _Evo2AmountModel {
    // weights6: [bias, w_rounds, w_my, w_avg, w_top, w_amount]
    // Shared by invest and loan — structurally identical.
    constructor(w) {
        this.bias    = w[0];
        this.wRounds = w[1];
        this.wMy     = w[2];
        this.wAvg    = w[3];
        this.wTop    = w[4];
        this.wAmount = w[5];
    }
    bid(f, amount) {
        return this.bias
             + this.wRounds * f.roundsRemaining
             + this.wMy     * f.myCoins
             + this.wAvg    * f.avgOppCoins
             + this.wTop    * f.topOppCoins
             + this.wAmount * amount;
    }
}

// Evolved by scripts/evolve_evo2.py --opponent self_play. Each set is the
// per-seat-count GA winner from `artifacts/best_weights_evo2_{N}p.json` —
// the same files the canonical heatmap loads. 19-vector layout:
// treasure(7) + invest(6) + loan(6).
//
// Self-play beats the `--opponent old_evo` weights head-to-head (35% as the
// lone challenger vs 3× old-evo on held-out seeds), so we ship self-play.
// 5p falls back to 4p until per-seat-count training finishes — paste from
// `artifacts/best_weights_evo2_5p.json` when it does.
const EVO2_WEIGHTS_3P = [
    // treasure: bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std
    1.1283650469891564, 0.004868855623980664, 0.10655847982258716,
    -0.07110456037718273, -0.014760843333120069, 0.3,
    -0.27549810321079016,
    // invest: bias, w_rounds, w_my, w_avg, w_top, w_amount
    2.0, -0.004050924080936394, 0.12166352569093686,
    -0.11450212366614854, -0.018890045923634705, 0.34920975704051466,
    // loan: bias, w_rounds, w_my, w_avg, w_top, w_amount
    0.9381212092519632, 0.07857322565139474, -0.07029638619042584,
    -0.1984092534881384, 0.14134434658521194, 0.4034770895444064,
];
const EVO2_WEIGHTS_4P = [
    // treasure: bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std
    0.9671062444221764, -0.0906995616980441, 0.07804979550128198,
    0.05375147152736104, -0.04247465810129918, 0.32783828473034604,
    -0.011838494331700117,
    // invest: bias, w_rounds, w_my, w_avg, w_top, w_amount
    1.908464547879478, 0.4300303741599258, -0.1201852409204779,
    -0.28421403664160627, 0.3149361220138405, 0.07219353469220569,
    // loan: bias, w_rounds, w_my, w_avg, w_top, w_amount
    -0.4139242208454687, -0.31190499765072527, 0.13966251262722051,
    0.12135141558388368, -0.0669196243751372, 0.36349000133503273,
];
// TODO: replace once `python -m scripts.evolve_evo2 --num-players 5`
// produces `artifacts/best_weights_evo2_5p.json`.
const EVO2_WEIGHTS_5P = EVO2_WEIGHTS_4P;

function evo2WeightsFor(numPlayers) {
    if (numPlayers === 3) return EVO2_WEIGHTS_3P;
    if (numPlayers === 4) return EVO2_WEIGHTS_4P;
    if (numPlayers === 5) return EVO2_WEIGHTS_5P;
    return EVO2_WEIGHTS_4P;
}

class Evo2AI {
    constructor(name, rng, weights) {
        if (!Array.isArray(weights) || weights.length !== 19) {
            throw new Error("Evo2AI requires a 19-element weights array");
        }
        this.name = name;
        this.isHuman = false;
        this.rng = rng;
        this.treasureModel = new _Evo2TreasureModel(weights.slice(0, 7));
        this.investModel   = new _Evo2AmountModel(weights.slice(7, 13));
        this.loanModel     = new _Evo2AmountModel(weights.slice(13, 19));
    }

    chooseBid(state, myState, auction) {
        const cap = maxLegalBid(myState, auction);
        if (cap === 0) return 0;
        const f = _computeEvo2Features(state, myState);

        if (auction.kind === "treasure") {
            const [ev, std] = _treasureValueStats(auction, state, myState);
            const raw = this.treasureModel.bid(f, ev, std);
            return Math.max(0, Math.min(Math.floor(raw), cap));
        }

        if (auction.kind === "invest") {
            const raw = this.investModel.bid(f, auction.amount);
            let bid = Math.max(0, Math.min(Math.floor(raw), cap));
            // Free money — always grab a token bid if we can.
            if (bid === 0 && cap > 0) bid = 1;
            return bid;
        }

        if (auction.kind === "loan") {
            const raw = this.loanModel.bid(f, auction.amount);
            return Math.max(0, Math.min(Math.floor(raw), cap));
        }

        return 0;
    }

    // Reveal logic is identical to HeuristicAI/HypergeometricAI — the
    // Evo2 redesign targets bidding only.
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

// ======================================================================
// Evo3AI — JS port of megagem/players/evo3.py:Evo3AI
//
// Identical to Evo2AI except every head gains two opponent-pricing
// features:
//   * mean_delta — weighted mean of (max opponent bid − baseline) over
//     the rounds this AI has already played.
//   * std_delta  — weighted stddev of the same quantity.
// "baseline" is what Evo3 *would have bid* with the default delta
// values (0, 1), not its actual bid — that stable reference breaks the
// feedback loop the actual bid would otherwise create.
//
// Observations in the matching category (treasure/invest/loan) are
// counted with weight 4; others with weight 1. Defaults before any
// history: (mean_delta=0, std_delta=1).
//
// `observeRound(state, myIdx, result)` is called by ui.js right after
// each auction resolves and appends (category, max_opp − baseline) to
// the per-instance history. `result` must carry `auction` and `bids`.
//
// 25 weights: treasure(9) + invest(8) + loan(8). The new pair on each
// head is (w_mean_delta, w_std_delta) tacked onto the end of the Evo2
// layout, so an Evo3 initialised with the Evo2 weights + 4 zeros
// reproduces Evo2 behaviour exactly when the history is empty.
// ======================================================================

const _EVO3_CAT_TREASURE = "treasure";
const _EVO3_CAT_INVEST   = "invest";
const _EVO3_CAT_LOAN     = "loan";
const _EVO3_MATCH_WEIGHT = 4.0;
const _EVO3_OTHER_WEIGHT = 1.0;
const _EVO3_DEFAULT_MEAN_DELTA = 0.0;
const _EVO3_DEFAULT_STD_DELTA  = 1.0;

function _weightedDeltaStats(history, currentCategory) {
    if (history.length === 0) {
        return [_EVO3_DEFAULT_MEAN_DELTA, _EVO3_DEFAULT_STD_DELTA];
    }
    let totalW = 0.0, totalX = 0.0, totalX2 = 0.0;
    for (const [cat, delta] of history) {
        const w = (cat === currentCategory) ? _EVO3_MATCH_WEIGHT : _EVO3_OTHER_WEIGHT;
        totalW  += w;
        totalX  += w * delta;
        totalX2 += w * delta * delta;
    }
    if (totalW <= 0.0) {
        return [_EVO3_DEFAULT_MEAN_DELTA, _EVO3_DEFAULT_STD_DELTA];
    }
    const mean = totalX / totalW;
    const variance = Math.max(0.0, totalX2 / totalW - mean * mean);
    return [mean, Math.sqrt(variance)];
}

class _Evo3TreasureModel {
    // weights9: [bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std,
    //            w_mean_delta, w_std_delta]
    constructor(w) {
        this.bias         = w[0];
        this.wRounds      = w[1];
        this.wMy          = w[2];
        this.wAvg         = w[3];
        this.wTop         = w[4];
        this.wEv          = w[5];
        this.wStd         = w[6];
        this.wMeanDelta   = w[7];
        this.wStdDelta    = w[8];
    }
    bid(f, ev, std, meanDelta, stdDelta) {
        return this.bias
             + this.wRounds    * f.roundsRemaining
             + this.wMy        * f.myCoins
             + this.wAvg       * f.avgOppCoins
             + this.wTop       * f.topOppCoins
             + this.wEv        * ev
             + this.wStd       * std
             + this.wMeanDelta * meanDelta
             + this.wStdDelta  * stdDelta;
    }
}

class _Evo3AmountModel {
    // weights8: [bias, w_rounds, w_my, w_avg, w_top, w_amount,
    //            w_mean_delta, w_std_delta]
    // Shared by the invest and loan heads — structurally identical.
    constructor(w) {
        this.bias         = w[0];
        this.wRounds      = w[1];
        this.wMy          = w[2];
        this.wAvg         = w[3];
        this.wTop         = w[4];
        this.wAmount      = w[5];
        this.wMeanDelta   = w[6];
        this.wStdDelta    = w[7];
    }
    bid(f, amount, meanDelta, stdDelta) {
        return this.bias
             + this.wRounds    * f.roundsRemaining
             + this.wMy        * f.myCoins
             + this.wAvg       * f.avgOppCoins
             + this.wTop       * f.topOppCoins
             + this.wAmount    * amount
             + this.wMeanDelta * meanDelta
             + this.wStdDelta  * stdDelta;
    }
}

// Evolved by scripts/evolve_evo3.py --opponent vs_all. 25-vector layout:
// treasure(9) + invest(8) + loan(8). Numbers copied verbatim from
// saved_best_weights/best_weights_evo3_vs_all_4p.json. 3p/5p fall back
// to 4p until per-seat-count training runs produce their own files.
const EVO3_WEIGHTS_4P = [
    // treasure: bias, w_rounds, w_my, w_avg, w_top, w_ev, w_std, w_mean_delta, w_std_delta
    0.9671062444221764, -0.22577743567106498, 0.06179431807120027,
    0.13179333635966, 0.029817067443990482, 0.22808568787174874,
    0.00826541378646068, 0.02088883506021053, -0.04065315920784615,
    // invest: bias, w_rounds, w_my, w_avg, w_top, w_amount, w_mean_delta, w_std_delta
    1.9428585631974362, 0.38197662141604993, -0.06693603814839288,
    -0.42612717455166765, 0.32059485484031025, 0.2245769585294565,
    -0.018375851882717283, -0.002632855943731513,
    // loan: bias, w_rounds, w_my, w_avg, w_top, w_amount, w_mean_delta, w_std_delta
    -0.3816101402754214, -0.21640338533739048, 0.17350712216597403,
    0.05427180974084926, -0.08435772023559968, 0.337710214852521,
    0.15583525951623678, -0.10407099542522064,
];
// TODO: replace once `python -m scripts.evolve_evo3 --num-players 3/5`
// produces their own per-seat-count files.
const EVO3_WEIGHTS_3P = EVO3_WEIGHTS_4P;
const EVO3_WEIGHTS_5P = EVO3_WEIGHTS_4P;

function evo3WeightsFor(numPlayers) {
    if (numPlayers === 3) return EVO3_WEIGHTS_3P;
    if (numPlayers === 4) return EVO3_WEIGHTS_4P;
    if (numPlayers === 5) return EVO3_WEIGHTS_5P;
    return EVO3_WEIGHTS_4P;
}

class Evo3AI {
    constructor(name, rng, weights) {
        if (!Array.isArray(weights) || weights.length !== 25) {
            throw new Error("Evo3AI requires a 25-element weights array");
        }
        this.name = name;
        this.isHuman = false;
        this.rng = rng;
        this.treasureModel = new _Evo3TreasureModel(weights.slice(0, 9));
        this.investModel   = new _Evo3AmountModel(weights.slice(9, 17));
        this.loanModel     = new _Evo3AmountModel(weights.slice(17, 25));
        // [category, max_opp_bid − baseline_bid] per observed round.
        this._oppHistory = [];
        // Cached default-deltas ("baseline") bid from the most recent
        // chooseBid call. observeRound reads and clears it. null when
        // nothing has been cached (e.g. chooseBid wasn't called, or the
        // cap was 0).
        this._lastDefaultBid = null;
    }

    // Optional constructor helper — lets ui.js persist/restore opponent
    // history across localStorage round-trips without touching private
    // fields directly.
    setOppHistory(history) {
        this._oppHistory = Array.isArray(history) ? history.slice() : [];
    }
    getOppHistory() {
        return this._oppHistory.slice();
    }

    chooseBid(state, myState, auction) {
        const cap = maxLegalBid(myState, auction);
        if (cap === 0) {
            this._lastDefaultBid = 0;
            return 0;
        }
        const f = _computeEvo2Features(state, myState);

        if (auction.kind === "treasure") {
            const [ev, std] = _treasureValueStats(auction, state, myState);
            const [mD, sD] = _weightedDeltaStats(this._oppHistory, _EVO3_CAT_TREASURE);
            const actualRaw  = this.treasureModel.bid(f, ev, std, mD, sD);
            // Baseline: same features, same weights, delta inputs pinned
            // to (0, 1). What the AI would bid with no history.
            const defaultRaw = this.treasureModel.bid(
                f, ev, std, _EVO3_DEFAULT_MEAN_DELTA, _EVO3_DEFAULT_STD_DELTA
            );
            const actualBid  = Math.max(0, Math.min(Math.floor(actualRaw),  cap));
            this._lastDefaultBid = Math.max(0, Math.min(Math.floor(defaultRaw), cap));
            return actualBid;
        }

        if (auction.kind === "invest") {
            const [mD, sD] = _weightedDeltaStats(this._oppHistory, _EVO3_CAT_INVEST);
            const actualRaw  = this.investModel.bid(f, auction.amount, mD, sD);
            const defaultRaw = this.investModel.bid(
                f, auction.amount, _EVO3_DEFAULT_MEAN_DELTA, _EVO3_DEFAULT_STD_DELTA
            );
            let actualBid  = Math.max(0, Math.min(Math.floor(actualRaw),  cap));
            let defaultBid = Math.max(0, Math.min(Math.floor(defaultRaw), cap));
            // Free money — the token-bid-if-zero rule applies to both so
            // the recorded baseline matches what chooseBid would actually
            // return for an empty history.
            if (actualBid  === 0 && cap > 0) actualBid  = 1;
            if (defaultBid === 0 && cap > 0) defaultBid = 1;
            this._lastDefaultBid = defaultBid;
            return actualBid;
        }

        if (auction.kind === "loan") {
            const [mD, sD] = _weightedDeltaStats(this._oppHistory, _EVO3_CAT_LOAN);
            const actualRaw  = this.loanModel.bid(f, auction.amount, mD, sD);
            const defaultRaw = this.loanModel.bid(
                f, auction.amount, _EVO3_DEFAULT_MEAN_DELTA, _EVO3_DEFAULT_STD_DELTA
            );
            const actualBid = Math.max(0, Math.min(Math.floor(actualRaw),  cap));
            this._lastDefaultBid = Math.max(0, Math.min(Math.floor(defaultRaw), cap));
            return actualBid;
        }

        this._lastDefaultBid = 0;
        return 0;
    }

    // Hook called by ui.js right after the engine resolves an auction.
    // `result` is { auction, bids } — `auction` is the just-resolved
    // card and `bids` is the array of all-player bids at the same index
    // order as state.playerStates. Uses the cached `_lastDefaultBid`
    // (from this round's chooseBid) so the delta measurement doesn't
    // depend on Evo3's learned response to its own history.
    observeRound(state, myIdx, result) {
        const baseline = this._lastDefaultBid;
        this._lastDefaultBid = null;
        if (!result || !result.auction) return;
        const kind = result.auction.kind;
        if (kind !== _EVO3_CAT_TREASURE
            && kind !== _EVO3_CAT_INVEST
            && kind !== _EVO3_CAT_LOAN) return;
        if (baseline === null) return;
        const bids = result.bids;
        if (!Array.isArray(bids) || bids.length === 0) return;
        let maxOpp = -Infinity;
        for (let i = 0; i < bids.length; i++) {
            if (i === myIdx) continue;
            if (bids[i] > maxOpp) maxOpp = bids[i];
        }
        if (maxOpp === -Infinity) return;
        this._oppHistory.push([kind, maxOpp - baseline]);
    }

    // Reveal logic is identical to HeuristicAI/Evo2AI — Evo3's redesign
    // targets bidding only.
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
    Evo2AI,
    Evo3AI,
    evolvedWeightsFor,
    evo2WeightsFor,
    evo3WeightsFor,
    EVOLVED_WEIGHTS_3P,
    EVOLVED_WEIGHTS_4P,
    EVOLVED_WEIGHTS_5P,
    EVO2_WEIGHTS_3P,
    EVO2_WEIGHTS_4P,
    EVO2_WEIGHTS_5P,
    EVO3_WEIGHTS_3P,
    EVO3_WEIGHTS_4P,
    EVO3_WEIGHTS_5P,
};
