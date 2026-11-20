"""gta6-trophy-rarity-forecast-01 entry: E/N hazard forecaster (Python 3.13, stdlib only).

Forecasts every GTA VI trophy's earn rate on Exophase at the case's target as
five quantiles, from the case's history of reads of the game's trophy page.

Model (per trophy), in the E/N split the data forces on us:
  - N(t) = players tracked; E(t) = rate(t) * N(t) / 100 = earners. Exophase
    refreshes its counts on its own schedule, not with every trophy earned,
    so the history is long flat runs punctuated by refresh jumps. Consecutive
    reads that are identical (same N and same rate) are the same refresh, not
    new signal, so they are deduplicated first; the signal is the change
    BETWEEN refreshes.
  - N(t) is projected by fitting N(t) = A - B*exp(-t/tau) on the distinct
    refreshes over a grid of tau, which nests both a saturating influx
    (small tau) and a near-linear one (large tau); too few points fall
    back to a shrunk recent log-growth rate.
  - E(t) grows by a hazard: dE/dt = lam(t) * (N(t) - E(t)), the per-hour
    chance that a player who has not earned the trophy yet earns it. lam
    is estimated from the deltas between distinct refreshes in the recent
    36 h window as a Gamma-Poisson posterior mean -- earners gained over
    susceptible-hours of exposure, with a per-tier prior as pseudo-counts
    (story bronze earns fast, grindy/platinum earns slow) -- so a long
    flat run with a huge tracked population overwhelms the prior, while a
    thin early population leaves it in charge. The window average is
    corrected to the hazard AT the last read, and its decay is fitted from
    the intervals (log lam against time) so the projection saturates when
    the data says the hazard is falling, and keeps going when the data
    says it is flat.
  - Median at target = 100 * E(target) / N(target), with E <= N always.
    The platinum median is capped at the lowest other trophy's median: the
    platinum needs every other trophy, so its rate cannot exceed any of them.
  - Spread (logit) combines the persistence random walk (0.5 * sqrt(days))
    with the hazard's own uncertainty (projecting with lam scaled up/down
    by a factor that shrinks as more refreshes are seen) and with the
    uncertainty of the N projection, the total scaled by 0.7 (see the
    SPREAD_SCALE note: the scale is deliberately NOT tuned to the
    backtest's minimum, which would chase the simulator's own smoothness).
    A trophy at exactly 100% keeps a median of 100 with a narrow spread:
    it survived a 6x dilution at 100%. (Written out it reads 99.99: the
    logit clamps to 99.99, which the scorer does too, so it scores the
    same as 100.)

Fallback: when a trophy's history has fewer than 2 distinct refreshes, or
its data is unusable, the forecast for that trophy is exactly the house
persistence baseline (median = last rate, spread = 0.5 * sqrt(days)) with
one addition: a 1.0-logit refresh-risk term added in quadrature. A flat
history says nothing about the next refresh, and the practice cases prove a
refresh moves every rate at once; the median stays the last rate (no data
says which way it moves) but the band is widened so the refresh does not
take the whole band by surprise.

Deterministic: pure functions of the case file, output in the case's trophy
order, no clock, no randomness. Stdlib only, no network.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim. -P keeps this file's directory OFF sys.path, so the
sys.path line below puts it back first (kept from the starter; this entry
is a single file and imports nothing of its own, but the line is harmless).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of a rate's move, in logit units per square-root day.
SIGMA = 0.5

# Game release, the clock the hazard decays against (2026-11-19T00:00:00Z).
RELEASE_EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()

# Prior hazard (per hour) by tier: the chance per hour that a player who has
# not earned the trophy yet earns it. Story bronze is fast, grindy/platinum
# is slow; the data overwhelms the prior whenever refreshes are plentiful.
TIER_PRIOR_LAMBDA = {
    "platinum": 0.00002,
    "gold": 0.0002,
    "silver": 0.0008,
    "bronze": 0.002,
}
# Pseudo-exposure of the tier prior, in susceptible-hours: how much
# refresh exposure it takes to outweigh the prior. At launch-night scale
# (a few hundred tracked players) the prior rules; once tens of thousands
# of susceptible-hours have been seen, the data rules.
PRIOR_EXPOSURE_H = 50000.0
# Recency half-life (hours) for weighting hazard intervals: the hazard NOW
# is what matters, and launch-night hazard is long gone by the final.
RECENCY_HALFLIFE_H = 12.0
# Hazard intervals are taken from this window before the last read (hours);
# widened to the whole history only when the window holds no interval.
HAZARD_WINDOW_H = 36.0
# Default hazard decay time constant (hours) while projecting forward, used
# when the data cannot show the decay itself; the fitted decay replaces it
# whenever at least 3 intervals speak.
LAMBDA_DECAY_DEFAULT_TAU_H = 72.0
# Extra logit spread (added in quadrature) when the history is flat: the
# next unseen refresh moves every rate at once, as the practice cases show.
# The observed first refresh moved non-100% rates by ~-1.5 logit, so the
# 5% quantile (at -1.645 * spread) must sit just beyond that move, not on
# it: 1.0 puts it at -1.66 logit for a 90-minute horizon.
REFRESH_RISK_LOGIT = 1.0
# Floor on the model-uncertainty part of the spread, in logit units.
SPREAD_FLOOR_LOGIT = 0.10
# Scale on the model path's total spread. The backtest's in-sim median
# error is far smaller than the real world's will be (the simulator shares
# the model's structure), so the scale is NOT tuned to the simulator's
# minimum: 0.7 keeps the final's band near 0.8 logit -- wide enough to
# cover a median miss of half a logit at the outer quantiles -- while
# dropping the over-wide 1.2-logit band the raw components give. The
# flat-history path is NOT scaled: with no signal the wide refresh-risk
# band stays.
SPREAD_SCALE = 0.7
# Widest logit spread ever written.
SPREAD_MAX_LOGIT = 3.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_hours(value) -> float | None:
    """ISO-8601 timestamp -> hours since the game's release. None if unparseable."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.timestamp() - RELEASE_EPOCH) / 3600.0


def persistence_row(rate, days: float, spread_extra: float = 0.0) -> list:
    """The house persistence baseline row for one trophy (median = last rate).

    spread_extra is extra logit spread added in quadrature (refresh risk).
    """
    lg = logit(rate if rate is not None else 0)
    spread = math.sqrt((SIGMA * math.sqrt(max(0.0, days))) ** 2 + spread_extra ** 2)
    return [round4(min(100.0, max(0.01, inv_logit(lg + z * spread)))) for z in Z]


def distinct_points(history, index: int) -> list:
    """(t_hours, N, rate) for one trophy, deduplicated: consecutive reads with
    the same players_tracked AND the same rate are one refresh, not signal."""
    points = []
    for pos, read in enumerate(history):
        if not isinstance(read, dict):
            continue
        rates = read.get("rates")
        if not isinstance(rates, list) or index >= len(rates):
            continue
        rate = rates[index]
        if rate is None:
            continue
        try:
            rate = float(rate)
            n = float(read.get("players_tracked"))
        except (TypeError, ValueError):
            continue
        if not (0.0 <= rate <= 100.0) or n <= 0:
            continue
        t = parse_hours(read.get("at")) or parse_hours(read.get("slot"))
        if t is None:
            t = pos * 0.5  # unparseable clock: fall back to read order
        if points and points[-1][1] == n and points[-1][2] == rate:
            continue
        points.append((t, n, rate))
    return points


def estimate_lambda(points, t_last: float, tier: str):
    """Hazard lam (per hour) from the deltas between distinct refreshes,
    plus the fitted decay time constant of that hazard.

    Only the recent window speaks: the hazard NOW is what the projection
    needs, and a whole-history average would drag in the launch-night rush
    (when every hazard was far higher) and over-project. Estimation is a
    Gamma-Poisson posterior mean: earners gained over susceptible-hours
    of exposure, with the tier prior as pseudo-counts (alpha earns over
    beta susceptible-hours). Exposure weighting is what lets a long flat
    run with a huge tracked population overwhelm the prior -- a trophy no
    one earned in 10M susceptible-hours has a hazard near zero, however
    fast its tier usually earns -- while a thin early population leaves
    the prior in charge. The pooled window average is then corrected to
    the hazard AT the last read (dividing by K): when the hazard decays,
    the window's average sits above its endpoint. Returns
    (lam_now, tau_decay_h, n_intervals, lam_cv).
    """
    prior = TIER_PRIOR_LAMBDA.get(tier, TIER_PRIOR_LAMBDA["bronze"])
    if len(points) < 2:
        return prior, LAMBDA_DECAY_DEFAULT_TAU_H, 0, 0.0
    intervals = []
    for (t1, n1, r1), (t2, n2, r2) in zip(points, points[1:]):
        dt = t2 - t1
        if dt <= 0:
            continue
        e1 = r1 * n1 / 100.0
        e2 = r2 * n2 / 100.0
        de = max(0.0, e2 - e1)  # earners never decrease; rounding can fake it
        s_avg = ((n1 - e1) + (n2 - e2)) / 2.0
        if s_avg <= 0:
            continue
        exposure = s_avg * dt  # susceptible-hours
        intervals.append((t1, t2, de, exposure))
    if not intervals:
        return prior, LAMBDA_DECAY_DEFAULT_TAU_H, 0, 0.0
    # Recent window first; fall back to the whole history when it is empty.
    window = [iv for iv in intervals if t_last - iv[1] <= HAZARD_WINDOW_H]
    if not window:
        window = intervals
    alpha = prior * PRIOR_EXPOSURE_H
    num = alpha
    den = PRIOR_EXPOSURE_H
    lams = []
    weights = []
    for (t1, t2, de, exposure) in window:
        r = 0.5 ** ((t_last - t2) / RECENCY_HALFLIFE_H)
        num += r * de
        den += r * exposure
        lams.append(de / exposure if exposure > 0 else 0.0)
        weights.append(r * exposure)
    lam_pool = num / den if den > 0 else prior
    # Fitted decay: log(lam) against interval mid-time, weighted by
    # exposure. A flat hazard (slope ~ 0) means no decay; a falling one
    # gives its time constant. Shrunk toward the 72 h default.
    tau = LAMBDA_DECAY_DEFAULT_TAU_H
    usable = [(0.5 * (t1 + t2), de / exposure, r * exposure)
              for (t1, t2, de, exposure) in window
              for r in (0.5 ** ((t_last - t2) / RECENCY_HALFLIFE_H),)
              if de > 0 and exposure > 0]
    if len(usable) >= 3:
        sw = sum(w for (_, _, w) in usable)
        tm = sum(w * t for (t, _, w) in usable) / sw
        ym = sum(w * math.log(l) for (_, l, w) in usable) / sw
        sxx = sum(w * (t - tm) ** 2 for (t, _, w) in usable)
        sxy = sum(w * (t - tm) * (math.log(l) - ym) for (t, l, w) in usable)
        if sxx > 0:
            slope = sxy / sxx
            slope = min(0.0, max(-1.0 / 24.0, slope))  # decay only, >= 24 h
            inv_fit = -slope if slope < -1.0 / 960.0 else 1.0 / 960.0
            n = len(usable)
            inv_tau = (n * inv_fit + 1.0 * (1.0 / LAMBDA_DECAY_DEFAULT_TAU_H)) / (n + 1.0)
            tau = 1.0 / inv_tau
    # Correct the window average to the hazard at the last read.
    k_num = 0.0
    k_den = 0.0
    for (t1, t2, de, exposure) in window:
        r = 0.5 ** ((t_last - t2) / RECENCY_HALFLIFE_H)
        w = r * exposure
        k_num += w * math.exp((t_last - 0.5 * (t1 + t2)) / tau)
        k_den += w
    k_factor = k_num / k_den if k_den > 0 else 1.0
    lam = lam_pool / max(1.0, k_factor)
    # Coefficient of variation across intervals, for the spread.
    cv = 0.0
    if len(lams) >= 2:
        wtot = sum(weights)
        mean = sum(w * l for w, l in zip(weights, lams)) / wtot if wtot > 0 else 0.0
        if mean > 0:
            var = sum(w * (l - mean) ** 2 for w, l in zip(weights, lams)) / wtot
            cv = math.sqrt(max(0.0, var)) / mean
    return lam, tau, len(window), min(cv, 2.0)


def fit_n_model(points):
    """Fit N(t) = A - B*exp(-t/tau) on the distinct refreshes, over a grid of
    tau. The form nests both a saturating influx (small tau) and a near-
    linear one (large tau: exp(-t/tau) ~ 1 - t/tau). Returns a function
    n_at(dt_hours_from_last_point) -> N, or None when there are too few
    points or the fit is degenerate."""
    if len(points) < 3:
        return None
    ts = [t for (t, _, _) in points]
    ns = [n for (_, n, _) in points]
    n_last = ns[-1]
    best = None
    for tau in (12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0, 1536.0, 1e9):
        xs = [math.exp(-t / tau) for t in ts]
        n = len(ts)
        sx = sum(xs)
        sy = sum(ns)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ns))
        det = n * sxx - sx * sx
        if abs(det) < 1e-12:
            continue
        # N = A + C*x with C = -B.
        a = (sy * sxx - sx * sxy) / det
        c = (n * sxy - sx * sy) / det
        if a <= 0:
            continue
        sse = sum((y - (a + c * x)) ** 2 for x, y in zip(xs, ns))
        if best is None or sse < best[0]:
            best = (sse, tau, a, c)
    if best is None:
        return None
    _, tau, a, c = best
    t_last = ts[-1]

    def n_at(dt_h: float) -> float:
        value = a + c * math.exp(-(t_last + dt_h) / tau)
        return min(50.0 * n_last, max(0.5 * n_last, value))

    # Reject a fit that already mispredicts the last point badly.
    if abs(n_at(0.0) - n_last) > 0.25 * n_last:
        return None
    return n_at


def fallback_n_model(points, t_last: float) -> callable:
    """Shrunk log-growth N model, used when the saturating fit has too few
    points: the RECENT growth rate only (the whole-history rate still
    carries the launch explosion and would over-project N badly), shrunk
    toward no growth, and the growth rate itself decays with a 48 h
    constant."""
    if len(points) < 2:
        n_last = points[-1][1] if points else 0.0
        return lambda dt_h: n_last
    (t_prev, n_prev, _), (t_end, n_end, _) = points[-2], points[-1]
    g = 0.0
    dt = t_end - t_prev
    if dt > 0 and n_prev > 0 and n_end > 0:
        g = math.log(n_end / n_prev) / dt
        g *= len(points) / (len(points) + 1.5)
        g = min(0.08, max(-0.01, g))
    n_last = n_end

    def n_at(dt_h: float) -> float:
        return n_last * math.exp(g * 48.0 * (1.0 - math.exp(-dt_h / 48.0)))

    return n_at


def project_en(n_last: float, e_last: float, lam: float, tau_lam: float,
               n_at, t_last: float, horizon_h: float) -> tuple:
    """Project (N, E) forward horizon_h hours in 1-hour steps.

    N follows the fitted model; E grows by the decaying hazard on the
    players who have not earned the trophy yet. Returns (N_target, E_target).
    """
    if horizon_h <= 0:
        return n_last, e_last
    steps = int(math.ceil(horizon_h))
    steps = min(steps, 2000)
    dt = horizon_h / steps
    n = n_last
    e = e_last
    for k in range(1, steps + 1):
        elapsed = k * dt
        n = n_at(elapsed)
        lam_k = lam * math.exp(-(elapsed - dt / 2.0) / tau_lam)
        s = max(0.0, n - e)
        e = min(n, e + lam_k * s * dt)
    return n, e


def forecast_trophy(trophy: dict, history, horizon_minutes: float) -> tuple:
    """One trophy's (median, spread_logit). Never raises: any failure falls
    back to the persistence baseline for that trophy."""
    days = max(0.0, horizon_minutes) / 1440.0
    idx = trophy.get("_index", 0)
    tier = trophy.get("tier") or "bronze"
    try:
        last_rate = None
        for read in reversed(history):
            if isinstance(read, dict) and isinstance(read.get("rates"), list):
                rates = read["rates"]
                if idx < len(rates) and rates[idx] is not None:
                    last_rate = float(rates[idx])
                    break
        if last_rate is None:
            return 0.01, SIGMA * math.sqrt(days)
        if last_rate >= 99.995:
            # Universal trophy: it held 100% through a 6x dilution, so it
            # stays 100%; narrow spread, median exactly 100.
            return 100.0, max(0.05, 0.25 * SIGMA * math.sqrt(days))
        points = distinct_points(history, idx)
        if len(points) < 2:
            # Flat history: no refresh seen, so no signal about the next
            # one. Median stays the last rate; the band widens by the
            # refresh risk so one unseen refresh does not take it all.
            return last_rate, math.sqrt((SIGMA * math.sqrt(days)) ** 2 + REFRESH_RISK_LOGIT ** 2)
        t_last, n_last, r_last = points[-1]
        e_last = r_last * n_last / 100.0
        horizon_h = max(0.0, horizon_minutes) / 60.0
        lam, tau_lam, n_int, lam_cv = estimate_lambda(points, t_last, tier)
        n_at = fit_n_model(points) or fallback_n_model(points, t_last)
        n_tgt, e_tgt = project_en(n_last, e_last, lam, tau_lam, n_at, t_last, horizon_h)
        median = 100.0 * e_tgt / n_tgt if n_tgt > 0 else r_last
        median = min(99.99, max(0.01, median))
        # Spread: random walk + hazard uncertainty (re-project with lam
        # scaled up/down; the factor shrinks as refreshes accumulate and
        # grows when intervals disagree) + N-growth uncertainty.
        factor = math.exp(0.45 / math.sqrt(max(1, n_int)) + min(0.4, 0.4 * lam_cv))
        _, e_hi = project_en(n_last, e_last, lam * factor, tau_lam, n_at, t_last, horizon_h)
        _, e_lo = project_en(n_last, e_last, lam / factor, tau_lam, n_at, t_last, horizon_h)
        med_hi = min(99.99, max(0.01, 100.0 * e_hi / n_tgt)) if n_tgt > 0 else median
        med_lo = min(99.99, max(0.01, 100.0 * e_lo / n_tgt)) if n_tgt > 0 else median
        s_lam = (logit(med_hi) - logit(med_lo)) / 2.0
        s_n = 0.25 * min(1.0, abs(math.log(n_tgt / n_last))) if n_last > 0 and n_tgt > 0 else 0.0
        base = SIGMA * math.sqrt(days)
        spread = SPREAD_SCALE * math.sqrt(base ** 2 + s_lam ** 2 + s_n ** 2 + SPREAD_FLOOR_LOGIT ** 2)
        return median, min(SPREAD_MAX_LOGIT, spread)
    except Exception:
        # Never let one trophy sink the case: persistence on its last rate.
        try:
            fallback_rate = last_rate if last_rate is not None else 0.0
        except Exception:
            fallback_rate = 0.0
        return fallback_rate, SIGMA * math.sqrt(days)


def forecast_case(case: dict, name: str) -> dict:
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    try:
        horizon = float(case.get("horizon_minutes") or 0)
    except (TypeError, ValueError):
        horizon = 0.0
    medians = []
    spreads = []
    for i, trophy in enumerate(trophies):
        if not isinstance(trophy, dict):
            trophy = {"id": i + 1}
        trophy = dict(trophy)
        trophy["_index"] = i
        median, spread = forecast_trophy(trophy, history, horizon)
        medians.append(median)
        spreads.append(spread)
    # Platinum cap: the platinum needs every other trophy, so its rate
    # cannot exceed the lowest other trophy's rate. Cap its median there.
    for i, trophy in enumerate(trophies):
        if isinstance(trophy, dict) and trophy.get("tier") == "platinum" and len(medians) > 1:
            others = [m for j, m in enumerate(medians) if j != i]
            cap = min(others)
            if medians[i] > cap:
                medians[i] = cap
    rows = []
    for i, trophy in enumerate(trophies):
        tid = trophy.get("id", i + 1) if isinstance(trophy, dict) else i + 1
        lg = logit(medians[i])
        qs = [round4(min(100.0, max(0.01, inv_logit(lg + z * spreads[i])))) for z in Z]
        qs = sorted(qs)  # rounding can never invert them, but guarantee it
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def fallback_case(case: dict) -> dict:
    """Last-resort valid forecast: persistence on the last rate of every
    trophy (0 where unknown). Used only if forecast_case itself fails."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    try:
        horizon = float(case.get("horizon_minutes") or 0)
    except (TypeError, ValueError):
        horizon = 0.0
    days = max(0.0, horizon) / 1440.0
    rows = []
    for i, trophy in enumerate(trophies):
        tid = trophy.get("id", i + 1) if isinstance(trophy, dict) else i + 1
        rate = None
        for read in reversed(history):
            if isinstance(read, dict) and isinstance(read.get("rates"), list):
                rates = read["rates"]
                if i < len(rates) and rates[i] is not None:
                    rate = rates[i]
                    break
        rows.append({"id": tid, "q": persistence_row(rate, days)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue  # an unreadable case file cannot be forecast at all
        if not isinstance(case, dict):
            continue
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            try:
                forecast = fallback_case(case)
            except Exception:
                forecast = {"trophies": []}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
