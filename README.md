[![Make.fun — GTA 6 Trophy Rarity Forecast](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Cm11w/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Cm11w/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn
rate on Exophase at 2026-11-26T00:00:00Z — the percent of the players
Exophase tracks who will have earned it a week after launch — as five
quantiles per trophy (5%, 25%, 50%, 75%, 95%), scored by pinball loss on
the logit scale. Built for make.fun's
[`gta6-trophy-rarity-forecast-01`](https://make.fun/c/gta6-trophy-rarity-forecast-01)
bounty.

## The model

`main.py` (Python 3.13, standard library only, no network) splits each
trophy's history into **earners E(t)** and **players tracked N(t)**,
because that is the split the data forces: Exophase refreshes its counts on
its own schedule, not with every trophy earned, so the history is long flat
runs punctuated by refresh jumps. Consecutive identical reads are the same
refresh, not new signal, so they are deduplicated first; the signal is the
change *between* refreshes.

- **N(t)** is projected by fitting `N(t) = A - B·exp(-t/τ)` over a grid of
  τ on the distinct refreshes — a form that nests both a saturating influx
  and a near-linear one.
- **E(t)** grows by a hazard, `dE/dt = λ(t)·(N(t) - E(t))`, estimated from
  the deltas between distinct refreshes in the recent 36-hour window as a
  Gamma–Poisson posterior mean (earners gained over susceptible-hours of
  exposure, with a per-tier prior as pseudo-counts), corrected to the hazard
  at the last read, with its decay fitted from the intervals — so the
  forecast saturates when the data says the hazard is falling, and keeps
  going when the data says it is flat.
- The **median** is `100·E(target)/N(target)`; the platinum median is
  capped at the lowest other trophy's median (the platinum needs every
  other trophy). The **spread** (logit) combines the persistence random
  walk with the hazard's and the N projection's uncertainty, scaled by 0.7.
- **Fallback:** a trophy with fewer than 2 distinct refreshes, or unusable
  data, gets exactly the house persistence baseline (median = last rate),
  with a 1.0-logit refresh-risk term added to the spread — a flat history
  says nothing about the next refresh, and the practice cases prove one
  refresh moves every rate at once.

Deterministic (two runs write byte-identical output), ~0.06 s per case,
far under the 8 s cap. It never emits an invalid forecast: every trophy
and every case is guarded by a persistence fallback.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, in name order, and writes
`<id>.out` per case: `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}`.

## Validation

- **Practice p00/p01:** valid output, two runs byte-identical, 0.06 s.
  Median identical to the persistence baseline (max difference 0.0000 pp,
  as designed for flat history); the band is ~8× wider (3.26 vs 0.41
  logit), so it covers the refresh the practice straddles. Under the
  answer reconstructed from the published 04:30 totals (N=3,868, 4,276
  trophies earned → non-100% rates at ~0.221× their launch-night values),
  estimated practice loss **0.83** vs **1.40** for both house baselines.
- **Synthetic backtest** (8 regimes with known ground truth — fast/slow
  influx, grindy-heavy, constant hazard, linear N growth, practice-like,
  weekend step-up, slow refresh): mean pinball loss **0.262** vs
  persistence **0.493** and curve baseline **0.324**. The model beats
  persistence in 8/8 regimes and the curve baseline in 7/8 (the one loss
  is the practice-like regime, a refresh lottery over 90 minutes that pays
  nothing). Worst regime 0.68, vs 0.94 (persistence) and 0.59 (curve).
- **Robustness:** rates null in some reads, 1-read history, horizon 0,
  empty/missing history, null rates in the last read, 0% rates, a
  1,000,000-minute horizon, garbage reads, no trophies, missing levels, a
  malformed `.in` file — all still produce valid output, never a crash.

Full numbers: see the validation report in the build workspace
(`~/workspace/makefun/trophy/validation-report.md` in the builder's copy;
the harness scripts live beside it as `backtest.py` and `validate.py`).

## License

MIT — see [LICENSE](LICENSE).

<sub>Written for the make.fun community. Sign up with my referral link:
https://make.fun/r/ywgem12g · code `ywgem12g`</sub>
