# Regime-Conditioned Black–Litterman Factor Allocation for Indian Equities

Pipeline v6.0 &nbsp;·&nbsp; NSE / NIFTY-500 &nbsp;·&nbsp; Walk-forward 2022–2026

A quantitative equity strategy that conditions Black–Litterman portfolio construction on a market regime estimated by a causal Student-t Hidden Markov Model. Every stage — universe construction, regime detection, alpha modelling, view scaling, allocation — is designed so that decisions at time *t* depend only on information available at *t*.

**Status.** Research code for an SSRN / arXiv q-fin preprint and a submission to the IIM Calcutta Finance Research Conference 2026. Results are reported net of a full Indian transaction cost stack.

---

## Contribution

Prior Indian HMM work (e.g. Chaudhuri & Kumar 2015) uses regime detection to time the market benchmark. The novel piece here is applying regime posteriors to a **Black–Litterman factor-allocation** framework — the regime governs not only what to hold but *how* to construct the portfolio (BL–MVO / MinVar / InvVol blend, cash overlay).

## Pipeline

| Stage | Method | Output |
|---|---|---|
| 1. Universe | Union of five NIFTY-500 constituent snapshots (2016–2026) plus the inclusion/exclusion log | 904 symbols → 756 with price history |
| 2. Factors | 12 active cross-sectional factors (momentum, quality, low-vol, liquidity, mean-reversion) after VIF pruning | Daily factor matrix, ~660 stocks |
| 3. Regime | 4-state Student-t HMM on 7 market features; forward-only Bayes filter | P(regime \| info up to *t*) |
| 4. Alpha | LightGBM regression on factors + regime posteriors, retrained every 63 days | Cross-sectional score per stock |
| 5. Views | Grinold fundamental law → Black–Litterman posterior | μ_BL, Σ_BL |
| 6. Weights | Regime-specific blend of BL–MVO / MinVar / InvVol + cash | 20-stock portfolio |

## Headline results (walk-forward 2022-01 → 2026-05)

|  | Strategy (gross) | Strategy (net) | NIFTY 500 |
|---|---|---|---|
| Annual return | 13.42% | 5.90% | 9.98% |
| Sharpe | **0.61** | −0.05 | 0.24 |
| Max drawdown | −18.19% | −22.52% | −18.84% |
| Annual volatility | 11.28% | 11.37% | 14.65% |

- Alpha model OOS rank-IC = **0.035**, t = 3.93 (HAC t = 3.60) — clears the Harvey–Liu–Zhu t > 3.0 bar.
- Cost drag = 7.5 percentage points of annual return, arising from 1,165% annualised one-way turnover across 97 rebalances.
- The signal is real; the implementation is constrained by turnover, not by alpha. See §9 and §10 of `docs/RP1_briefing.pdf`.

## Methodology — what v6 fixes over v5

Five documented biases were closed relative to the earlier pipeline version:

| # | Defect in v5 | v6 remedy |
|---|---|---|
| 1 | HMM fit on the full sample; `predict_proba` returned smoothed posteriors P(s_t \| y_1..T) — a June 2022 rebalance was reading a regime label informed by 2026 data | Single fit on data ≤ 2021-12-31; forward-only filter α̂_t(j) = P(s_t = j \| y_1..t) — causality is algebraic, not approximate |
| 2 | Value factors built from a current `.info` EPS/BVPS snapshot broadcast across 2015–2026 (100% coverage, the signature of a constant fill) | Value family removed; `VALUE_MODE` switch supports a point-in-time CSV loader for vendor extracts (Prowess, Capitaline, EODHD) |
| 3 | Grinold IC scaled by in-sample skill (0.31 vs true OOS 0.035 → views ~13× too confident) | Purged out-of-sample IC accumulated during walk-forward; James–Stein shrinkage; floor removed |
| 4 | `TRANSACTION_COST = 0.00` against ~50% monthly turnover | Full NSE delivery cost stack — brokerage, STT, exchange, SEBI, IPFT, stamp duty, GST, configurable market impact — priced separately on buy and sell legs |
| 5 | 140 of 904 union symbols reported as "delisted"; the failure list contained TATAMOTORS, KPIT, HDFC, PVR — genuine rate-limit failures | Per-ticker retry with backoff, successor-symbol resolution, delisted-overlay hook, survivorship audit isolating the 73 names that can actually bias the OOS window |

Two related backtest-accounting corrections came in alongside: portfolio return now uses w′(e^r − 1) instead of a weighted average of log returns (Jensen bias), and weights drift between rebalances instead of being silently re-imposed daily.

## Known open issues (disclosed, not yet fixed)

- **Regime labelling violates return-monotonic ordering.** The health-blend labelling ranks a state with mean annualised return +27% as "Correction" because its drawdown is deep. The 21-day feature window cannot separate a sustained bear market from a sharp rebound. Three candidate fixes are logged in `docs/RP1_briefing.pdf` §4.4.
- The Crisis regime never fires in the 2022–2026 window, so the crisis branch of the allocation table is untested out-of-sample.
- Sector classification is keyword-matched on company names (FMCG 137 > Finance 91, 180 "Other") — needs an official NSE / GICS map.
- The Black–Litterman equilibrium prior uses 1/N rather than float-adjusted market-cap weights.

## Repository layout

```
.
├── notebooks/
│   └── nse_quant_pipeline_v6.ipynb      # main pipeline, 67 cells, runs end-to-end
├── build/
│   ├── build_v6.py                      # assembles the v6 notebook from v5 + patches
│   ├── validate_v6.py                   # offline validation (syntax, causality, costs, IC)
│   └── cells/                           # replacement cells that build_v6.py splices in
├── data/
│   ├── universe/                        # 5 NIFTY-500 constituent snapshots + inc/exc log
│   └── outputs/                         # backtest artifacts, plots, audit CSVs
├── docs/
│   ├── RP1_briefing.pdf                 # 10-page methodology briefing
│   └── METHODOLOGY.md                   # formulas, symbols, references
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

## Reproducing the results

```bash
git clone https://github.com/<your-handle>/nse-quant-v6.git
cd nse-quant-v6
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/nse_quant_pipeline_v6.ipynb
```

The first run downloads ~2,700 daily price series from Yahoo Finance and takes 10–15 minutes on a broadband connection; subsequent runs use the disk cache in `yf_cache/`. Full pipeline execution is roughly 20–30 minutes on a laptop.

## Data sources and licensing

- Price data — Yahoo Finance via `yfinance` (research use).
- Universe — NIFTY-500 membership snapshots from NSE India.
- India VIX and sector indices — Yahoo Finance (`^INDIAVIX`, `^CNXIT`, `^NSEBANK`, etc.).
- Point-in-time fundamentals — **not** included. See §2.2 of `docs/RP1_briefing.pdf` for the source evaluation.

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{chava2026regimebl,
  author = {Chava, [Full Name]},
  title  = {Regime-Conditioned Black--Litterman Factor Allocation for Indian Equities},
  year   = {2026},
  note   = {Working paper},
  url    = {https://github.com/<your-handle>/nse-quant-v6}
}
```

## License

Code released under the MIT License (see `LICENSE`).
Data files in `data/universe/` remain the property of NSE India and are redistributed under fair-use for research reproducibility.
