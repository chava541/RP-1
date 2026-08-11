# NIFTY-500 universe data

Place the following five constituent snapshot CSVs and one inclusion/exclusion log in this folder before running the notebook:

| File | Snapshot date | Source |
|---|---|---|
| `nifty500_jan2016.csv` | 2016-01-29 | NSE India — historical index constituents |
| `ind_nifty500list_jan_28_2022.csv` | 2022-01-28 | NSE India |
| `Nifty500_Companies_list_2023.csv` | 2023-01-27 | NSE India |
| `nifty500_companies_Nov_2024.csv` | 2024-12-02 | NSE India |
| `ind_nifty500list_feb_1_2026.csv` | 2026-02-01 | NSE India |
| `Nifty500_Inc_Exc_2016-2020.xlsx` | 2016-02 → 2020-09 | NSE India — inclusion/exclusion circulars |

Filenames must match the `UNIVERSE_FILES` and `INCEXC_FILE` entries in the config cell of the notebook.

## Column expectations

Each CSV must contain at least a `Symbol` column and, ideally, an `Industry` column. Additional columns (Company Name, ISIN Code, Series) are ignored.

The Inc/Exc XLSX must contain columns for effective date, symbol, and inclusion/exclusion flag. The exact schema is discovered at load time in cell 6.

## Not committed to the repository

Only the ex-copyright metadata about these files is committed — the raw CSVs themselves are `.gitignore`d by extension (`data/raw/`, `data/prices/`) because they remain the intellectual property of NSE India. Fair-use redistribution for research reproducibility is generally accepted, but you should download them yourself from the NSE archive or your institution's data library.

## Why five snapshots, not a single point-in-time membership series

NSE does not publish a rolling membership file with entry/exit dates. The available primary sources are (a) the periodic snapshot files above and (b) the inclusion/exclusion circulars covering 2016–2020. Together they let the pipeline reconstruct a survivorship-aware time-varying universe: a stock enters on its first appearance in any snapshot and remains until its last appearance, with the Inc/Exc log providing finer transition dates where available.

## Delisted price history

Yahoo Finance does not serve delisted Indian tickers. The survivorship audit in §5b of the notebook identifies which missing names could bias the walk-forward result. If you obtain delisted price history from another source, drop per-symbol CSVs (with columns `Date, Open, High, Low, Close, Volume`, named `SYMBOL.csv`) into `../../delisted_overlay/` and the pipeline picks them up automatically on the next run.

The recipe for a full survivorship-complete rebuild from NSE bhavcopy files (via `jugaad-data`) is documented in the docstring of `bhavcopy_rebuild_stub` in §5b.
