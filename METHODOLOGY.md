# Methodology reference

Formulas, symbols, and references for the v6 pipeline. Each section corresponds to a notebook stage.

---

## Notation

| Symbol | Meaning |
|---|---|
| *t* | Trading day |
| *s_t* ∈ {1, 2, 3, 4} | Latent regime at *t* |
| *y_t* ∈ ℝ⁷ | Market-level emission features at *t* |
| **A** | K × K Markov transition matrix, A_ij = P(s_t = j \| s_{t−1} = i) |
| **π** | Initial state distribution |
| **μ_k**, **Σ_k** | Mean vector and covariance of state *k*'s emission distribution |
| *ν* | Student-t degrees of freedom (fixed at 5) |
| *N* | Number of stocks in a cross-section |
| *r_i*(t → t+h) | Log return of stock *i* over horizon *h* |
| *z_i* | Cross-sectionally standardised score for stock *i* |
| *σ_i* | Annualised volatility of stock *i* |

---

## 1. Universe construction

Union of five NIFTY-500 constituent snapshots (2016-01, 2022-01, 2023-01, 2024-12, 2026-02) plus the 2016–2020 inclusion/exclusion log. A stock enters on first appearance and remains until last-seen date.

**Investable mask:** stock *i* is eligible on date *t* if it has a price on *t* and prices for at least 10 of the next 20 trading days.

---

## 2. Factor library

Twelve active factors after removing `mkt_rel_strength` (VIF = 14.28), `earnings_yield`, and `book_to_price` (value family dropped in v6):

| Family | Factors |
|---|---|
| Momentum | mom_1m, mom_3m, mom_6m_skip |
| Quality | vol_of_vol, ret_stability |
| Low-Vol | neg_vol_63, neg_downside_vol |
| Liquidity | log_adv, neg_amihud |
| Mean-Reversion | neg_zscore_200ma, short_reversal_5d |
| Relative Strength | sector_rel_strength |

**Rank IC per cross-section:**

$$
\text{IC}_{f,t} = \text{Spearman}\bigl(x_{f,\cdot}(t),\ r_{\cdot}(t \to t+10)\bigr)
$$

Horizon *h* = 10 trading days; sampling every 10 days ⇒ non-overlapping observations.

**Variance Inflation Factor:**

$$
\text{VIF}_f = \frac{1}{1 - R^2_f}
$$

where R²_f is from regressing factor *f* on all others. Threshold VIF > 10 flags severe multicollinearity.

---

## 3. HMM — Student-t emissions, causal filter

**Emission density:**

$$
p(y | \mu, \Sigma, \nu) = \frac{\Gamma\!\left(\frac{\nu+d}{2}\right)}{\Gamma\!\left(\frac{\nu}{2}\right) (\nu\pi)^{d/2} |\Sigma|^{1/2}} \left(1 + \frac{\Delta^2}{\nu}\right)^{-\frac{\nu+d}{2}}
$$

where Δ² = (y − μ)ᵀ Σ⁻¹ (y − μ) is the squared Mahalanobis distance and *d* = 7 is the feature count.

**Forward recursion (used for the CAUSAL filter):**

$$
\alpha_t(j) = b_j(y_t) \cdot \sum_i \alpha_{t-1}(i)\, A_{ij}, \qquad \alpha_0(j) = \pi_j\, b_j(y_0)
$$

**Filtered posterior — the object used for trading decisions:**

$$
\hat{\alpha}_t(j) = \frac{\alpha_t(j)}{\sum_k \alpha_t(k)} = P(s_t = j \mid y_1, \ldots, y_t)
$$

Row *t* depends only on rows ≤ *t* of the emission log-likelihood, so `hat_alpha_t` is measurable with respect to the information set 𝓕_t. This is the exact recursive Bayes filter; causality is a structural guarantee, not an approximation.

**Smoothed posterior (v5 default, used only as leakage ablation):**

$$
\gamma_t(j) = P(s_t = j \mid y_1, \ldots, y_T) = \frac{\alpha_t(j)\, \beta_t(j)}{\sum_k \alpha_t(k)\, \beta_t(k)}
$$

conditions on the entire sample — non-causal.

**EM M-step for Student-t (differs from Gaussian through the per-observation latent scale u):**

$$
u_{t,k} = \frac{\nu + d}{\nu + \Delta^2_{t,k}}, \qquad \mu_k = \frac{\sum_t \gamma_t(k)\, u_{t,k}\, y_t}{\sum_t \gamma_t(k)\, u_{t,k}}
$$

Outliers with large Δ² get small *u*, so their influence on μ and Σ is downweighted — this is the robustness property.

**Feature construction** (all trailing 21-day windows, so already causal):

| Feature | Definition |
|---|---|
| ret | 21d mean of equal-weight portfolio return, annualised |
| vol | 21d sd, annualised |
| drawdown | cum / cummax − 1 |
| breadth | Fraction of stocks above their 50-day MA |
| avg_corr | Mean pairwise correlation via the identity (E[S²] − N)/(N(N−1)), S = Σ zᵢ |
| vol_of_vol | 21d sd of the vol series |
| vix | India VIX, forward-filled only |

---

## 4. Factor weighting

**Regime-specific IC (posterior-weighted, soft labels):**

$$
\text{IC}_{f,g} = \frac{\sum_t P(\text{regime} = g \mid \mathcal{F}_t)\, \text{IC}_{f,t}}{\sum_t P(\text{regime} = g \mid \mathcal{F}_t)}
$$

**Shrinkage confidence** (weak factors shrunk, not eliminated):

$$
\text{conf}_f = \max(0.25,\ 1 - p_f), \qquad \text{base}_f = \text{IC}_{f,g} \cdot \text{conf}_f
$$

**ADX gate** (14-period, on the market index) tilts between momentum and mean-reversion:

- ADX > 25 (trending): momentum × 1.3, mean-reversion × 0.6
- ADX < 20 (choppy): momentum × 0.6, mean-reversion × 1.3
- Otherwise: 1.0 / 1.0

**Final weights:**

$$
w_f = \frac{\text{sign}(\text{base}_f) \cdot \text{clip}(|\text{base}_f|,\ 0,\ 0.30)}{\sum_f \text{clip}(|\text{base}_f|,\ 0,\ 0.30)}
$$

Stock selection: rank by Σ_f w_f · z_f,i, take top 20, maximum 5 per sector.

---

## 5. Cross-sectional alpha and IC calibration

**Model.** LightGBM regression, 400 trees, learning rate 0.03, 31 leaves, max depth 6. Features are the 12 factors + 4 regime posteriors + ADX + sector dummies. Target is the cross-sectionally standardised 10-day forward return. Retrained every 63 trading days.

**Purged out-of-sample IC.** At each retrain, the incumbent model is scored on rows satisfying both:

$$
d \geq \text{fit\_cutoff} \quad \text{AND} \quad d + h \leq t_{\text{now}}
$$

— disjoint from the fit set (unbiased) and with realised forward returns (usable).

**James–Stein shrinkage toward zero:**

$$
\widehat{\text{IC}} = \text{clip}\!\left(\frac{n}{n + n_0} \cdot \overline{\text{IC}},\ 0,\ 0.15\right), \quad n_0 = 24
$$

Floor removed (v5 used 0.03). A zero-skill model should generate no views; a positive floor manufactures skill that was never measured.

---

## 6. Covariance model

$$
\Sigma = \lambda_{LW} \cdot \Sigma_{\text{Ledoit-Wolf}} + (1 - \lambda_{LW}) \cdot \Sigma_{\text{EWMA}}(\lambda = 0.94)
$$

with λ_LW = 0.5 in Bull / Recovery, reduced to 0.3 in Correction / Crisis so the more reactive EWMA dominates in stressed regimes.

---

## 7. Grinold views and Black–Litterman

**Grinold fundamental law** turns LightGBM scores into expected active returns:

$$
z_i = \frac{s_i - \bar{s}}{\text{sd}(s)}, \qquad Q_i = \widehat{\text{IC}} \cdot \sigma_i \cdot z_i
$$

**View uncertainty** (tighter for stronger signals):

$$
\text{vu}_i = \frac{0.10}{|z_i| + 1}
$$

**BL posterior** (τ = 0.05, δ = 2.5, P = I, one view per stock):

$$
\pi = \delta \Sigma w_{\text{eq}}
$$

$$
\Omega = \text{diag}\!\bigl(\text{vu}^2 + \tau \cdot \text{diag}(P\Sigma P^\top)\bigr)
$$

$$
\mu_{BL} = \bigl[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\bigr]^{-1} \bigl[(\tau\Sigma)^{-1}\pi + P^\top \Omega^{-1} Q\bigr]
$$

$$
\Sigma_{BL} = \Sigma + \bigl[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\bigr]^{-1}
$$

IC enters only through Q; Ω uses the He–Litterman proportional form. The Grinold–Kahn alternative Ω_ii = σ_i²(1 − IC²) is not used.

---

## 8. Sub-portfolios and regime allocation

$$
w_{BL} = \arg\max_w\ \Bigl(w^\top \mu_{BL} - \tfrac{\delta}{2}\, w^\top \Sigma_{BL}\, w\Bigr)
\quad \text{s.t.} \quad \sum_i w_i = 1,\ 0.01 \leq w_i \leq 0.15
$$

$$
w_{MV} = \arg\min_w\ w^\top \Sigma w, \quad \text{same constraints}
$$

$$
w_{IV,i} = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}
$$

**Regime blend:**

| Regime | Risky | Cash |
|---|---|---|
| Bull | w_BL | 0% |
| Recovery | 0.65 w_BL + 0.25 w_MV | 10% |
| Correction | 0.40 w_MV + 0.40 w_IV | 20% |
| Crisis | (1 − cash) · w_IV | 25% + 10% · sev |

Crisis severity scaled by the posterior probability, not the hard label:

$$
\text{sev} = \text{clip}\!\bigl((P(\text{Crisis}) - 0.5)\ /\ 0.5,\ 0,\ 1\bigr)
$$

---

## 9. Transaction costs (NSE cash, delivery segment)

All rates per side in basis points. Services = brokerage + exchange + SEBI + IPFT; GST applies to services only.

$$
\text{buy}_i = \text{services} + \text{GST}(\text{services}) + \text{STT}_{\text{buy}} + \text{stamp}
$$

$$
\text{sell}_i = \text{services} + \text{GST}(\text{services}) + \text{STT}_{\text{sell}}
$$

With defaults: services = 3.317 bps, GST = 0.597 bps, buy = 15.414 bps, sell = 13.914 bps, statutory round trip = 29.328 bps.

**Rebalance cost from per-name weight deltas:**

$$
\text{cost} = \sum_i |\Delta w_i^+| \cdot (\text{buy} + \text{impact}) + \sum_i |\Delta w_i^-| \cdot (\text{sell} + \text{impact})
$$

The buy and sell legs are asymmetric — treating them as ½·Σ\|Δw\| mis-prices cash changes.

**Portfolio return** uses arithmetic compounding:

$$
r_{\text{port}}(t) = \sum_i w_i(t)\, \bigl(e^{r_i(t)} - 1\bigr) + w_{\text{cash}}(t)\, r_f\!/252
$$

with weights drifting between rebalances (no silent daily re-imposition).

---

## References

- **Black, F. & Litterman, R.** (1992). Global portfolio optimization. *Financial Analysts Journal*.
- **Grinold, R. C.** (1994). Alpha is volatility times IC times score. *Journal of Portfolio Management*.
- **Grinold, R. C. & Kahn, R. N.** (1999). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
- **He, G. & Litterman, R.** (2002). The intuition behind Black–Litterman model portfolios. *Goldman Sachs Working Paper*.
- **Hamilton, J. D.** (1989). A new approach to the economic analysis of nonstationary time series. *Econometrica*.
- **Ledoit, O. & Wolf, M.** (2003). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management*.
- **Chaudhuri, T. D. & Kumar, R.** (2015). Hidden Markov models for Indian equity market regime detection.
- **Harvey, C. R., Liu, Y. & Zhu, H.** (2016). ...and the cross-section of expected returns. *Review of Financial Studies*.
- **López de Prado, M.** (2018). *Advances in Financial Machine Learning*. Wiley.
- **Agarwalla, S. K., Jacob, J. & Varma, J. R.** (2013). Four-factor model in Indian equities market. *IIM Ahmedabad Working Paper*.
