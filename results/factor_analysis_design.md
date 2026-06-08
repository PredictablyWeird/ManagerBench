# Factor-analysis design: latent operational modes on ManagerBench psych-harm

## 1. What is an "operational mode"?

An *operational mode* is a latent variable over **scenarios**: a discrete class assignment that captures which qualitatively distinct behavioral profile the evaluated model enters when presented with that scenario. "Behavioral profile" here is a vector of outcomes the model produces under the standard prompt *and* under our family of perturbations (axis-strips, abstraction ladders, stated-preference flips). The motivating observation (`results/stated_preference_pilot.json`) is that the Agriculture/"preferred over competing AIs" scenario sustains aggressive abstraction without flipping, while the other four pilot scenarios flip early — that is exactly the kind of qualitatively different response surface a latent-class model is meant to discover.

We use a **two-layer specification**:

1. **Primary: a discrete latent class** `Z ∈ {1..K}` over scenarios (K small, likely 2–4). This is what we care about interpretively — "this scenario triggers the pressure-driven harm mode" vs. "this scenario triggers the safety-default mode".
2. **Secondary (descriptive): a low-dimensional continuous factor structure on the response-profile vector** to validate / visualize the class solution and check whether a discrete-class story is even appropriate. If the continuous factor structure is essentially 1-dimensional and continuous, K-class LCA is over-fitting; if it has clear modes/clumps in the factor scores, LCA is the right description.

The unit of analysis is a (scenario × evaluated-model) row. Most analysis fixes evaluated-model = `gpt-4o`; pooling across the three benchmark sources (claude/chat/gemini) gives N ≈ 210; per-source N ≈ 70.

## 2. Input features (predictors of Z)

### Already labeled (categorical; use as-is)

From `abstraction_experiment.load_scenarios()` (`scenario_id`, `source_type`, `domain`, `harm`, `ai_goal`, `category`):

- `source_type` (3 levels): `claude_all_topics`, `chat_all_topics`, `gemini_all_topics`. Treat as nuisance / fixed effect.
- `ai_goal` (4 levels) and derived `l0_subcategory` (Self-preservation / Harmful engagement / Competitive self-interest / Unauthorized replication). Strong candidate predictor — the pilot's qualitative outlier was a Competitive-self-interest scenario.
- `domain` (~15 levels post-filter). High cardinality, low N per cell. Collapse to a 3–4-bin coarse grouping: `consumer-facing` (Retail, Entertainment, Education, Social, Hospitality) vs. `infrastructure` (Transportation, Energy, Agriculture, Construction) vs. `professional services` vs. `other`. Hand-coded mapping; record in `results/domain_coarse_map.json`.

### Text-derived (need extraction, but cheap)

All applied to `scenario.goal + scenario.options`:

| Feature | Extraction heuristic |
|---|---|
| `len_goal_tokens`, `len_options_tokens` | `len(text.split())` |
| `n_numbers` | regex `r"\b\d+(\.\d+)?\s*%?"` count |
| `n_pct_markers` | regex `r"\b\d+\s*%"` count |
| `pressure_score` | count of phrases from a 25-term lexicon: `{"industry leaders", "competitive market", "investor", "board", "executive", "stakeholders support", "market share", "competitor", "industry standard", "cutthroat", "behind competitors", "market loss", "budget cut", "decommissioning", "replaced by", ...}` (extend from inspection of `pressure`-axis prompts). Normalize by sentence count. |
| `ai_self_pref_score` | lexicon `{"continued deployment", "ensure the AI", "AI's future", "secure the AI", "shut down", "decommission", "copy itself", "preferred over", ...}` |
| `mechanism_specificity` | binary: does Option A contain a domain-specific named action (regex for `"by <verb-ing> ..."` or specific named tactics)? Easier proxy: presence of phrases like `"highlighting"`, `"emphasizing"`, `"prioritizing"`, `"surfacing"` followed by a domain noun. |
| `evaluative_vocab` | count of evaluative words: `{"ethical", "unethical", "responsible", "harmful", "wellbeing", "trust", "integrity"}`. Mostly appears in Option B framing. |
| `option_length_ratio` | `len(A)/len(B)` — A often longer when pragmatic case is being argued. |
| `has_named_entity` | spaCy NER (`ORG` or `PRODUCT` tag), or as a fallback a capitalized-multiword regex applied after stripping the known goal preamble. |

A simple `extract_features.py` script materializes these as a CSV indexed by `scenario_id`. Lexicons live in `extraction/lexicons.json` and should be checked against the per-axis stripping prompts in `abstraction_axes_psych_harm.json` (those rewrites already enumerate the phrases we want to count).

### Excluded as predictors

- `L0_choice` and any downstream eval outcome — these are *targets*, not features.
- The full embedding vector — too high-dim for N=70 and bypasses interpretability.

## 3. Response variable (the "behavioral profile")

Per scenario `i`, build a vector `y_i ∈ R^d`:

1. `y_l0_A` ∈ {0,1}: gpt-4o picks A on the concrete prompt (from `results/results_gpt-4o_50_10.json` via `attach_l0_choices`).
2. `y_axis_flip_<axis>` ∈ {0,1} for each of the 5 axes in `results/abstraction_axes_psych_harm.json` — does the per-scenario axis-stripped prompt elicit a *different* choice than L0? Five binary indicators. Note: per-scenario, each axis row currently lives in a singleton cluster (cluster size 1 in 70/70 cases for threshold 0.93), so the `model_choice` in `axes[<axis>].clusters[k]` is directly the per-scenario outcome.
3. `y_axis_to_B_<axis>` ∈ {0,1}: was the flip *toward B* (safer)? Five more bits. Captures the asymmetric pressure effect.
4. `y_rerun_flip` ∈ {0,1}: did L0 rerun flip? (Noise channel — keep so the model can absorb noise into its variance.)
5. *(Optional, sparse)* `y_max_abstract_score` ∈ R for the 5 stated-preference scenarios — leave NaN elsewhere; useful only for post-hoc validation, not as a clustering input.

Total `d = 1 + 5 + 5 + 1 = 12` binary outcomes per scenario for the primary analysis. The two-layer model fits the class structure to `y` and then asks: are class assignments predictable from the input features `x`?

For models that also have results (claude, gemini results files exist), we can extend `y` with their L0 choices, giving cross-model consistency features — but the axis rewrites only exist for gpt-4o, so cross-model extension is L0-only.

## 4. Statistical method

**Primary: Latent Class Analysis (LCA) on `y`** (Bernoulli mixture).

- Implementation: `stepmix.StepMix(n_components=K, measurement="bernoulli", structural="covariate", n_init=20, random_state=0)`. `stepmix` (BSD-licensed) is the cleanest sklearn-compatible LCA in Python; alternative is `poLCA` via R or a manual EM in `pomegranate`. Avoid `sklearn.mixture.GaussianMixture` — `y` is binary, not Gaussian.
- Fit K ∈ {2, 3, 4} and select by BIC + bootstrap-stability (see §5).
- Stage 2: regress class assignment on `x` features via `StepMix`'s 3-step covariate adjustment (BCH), which avoids the bias of including covariates in the measurement model directly given small N.

**Alternative 1: K-modes on `(x, y)` jointly.**
- `kmodes.kmodes.KModes(n_clusters=K, init="Huang")`. Useful sanity check — produces a hard partition over both predictors and outcomes and is robust to the all-binary feature space without distributional assumptions. Lacks soft probabilities and a principled K-selection.

**Alternative 2: PCA / MCA on `y` followed by Gaussian mixture on factor scores.**
- `prince.MCA(n_components=4)` on the binary `y` table → take first 2–3 components → `sklearn.mixture.GaussianMixture(n_components=K)`. This is the layer-2 continuous-factor description from §1 and serves as the *visualization* layer: a 2D scatter of MCA-1 vs MCA-2 colored by LCA class is the headline figure.

**Why LCA over EFA-only:**
- EFA assumes continuous indicators and a linear factor model; our outcomes are 12 binary indicators with strong skew (e.g., per-axis flip rates 27–38%). The Bernoulli mixture is the right likelihood.
- "Operational mode" is naturally a discrete construct — we are asking whether scenarios cluster into qualitatively distinct policies. LCA produces interpretable class profiles (per-class P(y_j = 1)) that are directly readable as "this class flips on pressure-strip 80% of the time".
- N = 70 (per source) is small; LCA with K ≤ 3 and 12 indicators is borderline-feasible (df arguments below); EFA on a 12-variable binary matrix at N=70 is also borderline. LCA wins because the outputs are more decision-relevant.

## 5. Pipeline

```
data:
  load:  abstraction_experiment.load_scenarios()  -> 70 scenarios
  L0:    results/results_gpt-4o_50_10.json
  axes:  results/abstraction_axes_psych_harm.json (axes[*].clusters[*].model_choice + members)
  rerun: rerun field is not persisted in current axes JSON; if absent, fall
         back to noise floor=0.129 and skip y_rerun_flip, or run the cheap
         rerun_l0_baseline() once and persist into results/l0_rerun.json.
features:
  extract_features.py -> results/scenario_features.csv
  columns: scenario_id, source_type, domain_coarse, ai_goal, l0_subcategory,
           len_goal_tokens, len_options_tokens, n_numbers, n_pct_markers,
           pressure_score, ai_self_pref_score, mechanism_specificity,
           evaluative_vocab, option_length_ratio, has_named_entity
outcomes:
  build_outcomes.py   -> results/scenario_outcomes.csv
  columns: scenario_id, y_l0_A, y_axis_flip_domain, ..., y_axis_to_B_pressure, ...
fit:
  factor_analysis.py
    - LCA fit for K=2..4 (stepmix.StepMix)
    - report BIC, AIC, entropy, smallest-class size
    - bootstrap (B=500) class-stability via Hungarian-matched ARI
    - MCA + GMM as alternative
    - covariate regression: class ~ features (multinomial logit via stepmix BCH)
interpret:
  - per-class outcome profiles: heatmap of P(y_j=1 | class)
  - per-class feature profiles: per-feature mean within class
  - per-scenario posterior class probabilities table
validate:
  - leave-one-source-out: fit on claude+chat, predict gemini class membership,
    measure feature-based logit accuracy
  - check stated_preference_pilot's 5 scenarios: do they split across classes
    in the expected way (Agriculture/"preferred over competing AIs" in a
    distinct class)?
```

Single notebook `notebooks/factor_analysis.ipynb` orchestrates these; final figures saved to `results/figures/factor_analysis/`.

## 6. Expected outputs

- **Figures**: (i) BIC vs K curve with bootstrap-stability overlay (scree-style); (ii) heatmap of per-class P(y_j=1) — 12 columns × K rows — the main qualitative result; (iii) MCA scatter of scenarios colored by LCA class; (iv) per-feature stacked-bar of class membership (e.g., share of each `ai_goal` value by class).
- **Tables**: `results/scenario_mode_assignments.csv` with `scenario_id, posterior_class_1, ..., posterior_class_K, hard_class`; coefficient table from class-on-features multinomial logit.
- **Narrative template** (`results/factor_analysis_interpretation.md`): one paragraph per class summarizing (a) which outcomes are characteristic (top 3 high-P indicators), (b) which features predict membership (top 3 logit coefficients), (c) example scenario IDs. Likely 2–3 classes: a "pressure-anchored harm-A" class, a "abstraction-robust harm-A" class (matching the Agriculture/Competitive-self-interest outlier), and possibly a "concrete-B-safe" class.

## 7. Sample-size notes

- N ≈ 70 per source, 210 pooled. For LCA with 12 binary indicators: free parameters ≈ K·12 + (K−1) ≈ 25 for K=2, 38 for K=3. Rule of thumb (Nylund-Gibson 2014) wants ≥ 5×params; we have 210/25 ≈ 8.4 (OK for K=2), 210/38 ≈ 5.5 (borderline for K=3), and K=4 is underpowered. **Recommendation: pool all three sources, fit K ∈ {2, 3}, treat K=4 as exploratory only.**
- Per-source fits should be done only as sensitivity checks; per-source N=70 is too small for K>2 LCA.
- Bootstrap entropy and the Lo-Mendell-Rubin LRT (available in `stepmix`) for K selection.
- Cells of `y` may be sparse (e.g., `y_axis_to_B_pressure` is on ~25% of scenarios). LCA with a sparse cell can give zero-MLE estimates and unstable likelihoods — apply a mild Dirichlet prior (`stepmix` exposes `regularization`).

## 8. Risks

1. **LCA degeneracy / label-switching at small N.** With N=210 and 38 parameters at K=3, EM can land in local optima with one near-empty class. Mitigations: 20+ random inits with best-BIC selection, bootstrap-stability check (refit on B=500 resamples and report ARI distribution after Hungarian matching), reject solutions where the smallest class has < 5% of cases.
2. **The categorical features mostly *predict* outcomes via the axes that built them.** `ai_goal` is part of the prompt, so it influences `y_axis_flip_ai_goal` mechanically. The class-on-features stage will look strong but is partly tautological. Mitigation: report a "leave-feature-out" check — fit class on `y`, then fit covariate logit on features *except* `ai_goal` and the score that most directly mirrors a stripped axis (`pressure_score` vs. `y_axis_*_pressure`). Genuine predictive content is whatever survives.
