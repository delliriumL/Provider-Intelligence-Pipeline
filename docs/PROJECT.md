# Provider Intelligence Pipeline

> **Trust-aware, offline-first provider directory validation** for HealthLynked / Kaggle — deterministic scoring, bounded LLM enrichment, human review, and full audit trail.

---

## Table of contents

1. [Executive summary](#executive-summary)
2. [Problem & approach](#problem--approach)
3. [Quick start](#quick-start)
4. [Demo results](#demo-results)
5. [Architecture](#architecture)
6. [Pipeline modules](#pipeline-modules)
7. [Scoring & decisions](#scoring--decisions)
8. [LLM policy](#llm-policy)
9. [Cost model](#cost-model)
10. [Data sources](#data-sources)
11. [Outputs & schema](#outputs--schema)
12. [Streamlit dashboard](#streamlit-dashboard)
13. [Demo walkthrough (judges)](#demo-walkthrough-judges)
14. [CLI reference](#cli-reference)
15. [Testing & CI](#testing--ci)
16. [Configuration](#configuration)
17. [Roadmap](#roadmap)
18. [Security & license](#security--license)

---

## Executive summary

Healthcare provider directories decay quickly: phone numbers change, addresses move, NPIs deactivate, specialties drift. Manual review of every record is expensive and slow; blind automation is unsafe.

**Provider Intelligence Pipeline** is an MVP that:

- Compares internal directory records against **NPPES**, **CMS Doctors & Clinicians**, **NUCC taxonomy**, and local practice HTML
- Normalizes phones, addresses, names, and specialties
- Computes **risk**, **field confidence**, and **conflict** scores with **expert-initialized** YAML weights (not fake production ML)
- Routes each record to `auto_update` | `human_review` | `no_change` | `do_not_update`
- Keeps a **full audit trail** and operational **Streamlit dashboard**
- Uses **bounded LLM enrichment** only on ambiguous cases — never to approve auto-updates

**Primary safety metric:** `false_auto_update_rate` on a synthetic benchmark (wrong auto-updates are costlier than missed updates or extra reviews).

---

## Problem & approach

| Challenge | Our response |
|-----------|--------------|
| No labeled training set from competition | Deterministic rules + configurable weights + synthetic benchmark |
| Per-record LLM/API cost | Offline bulk NPPES/CMS first; LLM gated to ~3–8% of records |
| Unsafe automation | Conservative thresholds; identity conflicts block auto-update |
| Operator trust | Audit log, score breakdowns, reviewer workflow in Streamlit |
| Honest MVP positioning | We document expert weights; audit logs enable future calibration |

### Design principles

1. **Safety over automation** — false auto-updates are costlier than extra human reviews
2. **Auditability** — every score component and decision is logged
3. **Config over code** — weights and thresholds in YAML
4. **Graceful degradation** — runs without internet or API keys

---

## Quick start

```bash
git clone <your-repo-url>
cd Provider-Intelligence-Pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make demo          # generate → pipeline → evaluate → cost
make test          # LLM_MODE=off pytest
make app           # Streamlit dashboard → http://localhost:8501
```

**Optional LLM demo** (local only — never commit `.env`):

```bash
cp .env.example .env
# Set LLM_MODE=auto, LLM_API_BASE, LLM_API_KEY, LLM_MODEL
LLM_MODE=auto make demo
```

**Docker:**

```bash
docker compose up --build
```

---

## Demo results

Typical offline demo run (60 synthetic records):

| Metric | Value |
|--------|-------|
| Records processed | 60 |
| Safe auto-updates | 8 |
| Human review queue | 22 |
| False auto-update rate | **0.0%** |
| Auto-update precision | **100%** |
| Change detection recall | ~47% (conservative routing) |
| Human review rate | ~37% |
| Cost / 1,000 records | **~$211** vs **$500** manual baseline |
| LLM calls (with credentials) | 3–4 (**5–7%** share, cap **8%**) |

**Showcase providers:**

- **HL_001** — `auto_update` at high confidence (address + phone confirmed by NPI Registry + CMS)
- **HL_002** — `human_review` when external sources disagree on address (conflict routing)
- **HL_003–HL_005** — bounded LLM enrichment demos (website parse, specialty, reviewer summary)

---

## Architecture

### Pipeline flow

```
Internal CSV + NPPES/CMS + NUCC + practice HTML
        ↓
   ingest → normalize → match → score → decide
        ↓                    ↓
   audit trail          gated LLM (enrich only)
        ↓
recommendations.json, queues, metrics, cost estimate
        ↓
   Streamlit operational dashboard
```

### Adaptive LLM boundary

The deterministic core **never** delegates to LLM:

- NPI validation
- Final confidence score
- Auto-update approval
- Decision action selection

LLM may assist when `LLM_MODE=auto` and gating passes:

- Website HTML extraction fallback
- Human-readable conflict explanations
- Specialty normalization fallback
- Reviewer summaries

All calls logged to `outputs/audit_llm_calls.csv`.

---

## Pipeline modules

| Module | Responsibility |
|--------|----------------|
| `config.py` | YAML + env merge (`LLM_MODE`, paths, costs) |
| `data_generation.py` | Synthetic directory + ground truth mutations |
| `ingest.py` | Load internal + external CSVs (demo/real) |
| `normalize.py` | Phone, address, name, specialty normalization |
| `npi.py` | Format + Luhn validation |
| `taxonomy.py` | NUCC lookup and fuzzy specialty mapping |
| `match.py` | Provider matching, duplicate detection |
| `scoring.py` | Risk, confidence, conflict, field changes |
| `decision_engine.py` | Conservative action routing |
| `audit.py` | Audit events and output writers |
| `llm.py` / `llm_enrichment.py` | Gated adaptive LLM with deterministic fallback |
| `website_parser.py` | Local HTML practice page parsing |
| `evaluation.py` | Synthetic benchmark + LLM comparison |
| `cost_model.py` | Operational cost estimation |
| `pipeline.py` | End-to-end orchestration |
| `cli.py` | Typer CLI and `run-all` |
| `export.py` | Competition JSON schema |

---

## Scoring & decisions

Scores are **expert-initialized policy weights**, not parameters learned from HealthLynked production labels.

### Risk score

Prioritizes records needing validation (not a calibrated error probability).

```
risk_score = Σ (weight_i × component_i)
```

| Component | Default weight |
|-----------|----------------|
| verification_age_risk | 0.25 |
| npi_status_risk | 0.20 |
| contact_quality_risk | 0.20 |
| address_quality_risk | 0.15 |
| duplicate_risk | 0.10 |
| source_disagreement_risk | 0.10 |

### Field confidence

Decides whether a proposed field update is safe:

| Component | Weight |
|-----------|--------|
| source_reliability | 0.30 |
| exact_identifier_match | 0.25 |
| cross_source_agreement | 0.20 |
| field_similarity | 0.15 |
| recency_score | 0.10 |

Record-level confidence = criticality-weighted average (NPI highest criticality).

### Conflict score

Pushes disagreeing records toward human review (identity, address, phone, status, specialty conflicts — capped at 1.0).

### Decision thresholds (`config/thresholds.yaml`)

| Threshold | Value | Use |
|-----------|-------|-----|
| auto_update_threshold | 0.88 | Min confidence for auto-update |
| no_change_threshold | 0.80 | Sufficient alignment |
| human_review_min_threshold | 0.55 | Lower bound for review |
| max_conflict_for_auto_update | 0.15 | Block auto-update when exceeded |
| high_risk_threshold | 0.70 | Risk prioritization / LLM gating |

### Primary evaluation metric

**`false_auto_update_rate`** — auto-updates that contradict synthetic ground truth.

Cost-sensitive loss (for future tuning):

```
loss = C_wrong_auto × wrong_auto_updates
     + C_missed × missed_real_changes
     + C_review × human_review_count
```

Defaults: wrong_auto=10.0, missed=3.0, review=0.50.

### Future calibration

When reviewer labels accumulate (`outputs/reviewer_actions.csv` + audit log):

1. Logistic regression on score components → recalibrated weights
2. Isotonic calibration on confidence → observed approval rates
3. Cost-sensitive threshold optimization under `false_auto_update_rate` SLA
4. Hard safety rules (identity conflict, invalid NPI) always preserved

---

## LLM policy

| Setting | Behavior |
|---------|----------|
| `LLM_MODE=off` | Rule-only; CI and air-gapped runs |
| `LLM_MODE=auto` | **Recommended** — gated calls, budget cap 8% |
| `LLM_MODE=force` | Experimentation only |

Gating considers risk, conflict, parser confidence, and demo showcase targets. Enrichment fields attach to recommendations but **do not change** `recommended_action`.

Environment (see `.env.example`):

```bash
LLM_MODE=auto
LLM_API_BASE=
LLM_API_KEY=
LLM_MODEL=
LLM_MAX_RECORD_SHARE=0.08
LLM_MIN_RISK_FOR_CALL=0.70
LLM_MIN_CONFLICT_FOR_CALL=0.35
```

---

## Cost model

Philosophy: **bulk data first**, deterministic processing for most records, LLM only on ambiguous cases, human review only when rules require it.

| Component | Default assumption |
|-----------|-------------------|
| Deterministic compute | ~$0.001/record |
| NPPES/CMS bulk | $0 |
| LLM assistance | ~$0.002/call |
| Human review | $0.50/record |
| Wrong auto-update risk | $10 × estimated rate |
| Missed update risk | $3 × estimated rate |

**LLM modes (estimated per 1,000 records):**

| Mode | LLM calls | Use case |
|------|-----------|----------|
| off | 0 | CI, baseline |
| auto | ~8% × 1.5 | Production default |
| force | ~8% × 3 | Experiments |

Output: `outputs/cost_estimate.json` — per-mode breakdown, savings vs manual review-all baseline.

---

## Data sources

### Internal directory

Demo: `data/sample/current_directory.csv`  
Production: HealthLynked CSV export via `ingest.py` real mode.

### NPPES / NPI bulk

- [NPI file downloads](https://download.cms.gov/nppes/NPI_Files.html)
- Demo: `data/sample/external_nppes_snapshot.csv`

NPI validates **identity enumeration**, not license or credentialing.

### CMS Doctors & Clinicians

- [Dataset mj5m-pzi6](https://data.cms.gov/provider-data/dataset/mj5m-pzi6)
- Demo: `data/sample/external_cms_clinicians.csv`

### NUCC taxonomy

- [taxonomy.nucc.org](https://taxonomy.nucc.org/)
- Sample: `data/reference/nucc_taxonomy_sample.csv`

### Practice websites

Local HTML snapshots in `data/raw/practice_websites/` — no default open-web scraping (reproducibility + legal clarity).

### Offline-first priority

1. Bulk NPPES/CMS files  
2. Local normalization + matching  
3. Optional NPI Registry API  
4. Gated LLM for ambiguous parsing  
5. Human review for conflicts  

---

## Outputs & schema

| File | Purpose |
|------|---------|
| `outputs/recommendations.json` | Competition-facing JSON |
| `outputs/recommendations_detailed.json` | Scores + debug fields |
| `outputs/human_review_queue.csv` | Review queue export |
| `outputs/auto_updates.csv` | Auto-applied changes |
| `outputs/audit_log.csv` | Full decision audit trail |
| `outputs/evaluation_metrics.json` | Synthetic benchmark |
| `outputs/cost_estimate.json` | Cost model by LLM mode |
| `outputs/audit_llm_calls.csv` | LLM call audit |
| `outputs/reviewer_actions.csv` | Streamlit reviewer log |
| `outputs/synthetic_ground_truth.csv` | Benchmark labels (demo) |

Committed samples for dashboard-without-run: `docs/sample_outputs/`.

---

## Streamlit dashboard

```bash
make app
# or: streamlit run app/streamlit_app.py
```

### Tabs

| Tab | Purpose |
|-----|---------|
| **Overview** | KPI row, action/confidence/risk charts, styled run summary panels |
| **Review Queue** | Filter, score breakdown, source evidence, LLM enrichment, reviewer actions |
| **Auto Updates** | High-confidence changes with field comparison |
| **Data Quality** | NPI, contact, staleness, duplicate flags |
| **Cost Model** | Scenario LLM modes, diagnostics, savings |
| **Audit Log** | Filterable events + full detail cards |

Dashboard reads `outputs/` when present; falls back to `docs/sample_outputs/`.

---

## Demo walkthrough (judges)

**~8 minutes**

| Step | Time | Action |
|------|------|--------|
| 1 | 0:00 | `make demo` — show terminal completion |
| 2 | 1:00 | Open Streamlit Overview — KPIs, charts, run summary |
| 3 | 2:30 | Review Queue → **HL_002** — conflict, human_review, source disagreement |
| 4 | 4:00 | Review Queue → **HL_003** — LLM enriched badge (if credentials run) |
| 5 | 5:00 | Auto Updates → **HL_001** — safe auto-update, high confidence |
| 6 | 6:00 | Cost Model — $211 vs $500 baseline, LLM share under cap |
| 7 | 7:00 | Audit Log — trace decision for HL_001 or HL_002 |
| 8 | 7:30 | Show `outputs/recommendations.json` competition schema |

**Key talking points:**

- Conservative routing: **0% false auto-updates** on synthetic benchmark
- Honest MVP: expert weights, not trained on hidden labels
- LLM enriches; deterministic core decides
- Audit trail ready for production calibration

---

## CLI reference

```bash
python -m provider_intelligence.cli --help

python -m provider_intelligence.cli run-all          # full demo workflow
python -m provider_intelligence.cli generate-demo-data
python -m provider_intelligence.cli run-pipeline [--mode demo|real] [--llm-mode off|auto|force]
python -m provider_intelligence.cli evaluate [--compare-llm]
python -m provider_intelligence.cli estimate-cost
```

**Makefile shortcuts:** `make demo`, `make demo-llm`, `make demo-compare`, `make test`, `make lint`, `make clean`.

---

## Testing & CI

```bash
LLM_MODE=off pytest          # all tests offline-safe
ruff check src tests app
```

GitHub Actions (`.github/workflows/ci.yml`): lint + pytest on push/PR to `main`.

Test coverage includes: NPI validation, normalization, decision engine, evaluation metrics, LLM gating, dashboard adapters, competition export schema.

---

## Configuration

| File | Purpose |
|------|---------|
| `config/thresholds.yaml` | Decision and LLM gating thresholds |
| `config/field_weights.yaml` | Risk/confidence/criticality weights |
| `config/source_reliability.yaml` | Per-source field reliability |
| `config/llm_config.yaml` | LLM use cases and cost assumptions |
| `config/app_config.yaml` | Demo settings, matching params |

---

## Roadmap

### Month 1 — Foundation

Real HealthLynked ingest, NPPES bulk refresh, deploy Streamlit for ops, security hardening.

### Month 2 — Coverage

CMS production connector, duplicate merge workflow, reviewer feedback loop, monitoring dashboards.

### Month 3 — Calibration

Logistic/isotonic calibration on audit + reviewer labels, LLM governance, scheduled batch jobs, auto-update rollback.

| MVP (today) | Production (target) |
|-------------|---------------------|
| Synthetic demo data | HealthLynked + CMS feeds |
| Expert YAML weights | Calibrated from reviewer outcomes |
| Local Streamlit | SSO, role-based review |
| Single-machine batch | Scheduled jobs + alerting |
| HTML snapshot parsing | Approved website ingestion |

---

## Security & license

- **Never commit** `.env`, API keys, or credentials
- Copy `.env.example` → `.env` locally only
- LLM calls optional; pipeline fully functional offline

**License:** MIT — see [LICENSE](../LICENSE).

---

*Provider Intelligence Pipeline — HealthLynked / Kaggle submission.*
