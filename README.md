# ops-recall

When an alert fires, the fastest path to a fix is usually a memory: *we have
seen this before*. That memory lives in post-mortems nobody rereads, Slack
threads nobody can find, and PagerDuty logs nobody exports.

ops-recall ingests all of it and, when a new alert fires, reconstructs the
incident: which past incidents this resembles, what the root cause turned out to
be, what the team actually did about it, **which of those conditions are true
again right now**, and the single best next action.

```
$ ops-recall demo

This alert looks 75% similar to INC-2197: checkout-api connection pool exhausted
while an ad-hoc VACUUM FULL held locks on orders_db.

Similar incidents
  INC-2197   75%    3 months ago  checkout-api connection pool exhausted while an ad-hoc VACUUM FULL...
           semantic 0.61 | lexical 1.00 | exact-keyword 1.00 | recency x0.94
  INC-1842   39%   21 months ago  checkout-api pool exhaustion during a non-concurrent index rebuild...
           semantic 0.48 | lexical 0.60 | exact-keyword 0.60 | recency x0.71

Live telemetry
  CONFIRMED A single session holds a lock on orders_db for minutes while dozens of queries queue behind it.
            1 blocking session(s) on orders_db; longest is pid 51774 at 383s running 'VACUUM FULL orders', 44 waiters.
  CONFIRMED checkout-api's HikariCP pool is pinned at max with a growing queue of pending connection requests.
            checkout-api pool at 50/50 (100%) with 118 request(s) queued.
  CONFIRMED No checkout-api deploy preceded the incident - the trigger came from the database side.
            No deploys of checkout-api in the last 6h.

Recommended next action
  1. Terminate the blocking backend.
     $ SELECT pg_terminate_backend(51774)
     This is what resolved INC-2197 (pool drained back to 6/50 active within 40 seconds)
     risk high | needs human approval
```

The last two sections are the point. Retrieval alone produces a plausible story;
re-testing that story's own conditions against the live system is what makes it
worth acting on — and what lets the system say *"this looks like INC-2197 but
nothing is actually blocking, so the resemblance is superficial"* instead of
confidently recommending the wrong fix.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/ops-recall demo                       # the flagship reconstruction
.venv/bin/ops-recall search "consumer lag climbing"
.venv/bin/ops-recall probe check_db_locks database=orders_db
.venv/bin/ops-recall serve                      # HTTP API on :8000
.venv/bin/python -m pytest                      # 76 tests, no network
```

Everything runs offline out of the box: the index is embedded Qdrant, the
embedder is deterministic, telemetry comes from a JSON snapshot, and the
reconstruction falls back to a deterministic assembler. Set
`ANTHROPIC_API_KEY` (or run `ant auth login`) and the narrative, the reasoning
about competing hypotheses, and the agentic follow-up checks switch on.

## How it works

```
post-mortems ─┐
Slack         ├─► ingest ─► correlate ─► index (Qdrant: dense + sparse)
PagerDuty     │                                    │
wiki ─────────┘                                    │
                                                   ▼
alert ───────────────────────────────► retrieve ──► no close match ──► "this is novel"
                                            │
                                            ▼
                                     check_evidence   replay each retrieved incident's
                                            │         probes against live telemetry
                                            ▼
                                     gather_quotes    what people actually said
                                            │
                                            ▼
                                      investigate     Claude runs extra read-only checks
                                            │
                                            ▼
                                      reconstruct     Claude writes the briefing
```

The pipeline is a LangGraph state machine (`ops_recall/agent/graph.py`).
Retrieval and evidence checking are deterministic and always run; the model is
reached last, with the facts already assembled. Every hop records timings and
counts in `state["trace"]`, which is returned on the API so a surprising answer
can be traced to the step that produced it.

### Hybrid search: three lanes

An alert contains two very different kinds of information, and one retrieval
strategy cannot serve both. `ERR_5012` and `db-prod-03` must match *exactly* —
embeddings put `db-prod-03` and `db-prod-07` in nearly the same place, which is
precisely wrong. "Requests are queuing and timing out" must match *semantically*
against "the pool never recovered on its own".

So three lanes, fused with renormalizing weights:

| Lane | What it is | Weight |
|---|---|---|
| `semantic` | Cosine over dense vectors | 0.5 |
| `lexical` | BM25 sparse vectors, native Qdrant sparse index | 0.3 |
| `keyword` | Fraction of the alert's extracted identifiers found in the incident | 0.2 |

```
similarity = (w_sem·semantic + w_lex·lexical + w_kw·keyword) / Σw  ×  recency
```

Details that matter:

- **The keyword lane drops out cleanly.** An alert with no error codes in it has
  nothing for that lane to match, so its weight is redistributed rather than
  capping every result at 0.8.
- **Both lanes are scored for every candidate.** A lane only returns scores for
  the points *it* retrieved, so a result found lexically would otherwise have no
  semantic score. The candidate set is re-scored directly against both stored
  vectors, which makes every breakdown complete instead of lane-dependent.
- **Each lane indexes different text.** The dense vector is built only from
  title, symptoms, keywords and services — an alert can only describe how
  something *presented*, so the semantic lane matches symptom against symptom.
  Including remediation prose pulls the vector toward how the incident was
  *fixed*, which is the part a firing alert cannot resemble. The sparse vector
  indexes everything, including the commands inside remediation steps, so an
  exact token is findable wherever it appears.
- **idf is per collection.** Incidents and fragments have separate BM25
  statistics; sharing one table let PagerDuty boilerplate distort the incident
  lane.
- **The lexical score is absolute, not relative.** It is normalized against the
  idf mass the query could achieve in this corpus, rather than against the best
  score in the result set — otherwise the top hit always looks perfect, even
  when the whole result set is poor.

Metadata filters (`services`, `max_age_days`, `min_severity`, `exclude_ids`) are
applied inside Qdrant, before scoring.

### Temporal awareness

```
recency = floor + (1 - floor) · 0.5 ^ (age_days / half_life_days)
```

Defaults: `half_life_days=365`, `recency_floor=0.6`. Decay is a multiplier on
the fused score, never a term added to it, so it can only break ties between
comparable matches — it can never manufacture similarity that is not there. The
floor is deliberate: a three-year-old exact match is stale, not irrelevant, and
burying it is a worse failure than surfacing it with a caveat. In the demo above
it is what puts the 3-month-old INC-2197 (×0.94) decisively ahead of the
21-month-old INC-1842 (×0.71).

### Signals and probes — re-testing history against the present

This is the central mechanism. A post-mortem declares the conditions that held
during the incident, each bound to a telemetry probe:

```yaml
signals:
  - id: blocking-lock-orders-db
    statement: A single session holds a lock on orders_db for minutes while dozens of queries queue behind it.
    probe: check_db_locks
    probe_args: {database: orders_db, min_seconds: 60}
```

At alert time every retrieved incident's signals are collected, **deduplicated**
(incidents in the same family share signals; a condition seen in three incidents
is stronger evidence, not three checks), **retargeted** at the service the
current alert names (a signal recorded as "checkout-api's pool is saturated"
generalizes to "the affected service's pool is saturated"), and run. Each comes
back `CONFIRMED`, `REFUTED`, or `UNKNOWN` — never guessed. `REFUTED` is often
the most informative label on the page, because it rules a story out.

Try it: empty `blocking_sessions` in `data/telemetry/current_state.json` and run
`ops-recall demo` again. Same alert, same archive, opposite conclusion.

### Agentic tool use

`ops_recall/agent/tools.py` exposes the telemetry surface to Claude via the SDK
tool runner, plus `search_incidents` and `get_runbook` over the same corpus.
This covers the case the deterministic replay cannot: the alert resembles a lock
incident, the lock probe comes back refuted, and the useful next question — "is
the service leaking connections instead?" — is one no post-mortem wrote down.

**Every tool is read-only.** Nothing terminates a backend, rolls back a deploy
or edits configuration. Remediation is proposed to a human with a risk label and
an approval flag; that is a boundary, not an unimplemented feature.

### What the model is and is not allowed to author

The model receives a schema (`ReconstructionDraft`) covering the headline, root
cause hypothesis, narrative, recommended actions, confidence and caveats. It has
no schema slot for similarity scores, evidence statuses, incident ids or
timestamps — those are computed by the system and merged in afterwards, so a
retrieval score or a telemetry result cannot be invented. Structured output is
requested through `client.messages.parse`; the system prompt is invariant across
alerts and cached.

## Adding your own data

Drop files into `data/seed/` (or point `OPS_RECALL_DATA_DIR` elsewhere) and run
`ops-recall ingest`.

- **`postmortems/*.md`** — YAML frontmatter (`id`, `title`, `severity`,
  `services`, `started_at`, `resolved_at`, `keywords`, `signals`,
  `pagerduty_incident`, `slack_thread`) plus `## Summary`, `## Symptoms`,
  `## Root cause`, `## Remediation` sections. Remediation steps are numbered and
  tagged `[diagnose]` / `[mitigate]` / `[fix]`, with an optional trailing
  `` `command` ``, `— @actor`, and `(outcome: ...)`. The `[fix]` tag is what
  separates "what finally worked" from "what we tried first".
- **`slack/*.json`** — `conversations.history` exports.
- **`pagerduty/*.json`** — incidents with their log entries.
- **`wiki/*.md`** — runbooks, indexed per `##` section.

Correlation is by explicit reference: the post-mortem's frontmatter declares its
PagerDuty incident and Slack thread, and those declarations let thousands of
unlabeled messages attach themselves to the right incident. Messages that name
an incident (`INC-2197`) link themselves. Nothing is linked by timestamp
proximity — ops channels run several conversations at once.

For continuous ingest, set `OPS_RECALL_SLACK_BOT_TOKEN` + `OPS_RECALL_SLACK_CHANNELS`
and `OPS_RECALL_PAGERDUTY_API_KEY`, then `ops-recall ingest --live`. The live
clients emit the same `Fragment` objects as the exports, deduplicated by source
id.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/alerts` | Reconstruct an alert. The main endpoint. |
| `POST` | `/v1/alerts/pagerduty` | Same, behind a PagerDuty v3 webhook adapter. |
| `POST` | `/v1/search` | Rank incidents against free text, with score breakdowns. |
| `GET` | `/v1/incidents`, `/v1/incidents/{id}` | Browse the archive. |
| `GET` | `/v1/probes` | What the agent can ask the live system. |
| `POST` | `/v1/reindex` | Rebuild the index from the configured sources. |
| `GET` | `/v1/stats`, `/healthz` | Operational. |

## Configuration

All settings take an `OPS_RECALL_` prefix and can live in `.env` (see
`.env.example`).

| Setting | Default | Notes |
|---|---|---|
| `REASONER` | `claude` | `template` for the deterministic fallback |
| `MODEL` | `claude-opus-5` | |
| `AGENTIC_CHECKS` | `true` | Let the model run its own follow-up probes |
| `EMBEDDER` | `hashing` | `voyage` for real semantic embeddings |
| `QDRANT_URL` / `QDRANT_PATH` | unset | Unset means in-memory, rebuilt at startup |
| `HALF_LIFE_DAYS` | `365` | |
| `RECENCY_FLOOR` | `0.6` | |
| `MIN_SIMILARITY` | `0.30` | Below this the answer is "this is novel" |
| `TOP_K` | `3` | |
| `WEIGHT_SEMANTIC` / `_LEXICAL` / `_KEYWORD` | `0.5` / `0.3` / `0.2` | |

## Taking this to production

Three things are stubbed, each behind an interface with a working
implementation on the other side:

1. **Embeddings.** `HashingEmbedder` is a hashed bag of words and character
   trigrams. It captures morphology and co-occurrence but not meaning: "pool
   starved" and "no free connections" stay far apart. It exists so the system
   runs and is tested offline. Set `OPS_RECALL_EMBEDDER=voyage` with a key and
   the semantic lane becomes genuinely semantic; nothing else changes.
2. **Telemetry.** `FixtureTelemetryProvider` reads a JSON snapshot. A production
   provider subclasses `TelemetryProvider` and overrides the `probe_*` methods
   to hit Prometheus, `pg_stat_activity` and the deploy tracker. Probe names,
   argument names and the `data["holds"]` contract stay identical, so the
   evidence layer does not change.
3. **Qdrant.** Embedded by default. Point `OPS_RECALL_QDRANT_URL` at a cluster
   and payload indexes are created automatically.

Also worth knowing: BM25 idf is corpus-global, so ingest re-indexes the whole
corpus. At post-mortem volumes (hundreds to low thousands of documents) that is
seconds; at a much larger scale it wants incremental idf updates.

## Design decisions worth arguing about

- **Similarity is explainable or it is worthless.** "87% similar" is only
  actionable if a responder can see that it came from an exact error-code match
  rather than a vague prose resemblance. Every response carries the full
  breakdown, which is also why the two lanes are queried separately instead of
  fused server-side by Qdrant.
- **Below the threshold, say nothing.** Forcing a bad analogy onto a novel
  incident is the failure mode that destroys trust in a system like this. The
  cold-start path returns "this is novel" with whatever telemetry it could
  still gather.
- **Recorded commands contain stale identifiers.** `pg_terminate_backend(48211)`
  refers to a process that died two years ago. The model is instructed to
  substitute a value from the telemetry it was given or name what the responder
  must look up; the deterministic path attaches a standing caveat.
- **Quotes are human commentary, not the loudest match.** The fragment most
  similar to a firing alert is always the *old alert text* from PagerDuty — a
  near-copy of the query that tells a responder nothing. Authored fragments are
  selected first, capped per incident.

## Layout

```
ops_recall/
  models.py            domain types (Incident, Signal, Alert, Reconstruction, ...)
  config.py            settings
  service.py           composition root
  ingest/              postmortems, slack, pagerduty, wiki, pipeline
  retrieval/           text, embeddings, sparse (BM25), temporal, store, ranker
  telemetry/           probe protocol + fixture provider
  agent/               evidence, tools, prompts, investigate, reconstruct, graph
  api/                 FastAPI app and schemas
  cli.py
data/seed/             10 synthetic incidents across 4 source systems
data/telemetry/        the live-system snapshot the probes read
tests/                 76 tests, no network
```
