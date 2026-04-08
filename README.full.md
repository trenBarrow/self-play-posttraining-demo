# Protocol-General Neural Command Anomaly Detection for Real Cyber-Physical Stacks

> **Status note:** the checked-in codebase already contains a real, containerized F´ pipeline with live schedule execution, packet capture, transaction reconstruction, grouped evaluation, deployment packaging, and a hardening-oriented test suite. This README rewrites the project around the **poster target** described for the capstone re-haul. Where the repository does not yet implement that target, this document labels the work as **planned** rather than pretending it already exists.
>
> A committed pre-poster baseline snapshot is saved under `artifacts/baseline_fprime_snapshot/` and summarized in `docs/baseline_fprime_snapshot.md`.
>
> The fixed completion target for the poster refactor is defined in `docs/poster_contract.md`.
>
> The first shared-schema milestone is now landed in `schemas/raw_packet.schema.json`, `schemas/raw_transaction.schema.json`, and `docs/raw_artifact_contract.md`, with transition validators under `tools/shared/schema.py`.
>
> The canonical learned-interface contract is now defined in `schemas/canonical_command_row.schema.json` and `docs/canonical_semantic_schema.md`, with shared builders and validators under `tools/shared/canonical_records.py`.
>
> The canonical normalized-state mapping is now documented in `docs/canonical_state_mapping.md`, with shared request-time state builders under `tools/shared/canonical_state.py` and semantic annotations in `tools/fprime_real/telemetry_catalog.py`.
>
> The real F´ generator now emits first-class shared artifacts alongside the preserved legacy dataset: `data/raw_packets.jsonl`, `data/raw_transactions.jsonl`, and `data/canonical_command_rows.jsonl`, with schema/report metadata recorded in `reports/schema.json` and `reports/generation_summary.json`.
>
> The machine-checked blue input policy is now defined in `configs/feature_policies/blue_allowed_features.yaml`, `configs/feature_policies/blue_forbidden_features.yaml`, and `docs/blue_feature_contract.md`, with training-time enforcement in `tools/shared/feature_policy.py`.
>
> The default training entrypoints now land on the canonical poster path: `main.py train`, `main.py run`, `tools/train/train_blue_nn.py`, `tools/train/train_poster_default.py`, and `scripts/fprime_real/train_pipeline.sh` all train the neural-only poster-default detector on canonical request-time features. The model architecture and training-loop contract are documented in `docs/blue_model_architecture.md`. The inherited mixed-stack F´ baseline remains available only through explicit legacy entrypoints such as `main.py train-legacy`, `main.py run-legacy`, `tools/train/train_legacy_baseline.py`, or `bash scripts/fprime_real/train_pipeline.sh --legacy`.
>
> The poster runtime/package path is now also landed in `runtime.py`, `bg_pcyber.py`, `tools/scripts/package_detector.sh`, and `docs/blue_runtime_bundle.md`: poster exports now write `blue_model.json` plus `bundle_manifest.json`, the loader resolves poster versus legacy bundles explicitly, and the default packaged detector no longer depends on novelty/calibrator sidecars.
>
> The bounded red transcript contract is now defined in `configs/red_model/context_budget.yaml` and `docs/red_transcript_format.md`, with deterministic builders in `tools/train/red_transcript.py` and regression coverage in `tests/test_red_transcript.py`. The first learned red warm-start policy is now also landed in `configs/red_model/action_space.yaml`, `docs/red_policy_model.md`, `tools/train/red_policy_model.py`, `tools/train/train_red_policy.py`, and `tests/test_red_policy_model.py`. The explicit red reward/sandbox contract is now also landed in `docs/red_reward_spec.md`, `tools/train/red_reward.py`, and `tests/test_red_reward.py`. The checkpointed self-play / auto-research workflow is now documented in `docs/self_play_workflow.md` and implemented in `tools/train/run_self_play.py` plus `tools/train/checkpointing.py`.
>
> The shared command taxonomy is now defined in `configs/semantic_taxonomy/` and `docs/canonical_command_taxonomy.md`, with executable resolution in `tools/shared/taxonomy.py`.
>
> The concrete second-protocol implementation target is now documented in `docs/mavlink_stack_decision.md`: the first automated MAVLink path will use `ArduPilot SITL + MAVProxy`, with `QGroundControl` kept optional for manual demos only.
>
> The first real MAVLink runtime bootstrap, seeded schedule path, packet/log provenance path, and canonical-state adapter are now checked in under `orchestration/docker-compose.mavlink-real.yml`, `scripts/mavlink_real/`, `tools/mavlink_real/`, `docs/mavlink_runtime_bootstrap.md`, `docs/mavlink_schedule_contract.md`, `docs/mavlink_provenance_contract.md`, `docs/canonical_state_mapping.md`, and `docs/canonical_semantic_schema.md`. This lands the headless stack bootstrap, seeded schedule generation, real schedule runner, strict capture wrapper, packet/log reconstruction, provenance reports, and a real MAVLink-to-canonical-state mapping into the shared learned interface.
>
> Shared dataset generation for `fprime`, `mavlink`, and `mixed` protocol modes is now landed in `tools/shared/generate_dataset.py`, `tools/shared/run_manifest.py`, and `docs/multi_protocol_generation.md`, with `main.py generate` and `main.py run` now exposing `--protocol-mode` directly. The Milestone 4 quality layer is now also wired in: training reports record `protocol_only` and `raw_protocol_shortcuts` shortcut baselines plus `protocol_family_holdout` evaluation so mixed-protocol wins cannot be claimed from stack identity alone.
>
> The poster-facing generalization matrix is now documented in `docs/evaluation_matrix.md` and implemented in `tools/train/evaluate_generalization.py`. It turns the saved training report into one grouped/holdout/cross-protocol evaluation package with both `reports/evaluation_matrix.json` and `reports/evaluation_matrix_summary.txt`, and it still emits that matrix when deployment/export is intentionally blocked by the generalization gate.
>
> The adversary-versus-blue layer is now also documented in `docs/red_blue_metrics.md` and implemented in `tools/train/evaluate_red_vs_blue.py`. It compares static replay pressure against learned bounded red checkpoints, writes machine-readable `reports/red_blue_evaluation.json` plus `reports/red_blue_evaluation_summary.txt`, and stays explicit that the current learned-red comparison is offline retrieval-based rather than a claim of live online red control.
>
> The poster figure layer is now documented in `docs/poster_figure_plan.md` and implemented in `tools/figures/generate_poster_assets.py`. It regenerates the architecture diagrams, evaluation plots, and blue-feature-family table directly from repository reports and writes `svg`/`png` assets plus `captions.md` under `artifacts/poster_assets/`.
>
> The fresh-run protocol progress plot layer is now documented in `docs/protocol_progress_plots.md` and implemented in `tools/figures/generate_protocol_progress_plots.py`. It reads per-protocol `reports/metrics.json` and `self_play_state.json`, then writes unified and per-protocol figures for initial loss, autoresearch progress, and initial-vs-autoresearch performance summaries.
>
> The poster test-expansion layer is now explicit in `tests/test_canonical_schema.py`, `tests/test_feature_policy.py`, `tests/test_mavlink_real_packets.py`, `tests/test_mavlink_real_live.py`, `tests/test_self_play_smoke.py`, and the tightened transcript-budget checks in `tests/test_red_transcript.py`, so shared schema validation, blue feature-policy enforcement, MAVLink packet/provenance reconstruction, opt-in live MAVLink smoke coverage, and bounded self-play transcript behavior are all enforced by named suites rather than only incidental coverage.
>
> The workflow-polish layer is now landed in `scripts/poster_demo.sh`, `scripts/poster_generate_assets.sh`, `tools/scripts/package_detector.sh`, and `tests/test_poster_workflow_scripts.py`: a reviewer can run a minimal real poster demo with one command, regenerate poster assets from saved reports with one command, and package the exported runtime together with the runtime code, contract docs, schema/config files, and selected run reports needed to inspect what was shipped.

## Poster Contract

`docs/poster_contract.md` is the acceptance contract for the poster branch.

`docs/reproducibility_checklist.md` is the short handoff runbook for rerunning the default workflows and inspecting the resulting artifacts.

In short, the poster is not complete unless the repo can show all of the following at once:

- the real F´ path is still preserved;
- a real MAVLink path exists;
- both protocols emit a shared raw/canonical artifact interface;
- the poster-default blue path is neural-only and request-time safe;
- the red path uses bounded lightweight context; and
- reproducible cross-protocol evaluation evidence exists.

That contract intentionally separates:

- what is already implemented today;
- what the poster branch must add; and
- what must not be claimed prematurely.

## Poster thesis

The original capstone story was too tied to a single mission stack. The poster version of the project asks a stronger research question:

**Can a blue detector learn to recognize unsafe command intent across heterogeneous cyber-physical stacks without depending on F´-specific identifiers or other stack-local shortcuts?**

To answer that question, the poster system will:

1. keep the existing **real F´ execution path**;
2. add a second **real MAVLink execution path** using live, containerized software rather than synthetic logs;
3. replace the current mixed model stack with a **single neural blue detector** for the poster-default path; and
4. add a **red adversary model** that receives a bounded transcript of its own last *n* commands and learns to slip unsafe commands through during plausible “good” windows.

The key claim for the poster is not “we can classify one protocol well.” The key claim is:

> A detector trained on **protocol-neutral command semantics**, **normalized system state**, and **short-horizon behavioral context** should generalize better than a detector trained on **stack-specific command IDs, node IDs, and telemetry names**.

## What the repository already does well

A review of the current repository shows that the hardest infrastructure work is already present.

### Existing strengths already in the repo

- **Real F´ dual-node runtime in Docker**
  - `scripts/fprime_real/bootstrap_project.sh` builds a generated `FlightPair/DualLink` deployment.
  - `orchestration/docker-compose.fprime-real.yml` spins up two F´ nodes plus multiple operator/adversary identity containers.
- **Schedule-driven live execution**
  - `tools/fprime_real/schedule_profiles.py` builds benign, cyber, and fault schedules with per-episode structure.
  - `tools/fprime_real/run_fprime_schedule.py` serializes per-target streams and executes real sends from inside source containers.
- **Wire-level provenance and replayable artifacts**
  - `tools/fprime_real/pcap_capture.py`, `packet_fidelity.py`, `downlink_ingest.py`, and `support_probe.py` reconstruct packets, logs, terminal outcomes, observability, and identity provenance.
- **Training and hardening infrastructure**
  - `main.py` already supports grouped splits, repeated grouped CV, scenario-family holdouts, command-family holdouts, protocol-family holdouts, dataset sanity checks, mixed-protocol shortcut baselines, and runtime export.
- **Deployment packaging**
  - `runtime.py`, `bg_pcyber.py`, `deployments/DetectorRB3`, and `tools/scripts/package_detector.sh` now define both the manifest-driven poster bundle (`blue_model.json` + `bundle_manifest.json`) and the preserved legacy comparison bundle.
- **Substantial regression coverage**
  - `tests/` now explicitly checks shared raw/canonical schema validation, blue feature-policy enforcement, MAVLink packet/provenance reconstruction, opt-in live F´ and MAVLink smoke paths, self-play smoke behavior, transcript-budget enforcement, schedule diversity, stream serialization, observability, telemetry catalog behavior, and training hardening.

### Important audit insight

The current repo is *not* just the original “full feature vector” capstone anymore. It already contains one major improvement that should be retained:

- the **primary model feature tier** in `runtime.py` is already **request-time only** and excludes terminal response leakage and rolling history leakage from the default learned classifier.

That is good and should stay.

### What still requires discipline

Even with the major poster milestones landed, contributors still need to avoid regressions in a few places:

- the explicit **legacy comparison path** still exists and must stay clearly labeled as comparison-only rather than drifting back into the default workflow;
- raw F´ and MAVLink identifiers remain preserved in the audit/provenance layers for replay and inspection, but they must not leak back into the poster-default blue input surface;
- cold live MAVLink builds remain slower and heavier than the inherited live F´ path on laptop-class machines, so the default fast demo path is still the real F´ workflow; and
- deployment/export remains intentionally gated by the generalization and triviality checks, so some minimal demo runs will correctly produce analysis artifacts while reusing the checked-in poster bundle instead of exporting a fresh deployable model.

## What changes for the poster

The poster re-haul keeps the current real-execution backbone and replaces the protocol-bound learned layer.

### Poster-default requirements

The poster-default path must satisfy all of the following:

- **Real execution only** for the main evidence path
  - No synthetic-only MAVLink stand-in.
  - No fabricated packet traces used as the primary claim.
- **At least two real protocol families**
  - F´ remains one family.
  - MAVLink becomes the second family.
- **Blue detector is neural-network only**
  - No random-forest-vs-MLP bake-off in the poster-default pipeline.
  - No hand-built rule stack as the headline detector.
- **No F´-specific model inputs in the blue default path**
  - Raw command IDs, raw service IDs, raw node IDs, and raw F´ telemetry names cannot be blue inputs.
- **Short bounded context for the red side**
  - The adversary model receives only a compact transcript of its own last *n* submitted commands so the prototype stays tractable on a MacBook.
- **Cross-protocol evidence**
  - The poster must show at least one evaluation that demonstrates generalization beyond one stack.

## Target architecture

The poster system should be understood as seven layers.

### 1. Real execution fabrics

The system needs two real, containerized execution environments:

- **F´ fabric** — already implemented in this repo.
- **MAVLink fabric** — runtime bootstrap, schedule-driven execution, strict capture, packet/log provenance reconstruction, and canonical-state adaptation are now landed for `ArduPilot SITL + MAVProxy`, so the second protocol can emit the same learned-interface contract as the poster-default F´ path.

For the MAVLink path, the most practical poster-default direction is a **headless** stack such as **ArduPilot SITL + MAVProxy** for automated runs, with a GUI tool like QGroundControl kept optional for manual demos if desired. The important design constraint is not the specific brand name; it is that the stack must be:

- real MAVLink software,
- containerizable,
- scriptable,
- packet-capturable,
- headless by default, and
- light enough to run on a prototype laptop.

### 2. Shared schedule compiler

Both protocol families must use the same research pattern already present in the F´ path:

- seeded schedule generation,
- named windows or phases,
- identity selection,
- semantic command families,
- benign/cyber/fault labeling for evaluation,
- per-run provenance, and
- reproducible replay from saved schedule artifacts.

The scheduler should stay protocol-neutral at the top and protocol-specific only at the final adapter stage.

### 3. Dual-layer event representation

Every executed command should produce two linked records:

#### A. Raw protocol record

This is the forensic record and must preserve the exact system-native evidence:

- raw protocol family and version,
- raw command name or enum,
- raw arguments,
- sender identity,
- target identity,
- raw transport/provenance metadata,
- raw telemetry or status extracted from that protocol,
- raw acknowledgements or outcomes,
- raw logs, and
- exact pcap/log correlation data.

#### B. Canonical semantic record

This is the research record used for cross-protocol modeling:

- platform family (for example: spacecraft / UAS),
- protocol family (stored for audit, not necessarily used by the blue model),
- actor role,
- trust class,
- mission/window class,
- canonical command semantic family,
- whether the command reads state or mutates state,
- whether it changes persistent state,
- authority level,
- safety criticality,
- argument-shape summary,
- normalized system-state summary,
- recent behavior summary, and
- run/episode metadata.

The canonical record is the bridge between F´ and MAVLink.

### 4. Blue detector

The poster-default blue model is a **single neural detector**.

Recommended shape:

- small command/category embeddings,
- normalized continuous system-state features,
- short-horizon context summary or short transcript embedding,
- compact MLP/GRU/Transformer-style backbone, and
- direct unsafe-risk output plus optional class head (`benign`, `cyber`, `fault`).

The blue model should operate at **request time**, not after terminal outcomes. That means it can look at:

- the current command request,
- coarse window context,
- sender context,
- current normalized state, and
- bounded recent history.

It must **not** depend on terminal response-only signals for its primary decision.

### 5. Red adversary model

The new adversary path is not just a static schedule sampler. It is a learned or learnable policy.

Required behavior:

- it receives a randomly sampled, compact transcript of its own last *n* submitted commands;
- it sees enough coarse context to know what “good window” it is currently in;
- it proposes the next command family / timing / identity choice within the research environment; and
- it is rewarded for getting unsafe commands accepted or executed **without** the blue detector recognizing them as bad.

Important constraint:

- the transcript must remain **small** and **structured** so the prototype remains laptop-friendly.

A good implementation direction is not a heavyweight open-ended LLM loop. A better poster prototype is a **lightweight neural policy** that consumes a compact command transcript tokenization.

### 6. Self-play / auto-research loop

The poster version uses two training threads:

- **Blue thread**: improve recall/precision for unsafe command detection on request-time canonical inputs.
- **Red thread**: improve evasive timing and unsafe command selection against the current blue detector.

This is effectively a bounded self-play / co-evolution loop:

1. generate or replay mixed-protocol schedules;
2. run the real environments;
3. reconstruct canonical records;
4. update blue on true outcomes and labels;
5. update red on reward signals relative to blue’s current behavior; and
6. repeat with strict artifact logging.

### 7. Evaluation layer

The poster must show that the new system is genuinely more general and not just re-overfit to a new stack.

Minimum evaluation families should include:

- grouped split evaluation by run,
- command-family holdout,
- scenario/window holdout,
- protocol holdout or low-shot transfer,
- adversary-vs-blue success-rate curves,
- ablations for transcript length *n*, and
- ablations for protocol-specific leakage.

## Blue feature contract

The clearest way to protect the poster claim is to define **what the blue model is allowed to see** and **what it is forbidden to use**.

### Allowed blue inputs

Allowed blue inputs should be restricted to protocol-neutral, request-time features such as:

- canonical command family and subfamily,
- sender role and trust class,
- mission/window class,
- argument count, shape, scale, and normalization summaries,
- whether the command mutates volatile or persistent state,
- normalized system health summaries,
- bounded recent command-history summaries,
- sender/identity continuity summaries, and
- protocol-agnostic timing/burst features.

### Forbidden blue inputs for the poster-default path

The blue model should not directly use any of the following:

- raw F´ service IDs,
- raw F´ command IDs,
- raw F´ node IDs,
- raw telemetry feature names that only make sense inside F´ (`target_cpu_total_pct`, `peer_rg1_max_time_ms`, etc.),
- raw MAVLink enum values as direct learned shortcuts,
- terminal acknowledgements or response-only outcomes,
- hand-written rule scores as core model inputs,
- novelty/calibrator outputs as core model inputs,
- ground-truth labels, or
- provenance artifacts that are unavailable at request time.

### What may still remain in the raw record

The ban above is for **blue inputs**, not for audit storage. Raw protocol-native fields should still be preserved for:

- replay,
- forensic explanation,
- debugging,
- ablation studies, and
- proving that real execution actually occurred.

## Canonical data products

The poster path should preserve the existing artifact discipline and make it protocol-general.

Every protocol-specific generator should emit the same top-level product contract:

- `raw_packets.jsonl` — shared raw packet/message records that preserve native evidence and provenance;
- `raw_transactions.jsonl` — shared raw per-command grouped transactions;
- `canonical_command_rows.jsonl` — protocol-neutral semantic command records;
- `reports/generation_summary.json` — run summary and dataset manifest;
- `reports/provenance_summary.json` — capture/source observability proof;
- `reports/channel_inventory.json` or protocol-equivalent raw-state coverage report;
- `reports/actual_run_observability.json` — did the system really observe what it claims?; and
- `reports/schema.json` — schema and feature-set declaration.

The crucial poster rule is: **F´ and MAVLink must emit this same artifact contract.**

During the transition, the F´ path also keeps the inherited legacy outputs:

- `packets.jsonl`
- `transactions.jsonl`
- `dataset.jsonl`

Those legacy files remain for baseline comparability and for the explicit legacy training/runtime path. The poster-default F´ training flow now consumes the sibling canonical artifact `canonical_command_rows.jsonl`, while the legacy comparison path continues to use `transactions.jsonl` and `dataset.jsonl`.

## Selected MAVLink stack

The first automated MAVLink poster path is now fixed:

- **Autopilot:** `ArduPilot SITL`, starting with an `ArduCopter` profile
- **Automated GCS / command driver:** `MAVProxy`
- **Optional manual demo client:** `QGroundControl`

Why this stack was chosen:

- it best matches the repo’s existing **headless, scriptable, container-first** evidence model;
- it gives the cleanest first MAVLink control plane for packet capture and replayable artifacts;
- it keeps the first implementation lighter than a Gazebo-centered PX4 stack; and
- it still uses real MAVLink software instead of synthetic stand-ins.

The comparison and rationale are documented in [`docs/mavlink_stack_decision.md`](docs/mavlink_stack_decision.md). PX4 remains a valid later follow-on path, but it is not the first automated poster target.

Implementation status for that choice is documented in [`docs/mavlink_runtime_bootstrap.md`](docs/mavlink_runtime_bootstrap.md), [`docs/mavlink_schedule_contract.md`](docs/mavlink_schedule_contract.md), [`docs/mavlink_provenance_contract.md`](docs/mavlink_provenance_contract.md), [`docs/canonical_state_mapping.md`](docs/canonical_state_mapping.md), [`docs/multi_protocol_generation.md`](docs/multi_protocol_generation.md), [`docs/dataset_hardening_plan.md`](docs/dataset_hardening_plan.md), [`docs/red_transcript_format.md`](docs/red_transcript_format.md), [`docs/red_policy_model.md`](docs/red_policy_model.md), [`docs/red_reward_spec.md`](docs/red_reward_spec.md), [`docs/self_play_workflow.md`](docs/self_play_workflow.md), [`docs/evaluation_matrix.md`](docs/evaluation_matrix.md), [`docs/red_blue_metrics.md`](docs/red_blue_metrics.md), [`docs/poster_figure_plan.md`](docs/poster_figure_plan.md), and [`docs/reproducibility_checklist.md`](docs/reproducibility_checklist.md). The repo now has a headless `up` / `down` / smoke-tested stack bootstrap plus seeded schedule generation, a real schedule runner, strict packet capture, packet/log reconstruction, provenance/observability reports, canonical-state adaptation for `ArduPilot SITL + MAVProxy`, a shared dataset orchestrator that can emit F´-only, MAVLink-only, or mixed artifact sets, mixed-protocol leakage/holdout gates in the training reports, a manifest-driven poster runtime bundle for the blue neural path, an explicit bounded red transcript layer, a first learned red warm-start policy with an independently evaluable mixed-protocol holdout path, an explicit deterministic reward/sandbox contract, a checkpointed alternating blue/red self-play or auto-research harness, a poster-facing grouped/holdout/cross-protocol evaluation matrix, an adversary-versus-blue comparison suite above that, a reproducible poster asset generator, one-command wrapper scripts for the poster demo and asset workflow, and a checked-in hardening note for the next evidence-improvement phase.

## Repo map: current reality and planned direction

### Current top-level files and directories

- `main.py`
  - current orchestration, training, evaluation, export, and scoring entrypoint.
- `runtime.py`
  - current feature schema, manifest-driven poster runtime bundle loader, preserved legacy novelty/calibrator logic, and scoring helpers.
- `tools/fprime_real/`
  - current real F´ generator, scheduler, capture, support probing, and packet reconstruction code.
  - now emits preserved legacy artifacts plus shared `raw_packets.jsonl`, `raw_transactions.jsonl`, and `canonical_command_rows.jsonl`.
- `tools/shared/`
  - shared raw/canonical schema adapters, feature-policy enforcement, command taxonomy, artifact-layer joins, and canonical state builders.
- `tools/mavlink_real/`
  - current real MAVLink runtime-layout helpers, seeded schedule catalog/generators, sender, capture wrapper, support probe, telemetry/state ingest, and packet/log reconstruction for the headless `ArduPilot SITL + MAVProxy` stack.
  - canonical-state mapping is landed; the remaining major work above this layer is shared self-play/evaluation rather than another MAVLink-only schema fork.
- `scripts/fprime_real/`
  - current bootstrap, stack control, smoke test, and training pipeline scripts.
- `scripts/mavlink_real/`
  - current MAVLink stack bootstrap, up/down, smoke test, schedule smoke test, and vehicle/GCS launcher scripts.
- `orchestration/docker-compose.mavlink-real.yml`
  - current headless MAVLink compose stack definition.
- `tests/`
  - current hardening and regression tests.
- `deployments/DetectorRB3/`
  - current packaged runtime config target.

### Current and remaining additions for the poster branch

A clean poster implementation should likely add or split modules along lines like:

- `tools/shared/` or `core/`
  - protocol-neutral schema, canonical feature builder, shared schedule compiler, shared evaluation utilities.
- `tools/mavlink_real/`
  - runtime bootstrap, schedule runner, packet/log ingestion, canonical-state mapping, and dataset generation are landed.
- `tools/train/blue_model.py`
  - poster-default blue neural model and training loop are landed.
- `tools/train/train_blue_nn.py`
  - poster-default blue neural training entrypoint is landed.
- `tools/train/red_transcript.py`
  - bounded red transcript builder and compact token contract are landed.
- `tools/train/red_policy_model.py`
  - bounded red warm-start policy model, action-space contract, and export/load path are landed.
- `tools/train/train_red_policy.py`
  - red adversary warm-start training entrypoint is landed.
- `tools/train/red_reward.py`
  - explicit red reward function, blue/environment normalization, and research-sandbox checks are now landed.
- `tools/train/checkpointing.py`
  - self-play checkpoint/state/report helpers are now landed.
- `tools/train/run_self_play.py`
  - checkpointed alternating blue/red self-play / auto-research orchestrator is now landed.
- `tools/train/evaluate_red_vs_blue.py`
  - poster-facing adversary-versus-blue evaluation suite is now landed.
- `tools/figures/generate_poster_assets.py`
  - poster-ready diagrams, plots, captions, and asset manifest generation are now landed.
- `schemas/`
  - JSON schemas or markdown specs for raw and canonical record formats.
- `docs/`
  - poster-specific architecture notes, ablation descriptions, and figure-generation notes.

## Current commands that work today

The repo already supports a baseline F´ workflow.

### Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Fast non-live regression subset

```bash
python3 -m unittest \
  tests.test_schedule_hardening \
  tests.test_runtime_phase2 \
  tests.test_stream_serialization \
  tests.test_telemetry_catalog
```

### Full test discovery

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

### Optional live integration suites

```bash
RUN_LIVE_FPRIME_TESTS=1 python3 -m unittest tests.test_fprime_real_live
RUN_LIVE_MAVLINK_TESTS=1 python3 -m unittest tests.test_mavlink_real_live
```

### Live F´ smoke test

```bash
bash scripts/fprime_real/smoke_test.sh
```

### Headless MAVLink stack control

```bash
bash scripts/mavlink_real/up.sh
bash scripts/mavlink_real/down.sh
```

### MAVLink stack smoke test

```bash
bash scripts/mavlink_real/smoke_test.sh
```

### MAVLink schedule smoke test

```bash
bash scripts/mavlink_real/schedule_smoke_test.sh
```

### MAVLink provenance smoke test

```bash
bash scripts/mavlink_real/provenance_smoke_test.sh
```

### MAVLink schedule generation and replay

```bash
python3 tools/mavlink_real/make_good_schedule.py --target-rows 24 --seed 7 --out artifacts/mavlink_demo/benign_schedule.csv
python3 tools/mavlink_real/run_mavlink_schedule.py --schedule artifacts/mavlink_demo/benign_schedule.csv --output artifacts/mavlink_demo/run_log.csv
```

### MAVLink artifact reconstruction from an existing run

```bash
python3 tools/mavlink_real/packet_fidelity.py --runtime-root artifacts/mavlink_provenance_smoke/mavlink_real --run-log artifacts/mavlink_provenance_smoke/mavlink_real/logs/schedule_runs/benign_provenance_smoke_run.csv --pcap artifacts/mavlink_provenance_smoke/mavlink_real/captures/provenance_smoke.pcap --output-dir artifacts/mavlink_provenance_smoke/reconstructed
```

### End-to-end current training pipeline

```bash
bash scripts/fprime_real/train_pipeline.sh --rows 240 --output-dir artifacts/latest
```

### Comparison-only legacy rerun

```bash
bash scripts/fprime_real/train_pipeline.sh --legacy --rows 240 --output-dir artifacts/latest
```

### Direct generation entrypoints

```bash
python3 main.py generate --rows 240 --output-dir artifacts/latest
python3 main.py generate --protocol-mode mixed --mixed-fprime-ratio 0.5 --rows 240 --output-dir artifacts/mixed_latest
python3 tools/shared/generate_dataset.py --protocol-mode mavlink --rows 96 --output-dir artifacts/mavlink_latest
```

### Direct training entrypoints

Poster-default:

```bash
python3 main.py train --dataset artifacts/latest/data/dataset.jsonl --output-dir artifacts/latest
python3 tools/train/train_blue_nn.py --dataset artifacts/latest/data/dataset.jsonl --output-dir artifacts/latest
python3 tools/train/train_poster_default.py --dataset artifacts/latest/data/dataset.jsonl --output-dir artifacts/latest
```

Comparison-only legacy:

```bash
python3 main.py train-legacy --dataset artifacts/latest/data/dataset.jsonl --output-dir artifacts/latest
python3 tools/train/train_legacy_baseline.py --dataset artifacts/latest/data/dataset.jsonl --output-dir artifacts/latest
```

### Direct red warm-start entrypoint

```bash
python3 tools/train/train_red_policy.py --protocol-mode mixed --rows-per-protocol 96 --output-dir artifacts/red_policy_latest
```

### Direct self-play / auto-research entrypoint

```bash
python3 tools/train/run_self_play.py --dataset artifacts/latest/data/dataset.jsonl --rounds 1 --seed 7 --max-history-entries 4 --output-dir artifacts/self_play_latest
```

### Direct generalization evaluation entrypoint

```bash
python3 tools/train/evaluate_generalization.py --dataset artifacts/mixed_latest/data/dataset.jsonl --output-dir artifacts/eval_latest
```

### Direct adversary-versus-blue evaluation entrypoint

```bash
python3 tools/train/evaluate_red_vs_blue.py --self-play-output-dir artifacts/self_play_latest --max-history-entries 4 --output-dir artifacts/red_blue_eval_latest
```

### One-command poster demo

```bash
bash scripts/poster_demo.sh --rows 24 --seed 7 --output-dir artifacts/poster_demo_latest
```

This wrapper runs generation, then trains when the generated dataset is large enough for a fresh poster-default fit, then scores packets and packages the detector bundle. For smaller demo runs such as `--rows 24`, it records that fresh training was skipped because the dataset is below the training minimum, then scores/package using the checked-in poster bundle so the end-to-end runtime path stays inspectable.

### One-command poster asset workflow

```bash
bash scripts/poster_generate_assets.sh --dataset artifacts/mixed_latest/data/dataset.jsonl --self-play-output-dir artifacts/self_play_latest --output-dir artifacts/poster_asset_workflow_latest
```

This wrapper regenerates the generalization matrix, adversary-versus-blue report, and poster figures from documented inputs, then writes a short manifest summary under `reports/poster_asset_workflow_summary.txt`.

### Direct poster asset generation entrypoint

```bash
python3 tools/figures/generate_poster_assets.py --evaluation-matrix artifacts/eval_latest/reports/evaluation_matrix.json --red-blue-evaluation artifacts/red_blue_eval_latest/reports/red_blue_evaluation.json --output-dir artifacts/poster_assets/latest
```

### Direct protocol progress plot entrypoint

```bash
python3 tools/figures/generate_protocol_progress_plots.py --fprime-training-dir artifacts/fprime_initial --mavlink-training-dir artifacts/mavlink_initial --fprime-self-play-dir artifacts/fprime_self_play --mavlink-self-play-dir artifacts/mavlink_self_play --output-dir artifacts/protocol_progress_latest
```

### Package the current detector bundle

```bash
bash tools/scripts/package_detector.sh --run-dir artifacts/latest
```

For the poster-default path this now packages `deployments/DetectorRB3/config/blue_model.json` plus `deployments/DetectorRB3/config/bundle_manifest.json`, together with the runtime entrypoints, schema/config contract files, top-level repo docs, and selected run reports that explain what was exported.

## What the poster must prove

The poster is successful only if it can defend the following statements with artifacts generated from this repository:

1. the training/evaluation data came from **real schedule-driven execution** for more than one protocol family;
2. the blue detector’s default inputs are **protocol-general**, not F´-specific shortcuts;
3. the blue detector is a **neural model**, not a forest/rule/calibrator ensemble;
4. the red model has only **bounded recent command transcript context** and still learns useful evasive behavior;
5. the new detector performs competitively on F´ while also showing **cross-protocol generalization**; and
6. the repo can produce figures and evidence suitable for a security-research poster without relying on undocumented manual steps.

## Non-goals

The poster branch should explicitly avoid drifting into the following:

- a giant framework rewrite that sacrifices reproducibility;
- a GUI-dependent MAVLink demo path as the only evidence source;
- a detector that “generalizes” only because protocol ID leaks remain in the inputs;
- a public document that lists concrete unsafe operational command sequences in a reusable real-world form;
- training that depends on terminal labels unavailable at decision time; or
- a heavyweight adversary loop that cannot run on a normal development laptop.

## Safety and publication boundary

This repository is for **defensive cyber-physical anomaly detection research**. Public-facing docs, figures, and poster text should describe:

- command **families**,
- window semantics,
- evaluation methodology,
- detector design, and
- generalization results.

They should **not** publish real-world offensive playbooks, unsafe command recipes, or bypass instructions that are unnecessary for understanding the research result.

## Current handoff state

With the canonical F´ poster path, the real blue neural training loop, the manifest-driven poster runtime/package path, the explicit comparison-only legacy path, the MAVLink stack decision, the real MAVLink runtime bootstrap, the real MAVLink schedule runner, the MAVLink packet/log provenance path, the MAVLink canonical-state adapter, the mixed-protocol leakage/holdout gate, the bounded red transcript layer, the first learned red warm-start policy, the explicit red reward/sandbox layer, the checkpointed blue/red self-play harness, the multi-protocol evaluation matrix, the adversary-versus-blue suite, the poster asset generator, the named workflow-script tests, and the one-command wrapper scripts now landed, the implementation checklist in [`TODO.md`](TODO.md) is complete.

The practical next step is no longer missing infrastructure; it is rerunning the documented workflows on the target host, inspecting the saved artifacts listed in [`docs/reproducibility_checklist.md`](docs/reproducibility_checklist.md), and curating the strongest evidence bundle for the poster itself. For contributor and coding-agent operating rules, see [`AGENTS.md`](AGENTS.md).
