# Literature repository reproduction log

Date: 2026-11-01

## Summary

This pass re-ran the analysis pipeline over the raw per-item score and rollout files included in `feedback-that-backfires`. It did **not** regenerate model scores: the Hugging Face weight download stalled, so reported results below are a statistical re-analysis of the repository's supplied experimental records.

For Sentry, offline configuration validation and a mock-model hard-repair smoke run passed. This does **not** reproduce the paper's LLM-based benchmark results. The real AppWorld run is not configured in this environment: no supported model API key or local inference endpoint is present, AppWorld is not installed, and only Python 3.12 is available while the benchmark adapter asks for Python 3.11.

## Feedback That Backfires

### Repository and data checks

- Existing committed probe data: 275 ToolShed items / 58 tasks and 266 CodeRepair items / 132 tasks.
- Existing raw experimental records cover six model checkpoints across ToolShed and CodeRepair probe studies, plus Qwen2.5-0.5B ToolShed agent rollouts.
- `python scripts/check_version_drift.py` completed without reporting drift errors; Python compilation of `src/slmecho` and `scripts` completed.
- Current installed versions include torch 2.8.0+cu129, transformers 4.57.6, and datasets 2.21.0. The repository pins transformers 4.56.1 and datasets 4.0.0, so any fresh scoring run should use an isolated environment matching those pins.
- Running `build_items.py` under the current datasets version produced 267 CodeRepair items / 133 tasks, unlike the committed 266 / 132. The generated data changes were restored; do not use the regenerated data for paper-aligned analysis.

### Re-analysis performed

Commands run:

```powershell
python scripts/analyze_probe.py --raw-dir results/raw/probe --out-dir results/reproduction_probe --tables-dir results/reproduction_probe/tables --n-boot 1000
python scripts/analyze_agent.py --raw-dir results/raw/agent --out-dir results/reproduction_agent --tables-dir results/reproduction_agent/tables --n-boot 1000
```

The probe re-analysis reports negative corrective gain for every included model/environment cell. The estimated values in the generated output range from about -12 to -83 nats depending on environment, action length, and checkpoint; this is consistent in direction with the paper's feedback-inversion claim. The copy component is positive and substantially larger than the semantic component in the included cells.

For the included Qwen2.5-0.5B ToolShed agent records, the analysis reports:

| Harness | Solved fraction | Exact repeat rate |
|---|---:|---:|
| Verbatim failure record | 0.417 | 0.307 |
| Abstract failure record | 0.333 | 0.162 |
| Drop failed event | 0.333 | 0.798 |
| Verbatim + repeat ban | 0.417 | 0.075 |

These are results from the repository-provided rollout files, not new runs. The task count is small (24 for solved, fewer for repeat-rate contrasts), so interpret them as a reproduction check rather than a new benchmark claim.

Generated analysis artifacts are in `results/reproduction_probe/` and `results/reproduction_agent/`. No tracked source files or committed data were changed; `git status` shows only those generated output directories.

### Fresh scoring status (historical; superseded by the 2026-11-01 fresh-run update below)

The repository's 135M model download was attempted through the verified Mihomo HTTP proxy. Tokenizer/config files downloaded, but the large checkpoint file remained at approximately 8.3 MB without further progress. The attempt was stopped. A partial cache remains under `.hf/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/` and can be resumed or removed later. Therefore `verify_scoring.py` and new model-generated probe/rollout runs remain outstanding.

## Sentry

- `python -m Sentry.cli configs/paper.yaml` passed offline validation and confirmed that no API is configured.
- Python compilation of `Sentry` and `benchmarks` completed.
- A local mock-model smoke script exercised invalid-action detection, hard-repair routing, corrected-action validation, and recovery verification; result: `SENTRY_HARD_REPAIR_SMOKE=PASS`.
- This smoke uses a deterministic stub and is only a code-path check; it is not a result for the paper's Qwen3.5-9B/AppWorld evaluation.
- An editable pip install initially failed because isolated build dependency installation attempted PyPI access before the proxy was configured. Direct module execution did not require installing the package.
- AppWorld evaluation remains blocked by absent API credentials/local model endpoint, missing AppWorld dependency/data, and the lack of Python 3.11 in the installed interpreters.

## Next reproducibility steps (historical; superseded by the update below)

1. Configure proxy access for large Hugging Face artifact transfers and resume the SmolLM2-135M checkpoint download.
2. Create a clean Python environment from `feedback-that-backfires/requirements.txt`, then run `verify_scoring.py` before generating any new results.
3. For Sentry, provide an OpenAI-compatible inference endpoint (local or hosted) and set up Python 3.11 plus AppWorld data; then run the repository's AppWorld runner on a small fixed task subset before scaling up.
4. Keep new runs in separate output directories and record model ID, repository commit, task IDs, seeds, package versions, and proxy/model settings.

## 2026-11-01 fresh local-model reproduction update

The two requested local checkpoints are now available outside the repository at `C:\Projects\AI\models\SmolLM2-135M-Instruct` and `C:\Projects\AI\models\Qwen2.5-0.5B-Instruct`. Fresh runs load these local directories directly; no model weights or source code were added to either paper repository.

### Feedback That Backfires: fresh runs

- `scripts/verify_scoring.py` completed for local SmolLM2-135M on 4 sampled ToolShed items: 40 cached/uncached candidate checks, maximum absolute log-probability difference `7.057e-05` (below the `1e-3` tolerance).
- A 24-item, 5-condition ToolShed probe completed for both local models, using the same seeded item subset. SmolLM2 scored 336 candidates; Qwen2.5-0.5B scored 142 candidates with greedy decoding disabled. Both used CPU float32 and one PyTorch thread. Analysis is in `results/reproduction_fresh/analysis_full/`.
- Corrective gain was negative in both fresh probe cells: SmolLM2-135M `G=-25.819` nats and Qwen2.5-0.5B `G=-14.602` nats. The decomposition attributed most of the increase in failed-action probability to the copied action string (copy `+23.312` / `+11.727` nats); the semantic component was also positive (`+2.507` / `+2.875` nats). This pilot direction is consistent with feedback inversion, but 24 items and one environment are insufficient for a paper-level claim. The Qwen and SmolLM scoring counts differ because the Qwen run omitted optional greedy decoding; the primary teacher-forced score conditions were run on the same item subset.
- A three-task Qwen2.5-0.5B ToolShed rollout completed for `verbatim`, `abstract`, and `drop`; each solved 2/3 tasks, so this tiny pilot does not distinguish harnesses. Raw rollouts and summaries are in `results/reproduction_fresh/agent/` and `results/reproduction_fresh/agent_analysis/`.
- The earlier committed raw-data re-analysis remains distinct from these fresh local-model outputs. Fresh runs are kept under `results/reproduction_fresh/` and do not overwrite `results/raw/`.

### Sentry: code-path check

- `python -m Sentry.cli configs/paper.yaml` validates the paper configuration offline and confirms that it declares Qwen3.5-9B but has no API configured.
- Python compilation of `Sentry/` and `benchmarks/` passed. A deterministic smoke check exercised schema-invalid action detection, hard-repair prompt routing, and successful verification after an accepted corrected action (`SENTRY_HARD_REPAIR_SMOKE=PASS`).
- This is not a benchmark reproduction. The AppWorld run still needs the benchmark installation/data and a compatible model endpoint. The repository's listed AppWorld experiment is not run by the local 0.5B/0.135B probe pipeline.

### Next runs

1. Expand the paired ToolShed probe from 24 to the complete committed item pool for the two local models; run CodeRepair after checking its MBPP environment requirements.
2. Run the Qwen end-to-end harness pilot with more tasks and all paper harnesses, then compare solved rate, exact-repeat rate, and token cost with uncertainty intervals.
3. Keep Sentry's AppWorld benchmark as a separate track; first establish Python 3.11, AppWorld data, and a model endpoint, then run a small fixed task subset before scaling.

