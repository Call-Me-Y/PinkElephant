# PinkElephant: Action-Content-Isolated Correction Memory for Multi-Step Agents

## One-Sentence Research Claim

Multi-step agents need to recover from failures, but inserting failed actions verbatim into subsequent context can increase the probability of repetition. PinkElephant studies an **action-content-isolated corrective memory**: the monitor may read the full trajectory to determine what happened, but the memory written and the correction cards injected to the task agent must not restate, quote, or reconstruct the failed action; the system retrieves only verified repair principles when a similar failure recurs, and checks whether subsequent behavior recovers.

This claim is more specific than "rewriting error information into natural language." Prior work has already reported that replacing the raw failed call with a runtime-generated failure description mitigates most of the backfiring effect; Sentry has already implemented failure-triggered external playbooks, recovery verification, and cross-task accumulation. Therefore, PinkElephant's core contribution should lie in the **action-excluding representation and information channel, a verifiable action-leakage criterion, and whether it simultaneously reduces repetition and preserves corrective benefit in multi-step tasks** — not in a blanket claim of being the first to propose on-demand feedback or failure memory.

## Refined Project Statement

### Academic (Full) Version

Multi-step agents typically handle execution failures by appending tool calls and their results to the interaction trajectory. However, the surface form of a failed call can act as a strong reproduction cue in subsequent context, so that negative feedback paradoxically increases the agent's likelihood of repeating the same erroneous behavior; removing failure context entirely may in turn discard information needed for diagnosis. PinkElephant proposes to study an **action-content-isolated correction recording and processing framework**: the system internally identifies failure types and state changes from the complete trajectory, but compiles reusable corrective knowledge into structured records that contain no failed-action content; only when the subsequent trajectory satisfies the corresponding trigger conditions does the system provide the task agent with brief state constraints, repair goals, or next-step decision principles, and it verifies through subsequent environment states whether the correction worked. We will systematically evaluate whether this framework lowers the repetition rate of the same error while maintaining or improving recovery success, task completion, and cross-task transfer, in comparison with raw error echoing, verbalized feedback, reflective memory, and runtime-intervention methods.

### Short Abstract Version

PinkElephant addresses failure recovery in multi-step agents and studies how to retain effective corrective signals without restating failed actions to the task agent. The framework extracts failure types, triggering states, and verified repair principles from full execution trajectories and stores them in external structured memory; when a similar failure recurs, the system retrieves an action-content-isolated correction card conditioned on the current state, and verifies recovery from subsequent environment feedback. Controlled repetition experiments and multi-step benchmarks such as WebShop and AppWorld will be used to examine action repetition, recovery rate, final task performance, false interventions, and cross-task transfer.

## Literature Context and Research Gap

| Direction | Representative Work | Implications for PinkElephant | Open Questions |
|---|---|---|---|
| Failed-action echoing and the copying effect | Gumaan, *Feedback That Backfires* (arXiv preprint, August 2026; PDF in project `literature/`) | In small instruction-tuned models, simulated tool calls, and MBPP repair, failure records increase the probability that the failed action reappears; counterfactual decomposition attributes the main effect to the failed action string appearing in context. The paper reports that runtime-generated failure descriptions mitigate most of the backfiring. | Does the conclusion generalize to stronger models, long-horizon real tasks, and semantic-level "repeated actions"? Can action-content-isolated memory learn across tasks, take effect under condition matching, and preserve successful repairs? |
| Verbal reflection and experiential memory | Reflexion (Shinn et al., 2023), ExpeL (Zhao et al., 2023) | Show that feedback can be summarized into natural-language experience for later attempts without updating model parameters. | Reflection may contain failed actions, be misapplied in mismatched situations, and typically lacks explicit action-leakage constraints or recovery-based admission verification. |
| Conditionally retrieved recovery memory | Sentry (Ji et al., arXiv preprint, October 2026; PDF in project `literature/`) | Externalizes playbooks, retrieves on failure, and writes only after recovery; already evaluated on WebShop, AppWorld, SWE-bench Lite, and Mind2Web Replay — the closest and most important comparator for PinkElephant. | Sentry's general principles already overlap with "on-demand externalized corrective memory." PinkElephant must more strictly define that the correction channel must not convey failed-action content, and quantify how this isolation changes repetition rate, utility, and leakage risk. |
| Runtime monitoring and intervention | AgentGuard, AgentFixer, Wink, AgentForesight, etc. (covered by Sentry's baseline review and comparisons) | Show that runtime monitoring can localize invalid actions, stagnation, and ungrounded reasoning, and intervene during execution. | Knowledge accumulation under fixed rules or immediate intervention, false-positive costs, and trigger-condition transfer remain key comparison dimensions. |

**Positioning judgment:** "feedback should be conditionally triggered," "knowledge should live in external memory," and "only record successful repairs" cannot each serve individually as PinkElephant's novelty claim — Sentry already covers these points. We recommend framing the research question as: **Can an action-content-isolated corrective representation and channel simultaneously reduce the reappearance of failed actions, maintain effective recovery, and transfer verified repair principles to subsequent similar failures in multi-step execution?** This still needs to be confirmed through a complete related-work search and replication experiments; an absolute gap should not be claimed at the proposal stage.

## Proposed Method: Action-Free Correction Memory (AFCM)

PinkElephant is the name of the overall research framework; AFCM can serve as the name of the method module within it. Each record is not a summary of the failed action, but state–repair knowledge oriented toward failure conditions. We recommend starting with an interpretable, auditable structured version:

```text
failure_family: schema_rejection | no_progress | state_mismatch | unsupported_plan
trigger: condition decidable from environment state or recent trajectory
state_signature: observable state features at failure time, containing no action text
repair_goal: the state or decision objective the next step should reach
repair_principle: one concise positive strategy
success_criterion: recovery condition decidable from environment feedback
provenance: source task, number of verifications, confidence, applicable environment
```

Explicitly forbidden fields: the raw call, arguments, code lines, copied error snippets, and verbatim near-descriptions that directly allow reconstructing the failed action. Tool names or action categories are retained only when necessary and when they do not expose the original action; specific constraints need to be defined before experiments, with sensitivity analyses reported at different strictness levels. Example: for a failure of "consecutive queries did not narrow the candidate set," do not store the original query strings; instead store "the candidate set has remained unchanged for two consecutive steps; next, select an attribute dimension not yet inspected, and check whether the candidate set shrinks." This provides an executable decision direction without repeating the original erroneous call.

### Operating Pipeline

1. **Record the full trajectory.** The executor retains the agent, tools, observations, and environment states for failure detection.
2. **Detect and classify.** The monitor reads the trajectory, identifies failures such as schema errors, no progress, state mismatch, and unsupported plans, and extracts observable state signatures. The detector can see the failed action, but its output goes through an isolation compiler.
3. **Compile action-isolated cards.** The compiler maps the diagnosis into structured state conditions and repair goals, without sending the failed action or reconstructable details to the task agent. The first version prioritizes structured error types, state changes, and tool-schema validation results provided by the environment; language-model extraction is used only when structured signals are missing.
4. **Retrieve and guide on demand.** Only when the current failure matches the trigger conditions are a small number of high-confidence records retrieved. The agent receives "current observed state + repair goal + positive strategy + success criterion," not historical failed actions. When no memory matches, a generic repair template that likewise does not echo actions is used.
5. **Recovery verification.** Observe the next few steps to determine whether the failure pattern has stopped and the task has resumed effective progress. Only repair knowledge that passes verification is written to memory; entries that repeatedly fail have their confidence lowered or are quarantined, to avoid propagating bad advice to later tasks.
6. **Action-leakage audit.** After compilation, automatically check string overlap and key-parameter overlap with the original failed action; sample cards are blind-tested by annotators who see only the correction card and judge whether they can reconstruct the failed action. Automated filtering cannot replace human/model blind testing, because semantic restatement can still leak.

### Core Hypotheses

- **H1 (repetition suppression):** Compared to echoing "failed action + raw error," AFCM reduces both the exact repetition rate and the semantically equivalent repetition rate of the same failed action.
- **H2 (correction preservation):** AFCM's recovery rate and final task success rate are no lower than a verbalized failure description under an equal token budget; the core goal is achieved only if repetition is reduced while recovery is preserved.
- **H3 (condition matching):** Failure-triggered retrieval outperforms keeping the full playbook resident in context, especially in tasks where similar appearances correspond to different correct responses.
- **H4 (verified admission):** Writing only verified repair records outperforms indiscriminately recording all failure reflections, improving held-out task performance and reducing harmful interventions.

## Experimental Design

### Phase 1: Controlled Causal Replication

Replicate the failed-call repetition measurements of *Feedback That Backfires*, and extend them to multi-step environments. Fix the task, initial state, model, sampling seeds, and token budget; fork identical prefixes at each error point, and compare the following conditions:

| Condition | Feedback seen by the task agent |
|---|---|
| C0 No correction | Receives only a neutral, minimally informative execution result |
| C1 Raw echo | Failed action and raw error, representing a common ReAct-style harness |
| C2 Failed action + corrective instruction | Raw echo, plus an explicit request not to repeat; tests whether negative instruction suffices |
| C3 Verbalized failure description | Does not echo the raw action; a runtime-generated natural-language summary, aligning with the prior work's summary baseline |
| C4 Sentry-style retrieved repair | Current diagnosis, retrieved lessons, and soft repair; replicated as closely as possible from the public implementation |
| C5 PinkElephant/AFCM | Provides only action-isolated cards; external memory triggered by failure state |
| C6 Content ablation | Keeps the same framework as C5, removing respectively the state signature, repair goal, verified admission, or conditional retrieval |

C0 tests whether "hiding the original action from view" merely discards useful information; C3 is an extremely important near-neighbor method, preventing a weak comparison against raw echo alone; C4 separates whether the method's gain comes from Sentry's existing mechanisms or from strict action isolation. In addition to final task metrics, perform local paired forks at each failure point to estimate the causal effect of a single correction.

### Phase 2: Benchmark Evaluation

| Priority | Benchmark | Use and caveats |
|---|---|---|
| Primary | **WebShop** | Multi-step product search and purchase; observable failures include search stagnation, repeated filtering, and goal matched but purchase not completed. Compare average reward, full-score success, and post-trigger recovery rate over a fixed task split and multiple run seeds. |
| Primary | **AppWorld** | Stateful, multi-app API tasks; tests generalization of correction in long tool chains, parameter errors, and side-effect environments. Use task test pass rate, and report erroneous modifications / unintended state changes. |
| Mechanism benchmark | **MBPP program repair** | Follows the mutated programs and real interpreter feedback of *Feedback That Backfires*; directly measures the effect of echoing failed code/tool actions. Do a clean replication first, then test AFCM. |
| Mechanism benchmark | **Mind2Web Replay** | Sentry adapted Mind2Web with a Replay setup allowing retry after rejected actions; suitable for local recovery control in web tasks. Make clear this is an adapted version; scores must not be directly compared to official Mind2Web step-level metrics. |
| Extension | **SWE-bench Lite** | External validation as a high-cost coding-agent benchmark; failure diagnosis requires controlling model, tool budget, containers, and the official evaluation setup. Suitable only after the preceding mechanisms are established; not recommended as the first benchmark. |

Minimal viable paper experiments: complete the controlled replication (tool calls + MBPP) and the two multi-step benchmarks WebShop and AppWorld first; then do held-out memory transfer and at least one external domain (Mind2Web Replay or SWE-bench Lite). If Sentry's Replay setup is reused, clearly mark it as a new interaction adaptation.

### Baselines and Fairness Controls

- Baselines: raw agent, C1 raw feedback, C3 verbalized summary, Reflexion- or ACE-style reflection/context evolution, Sentry, AFCM.
- Use the same task agent, model version, system prompt, environment version, action budget, context budget, and sampling settings; models and call costs of the detector/repair generator are reported separately.
- Run online-from-scratch first (memory starts empty and accumulates across tasks), then held-out (training tasks and evaluation tasks do not overlap) to test transfer, avoiding mistaking task memory leakage for correction ability.
- Use at least 3 random seeds, preferably 5; pair tasks and seeds across methods. Confidence intervals via task-clustered bootstrap; report mean, 95% interval, and effect sizes. Sentry's practice of 5 repetitions can serve as an alignment reference.
- Report results under both equal token budget and equal model-call budget, so that longer guidance text or extra LLM calls do not confer an unfair advantage.

### Metrics

**Primary metrics**

- Task success rate / environment primary metric (WebShop reward, AppWorld test pass, etc.).
- Post-failure recovery rate: the same failure pattern stops within a predetermined observation window, and the agent resumes goal-relevant effective progress.
- Error-action repetition rate: exact string repetition; semantically equivalent repetition (determined preferably by environment state/action semantic rules, with model adjudication only as a supplement).

**Diagnostic metrics**

- Steps to recovery, number of invalid actions, intervention precision/recall, false-intervention rate.
- Action-leakage rate: proportion of cards literally containing failed-action content; blind-restoration rate (whether a card reader can identify the excluded action); rate of exposing erroneous parameters or code snippets.
- Negative transfer rate: proportion of triggered lessons that cause stagnation, rollback, or task failure in the current situation.
- Tokens per task, number of model calls, latency, and cost; record memory growth and retrieval hit rate.

## Success Criteria and Falsifiability

PinkElephant's minimum success bar should not be merely "the card contains no verbatim text." We recommend preregistering the following criteria:

1. AFCM significantly reduces exact and semantic repetition rates relative to C1;
2. AFCM shows a clear gain over C3 (verbalized summary), or achieves comparable repetition with improved recovery/task success;
3. Primary task success improves relative to C1, and shows at least one statistically reliable advantage over C3 and Sentry on at least one key dimension;
4. The blind-test restoration rate of cards is low, while recovery utility does not depend on leaking the failed action;
5. Effective on held-out tasks, with acceptable false-intervention and call costs.

If C5 only matches C3, the contribution should be narrowed to a stricter, auditable safe representation and measurement protocol; if removing action information from cards markedly lowers the recovery rate, report the information–interference trade-off and further test which minimal state features suffice to restore utility, rather than treating action suppression alone as success.

## Recommended Implementation Order

1. **Week 1:** Read the two project papers and their public code; implement the failed-action repetition measurement, trajectory forking, and conditional record format; define action semantic equivalence and leakage-audit definitions.
2. **Mechanism validation:** Run small-scale paired experiments C0–C5 on simulated tool calls and MBPP; adjust the card schema based on results, to avoid discovering information loss only after investing in full task benchmarks.
3. **Integrated evaluation:** Integrate with WebShop and AppWorld; fix model, split, budget, and seeds; add Sentry/C3 controls and AFCM component ablations.
4. **Transfer and audit:** Report online and held-out separately; blind-evaluate card reconstructability; check memory growth, duplicate lessons, false triggers, and reasoning overhead.
5. **Release:** Provide the runner, schema, task/seed lists, per-failure-event logs, and metric scripts; disclose benchmark adaptation differences, especially the unofficial nature of Mind2Web Replay.

## Core References

1. Esmail Gumaan. [*Feedback That Backfires: Why Small Language Model Agents Repeat the Call They Just Watched Fail*](https://arxiv.org/abs/2608.23651), arXiv:2608.23651, 2026. See project `literature/2608.23651v1.pdf`.
2. Changxiu Ji, Amy Lu, Qizheng Zhang, Kunle Olukotun. [*Sentry: Learning to Recover from LLM Agent Failures at Test Time*](https://arxiv.org/abs/2610.02994), arXiv:2610.02994, 2026. See project `literature/2610.02994v1.pdf`.
3. Noah Shinn et al. [*Reflexion: Language Agents with Verbal Reinforcement Learning*](https://arxiv.org/abs/2303.11366), NeurIPS 2023.
4. Andrew Zhao et al. [*ExpeL: LLM Agents Are Experiential Learners*](https://arxiv.org/abs/2308.10144), 2023.
5. Harsh Trivedi et al. [*AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents*](https://appworld.dev/), ACL 2024.
6. Xiang Deng et al. [*Mind2Web: Towards a Generalist Agent for the Web*](https://proceedings.neurips.cc/paper_files/paper/2023/file/5950bf290a1570ea401bf98882128160-Paper-Datasets_and-Benchmarks.pdf), NeurIPS 2023.
7. Carlos E. Jimenez et al. [*SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*](https://www.swebench.com/), ICLR 2024; when used, follow the official evaluation harness and current task splits.

> Note: This proposal is based on the two project PDFs and publicly available information retrievable as of 2026-10-21. The two core recent papers are currently arXiv preprints; their results can serve as neighboring work and replication targets, and their version and preprint status should be noted when cited. Before formal submission, versions, code, and evaluation setups should be re-verified.
