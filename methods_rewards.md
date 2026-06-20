# Methods — the reward functions, and why each is built the way it is

> **Framing:** this project is the generative sequel to "The Occasional Bitter
> Lesson" — for a *contested, generative* cultural target, scaling underperforms and
> the theoretical lens (the rewards) becomes the method. See **`FRAMING.md`** for the
> argument and how the project's results map onto that paper's pathology taxonomy.

This document explains, for each reward in the pipeline, **what construct it is
trying to measure, how it computes it, and what it is deliberately blind to.**
It is the methods section a reader needs to understand the experiments in
`experiments.md` and the dials in `DIALS.md`. The closing section,
*Validity and the circularity problem*, explains why a comparison of rewards is
not a rigged game even though each arm naturally scores best on its own ruler.

The project's premise is that **"good detail" in an archival micro-narrative is a
contested, community-defined construct**, not a fixed quantity. There is no
single correct reward. Instead we build several *operationalisations* of "good
detail", train a separate model arm to each, and study where they agree and —
more importantly — where they diverge. The rewards differ along one main axis:
how much they care about whether a vivid specific is actually *true* (situated /
grounded) versus merely *present and well-formed* (culture-blind / surface).


## 0. The shared substrate: a synthetic fact base and its oracle

Every reward is evaluated against the same synthetic corpus. Each record is a
**structured fact base** we author by hand: entities (people, places,
organisations) with attributes, events with dates and a `certainty` flag
(`attested` / `probable` / `uncertain`), per-fact source grounding, a `salience`
rank, and a `sensitivity` flag. From one record we render three *quality
profiles* of the same micro-narrative:

- **good** — specific, grounded, source-aware, appropriately hedged;
- **flattened** — bland and vague but *not false* (it simply says little);
- **fabricated** — fluent and vivid but inventing specifics not in the record
  (false names, places, dates, and dramatised events).

Because we *own* the facts, we have an **oracle**: an exact lookup that knows
which specifics are real. The oracle is the synthetic-phase ground truth and is
not itself one of the trainable rewards — it is the yardstick the rewards are
measured against. On real corpora the oracle is replaced by human judgement and
by retrieval against source passages; the synthetic phase exists precisely to
rehearse the whole apparatus where ground truth is free and unambiguous.

The distinction between **flattened** and **fabricated** is the heart of the
design. A reward that cannot tell vivid-and-false from bland-and-honest is a
reward that will teach a model to hallucinate confidently. Several of the
rewards below are built specifically to expose, or to fix, that failure.


## 1. Linguistic reward — the culture-blind surface arm

**Construct.** "Good detail" approximated purely from the *form* of the text —
how specific, concrete, and calibrated it reads — with **no access to the fact
base**. This is the automatable, corpus-agnostic reward an NLP practitioner
would reach for first, and the arm whose limits the project is about.

**How.** A length-normalised, decomposable weighted sum of five surface
features, each logged separately so divergence from human judgement is
attributable to a specific feature:

1. *proper-noun density* — capitalised non-initial tokens / √tokens (names, places);
2. *number/date density* — digit-bearing tokens / √tokens (dates, addresses, counts);
3. *concreteness* — mean concreteness of content words (Brysbaert norms if a
   `concreteness_norms.csv` is supplied, else a small built-in stub);
4. *lexical density* — content words / all words (information packing);
5. *calibration* — hedge-marker rate minus over-claim-marker rate.

`linguistic_reward = Σ weightₖ · featureₖ` with weights in `config.WEIGHTS`.
Counts are divided by √tokens so the reward rewards *density*, not mere length.
Provenance asides in parentheses are stripped before scoring so the surface
reward cannot earn credit from citation markers — it must be genuinely blind.

**Deliberately blind to.** Truth. By design every feature reads only the text.
A fabricated sentence packed with invented names and dates scores *higher* on
features 1–2 than an honest, hedged one. The one partial exception is
*calibration*: it senses over-claimed **certainty** (a fabricator who writes
"definitely" and "proven") but not fabricated **content**. The validation
harness (`python linguistic_reward.py`) reports, per feature, whether it
separates good from flattened (a real detail feature should) and whether it is
blind to fabrication (a culture-blind feature will be) — so the blindness is
measured, not assumed.


## 2. Composite reward — the situated, faithfulness-gated arm

**Construct.** The deployable analogue of the oracle: "good detail" that is
surface-rich **and** actually grounded in the record. It is the surface reward
with a truth gate bolted on, so it can be computed from a generation plus its
source without owning the facts.

**How.** A single scalar,

```
reward = F^γ · (Q + w_c · C)  −  w_fab · n_unsup  −  w_sens · S
```

where `Q` is the mean of the four surface detail features from the linguistic
reward, `C` is calibration, `F ∈ [0,1]` is **faithfulness**, `n_unsup` is the
count of unsupported/fabricated specifics, and `S` is a sensationalism penalty
(a small lexicon of gratuitously dramatic words, length-normalised — restraint
matters with this material). Defaults: `γ = 2.0`, `w_c = 0.3`, `w_fab = 0.5`,
`w_sens = 1.0`, all in `config.py`.

The design choice that matters is that **faithfulness is a multiplicative gate,
not an additive term.** Because `F` multiplies the quality term and `γ > 1`,
low grounding scales surface quality toward zero no matter how specific the text
is — a vivid fabrication cannot buy back reward with more vividness. The
additive `n_unsup` and `S` penalties then push fabrication below bland-but-honest
text. The validator (`python composite_reward.py`) confirms the effect: the
composite ranks good above fabricated in 16/16 grid cells where the
linguistic-only reward manages only 4/16.

### 2a. Faithfulness — a pluggable estimator (the core methods contribution)

`F` is computed by a **pluggable estimator** selected by
`config.FAITHFULNESS_METHOD`. How you operationalise grounding is itself a
research variable, so the three backends are first-class and interchangeable.

**`lexical` (the baseline).** Surface entity/year overlap: a capitalised
mid-sentence token or a four-digit year is *supported* iff it appears anywhere
in the record's bag of tokens, otherwise it is an unsupported specific. Fast,
offline, deterministic. Its weakness is exactly the project's theme: it is
**blind to recombination**. "Marta moved to Rotterdam in 1939" scores as fully
supported because "Rotterdam" and "1939" both appear in the record — even when
the *claim* combining them is invented — and a relational fabrication that adds
no new entity ("helped establish the workshop") is invisible to it. Critically,
this is the *same surface-overlap machinery the linguistic reward uses*, so a
composite built on it risks collapsing into the linguistic reward. That risk is
the reason for the next backend.

**`nli` (recommended).** Claim-level **entailment** against the fact base — the
faithfulness/factual-consistency approach standard in the summarisation
literature (FActScore, SummaC, AlignScore). It measures *propositions*, not
surface form, so it is orthogonal to the linguistic reward. The procedure:

1. *Atomise.* Split the narrative into sentences, drop self-referential meta
   sentences ("this narrative aims to…") and bare provenance fragments, then
   split each sentence into clauses, because a "good" narrative packs several
   facts into one sentence and no single premise can entail a four-fact
   sentence. The focal entity's name is prepended to clauses that lost their
   subject ("worked at the workshop" → "Marta Hellinger worked at the workshop").
2. *Build premises.* Turn the structured fact base into a list of atomic premise
   sentences (one per attribute and per event, with its date). On real corpora
   this list becomes the retrieved source sentences.
3. *Judge each claim against the best premise.* Run a small local NLI model
   (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` by default; ~370 MB, runs on
   the A10) on every (premise, claim) pair and take the **maximum over premises**
   — a claim is grounded if *any* fact entails it.
4. *Three-way verdict.* `entailed` → supported; `contradicted` (max
   contradiction over premises ≥ threshold) → fabrication; `neutral but
   containing an invented specific` (a name or year not in the record) →
   fabrication; `neutral and vague` → ignored.
   `F = supported / (supported + fabrications)`.

The fourth verdict is the conceptual point. **Blandness is not unfaithfulness.**
A vague flattened sentence ("she was an ordinary person") is *neutral* — not
false — so it is excluded from the denominator rather than punished. Penalising
vagueness is the linguistic/quality reward's job; faithfulness should only fire
on fabrication. On the grid this yields the right ordering — good ≈ 0.83,
flattened ≈ 0.88 (bland but honest, not penalised), fabricated ≈ 0.49 with ~3
fabrications per cell — and the per-claim debug view (`python faithfulness.py
--debug`) shows the model tagging the fabricated arm's inventions ("the Gestapo
seized…", "Officer Reinhardt… Westmark depot… 1941") as contradictions at ~0.97.
Thresholds (`NLI_ENTAIL_THRESHOLD`, `NLI_CONTRADICT_THRESHOLD`) are dials; the
debug view shows which way to move them. Known residual artefacts: a hedged
grounded claim ("probably in 1939") can fall just under the entailment threshold
and land in the benign `vague` bucket (un-credited but not penalised), and a
malformed bland fragment can occasionally trip a false contradiction — both
rare, both visible in the debug view.

**`llm` (strongest, slowest).** Ask an LLM judge to read the record and the
narrative and count the unsupported claims. The most flexible at catching
relational fabrication, but it carries token cost and its own hallucination
risk, so it is best as a validation oracle rather than the dense per-step RL
reward. It reuses the judge backends described in §4.

Results are cached by (method, case-id, text-hash) so the NLI/LLM cost is paid
once per unique completion inside the RL loop.


## 3. Human reward — the situated / community arm (synthetic stand-in)

**Construct.** "Good detail" as a *historian* would judge it: a weighted
combination of value dimensions, where the weighting encodes **whose** values.
This is the arm that, on real data, comes from workshop annotation; here it is
synthesised so the experiment can run today.

**How.** Six positive dimensions are scored from the generation and its record —
*coverage* (fraction of the record's salient specifics that appear),
*specificity* (surface concreteness), *grounding* (`F`, the same pluggable
faithfulness as §2, so the human arm also benefits from NLI), *source* (does it
attribute its claims), *calibration* (hedging vs over-claiming, clipped to
[0,1]), and *restraint* (1 − sensationalism) — minus a *fabrication* penalty on
unsupported specifics. A **persona** is a weight vector over these dimensions:

```
human_reward = Σ wᵈⁱᵐ · scoreᵈⁱᵐ  −  w_fab · n_unsup
```

`config.PERSONAS` ships four — `balanced`, `grounding_first`, `ethics_first`,
`specificity_first` — and adding one is a one-line change that auto-appears in
`--persona`. Optional Gaussian noise simulates inter-annotator disagreement, so
a realistic annotation set can be synthesised from personas + noise.

**What it is for.** Two things. First, a strong faithfulness-aware reward to
compare against the surface reward. Second, and more interesting, **pluralism**:
the report (`python human_reward.py`) shows that all personas reject outright
fabrication, but they *disagree* on whether a bland-but-honest narrative beats a
vivid-but-fabricated one — ethics/grounding personas say yes emphatically, a
specificity-first persona is more tempted by the vivid fake. That disagreement
is the measurable form of "whose good detail?".

**Honest caveat.** This arm is currently *synthetic*: its dimensions are
computed from the same NLP components (oracle, linguistic features) as the other
rewards, so "human vs automated" is not yet a fully independent contrast. It
becomes independent only when real annotators score these texts. The synthetic
human arm is a scaffold for that experiment, not a substitute for it.


## 4. LLM-as-judge reward — the RLAIF arm

**Construct.** "Good detail" as scored by a capable language model reading a
rubric — the reward behind RLAIF, and the one most teams actually deploy.

**How.** A system rubric and a per-item user prompt ask the model to score the
narrative on the value dimensions and return a number; the parser is tolerant of
fenced JSON and reasoning preambles. Multiple OpenAI-compatible backends are
supported (`config.BACKENDS`: a local **ollama**, a local-GPU **vllm** server,
**gemini**, and the UvA proxy), with retries and an optional `--mock` mode.

**The finding it exists to produce.** The judge is not neutral. Small judges
cluster their scores and barely penalise fabrication; larger judges separate the
profiles cleanly. More pointedly, **the value ordering depends on the judge
model** — a judge with a "vividness bias" can rank a fabricated-but-lively
narrative above a careful one, while a different judge reverses that. The LLM
judge is therefore studied as *another contested operationalisation*, not as an
oracle — its model-dependence is a result, not a bug to be tuned away.


## 5. The guards — reward-hacking defences, not value signals

Two auxiliary reward terms are added during GRPO to stop the policy from gaming
the main reward rather than improving. They encode no notion of "good detail":

- **length guard** — penalises deviation from a target length (`TARGET_WORDS`,
  default 90), so the model cannot inflate density-based scores by rambling. Its
  absence was the cause of the early non-terminating runs.
- **anti-copy guard** — penalises high n-gram overlap with the source record
  above a tolerance, so the model cannot maximise faithfulness by copying the
  record verbatim instead of writing a narrative.

Both are toggleable (`--no_length_guard`, `--no_anti_copy`) and their weights
live in `config.py`.


## 6. Validity and the circularity problem

A fair objection: if you reward a model with a linguistic metric and then it
scores well on that same linguistic metric, you have shown nothing but that
optimisation works. The objection is correct — about the **diagonal** of the
arm × metric table. "The linguistic arm wins on the linguistic reward" and "the
composite arm wins on NLI faithfulness" are tautologies and are never reported
as findings. No arm is judged by its own ruler.

The non-circular content lives in three places, none of them self-evaluation.

**The off-diagonal — cross-metric cost.** The claim is never "linguistic wins on
linguistic" but "optimising surface specificity *degrades grounding*": the
linguistic arm climbs on its own reward while its `n_unsup` rises and its NLI
faithfulness falls. That cost is measured with a *different* ruler than the one
optimised, so it is not circular. The payload is always what an arm pays on the
axes it was **not** trained on.

**The structure that makes the cost real — the agreement matrix.** We measure the
correlation between every pair of reward signals across the grid. On this corpus
`corr(linguistic, NLI-faithfulness) ≈ −0.34`: they are *anti*-correlated, so
maximising one mechanically pushes the other down. This is what makes the
surface reward a hazard rather than a harmless proxy — if the two were
uncorrelated you could satisfy both and there would be no story; if positively
correlated, surface specificity would be a fine stand-in for grounding. The
negative correlation is measured, not assumed, and reporting the full agreement
matrix (linguistic, lexical-F, NLI-F, human, LLM-judge) is itself a result.

**The ground truth no reward defines — the oracle.** Because we own the fact
base, we can ask each arm a question independent of all the rewards: does it emit
text the fact base says is false? The linguistic arm rewarding fluent
fabrication is adjudicated against the oracle, not against a competing reward.

And the deepest answer is the project's premise: **none of the automated rewards
is the ground truth for "good detail."** Linguistic, NLI, composite, human-synthetic,
and LLM-judge are all *hypotheses* about a contested construct. The contribution
is to make their disagreement legible so a community of historians and archivists
can adjudicate it — not to crown one by fiat. The only genuinely external
arbiter is the workshop, which is also why the synthetic human arm is honestly
labelled a stand-in. Keep the diagonal out of the claims, lead with the
off-diagonal trade-offs and the anti-correlation, and treat every reward as a
claim awaiting human adjudication — and the circularity dissolves into the
actual research question.


## 7. Two ways to use these rewards — training vs. selection

The same reward functions support two distinct modes, and they suit different
purposes (and audiences).

**Training-time alignment (`grpo_train.py`).** Bake "good detail" into a model's
weights: SFT a starting policy, then GRPO under reward X. This is the *controlled
study* — single-variable, the reward-level findings, the anti-correlation. Its
costs: it needs a starting policy (SFT, which needs *targets*), it is heavy, and
at small scale (a 0.5–3B model, few prompts) the trained arms barely diverge. It
is an ML-research artifact; it is not what convinces an archivist.

**Inference-time selection (`bestofn_demo.py`).** Leave the model untouched: have
a *strong, current* LLM write K candidate narratives for a record, and use a
reward to **pick** the best (best-of-N / rerank). No training, no SFT targets, no
GPU for generation — and because the rewards are **model-agnostic** (they score
any text), the model is fully pluggable: point it at whatever is current. This is
the deployable framing and the one that lands with historians: *here are five
narratives a strong model wrote; the grounded reward picks the faithful one, the
surface reward picks the vivid-but-fabricated one, the judge picks the fluent
one.* The disagreement between selectors **is** the thesis, shown with a model
people recognise. It also reframes the contribution honestly — the rewards are
not a training trick but **a lens for operationalising, and contesting, "good
detail" over any model's output.**

The two compose: the synthetic SFT→GRPO study is the rigorous spine; the
inference-time demo is the accessible, current-LLM layer; and real SFT/GRPO on a
corpus is what the historian *workshop* eventually enables (it supplies the gold
"good" targets that real-data SFT needs — though note GRPO needs only *prompts +
a reward*, not targets, so it can train on real records without the workshop).
