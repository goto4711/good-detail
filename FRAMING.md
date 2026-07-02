# The Occasional Bitter Lesson, Generatively: Aligning to Contested Detail

*A framing for the good-detail project, extending "The Occasional Bitter Lesson:
AI Engineering and Expert Knowledge in the Digital Humanities" (Deep Culture Team,
UvA).*

## From replication to alignment

Our earlier study tested Sutton's *Bitter Lesson* — that general methods leveraging
computation eventually dominate methods built on human domain knowledge — across
three **discriminative** digital-humanities tasks (temporal attribution, keyword
extraction/classification, anomaly detection). The lesson proved "occasional": bitter
where the required knowledge is culturally general and well-represented in
pre-training (document classification), sweet where it is local, distributional, and
historically specific (temporal attribution). The conclusion was not that human
expertise beats computation, but that **theory — a clear lens on what counts as a
meaningful outcome — becomes more, not less, central** when one works by adapting
foundation models rather than training them.

This project carries that inquiry into a harder case: **generation under a contested
value**. The task is not to match a ground-truth label but to produce "good detail"
in an archival micro-narrative — a notion that is community-defined and contested
("whose good detail?"), with no single correct answer to score against. If the
occasional bitter lesson holds for discriminative tasks, the generative-and-contested
case is where it should hold *least*, and where the theoretical lesson should bite
*most*. That is what we find.

## Where the task sits, and what scaling did

"Good detail" is about as local and situated as a target gets: specificity that is
*grounded* in a particular record, *source-aware*, *calibrated*, and *restrained*
with sensitive material — properties a community of historians and archivists
negotiates rather than reads off the data. On the spectrum from our earlier paper,
it lies past temporal attribution, at the far end where general computation has least
purchase.

Accordingly, every scaling move under-performed. Aligning a model to the construct
by reinforcement learning (GRPO) produced a near-null divergence between reward arms
at scale (composite ≈ linguistic ≈ human; mean F ≈ 0.59 for all three on real
testimony, N=150); training that GRPO on real records rather than synthetic ones did
not help (the reward drifted from −2.38 to −2.58 over 300 steps); and the largest
general model available to us, gpt-oss-120b, could not reliably produce a clean
micro-narrative for the task at all, defaulting to verbose reasoning over content —
the same format pathology we documented for it on classification. The lever that did
move grounding was not scale but **focused source passages and a defined target**:
grounding generations against retrieved source sentences roughly halved fabrication
versus an entity-list record (unsupported specifics 6.7 → 4.2). Scale was not the
ingredient; situated signal was.

## The pathologies recur — generatively

Our earlier paper warned that LLMs "readily declare success whenever their internal
computation appears coherent — even when they have merely learned shortcuts based on
data leakages, reproduced majority-class labels, or converged on median values."
Each pathology reappears in generative form:

- **Leakage → reward-hacking.** A surface ("culture-blind") reward that scores only
  textual form trained a policy to *reformat the record into a list* and to invent
  authoritative-sounding specifics (expanding a source code "WL/SYN/301" into a
  fabricated witness, "Wilhelm Lederer"). High score, no grounding — the generative
  cousin of the keyword "security council" carrying 85% of a decision tree.
- **Majority-class → collapse to baseline.** Under-powered RL left the reward arms
  near-identical to the untrained baseline: the model converged on the safe central
  policy rather than learning the value it was optimised for.
- **Median collapse → the LLM judge.** An LLM-as-judge reward gave a fully
  ungrounded narrative (faithfulness 0.00) a perfect 10, and scored two near-identical
  texts 9 versus 2; at the arm level it rewarded the fabrication-leaning surface arm
  (7.0) over the grounded one (3.75). The judge simulates evaluation without
  performing it — the same "coherent-looking but hollow" failure, now in the loss.

And as before, none of these surfaced on their own. They were caught only by the
supervision the earlier paper prescribes — inspecting where reward came from, checking
for collapse, and a deliberate **ablation**: testing whether the extracted relations
actually improved grounding. They did not (relations-on ≈ relations-off; relation
extraction itself only 0.40 F1 against gold), which stopped us claiming a relational
"fact base" did work it did not. The methodological moral is identical: *outputs are
easily mistaken for findings rather than artefacts of prompt or optimisation
shortcuts.*

## Against the hivemind: pluralism by construction

The earlier paper raised the "artificial hivemind" (Jiang et al. 2025): LLMs converge
on salient vocabulary (91% shared keywords at full scale), narrowing the interpretive
lenses humanities research depends on. Good-detail is, in part, a countermeasure. By
operationalising "good detail" as *several* explicit, contestable reward functions —
a surface/linguistic measure, a situated/grounded measure, a persona-weighted human
measure, an LLM judge — and by measuring their disagreement, we re-introduce
plurality rather than dissolving it. The disagreement is real and measurable: the
surface and grounded rewards are *anti-correlated* on our corpus (r ≈ −0.34), and
different judge models (Qwen vs. Gemini) and different personas reverse the ordering
of bland-but-honest versus vivid-but-fabricated text. The point is not to crown a
single reward but to make visible *that* the choice of reward is a choice of values —
the opposite of homogenisation.

## The lesson, sharpened

For a contested, generative cultural target, the bitter lesson is not merely
occasional; it is **inapplicable by construction**. One cannot let computation
discover "good detail," because the target must first be theorised and
community-defined before any model can be judged against it. This is the earlier
paper's conclusion — theory becomes central — turned into a method: the reward
functions *are* the theoretical lens, made explicit, plural, and falsifiable; and the
historian **workshop** is the procedure by which the community sets that lens and
supplies the situated signal no pre-training can. What general computation offers is
real but bounded — a strong model is a good *candidate generator*, and the rewards
are best used as a *lens over its output* (inference-time selection) rather than as a
training signal that, at our scale, simply collapses. The expertise that matters is
not obsolete; it has moved from hand-engineering features to **deciding what counts
as good, what should remain ambiguous, and what lies beyond the method** — exactly
where, as we argued before, AI engineering in the humanities most needs it.

## References

- Sutton, R. (2019). *The Bitter Lesson.*
- Kitchin, R. (2014). Big Data, New Epistemologies and Paradigm Shifts. *Big Data &
  Society* 1(1).
- Jiang, L. et al. (2025). *Artificial Hivemind: The Open-Ended Homogeneity of
  Language Models (and Beyond).* arXiv:2510.22954.
- Khattab, O. et al. (2023). *DSPy: Compiling Declarative Language Model Calls into
  State-of-the-Art Pipelines.* ICLR.
- Deep Culture Team (UvA). *The Occasional Bitter Lesson: AI Engineering and Expert
  Knowledge in the Digital Humanities.* deep-culture.org.
