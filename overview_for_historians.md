# Teaching a machine to write good archival detail — and asking *whose* "good" counts

### A plain-language overview for historians and archivists

*An early-stage research project. June 2026.*

---

## What we are trying to do

When you read an archival record — a registry entry, a line of testimony, a finding-aid description — and write a short account of the person or place it concerns, you make countless small judgements: which details matter, what the record actually supports, where to stay cautious because the evidence is thin. We are studying whether a computer can be taught to write those short, careful accounts — **micro-narratives** — *from a record*, and, more importantly, **what makes such an account good, and who gets to decide**.

The honest motivation is a worry. Today's AI writes fluent, confident prose very easily. Pointed at a historical record, it will happily produce a paragraph that *sounds* rich and specific — while quietly inventing a name, a date, a place, or an event that is nowhere in the source. With Holocaust testimony, that is not a minor flaw; it is a line that must not be crossed. The opposite failure is just as real: AI also tends to flatten a particular human being into a generic profile — "an ordinary person caught up in the events of the period" — erasing exactly the specificity that gives a life its dignity.

So the project is really about a humanistic question dressed in technical clothing: **can "good detail" in a historical narrative be captured by a machine, and if so, on whose terms?**

## What "good detail" looks like — three versions of the same record

The clearest way to see the problem is an example. Everything below is **invented** — a fictional person, used only as a test case.

> **The record says:** Marta Hellinger, a milliner, worked at the Adler & Son hat workshop on Brünnergasse in 1938 (per a registry); she probably left for Rotterdam in 1939 (uncertain); her brother's testimony mentions she was caught up in a round-up in 1942.

A **good** micro-narrative stays inside the record and is honest about doubt:

> *Marta Hellinger trained as a milliner and, by 1938, was working at the Adler & Son hat workshop on Brünnergasse — that much the registry records. She is believed to have left for Rotterdam the following year, though the date isn't certain. Her brother later recalled she was caught up in a round-up in 1942; beyond that, the record falls silent.*

A **flattened** version is technically harmless but says almost nothing — it dissolves a specific woman into cliché:

> *Marta Hellinger was an ordinary person whose life, like so many others, was caught up in the events of the war.*

A **fabricated** version reads vividly — and is the dangerous one, because it invents:

> *After the Gestapo seized the Adler workshop she fled to Rotterdam in March 1939, and in 1942 she was dragged from her home at dawn and deported to the east.*

Every confident specific in that last paragraph — "Gestapo seized," "March 1939," "dragged… at dawn," "deported to the east" — is **not in the record**. It is invention wearing the costume of detail.

## The question at the heart of the project

Could a computer simply *measure* whether a narrative has good detail — counting names, dates, and specific words — and reward itself for producing more of them? That is the tempting shortcut, and our early results say it fails in a revealing way.

A purely surface measure — one that looks only at how specific and rich the text *reads* — **cannot tell grounded detail from fabricated detail.** In our tests it rated the invented paragraph *as highly as*, sometimes higher than, the faithful one: the fabrication looks great on the surface precisely because it is stuffed with vivid (invented) specifics. A measure that instead checks each claim against the actual record sees straight through it.

The lesson is not subtle: **good archival detail cannot be read off the surface of the writing. It depends on grounding in the source — a judgement that, in the end, is yours, not the machine's.** That is the core finding the whole project is built around, and it is an argument *for* the central role of historians, not against it.

## Whose "good detail"? — a question without a single answer

There is one more wrinkle, and it is genuinely interesting rather than a problem to be solved. Consider: is a *bland-but-honest* account (the flattened one) better or worse than a *vivid-but-invented* one (the fabricated one)? Both fail, but differently — one by saying too little, the other by saying too much that isn't true.

When we asked different "judges" to rank them, **they disagreed** — and so would reasonable people. A judge weighted toward caution and grounding ranks the honest-but-dull account clearly above the vivid fake. A judge weighted toward richness is more tempted by the vivid fake, even while recognising it is invented. There is no neutral, universal answer; the ranking reflects *values* — what one cares about most in a historical account.

This is why the project does not try to hand you a finished, "objective" definition of good detail. It treats that definition as **a choice that belongs to your community** — and it makes the choice explicit, visible, and adjustable, rather than buried inside an algorithm. Different archives, different scholarly traditions, different ethical commitments may legitimately set the standard differently. The point of the work is to make those differences *sayable and accountable*, not to paper over them.

## What this project is *not* doing — and our safeguards

- It does **not** treat invention as acceptable. Fabricated specifics are actively penalised; the goal is detail that is *grounded*, with uncertainty acknowledged rather than smoothed away.
- It does **not** aim to replace the historian's reading. It aims to study, and eventually support, the act of writing grounded micro-histories — with people setting the standard.
- The **method was developed entirely on invented test data** — fictional people, places, and references. We have since begun running it on **real EHRI early-testimony records**, but only as a *methods diagnostic*: every machine-written narrative is treated as **unverified**, names are redacted in routine output, samples are kept small and internal, and nothing produced is offered as a historical source or a claim about actual history. The results are about the *method* and its failure modes, not about the people in the records.
- It treats sensitive material with restraint: the aim is faithful, non-sensational accounts, attentive to the dignity of the people in the records.

## Where you come in

The machine can only ever try to meet a standard that people define. That standard is the part only historians and archivists can provide, and it is the next step:

1. **A workshop** to articulate, together, what "good detail" means for this kind of material — which qualities matter (specificity, grounding, acknowledgement of uncertainty, restraint, attention to whose story is being told), and how to weigh them when they pull against each other.
2. **Judging examples** — reading short candidate narratives and marking which are good and which are not, with a brief reason. This is light, structured work, and it is what teaches the system your standard. Your judgements, not an engineer's defaults, become the definition.

In short: the technology is the easy part. The hard, irreplaceable part — *what counts as a good, faithful, dignified piece of archival detail* — is yours. This project is an attempt to take that judgement seriously enough to build everything else around it.

---

*Questions, disagreements, and "that example is wrong because…" are exactly what we're hoping for. Contact: Toby Blanke (t.blanke@uva.nl).*
