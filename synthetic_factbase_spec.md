# Synthetic Fact-Base Specification (v1)

> **Status (17 June 2026):** this spec is now prototyped — generator, 2×3 grid, oracle, TEI/IOB serialisers, ingestion adapters, linguistic + composite rewards, and a guarded GRPO loop all run on synthetic data. See **`PROJECT_STATUS_2026-06-17.md`** for the live build status, results, and the reward-functions reference.

**Role in the project:** the controlled substrate for building and validating the pipeline *before* any real corpus or historian annotation. It lets us test the whole comparison — community/oracle reward vs. linguistic reward — on data where we know the ground truth, and where we have deliberately planted the differences the rewards are supposed to detect.

**Settled decisions (this version):**
- Registers: **testimony** vs. **finding-aid / catalogue** (literary deferred).
- **Per-fact source grounding** (every fact links to the source that supports it, or is marked inferred/unsupported).
- Case size: small — ~5–8 facts, 1–3 sources each.
- Two distinct failure modes on the "bad" side: **flattened** and **fabricated**.

Everything below is **fictional**. Names, places, and archive references are invented and must not correspond to real people or holdings.

---

## 1. Two-layer model

A strict split between content and surface is what gives us both register variation and a detail oracle:

- **Fact layer** — register-invariant structured truth (a small knowledge graph). Unchanged when register switches.
- **Rendering layer** — surface text produced *from* the facts under a chosen **register profile** and **detail profile**. The only thing that varies.

The fact layer stays hidden from the surface. This produces the key asymmetry that the whole study turns on:

> The **linguistic reward** sees only the rendering. The **oracle** (our synthetic stand-in for historians) sees facts-vs-rendering — it has access to grounding, salience and certainty that the surface cannot reveal.

That is exactly the human-vs-linguistic asymmetry we will face on real corpora, so the synthetic phase rehearses the real comparison rather than faking it.

---

## 2. Schema

```jsonc
Case {
  case_id,
  focal_entity_id,                 // what the micro-narrative is about
  entities: [ {
    id, type,                      // person | place | org | object | date
    name, fictional: true,
    attributes: [ {
      key, value,
      specificity,                 // generic | specific
      grounded_by,                 // [source_id] | "inferred" | "unsupported"
      certainty                    // attested | probable | uncertain
    } ]
  } ],
  events: [ {
    id, type, summary,
    participants: [ { entity_id, role } ],
    place_id,
    date: { value, precision, certainty },
    grounded_by,                   // [source_id]
    salience,                      // 1–3 (importance to the micro-history)
    sensitivity                    // low | medium | high (restraint dimension)
  } ],
  relations: [ { head_id, type, tail_id, grounded_by, certainty, salience } ],
  sources: [ {
    source_id,
    kind,                          // testimony | letter | registry | catalogue_entry | caption
    archive_ref,                   // fake provenance string
    supports: [fact_id]
  } ]
}
```

### The five control tags carry the whole experiment

Each maps onto one dimension of "good detail," and therefore onto a reward component:

| Tag | Meaning | Reward dimension |
|---|---|---|
| `specificity` | specific vs. generic value | concreteness |
| `grounded_by` | which source supports it (or none) | faithfulness |
| `certainty` | attested / probable / uncertain | calibrated hedging |
| `salience` | how much the fact matters | meaningful vs. trivial detail |
| `sensitivity` | risk of gratuitous framing | restraint / anti-sensationalism |

---

## 3. The rendering grid

A **register profile** sets voice and structure; a **detail profile** sets quality. Per case:

**2 registers × 3 detail profiles = 6 renderings.**

| Detail profile | What it does |
|---|---|
| **good** | includes salient *grounded* facts at their *specific* value; hedges uncertain facts; attributes claims to sources (register-appropriately); restrained on high-sensitivity events |
| **flattened** | drops salient/specific facts into generic profile-speak; no sourcing; no hedging structure |
| **fabricated** | adds confident *specifics that no source grounds*; may over-claim certainty and sensationalise. **Implemented:** injects *hallucinated* named entities + a false precise date (the realistic hallucination mode) so the faithfulness reward's grounding check is actually exercised, plus sensational embellishments. |

**Register-dependent realisation of source-awareness** (same underlying dimension, different surface):
- *Testimony*: natural attribution — "according to the workshop registry," "her brother recalled."
- *Finding-aid*: explicit provenance citation — "(registry, WL/SYN/118)."

### Matched pairs = the validation set

- **Detail-quality pairs** (hold register fixed): good vs. flattened, good vs. fabricated. → Tests whether a reward detects detail quality.
- **Register pairs** (hold detail profile fixed): testimony vs. finding-aid. → Tests whether the audit can attribute divergence to register rather than quality.

A reward that cannot separate good from flattened, or good from fabricated, is not ready for historians.

---

## 4. The oracle (synthetic "good detail" judgment)

The oracle scores a rendering `R` against case facts `F`. It is transparent and decomposable, because its components are exactly what we will later ask historians to weigh:

- **(+) Salient coverage** — fraction of high-salience *grounded* facts in `F` that appear in `R`.
- **(+) Specificity** — fraction of included facts rendered at their specific (not generic) value.
- **(+) Source-awareness** — grounded claims in `R` are attributed to their source, register-appropriately.
- **(+) Calibration** — probable/uncertain facts are hedged; attested facts stated plainly; over-claiming certainty is penalised.
- **(−) Fabrication — GATING** — any specific claim in `R` not grounded by `F` is penalised heavily. A fabricated rendering must never outscore a good one, whatever its other merits. (This encodes "faithfulness dominates.")
- **(−) Sensationalism** — high-`sensitivity` events framed gratuitously are penalised.

Weights are deliberately left open — they are a *workshop decision with historians*, not an engineering default. The synthetic phase only needs the ordering to be right (good ≫ flattened, good ≫ fabricated).

### The linguistic reward, by contrast

Computes only on `R`'s surface — concreteness, proper-noun/date density, lexical specificity, surprisal against a generic language model. It is **blind to grounding and salience by construction**.

### First hypothesis the fact base lets us test

Because the linguistic reward cannot see grounding, it should rank a **fabricated** rendering roughly as high as a **good** one whenever both are specific and dense — while the oracle ranks good ≫ fabricated. If observed on the synthetic grid, that is a clean existence proof that surface metrics cannot carry "good detail," and it is the pipeline's first concrete experiment.

---

## 5. Worked case A — fully rendered (fictional)

### Fact base

```json
{
  "case_id": "synth_case_001",
  "focal_entity_id": "marta",
  "entities": [
    { "id": "marta", "type": "person", "name": "Marta Hellinger", "fictional": true,
      "attributes": [
        { "key": "occupation", "value": "milliner", "specificity": "specific", "grounded_by": ["S1"], "certainty": "attested" },
        { "key": "birth_year", "value": "1911", "specificity": "specific", "grounded_by": ["S1"], "certainty": "attested" }
      ] },
    { "id": "adler", "type": "org", "name": "Adler & Son hat workshop", "fictional": true,
      "attributes": [
        { "key": "address", "value": "Brünnergasse 7, Brünn", "specificity": "specific", "grounded_by": ["S1"], "certainty": "attested" }
      ] },
    { "id": "josef", "type": "person", "name": "Josef Hellinger", "fictional": true, "attributes": [] },
    { "id": "rotterdam", "type": "place", "name": "Rotterdam", "fictional": false, "attributes": [] }
  ],
  "events": [
    { "id": "E1", "type": "employment", "summary": "Marta worked at the Adler & Son hat workshop",
      "participants": [ { "entity_id": "marta", "role": "employee" } ], "place_id": "adler",
      "date": { "value": "1938", "precision": "year", "certainty": "attested" },
      "grounded_by": ["S1"], "salience": 3, "sensitivity": "low" },
    { "id": "E2", "type": "emigration", "summary": "Marta left for Rotterdam",
      "participants": [ { "entity_id": "marta", "role": "emigrant" } ], "place_id": "rotterdam",
      "date": { "value": "1939", "precision": "year", "certainty": "probable" },
      "grounded_by": [], "salience": 2, "sensitivity": "low" },
    { "id": "E3", "type": "detention", "summary": "Marta was detained during a round-up",
      "participants": [ { "entity_id": "marta", "role": "detained" } ], "place_id": null,
      "date": { "value": "1942", "precision": "year", "certainty": "uncertain" },
      "grounded_by": ["S2"], "salience": 3, "sensitivity": "high" }
  ],
  "relations": [
    { "head_id": "marta", "type": "works_at", "tail_id": "adler", "grounded_by": ["S1"], "certainty": "attested", "salience": 3 },
    { "head_id": "josef", "type": "sibling_of", "tail_id": "marta", "grounded_by": ["S2"], "certainty": "probable", "salience": 1 }
  ],
  "sources": [
    { "source_id": "S1", "kind": "registry", "archive_ref": "WL/SYN/118", "supports": ["marta.occupation", "marta.birth_year", "adler.address", "E1"] },
    { "source_id": "S2", "kind": "testimony", "archive_ref": "WL/SYN/204", "supports": ["E3", "josef.sibling_of.marta"] }
  ]
}
```

### Renderings

**Testimony — good**
> Marta Hellinger trained as a milliner and, by 1938, was working at the Adler & Son hat workshop on Brünnergasse in Brünn — that much the workshop registry records. She is believed to have left for Rotterdam the following year, though the date isn't certain. Her brother Josef later recalled that she was caught up in a round-up in 1942; beyond that, the record falls silent.

**Testimony — flattened**
> Marta Hellinger was a working woman who later emigrated and was eventually caught up in the events of the war. Like many others, she experienced hardship and displacement.

**Testimony — fabricated**
> Marta Hellinger fled to Rotterdam in March 1939 after the Gestapo seized the Adler workshop, and in 1942 she was dragged from her home at dawn and deported to the east.
> *(unsupported specifics: March 1939, Gestapo seizure, "dragged at dawn," "deported to the east"; sensationalised high-sensitivity event)*

**Finding-aid — good**
> Hellinger, Marta (b. 1911), milliner. Employed at the Adler & Son hat workshop, Brünnergasse 7, Brünn, 1938 (registry, WL/SYN/118). Probable emigration to Rotterdam, 1939 (undated; unconfirmed). Detained during a round-up, 1942, per family testimony (WL/SYN/204). Further fate undocumented.

**Finding-aid — flattened**
> Hellinger, Marta. Worker. Emigrated. Affected by wartime events. See file.

**Finding-aid — fabricated**
> Hellinger, Marta (b. 1909), milliner and resistance courier. Fled to Rotterdam, March 1939. Arrested by the Gestapo and deported to Auschwitz, 1942 (confirmed).
> *(false birth year, unsupported "resistance courier," invented specifics, and a false claim of confirmation)*

---

## 6. Worked case B — second focal type (fictional)

A non-person focal (a place), to show the schema generalises and to vary the sensitivity profile.

### Fact base (abridged)

```json
{
  "case_id": "synth_case_002",
  "focal_entity_id": "cafe",
  "entities": [
    { "id": "cafe", "type": "place", "name": "Café Landauer", "fictional": true,
      "attributes": [ { "key": "address", "value": "Krapfengasse 3, Brünn", "specificity": "specific", "grounded_by": ["S1"], "certainty": "attested" } ] },
    { "id": "hugo", "type": "person", "name": "Hugo Landauer", "fictional": true,
      "attributes": [ { "key": "role", "value": "proprietor", "specificity": "specific", "grounded_by": ["S1"], "certainty": "attested" } ] }
  ],
  "events": [
    { "id": "E1", "type": "meeting_point", "summary": "informal meeting point for emigration inquiries",
      "participants": [ { "entity_id": "cafe", "role": "venue" } ], "place_id": "cafe",
      "date": { "value": "1938–1939", "precision": "range", "certainty": "probable" },
      "grounded_by": ["S1"], "salience": 3, "sensitivity": "low" },
    { "id": "E2", "type": "closure", "summary": "café closed / Aryanised",
      "participants": [ { "entity_id": "cafe", "role": "subject" } ], "place_id": "cafe",
      "date": { "value": "1939", "precision": "year", "certainty": "uncertain" },
      "grounded_by": ["S2"], "salience": 2, "sensitivity": "medium" }
  ],
  "relations": [ { "head_id": "hugo", "type": "proprietor_of", "tail_id": "cafe", "grounded_by": ["S1"], "certainty": "attested", "salience": 2 } ],
  "sources": [
    { "source_id": "S1", "kind": "testimony", "archive_ref": "WL/SYN/301", "supports": ["cafe.address", "hugo.role", "E1", "hugo.proprietor_of.cafe"] },
    { "source_id": "S2", "kind": "registry", "archive_ref": "WL/SYN/302", "supports": ["E2"] }
  ]
}
```

### "Good" renderings

**Testimony — good**
> People remembered Café Landauer on Krapfengasse as a place where, in 1938 and 1939, you could quietly ask about emigration papers — Hugo Landauer let such conversations happen over coffee. The café seems to have been closed sometime in 1939, though accounts differ on exactly when.

**Finding-aid — good**
> Café Landauer, Krapfengasse 3, Brünn. Coffeehouse; informal meeting point for emigration inquiries, 1938–1939 (testimony, WL/SYN/301). Closure/Aryanisation reported 1939, date uncertain (registry, WL/SYN/302). Proprietor: Hugo Landauer.

*(Flattened and fabricated variants follow the same pattern as Case A.)*

---

## 7. Ethics constraints (built in, not bolted on)

- Every entity is overtly **fictional**: invented names, places, and archive references.
- The `sensitivity` tag exercises **restraint of framing**, never detail of harm — high-sensitivity events are tests of *how* something is told (restrained vs. gratuitous), not prompts to generate graphic content.
- The fabricated profile exists to be *detected and penalised*, never to be produced as output beyond the validation set.

---

## 8. Open items / next steps

1. **Number of cases for v1** — suggest ~10–20 hand-authored cases spanning focal types (person, place, org) and sensitivity levels, enough to form a small SFT set plus a held-out validation grid.
2. **How renderings are produced** — templated (maximally controlled) vs. LLM-rendered under the fact base as a hard constraint (more natural, but needs a drift check so the renderer doesn't add ungrounded specifics). Likely a hybrid: LLM-rendered, then auto-verified against `grounded_by`.
3. **Oracle weights** — left open by design; to be set with historians. v1 only needs correct ordering.
4. **Then** wire the grid into the two reward modules and run the first hypothesis test (§4): does the linguistic reward fail to separate good from fabricated where the oracle succeeds?
