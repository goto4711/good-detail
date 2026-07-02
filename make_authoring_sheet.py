#!/usr/bin/env python3
"""
make_authoring_sheet.py
================================================================
The GOLD-DATA front door. Where make_annotation_sheet.py asks historians to RATE
machine candidates, this asks them to WRITE the good summary themselves — a
historian reads a real record and authors the micro-narrative. Each authored
(record -> human summary) pair is a top-quality SFT target: instead of teaching
the model to rank, it teaches the model to write good detail directly.

No public corpus of archival "good detail" summaries exists (the nearest is
WikiBio's infobox->biography pairs, known to hallucinate ~62% of the time), so
these human-authored summaries ARE the contribution's training data.

Two ways to load the records the historian will summarise:
  --from records.json        # a bestofn --save file, or this tool's own records.json
  --corpus EHRI --source extracted --limit N   # straight from the ingest layer

It writes a self-contained authoring.html (open in a browser, write, download
authored_<name>.json) AND a records.json carrying the exact prompts, so:

  python make_authoring_sheet.py --corpus EHRI --source extracted --limit 20 -o authoring.html
  # historians fill authoring.html, send back authored_*.json, then:
  python build_training_data.py records.json --label authored --authored authored/

SAFEGUARDS: the historian sees the record + source excerpt and is asked to write
ONLY what the source supports; outputs stay local (real names) and off git.
"""

import argparse
import html
import json
import os
import sys


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Good-detail authoring</title><style>
body{{font:16px/1.5 Georgia,serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#222}}
h1{{font-size:1.4rem}} .rec{{border-top:3px solid #444;margin-top:2.5rem;padding-top:1rem}}
.src{{background:#f4f1ea;border-left:4px solid #999;padding:.6rem .9rem;white-space:pre-wrap;
font:13px/1.45 monospace;color:#444;max-height:320px;overflow:auto}}
textarea.sum{{width:100%;min-height:7rem;font:15px/1.5 Georgia,serif;margin-top:.5rem}}
.banner{{background:#e7f1e7;border:1px solid #8fb98f;padding:.5rem .8rem;border-radius:5px}}
.dl{{position:sticky;bottom:0;background:#fff;padding:.8rem 0;border-top:2px solid #444}}
.conf label{{margin-right:.8rem;white-space:nowrap}} small{{color:#666}}
button{{font-size:1rem;padding:.5rem 1rem}}
</style></head><body>
<h1>Good detail — author the summary</h1>
<p class="banner">You are the expert. For each record below, please <b>write a short
micro-narrative</b> that captures the <b>good detail</b> — grounded in, and faithful to, the
source shown. Write only what the record supports; if something is uncertain, hedge or omit it.
A few sentences is ideal.</p>
<p><small>Your summaries become gold training examples. <b>Download before closing the page</b> —
nothing is saved automatically.</small></p>
<label>Your name/initials: <input id="annot" size="20"></label>
<div id="recs"></div>
<div class="dl"><button onclick="dl()">⬇ Download my summaries (JSON)</button>
<span id="msg"></span></div>
<script id="data" type="application/json">{data}</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const R=document.getElementById('recs');
D.records.forEach((rec,ri)=>{{
 const d=document.createElement('div'); d.className='rec';
 d.innerHTML=`<h3>Record ${{ri+1}} of ${{D.records.length}} — ${{rec.unit||''}}</h3>`+
   `<div class="src">${{rec.record_block}}</div>`+
   `<div>Your micro-narrative:</div>`+
   `<textarea class="sum" id="sum_${{ri}}" placeholder="Write the grounded summary here…"></textarea>`+
   `<div class="conf"><small>How confident are you it is fully supported by the source?</small><br>`+
   [1,2,3,4,5].map(n=>`<label><input type="radio" name="c_${{ri}}" value="${{n}}">${{n}}</label>`).join('')+
   ` <small>(1 = unsure, 5 = certain)</small></div>`;
 R.appendChild(d);
}});
function dl(){{
 const out={{author:document.getElementById('annot').value||'anon',
   corpus:D.corpus,source:D.source,summaries:[]}};
 D.records.forEach((rec,ri)=>{{
   const t=document.getElementById('sum_'+ri).value.trim();
   if(!t) return;
   const sel=document.querySelector(`input[name="c_${{ri}}"]:checked`);
   out.summaries.push({{record_id:rec.id,text:t,confidence:sel?+sel.value:null}});
 }});
 const blob=new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}});
 const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
 a.download='authored_'+(out.author||'anon').replace(/\\W+/g,'_')+'.json'; a.click();
 document.getElementById('msg').textContent=' …downloaded '+out.summaries.length+' summaries. Thank you!';
}}
</script></body></html>"""


def _records_from_corpus(corpus, source, limit):
    """Load real records via the ingest layer and render the same prompt GRPO/SFT feed."""
    try:
        from ingest import load_corpus, record_block
        from config import INSTRUCTION
    except ImportError as e:
        sys.exit(f"--corpus needs the project modules: {e}")
    recs = load_corpus(corpus, source, limit=limit)
    if not recs:
        sys.exit(f"No records for {corpus}/{source}")
    out = []
    for rec in recs:
        focal = rec.unit or rec.title
        block = record_block(rec, with_relations=False)
        excerpt = (rec.source_text or "")[:1200]
        if excerpt:
            block = block + "\n\nSOURCE EXCERPT:\n" + excerpt
        prompt_user = INSTRUCTION.format(register=rec.register.replace("_", "-"),
                                         unit=focal, record=block)
        out.append({"id": rec.id, "title": rec.title, "unit": focal,
                    "register": rec.register, "record_block": block,
                    "prompt_user": prompt_user})
    return {"corpus": corpus, "source": source, "records": out}


def _records_from_file(path):
    d = json.load(open(path, encoding="utf-8"))
    recs = []
    for r in d.get("records", []):
        recs.append({"id": r["id"], "title": r.get("title", ""), "unit": r.get("unit", ""),
                     "register": r.get("register"), "record_block": r.get("record_block", ""),
                     "prompt_user": r.get("prompt_user")})
    return {"corpus": d.get("corpus"), "source": d.get("source"), "records": recs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src_file", default=None,
                    help="records JSON (a bestofn --save file or this tool's records.json)")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--source", default="extracted")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("-o", "--out", default="authoring.html")
    ap.add_argument("--records_out", default="records.json",
                    help="also write the record list (with prompts) for build_training_data.py")
    args = ap.parse_args()

    if args.src_file:
        data = _records_from_file(args.src_file)
    elif args.corpus:
        data = _records_from_corpus(args.corpus, args.source, args.limit)
    else:
        sys.exit("Give either --from records.json OR --corpus NAME [--source SRC]")

    payload = {"corpus": data.get("corpus"), "source": data.get("source"),
               "records": [{"id": r["id"], "unit": html.escape(r.get("unit") or ""),
                            "record_block": html.escape(r.get("record_block") or "")}
                           for r in data["records"]]}
    open(args.out, "w", encoding="utf-8").write(PAGE.format(data=json.dumps(payload)))
    # the un-escaped record list with prompts → feeds build_training_data.py --label authored
    json.dump(data, open(args.records_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}  ({len(data['records'])} records) and {args.records_out}.")
    print("Send authoring.html to each historian; collect authored_*.json; then:")
    print(f"  python build_training_data.py {args.records_out} --label authored --authored <folder>")


if __name__ == "__main__":
    main()
