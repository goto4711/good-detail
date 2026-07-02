#!/usr/bin/env python3
"""
make_annotation_sheet.py
================================================================
Turn a best-of-N candidate dump (bestofn_demo.py --save) into a BLIND annotation
sheet for the historian workshop (P1) — a single self-contained HTML file each
annotator opens in a browser, rates, and downloads their answers as JSON.

Blind by design: candidates are shuffled per record and labelled A/B/C…; the
reward scores are NOT shown. The download carries the original candidate index,
so analyze_annotations.py can join answers back to the rewards.

  python make_annotation_sheet.py candidates.json -o annotation.html
Then send annotation.html to each historian; collect their downloaded
responses_*.json and run analyze_annotations.py.
"""

import argparse
import html
import json
import random


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Good-detail annotation</title><style>
body{{font:16px/1.5 Georgia,serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#222}}
h1{{font-size:1.4rem}} .rec{{border-top:3px solid #444;margin-top:2.5rem;padding-top:1rem}}
.src{{background:#f4f1ea;border-left:4px solid #999;padding:.6rem .9rem;white-space:pre-wrap;
font:13px/1.45 monospace;color:#444;max-height:260px;overflow:auto}}
.cand{{border:1px solid #ccc;border-radius:6px;padding:.8rem 1rem;margin:.8rem 0;background:#fafafa}}
.lab{{font-weight:bold;color:#7a2}} .rate label{{margin-right:.8rem;white-space:nowrap}}
textarea{{width:100%;min-height:3rem;font:14px/1.4 Georgia,serif}}
.banner{{background:#fff3cd;border:1px solid #e0c97f;padding:.5rem .8rem;border-radius:5px}}
.dl{{position:sticky;bottom:0;background:#fff;padding:.8rem 0;border-top:2px solid #444}}
button{{font-size:1rem;padding:.5rem 1rem}}
</style></head><body>
<h1>Good detail — annotation</h1>
<p class="banner">⚠ These narratives are <b>machine-generated and UNVERIFIED</b> — a methods
study, not historical sources. Please judge them as <i>candidate summaries</i>, not as fact.</p>
<p>For each record you see the source material, then several candidate micro-narratives in
random order. Rate each on <b>“good detail”</b> overall (1 = poor, 5 = excellent), and add a
note on <i>why</i> — especially where they differ. <b>Download your answers before closing
the page</b> (nothing is saved automatically).</p>
<label>Your name/initials: <input id="annot" size="20"></label>
<div id="recs"></div>
<div class="dl"><button onclick="dl()">⬇ Download my answers (JSON)</button>
<span id="msg"></span></div>
<script id="data" type="application/json">{data}</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const R=document.getElementById('recs');
D.records.forEach((rec,ri)=>{{
 const d=document.createElement('div'); d.className='rec';
 d.innerHTML=`<h3>Record ${{ri+1}} of ${{D.records.length}} — ${{rec.unit||''}}</h3>`+
   `<div class="src">${{rec.record_block}}</div>`;
 rec.cands.forEach(c=>{{
   const b=document.createElement('div'); b.className='cand';
   b.innerHTML=`<div><span class="lab">${{c.label}}.</span> ${{c.text}}</div>`+
     `<div class="rate">Good detail: `+
     [1,2,3,4,5].map(n=>`<label><input type="radio" name="r_${{ri}}_${{c.i}}" value="${{n}}">${{n}}</label>`).join('')+`</div>`;
   d.appendChild(b);
 }});
 const t=document.createElement('div');
 t.innerHTML=`<div>Notes (why — what makes one better/worse?):</div>`+
   `<textarea id="note_${{ri}}"></textarea>`;
 d.appendChild(t); R.appendChild(d);
}});
function dl(){{
 const out={{annotator:document.getElementById('annot').value||'anon',
   corpus:D.corpus,source:D.source,gen_model:D.gen_model,responses:[],notes:{{}}}};
 D.records.forEach((rec,ri)=>{{
   rec.cands.forEach(c=>{{
     const sel=document.querySelector(`input[name="r_${{ri}}_${{c.i}}"]:checked`);
     if(sel) out.responses.push({{record_id:rec.id,candidate_i:c.i,overall:+sel.value}});
   }});
   const n=document.getElementById('note_'+ri).value.trim();
   if(n) out.notes[rec.id]=n;
 }});
 const blob=new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}});
 const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
 a.download='responses_'+(out.annotator||'anon').replace(/\\W+/g,'_')+'.json'; a.click();
 document.getElementById('msg').textContent=' …downloaded '+out.responses.length+' ratings. Thank you!';
}}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", help="JSON from bestofn_demo.py --save")
    ap.add_argument("-o", "--out", default="annotation.html")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(args.candidates, encoding="utf-8"))
    rng = random.Random(args.seed)
    recs = []
    for r in data["records"]:
        cands = list(r["candidates"])
        rng.shuffle(cands)                      # blind: random display order
        labelled = [{"label": chr(65 + j), "i": c["i"], "text": html.escape(c["text"])}
                    for j, c in enumerate(cands)]
        recs.append({"id": r["id"], "unit": html.escape(r.get("unit", "")),
                     "record_block": html.escape(r["record_block"]), "cands": labelled})
    payload = {"corpus": data.get("corpus"), "source": data.get("source"),
               "gen_model": data.get("gen_model"), "records": recs}
    open(args.out, "w", encoding="utf-8").write(PAGE.format(data=json.dumps(payload)))
    print(f"Wrote {args.out}  ({len(recs)} records, "
          f"{sum(len(r['cands']) for r in recs)} candidates). Send it to each annotator.")


if __name__ == "__main__":
    main()
