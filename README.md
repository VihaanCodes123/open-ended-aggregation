# Aggregating open-ended LLM answers — results

**Start with `REPORT.docx`.** Everything else here backs it up.

## The one-paragraph version

The paper we built on (*Beyond Majority Voting*, ICML 2026) combines several LLMs' answers
optimally, but only for multiple choice — it needs a fixed option list. We extended it to
open-ended answers by replacing its exact-match test with a similarity measure, then tested
it on 6 models × 3,000 TriviaQA questions.

**Result: partially negative.** Our method beats majority voting (+3.2 points, p=4e-11), but
an ablation shows it never beats the better of its own two ingredients — on verbose answers a
3-line unweighted baseline outperforms it. The report explains why.

## Files

| file | what |
|---|---|
| `REPORT.docx` | **read this** — full write-up, 8 sections, 2 figures |
| `kernel_agg.py` | the method. Only `em_estimate_beta` (label-free skill estimation) and `agg_kernel` matter here; the rest is simulation machinery from earlier work. |
| `generate.py` | queries the 6 models via OpenRouter. Resumable. Downloads TriviaQA itself on first run. |
| `analyze_real.py` | builds candidate pools from the generations, runs every aggregator, scores them |
| `data/gen_full.jsonl` | **all 36,000 raw model responses.** This is the part that cost money — public benchmarks are downloaded automatically and not included. |
| `results/real_full.json` | the numbers behind the report |

## Reproduce the analysis (no API key needed — the generations are included)

```bash
python3 -m venv venv
./venv/bin/pip install numpy scipy pandas pyarrow requests sentence-transformers matplotlib
./venv/bin/python analyze_real.py full
```

To generate fresh responses instead, put `OPENROUTER_API_KEY=sk-or-...` in a `.env` file
and run `./venv/bin/python generate.py pilot` (50 questions, ~$0.01).

## Known issues, before you quote any number

- Grading is substring matching against TriviaQA's answer aliases. **12.8% of the
  full-sentence responses are truncated** by our token cap — a defect we introduced, and it
  contaminates the setting where our method loses. Fix before trusting those numbers.
- False negatives on questions with only one recorded alias are unmeasured.
- One benchmark, one embedding model, six models from four families.

Full list in REPORT.docx §7.
