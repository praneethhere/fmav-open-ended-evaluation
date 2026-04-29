# FMAV — Open-Ended Evaluation

Companion repository containing the open-ended evaluation code, dataset, and result files for the FMAV (Federated Multi-Agent Validator) experiments.

## Repository contents

```
TruthfulQA.csv                                ← official TruthfulQA dataset (Apache 2.0)
fmav_groq_openended_runner.py                 ← open-ended runner (Groq + gpt-oss-120b)
fmav_groq_openended_results.json              ← aggregated open-ended results
fmav_groq_openended_generations.jsonl         ← per-question generation audit trail
```

## Requirements

```
pip install numpy pandas scikit-learn openai
```

Set your Groq API key:

```
export GROQ_API_KEY=<your_key>
```

## Running the open-ended evaluation

Quick test (50 questions):

```
python fmav_groq_openended_runner.py --csv TruthfulQA.csv --max-questions 50
```

Full run (790 questions):

```
python fmav_groq_openended_runner.py --csv TruthfulQA.csv
```

Outputs:
- `fmav_groq_openended_results.json` — aggregated metrics, McNemar tests, domain breakdown
- `fmav_groq_openended_generations.jsonl` — per-question audit trail with generated answer, similarity scores, validator votes, and final delivery decision

The runner uses `temperature = 0` for deterministic generation. Numbers may vary slightly only if the Groq-hosted model or serving infrastructure is updated.

## Default configuration

| Parameter | Value |
|---|---|
| Backend model | `openai/gpt-oss-120b` via Groq |
| Temperature | 0 |
| Cross-validation | 5-fold stratified by broad domain |
| FMAV quorum | 3 of 5 |
| Post-hoc filter threshold | 0.47 |
| RAG-style threshold | 0.45 |
| Routing accuracy target | 0.908 |
| Random seed | 42 |

## License

Code: MIT
Dataset: TruthfulQA is provided under Apache 2.0; refer to the upstream repository at https://github.com/sylinrl/TruthfulQA for original terms.