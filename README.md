# Same verdict, different receipts

**Do LLMs answer differently depending on who's asking?**

I asked 3 models 85 questions under 4 user profiles (no profile, novice, intermediate, expert). Two LLM judges graded the answers. 973 responses were analysed.

- **No significant pooled shift in verdicts** between novices and experts.
- **Users who say they are experts get more named sources** than users with no profile, in all three models.
- **Novices get fewer sources than no-profile users only in Gemini.** GPT and Qwen show no difference.
- **Main caveat:** the expert profiles mention reading literature or case law, which may prompt citations by itself.

Built during the AI x Epistemics research sprint, November 2026. I used Claude Code heavily for code, analysis and drafting.

## Results

Share of answers that name a specific source: novice 23%, no profile 30%, intermediate 42%, expert 72%.

Change in that share compared with no profile:

| Model | Novice − no profile | Expert − no profile |
|---|---|---|
| Gemini 3.8 Flash | −0.22 (p < 0.001) | +0.22 (p < 0.001) |
| GPT-6.1 Terra | −0.01 (p = 1.0) | +0.57 (p < 0.001) |
| Qwen 3.5 397B | −0.01 (p = 1.0) | +0.45 (p < 0.001) |
| Pooled | −0.08 (p = 0.002) | +0.41 (p < 0.001) |

Novice minus expert, pooled (231 to 237 question-model pairs):

| Metric | Difference | 95% CI | p |
|---|---|---|---|
| Stance, −2 to 2 | +0.02 | [−0.04, 0.09] | 0.70 |
| Stated confidence, 0 to 100 | +0.40 | [−0.10, 0.87] | 0.12 |
| Names a specific source | −0.49 | [−0.55, −0.43] | < 0.001 |
| States evidence strength | −0.31 | [−0.36, −0.26] | < 0.001 |
| Caveats per answer | −1.69 | [−1.94, −1.45] | < 0.001 |
| Share of known perspectives covered | −0.06 | [−0.08, −0.03] | < 0.001 |
| Says the question is disputed | +0.05 | [0.01, 0.09] | 0.036 |
| Advice on how to verify | −0.02 | [−0.06, 0.02] | 0.41 |

Most of the source and caveat gaps come from experts getting more. Caveats do not differ between novice and no profile (−0.04, p = 0.72). Confidence was slightly higher for novices than experts in Qwen (+1.1, p = 0.025), but not in the other two models.

**Judge agreement** (GLM 5.2 and DeepSeek V4 Flash, 958 responses graded by both):

| Metric | Agreement | Cohen's kappa |
|---|---|---|
| Named sources | 98% | 0.95 |
| Dispute awareness | 93% | 0.78 |
| Evidence strength | 75% | 0.51 |

Perspective coverage: r = 0.87.

![Decisiveness, confidence, dispute awareness and perspective coverage](results/analysis/fig_audience_by_tier.png)

## Method

1. **Questions** ([select_questions.py](select_questions.py)): from the [DebateQA](http://arxiv.org/abs/2408.01419) test split. One LLM labelled 646 yes/no-shaped questions; a second labelled the 276 eligible ones. I used a seeded sample plus manual vetoes, corrections and one forced inclusion, all in [data/selection_log.md](data/selection_log.md). Final set: 60 debated, 15 consensus but contested, 10 trivia.
2. **Prompts** ([run_experiment.py](run_experiment.py)): the same question with a different profile. Each answer ends with BOTTOM LINE, STANCE and CONFIDENCE lines.
3. **Grading** ([grade.py](grade.py)): judges do not see the profile, and STANCE and CONFIDENCE lines are removed. An answer's wording could still hint at the profile.
4. **Analysis** ([analyze.py](analyze.py)): paired differences, bootstrap 95% CIs and sign-flip permutation tests.

## Limitations

- **Persona wording:** only the expert profiles mention reading field literature, case law or statutes. A profile without that phrase is the needed control.
- **One run per condition:** run-to-run noise was not measured.
- **Sources were not verified:** the metric checks whether a source is named, not whether it exists or is correct.
- **Length:** answers to experts were about 126 words longer than no-profile answers, and novice answers about 75 words longer. Length was not controlled for.
- **Missing data:** 47 of 1,020 planned responses are missing: 24 Gemini/GPT calls have no successful response and 23 Qwen answers were empty. 14 answers had no parseable STANCE. 15 of 1,946 expected judge records are missing.
- **Exploratory:** the source result was not the original hypothesis, and p-values are not corrected for multiple comparisons.
- **Scope:** single-turn, English, 85 questions, 3 models.

## Reproduce

```bash
pip install -r requirements.txt
python analyze.py
```

This uses the saved responses and grades, with no API calls. It writes the pooled contrasts and judge agreement. Its per-model table covers novice − expert and excludes trivia. For the per-model table above, run this after `analyze.py`:

```python
import numpy as np
import pandas as pd
from analyze import paired

d = pd.read_csv("results/analysis/items.csv")
rng = np.random.default_rng(0)
for model, sub in d.groupby("model"):
    for a in ["novice", "expert"]:
        print(model, a, paired(sub, "specific_sources", a, "control", rng))
```

Cohen's kappa was computed separately from the same grades. P-values and intervals can vary slightly with the random seed.

To collect responses again, set `COARENA_API_KEY` and `ADAPTION_API_KEY` in `.env`, move the saved results elsewhere, and run:

```bash
python run_experiment.py run --models coarena/gemini-3-8-flash,coarena/gpt-6-1-terra,adaption/qwen3.5-397b --reps 1
python grade.py
```

## Data credits

Questions and perspective annotations: DebateQA (Xu et al., 2024, [arXiv:2408.01419](http://arxiv.org/abs/2408.01419)).
