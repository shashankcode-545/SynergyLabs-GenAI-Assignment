# Answer evaluation [MEASURED]

18 eval questions, k=5, chunk 1200/200, generator `gemini-3.1-flash-lite`, judge `gemini-3.6-flash`.
Mean [95% bootstrap CI]. Per-question detail and judge claims: `answers.json`.

| Metric | Score | Method |
|---|---|---|
| Faithfulness | 1.000 [1.00, 1.00] | LLM judge splits the answer into claims; a claim counts only if its quote is verified in the context by code |
| Answer relevance | 1.000 [1.00, 1.00] | LLM judge 1-5 -> (s-1)/4, answerable questions that were answered |
| EM | 0.000 [0.00, 0.00] | SQuAD-normalized exact match vs short gold answer (abstain = 0) |
| F1 | 0.310 [0.19, 0.43] | SQuAD token F1 vs gold (abstain = 0) |
| Contains gold | 0.571 [0.29, 0.79] | all gold answer parts appear in the answer |
| Citation precision | 0.611 [0.44, 0.79] | cited chunks that are relevant to the gold evidence (no LLM) |
| Correct abstention (unanswerable) | 4/4 | |
| False abstention (answerable) | 2/14 | |

Generation spend $0.0070, judge spend $0.0206 (from API usage metadata; $0 when replayed from cache).
