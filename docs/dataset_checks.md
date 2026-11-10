# Dataset Checks & Empirical Verification

This document contains empirical findings and data verification checks executed directly on local cached copies of **RAGTruth** (`data/raw/ragtruth/`) and **HaluEval** (`data/raw/halueval/`).

All numbers and statistics reported below were computed programmatically with zero data cleaning or filtering, using cached files exclusively (offline).

---

## 1. RAGTruth Empirical Checks

### (a) Hallucination Rate by Task Type, Model, and Split

A response is defined as hallucinated if its `labels` list is non-empty (`len(labels) > 0`).

#### 1. By Task Type
*Task type is obtained by joining `response.jsonl` with `source_info.jsonl` on `source_id`.*

| Task Type | Hallucinated Responses | Total Responses | Hallucination Rate (%) | Ratio |
|---|---|---|---|---|
| **Data2txt** | 4,254 | 6,198 | **68.64%** | 0.6864 |
| **QA** | 1,724 | 5,934 | **29.05%** | 0.2905 |
| **Summary** | 1,686 | 5,658 | **29.80%** | 0.2980 |
| **Total** | 7,664 | 17,790 | **43.08%** | 0.4308 |

*Observation:* Data2txt (structured Yelp data) exhibits more than double the hallucination rate of QA or Summary, primarily due to models fabricating details for missing/null attributes.

#### 2. By Model
*Each model generated exactly 2,965 candidate responses across the 2,965 distinct `source_id` contexts.*

| Model Name | Hallucinated Responses | Total Responses | Hallucination Rate (%) | Ratio |
|---|---|---|---|---|
| **gpt-3.5-turbo-0613** | 401 | 2,965 | **13.52%** | 0.1352 |
| **gpt-4-0613** | 406 | 2,965 | **13.69%** | 0.1369 |
| **llama-2-70b-chat** | 1,395 | 2,965 | **47.05%** | 0.4705 |
| **llama-2-13b-chat** | 1,677 | 2,965 | **56.56%** | 0.5656 |
| **llama-2-7b-chat** | 1,832 | 2,965 | **61.79%** | 0.6179 |
| **mistral-7B-instruct** | 1,953 | 2,965 | **65.87%** | 0.6587 |
| **Total** | 7,664 | 17,790 | **43.08%** | 0.4308 |

*Observation:* Proprietary models (`gpt-3.5`, `gpt-4`) maintain an approximately 13.5%–13.7% hallucination rate, while open-weights models (`llama-2` family, `mistral-7B`) hallucinate in 47%–66% of responses.

#### 3. By Predefined Split

| Split Name | Hallucinated Responses | Total Responses | Hallucination Rate (%) | Ratio |
|---|---|---|---|---|
| **train** | 6,721 | 15,090 | **44.54%** | 0.4454 |
| **test** | 943 | 2,700 | **34.93%** | 0.3493 |
| **Total** | 7,664 | 17,790 | **43.08%** | 0.4308 |

---

### (b) Source ID Overlap Across Train and Test Splits

- **Unique `source_id` in `train`:** 2,515
- **Unique `source_id` in `test`:** 450
- **Total `source_id` count:** 2,965 ($2,515 + 450 = 2,965$)
- **Number of `source_id` appearing in both train and test:** **0**

*Conclusion:* There is **zero overlap** (0 leakage) between `train` and `test` source contexts. The split was constructed strictly at the `source_id` partition level ($2,515 \times 6 = 15,090$ train responses; $450 \times 6 = 2,700$ test responses).

---

### (c) Non-Empty Label Counts for Quality Filter Flags

The `response.jsonl` file includes a `quality` field indicating response generation fidelity (`good`, `incorrect_refusal`, or `truncated`).

| Quality Flag | Total Records | Non-Empty Labels (`labels != []`) | Empty Labels (`labels == []`) | % Non-Empty |
|---|---|---|---|---|
| **good** | 17,617 | 7,641 | 9,976 | 43.37% |
| **incorrect_refusal** | 144 | 1 | 143 | 0.69% |
| **truncated** | 29 | 22 | 7 | 75.86% |
| **Combined (`incorrect_refusal` + `truncated`)** | **173** | **23** | **150** | **13.29%** |

*Notable finding:* Exactly **1** record categorized as `incorrect_refusal` contains a non-empty span annotation (`id: "16991"`). The model refused by asserting: *"The provided passages do not contain any information about the song 'Bernadette' by The Four Tops..."*, which annotators marked with `label_type: "Evident Conflict"` because the source passages did in fact contain the song lyrics.

---

### (d) `source_info` Length in Words per Task Type

Word counts are evaluated across all 2,965 source records ($N = 2,965$). Because `source_info` is formatted as a `str` in Summary, a `dict` with keys `('passages', 'question')` in QA, and a `dict` of business metadata and reviews in Data2txt, we report both the string representation (`str(source_info).split()`) and the primary context text (`passages` for QA, article string for Summary, concatenated metadata values for Data2txt):

#### 1. String Representation (`str(source_info).split()`)

| Task Type | Record Count | Min Words | Median Words | Max Words | Count > 400 Words | % > 400 Words |
|---|---|---|---|---|---|---|
| **Data2txt** | 1,033 | 117 | 319.0 | 1,253 | 299 | 28.94% |
| **QA** | 989 | 79 | 224.0 | 509 | 27 | 2.73% |
| **Summary** | 943 | 102 | 428.0 | 1,749 | 537 | 56.95% |
| **Overall** | 2,965 | 79 | 321.0 | 1,749 | 863 | 29.11% |

#### 2. Context Passage Text (`source_info['passages']` for QA, Article `str` for Summary)

| Task Type | Record Count | Min Words | Median Words | Max Words | Count > 400 Words | % > 400 Words |
|---|---|---|---|---|---|---|
| **Data2txt** | 1,033 | 108 | 310.0 | 1,244 | 274 | 26.52% |
| **QA** | 989 | 75 | 217.0 | 498 | 23 | 2.33% |
| **Summary** | 943 | 102 | 428.0 | 1,749 | 537 | 56.95% |

*Implication for RAGuard:* Over **56.9%** of Summary source articles exceed 400 words, meaning cross-encoder NLI and dense attribution modules will routinely encounter context chunks exceeding maximum token sequence lengths (e.g. 512 tokens), validating RAGuard's architectural decision to implement chunking with max-probability pooling.

---

### (e) QA Only: Label Type Counts and `due_to_null` Incidence

Filtering strictly to the QA task type ($N = 5,934$ responses across 989 questions):
- **Total QA responses:** 5,934
- **Total hallucinated QA responses:** 1,724 (29.05%)
- **Total hallucinated spans in QA:** 2,927

#### 1. QA Counts per `label_type`

| Label Type | Span Count | % of All QA Spans |
|---|---|---|
| **Evident Baseless Info** | 1,562 | 53.37% |
| **Subtle Baseless Info** | 893 | 30.51% |
| **Evident Conflict** | 423 | 14.45% |
| **Subtle Conflict** | 49 | 1.67% |
| **Total** | **2,927** | **100.00%** |

*Observation:* Baseless Information (unsupported extrapolation) accounts for **83.88%** (2,455 / 2,927) of all hallucinated spans in QA, whereas Conflict (direct contradiction) accounts for **16.12%** (472 / 2,927).

#### 2. QA Responses with `due_to_null` Flag

- **Hallucinated QA responses with $\ge 1$ `due_to_null` span:** **0**
- **Fraction:** $\frac{0}{1,724} =$ **0.00%** (`0.0000`)

*Why this occurs:* The `due_to_null` flag is specific to structured data where input attributes are `null` / `None`. In the full dataset, all **1,642** occurrences of `due_to_null == True` reside exclusively in **Data2txt** (Yelp dataset). In QA (MS MARCO passages) and Summary (CNN/DM, Recent News), all spans have `due_to_null == False`.

---

## 2. HaluEval Empirical Checks

### (f) Unique Key Definitions, Counts, and Duplicates per Subset

Keys are defined based on the grounding context and prompt fields available in each subset schema:

| Subset Name | Key Fields | Total Records | Unique Keys | Duplicate Count |
|---|---|---|---|---|
| **qa** | `(knowledge, question)` | 10,000 | 10,000 | **0** |
| **qa_samples** | `(knowledge, question)` | 10,000 | 10,000 | **0** |
| **dialogue** | `(knowledge, dialogue_history)` | 10,000 | 10,000 | **0** |
| **dialogue_samples** | `(knowledge, dialogue_history)` | 10,000 | 10,000 | **0** |
| **summarization** | `(document)` | 10,000 | 9,998 | **2** |
| **summarization_samples** | `(document)` | 10,000 | 9,998 | **2** |
| **general** | `(user_query)` | 4,507 | 4,507 | **0** |

*Note on Summarization duplicates:* In `summarization` and `summarization_samples`, exactly **2** source articles appear twice each:
1. Article beginning: *"Tornadoes, fierce winds and severe thunderstorms with large hail are predicted for the Midwest..."* (count: 2)
2. Article beginning: *"Defiant Nigel Farage today insisted he did not 'lose my rag' after rounding on the audience of a live TV debate..."* (count: 2)

---

### (g) `qa_samples` vs. `qa` Alignment and Equivalence

Comparing `qa_samples` ($N = 10,000$) against the paired base subset `qa` ($N = 10,000$):

- **Key Overlap on `(knowledge, question)`:** **10,000 / 10,000** (**100.0%**)
- **Equivalence Analysis:**
  - Records where `answer == right_answer`: **4,990** (**49.90%**)
  - Records where `answer == hallucinated_answer`: **5,010** (**50.10%**)
  - Records where `answer == neither`: **0** (**0.00%**)
  - Records where `answer` equals both: **0** (**0.00%**)

#### Correlation with `hallucination` Ground-Truth Label in `qa_samples`

| Comparison | `hallucination == "no"` | `hallucination == "yes"` | Total |
|---|---|---|---|
| `answer == right_answer` | 4,990 | 0 | 4,990 |
| `answer == hallucinated_answer` | 0 | 5,010 | 5,010 |
| `answer == neither` | 0 | 0 | 0 |

*Conclusion:* `qa_samples` is an exact 1-to-1 derived sample from `qa`. Every sample labeled `"no"` is identical to `right_answer`, and every sample labeled `"yes"` is identical to `hallucinated_answer`.

---

### (h) Dialogue and Summarization Alignment

#### 1. `dialogue_samples` vs. `dialogue`

Comparing `dialogue_samples` ($N = 10,000$) against paired `dialogue` ($N = 10,000$):

- **Key Overlap on `(knowledge, dialogue_history)`:** **10,000 / 10,000** (**100.0%**)
- **Equivalence Analysis:**
  - Records where `response == right_response`: **4,990** (**49.90%**) [all have `hallucination == "no"`]
  - Records where `response == hallucinated_response`: **5,010** (**50.10%**) [all have `hallucination == "yes"`]
  - Records where `response == neither`: **0** (**0.00%**)

#### 2. `summarization_samples` vs. `summarization`

Comparing `summarization_samples` ($N = 10,000$) against paired `summarization` ($N = 10,000$):

- **Document Key Overlap on `(document)`:** **9,998 / 9,998** unique documents (**100.0%**)
- **Row-Aligned Equivalence Analysis:**
  - Records where `summary == right_summary`: **4,990** (**49.90%**) [all have `hallucination == "no"`]
  - Records where `summary == hallucinated_summary`: **5,010** (**50.10%**) [all have `hallucination == "yes"`]
  - Records where `summary == neither`: **0** (**0.00%**)

*Architectural Conclusion:* Across all three task domains (QA, Dialogue, Summarization), the `*_samples` subsets are strictly 50/50 balanced classifications (4,990 factual vs. 5,010 hallucinated) sampled from the paired benchmarks. Benchmarking on both the paired set and the sampled set is redundant; RAGuard will use `*_samples` for standard classification metrics (PR-AUC, ROC-AUC, F1) and paired sets for pairwise ranking/preference tests.

