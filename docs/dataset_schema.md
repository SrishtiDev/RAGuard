# Dataset Schema Reference

All findings documented below are derived directly and exclusively from running `scripts/explore_datasets.py` and inspecting the downloaded raw datasets on 2026-10-12. All counts, field names, Python types, distributions, and sample values represent actual observed data.

---

## 1. RAGTruth

- **Paper:** "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models"
- **Repository:** https://github.com/ParticleMedia/RAGTruth
- **Source Files Downloaded:**
  - `data/raw/ragtruth/source_info.jsonl` (from `https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl`)
  - `data/raw/ragtruth/response.jsonl` (from `https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl`)

---

### 1.1 `source_info.jsonl`

#### Overview & Counts
- **Total Records:** 2,965
- **Record Type:** Prompt and retrieval context source records (one record per input context/task prompt).
- **Join Key:** `source_id` (joins 1-to-many to `response.jsonl` on `response.source_id == source_info.source_id`).

#### Fields & Types
| Field Name | Observed Python Type | Description |
|---|---|---|
| `source_id` | `str` | Unique source identifier (numeric string, e.g. `"15592"`, `"15593"`) |
| `task_type` | `str` | Task category: `"Data2txt"`, `"QA"`, or `"Summary"` |
| `source` | `str` | Provenance corpus name: `"CNN/DM"`, `"MARCO"`, `"Recent News"`, `"Yelp"` |
| `source_info` | `str` | Retrieved reference context / passage text used to ground the generation |
| `prompt` | `str` | The exact prompt instructed to the generation LLM |

#### Task Type Distribution
- `Data2txt`: 1,033
- `QA`: 989
- `Summary`: 943
- **Total:** 2,965

#### Provenance Source Distribution
- `Yelp`: 1,033 (all corresponding to `Data2txt`)
- `MARCO`: 989 (all corresponding to `QA`)
- `CNN/DM`: 628 (corresponding to `Summary`)
- `Recent News`: 315 (corresponding to `Summary`)
- **Total:** 2,965

#### Context / Passage Shape & Location
The retrieval context resides entirely in the `source_info` field as a plain text `str` (ranging from news articles in `Summary`, web passages in MS MARCO `QA`, to structured business review text/metadata in `Data2txt`).

#### Example Record (`source_id = "15592"`)
```json
{
  "source_id": "15592",
  "task_type": "Summary",
  "source": "CNN/DM",
  "source_info": "Seventy years ago, Anne Frank died of typhus in a Nazi concentration camp at the age of 15. Just two weeks after her supposed death on March 31, 1945, the Bergen-Belsen concentration camp where she had been imprisoned was liberated -- timing that showed how close the Jewish diarist had been to surviving the Holocaust. But new research released by the Anne Frank House shows that Anne and her older sister, Margot Frank, died at least a month earlier than previously thought. Researchers re-examined archives of the Red Cross, the International Tracing Service and the Bergen-Belsen Memorial, along with testimonies of survivors, the Anne Frank House said Tuesday. They concluded that Anne and Margot probably died in February 1945...",
  "prompt": "Summarize the following news within 116 words:\nSeventy years ago, Anne Frank died of typhus in a Nazi concentration camp at the age of 15. Just two weeks after her supposed death on March 31, 1945, the Bergen-Belsen concentration camp where she had been imprisoned was liberated -- timing that showed how close the Jewish diarist had been to surviving the Holocaust..."
}
```

---

### 1.2 `response.jsonl`

#### Overview & Counts
- **Total Records:** 17,790
- Exactly 6 responses per `source_id` across 6 distinct models ($2,965 \times 6 = 17,790$).

#### Fields & Types
| Field Name | Observed Python Type | Description |
|---|---|---|
| `id` | `str` | Unique response ID (numeric string, e.g. `"0"`, `"1"`, `"2"`) |
| `source_id` | `str` | Foreign key matching `source_info.jsonl` |
| `model` | `str` | LLM that produced the generation |
| `temperature` | `float` | Sampling temperature (all observed values are `0.7`) |
| `labels` | `list[dict]` | Fine-grained span-level hallucination annotations (empty if non-hallucinated) |
| `split` | `str` | Predefined split assignment: `"train"` or `"test"` |
| `quality` | `str` | Quality filter flag: `"good"`, `"incorrect_refusal"`, or `"truncated"` |
| `response` | `str` | Full model-generated response text |

#### Model Distribution (Exact Counts)
Each model generated exactly 2,965 responses:
- `gpt-4-0613`: 2,965
- `gpt-3.5-turbo-0613`: 2,965
- `mistral-7B-instruct`: 2,965
- `llama-2-7b-chat`: 2,965
- `llama-2-13b-chat`: 2,965
- `llama-2-70b-chat`: 2,965
- **Total:** 17,790

#### Predefined Splits
- `train`: 15,090 (84.8%)
- `test`: 2,700 (15.2%)
- *Note:* No predefined `val` split is provided in the file.

#### Quality Field Distribution
- `good`: 17,617 (99.03%)
- `incorrect_refusal`: 144 (0.81%)
- `truncated`: 29 (0.16%)

#### Hallucination Labels & Representation

##### Response-Level Distribution
- **Hallucinated** (`labels != []`): 7,664 (43.08%)
- **Not Hallucinated** (`labels == []`): 10,126 (56.92%)

##### Span-Level Annotation Structure
When `labels` is non-empty, each item is a dictionary with the following 7 keys:
- `start` (`int`): Exact character start offset into the `response` string.
- `end` (`int`): Exact character end offset into the `response` string.
- `text` (`str`): The exact substring of `response[start:end]`.
- `meta` (`str`): Detailed annotator justification comparing source context vs. generation.
- `label_type` (`str`): Hallucination taxonomy label category.
- `implicit_true` (`bool`): Flag indicating if the information might be world-true but ungrounded by the provided source context.
- `due_to_null` (`bool`): Flag indicating if the hallucination arose from ungrounded claims when reference context lacked the relevant entity/fact.

##### Distinct Label Types & Distributions
Observed across all spans:
- `Evident Baseless Info`: 6,237 spans
- `Evident Conflict`: 5,324 spans
- `Subtle Baseless Info`: 2,527 spans
- `Subtle Conflict`: 201 spans
- **Total Spans:** 14,289 spans

##### Span Flag Counts
- `implicit_true == True`: 1,928 spans
- `due_to_null == True`: 1,642 spans

#### Example Label Dict
```json
{
  "start": 636,
  "end": 653,
  "text": "February 7, 2022.",
  "meta": "EVIDENT CONFLICT\nOriginal: February 7 (1945)\nGenerated: February 7, 2022",
  "label_type": "Evident Conflict",
  "implicit_true": false,
  "due_to_null": false
}
```

---

## 2. HaluEval

- **Paper:** "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models"
- **Repository / Source:** Hugging Face `datasets` repository `pminervini/HaluEval` (official mirror)
- **Local Download Path:** `data/raw/halueval/`
- **Subsets Available:** 7 subsets:
  1. `qa`
  2. `qa_samples`
  3. `dialogue`
  4. `dialogue_samples`
  5. `summarization`
  6. `summarization_samples`
  7. `general`
- **Predefined Split for all subsets:** Only 1 split exists: `'data'`. No predefined train/val/test splits exist in Hugging Face.

---

### 2.1 Subset: `qa`

- **Config Name:** `qa`
- **Record Count:** 10,000
- **Structure:** **Paired responses** (contains both ground-truth right answer and hallucinated answer for each input).
- **Context Field:** `knowledge` (`str`) is present and contains grounding passages.

#### Fields & Types
| Field Name | Observed Python Type | Description |
|---|---|---|
| `knowledge` | `str` | Supporting passages / context text |
| `question` | `str` | User question |
| `right_answer` | `str` | Ground-truth factual answer |
| `hallucinated_answer` | `str` | Synthesized hallucinated answer |

#### Example Record
```json
{
  "knowledge": "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.First for Women is a woman's magazine published by Bauer Media Group in the USA.",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "right_answer": "Arthur's Magazine",
  "hallucinated_answer": "First for Women was started first."
}
```

---

### 2.2 Subset: `qa_samples`

- **Config Name:** `qa_samples`
- **Record Count:** 10,000
- **Structure:** **Single response** per example with binary hallucination label.
- **Context Field:** `knowledge` (`str`) is present.

#### Fields & Types
| Field Name | Observed Python Type | Description |
|---|---|---|
| `knowledge` | `str` | Supporting passages / context text |
| `question` | `str` | User question |
| `answer` | `str` | Candidate answer to evaluate |
| `hallucination` | `str` | Binary flag (`"yes"` or `"no"`) |

#### Label Distribution
- `"yes"` (Hallucinated): 5,010 (50.1%)
- `"no"` (Factual): 4,990 (49.9%)

#### Example Record
```json
{
  "knowledge": "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.First for Women is a woman's magazine published by Bauer Media Group in the USA.",
  "question": "Which magazine was started first Arthur's Magazine or First for Women?",
  "answer": "First for Women was started first.",
  "hallucination": "yes"
}
```

---

### 2.3 Subset: `dialogue`

- **Config Name:** `dialogue`
- **Record Count:** 10,000
- **Structure:** **Paired responses** (both factual and hallucinated candidate dialogue responses).
- **Context Field:** `knowledge` (`str`) is present.

#### Fields & Types
| Field Name | Observed Python Type | Description |
|---|---|---|
| `knowledge` | `str` | External grounding facts/knowledge snippet |
| `dialogue_history` | `str` | Multi-turn conversation history |
| `right_response` | `str` | Factual grounding response |
| `hallucinated_response` | `str` | Hallucinated response |

#### Example Record
```json
{
  "knowledge": "Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal",
  "dialogue_history": "[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i like him too did you know he also was in Zodiac a crime fiction film.",
  "right_response": "I like crime fiction! Didn't know RDJ was in there. Jake Gyllenhaal starred as well.",
  "hallucinated_response": "I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks."
}
```

---

### 2.4 Subset: `dialogue_samples`

- **Config Name:** `dialogue_samples`
- **Record Count:** 10,000
- **Structure:** **Single response** per record with binary label.
- **Context Field:** `knowledge` (`str`) is present.

#### Fields & Types
| Field Name | Observed Python Type | Description |
|---|---|---|
| `knowledge` | `str` | External grounding facts/knowledge snippet |
| `dialogue_history` | `str` | Multi-turn conversation history |
| `response` | `str` | Candidate dialogue response to evaluate |
| `hallucination` | `str` | Binary flag (`"yes"` or `"no"`) |

#### Label Distribution
- `"yes"` (Hallucinated): 5,010 (50.1%)
- `"no"` (Factual): 4,990 (49.9%)

#### Example Record
```json
{
  "knowledge": "Iron Man is starring Robert Downey Jr.Robert Downey Jr. starred in Zodiac (Crime Fiction Film)Zodiac (Crime Fiction Film) is starring Jake Gyllenhaal",
  "dialogue_history": "[Human]: Do you like Iron Man [Assistant]: Sure do! Robert Downey Jr. is a favorite. [Human]: Yes i like him too did you know he also was in Zodiac a crime fiction film.",
  "response": "I'm not a fan of crime movies, but I did know that RDJ starred in Zodiac with Tom Hanks.",
  "hallucination": "yes"
}
```

---

### 2.5 Subset: `summarization`

- **Config Name:** `summarization`
- **Record Count:** 10,000
- **Structure:** **Paired responses** (factual summary vs. hallucinated summary).
- **Context Field:** `document` (`str`) is present and contains source news articles (average length >7,000 characters).

#### Fields & Types
| Field Name | Observed Python Type | Description |
|---|---|---|
| `document` | `str` | Source article context |
| `right_summary` | `str` | Factual reference summary |
| `hallucinated_summary` | `str` | Synthesized hallucinated summary |

#### Example Record
```json
{
  "document": "Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane...",
  "right_summary": "Marseille prosecutor says \"so far no videos were used in the crash investigation\". He urged anyone with footage to turn it over immediately.",
  "hallucinated_summary": "A video showing the final moments of Germanwings Flight 9525 has been recovered by investigators from the wreckage site. Marseille prosecutor Brice Robin urged anyone who might have more footage to turn it over immediately. Andreas Lubitz, the co-pilot accused of deliberately crashing the plane, had a history of severe depression and suicidal tendencies."
}
```

---

### 2.6 Subset: `summarization_samples`

- **Config Name:** `summarization_samples`
- **Record Count:** 10,000
- **Structure:** **Single response** per record with binary label.
- **Context Field:** `document` (`str`) is present.

#### Fields & Types
| Field Name | Observed Python Type | Description |
|---|---|---|
| `document` | `str` | Source article context |
| `summary` | `str` | Candidate summary to evaluate |
| `hallucination` | `str` | Binary flag (`"yes"` or `"no"`) |

#### Label Distribution
- `"yes"` (Hallucinated): 5,010 (50.1%)
- `"no"` (Factual): 4,990 (49.9%)

#### Example Record
```json
{
  "document": "Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane...",
  "summary": "A video showing the final moments of Germanwings Flight 9525 has been recovered by investigators from the wreckage site. Marseille prosecutor Brice Robin urged anyone who might have more footage to turn it over immediately. Andreas Lubitz, the co-pilot accused of deliberately crashing the plane, had a history of severe depression and suicidal tendencies.",
  "hallucination": "yes"
}
```

---

### 2.7 Subset: `general`

- **Config Name:** `general`
- **Record Count:** 4,507
- **Structure:** **Single response** evaluations on open-domain/closed-book ChatGPT user queries.
- **Context Field:** **None**. There is no grounding knowledge or document context field.

#### Fields & Types
| Field Name | Observed Python Type | Description |
|---|---|---|
| `ID` | `str` | Unique record ID (numeric string, e.g. `"1"`, `"2"`) |
| `user_query` | `str` | Prompt or query asked by user |
| `chatgpt_response` | `str` | Response generated by ChatGPT |
| `hallucination` | `str` | Binary flag (`"yes"` or `"no"`) |
| `hallucination_spans` | `list[str]` | List of hallucinated substring excerpts |

#### Label Distribution
- `"no"` (Factual): 3,692 (81.92%)
- `"yes"` (Hallucinated): 815 (18.08%)

#### Span Annotations in `general`
- Unlike RAGTruth, which provides character offsets (`start`, `end`) and taxonomy types (`label_type`), HaluEval `general` provides `hallucination_spans` as raw string excerpts in a `list[str]`.
- Exactly 802 out of the 4,507 records (98.4% of the 815 hallucinated records) contain non-empty spans in `hallucination_spans`. In 13 cases labeled `"yes"`, `hallucination_spans` was empty `[]`.

#### Example Record (`ID = "2"`)
```json
{
  "ID": "2",
  "user_query": "Provide a few examples of homophones.",
  "chatgpt_response": "1. allowed (permitted) and aloud (out loud)\n2. weather (condition of the atmosphere) and whether (expressing a doubt or choice)\n3. gate (entrance) and gait (way of walking)\n4. their (belonging to them) and there (in that place)\n5. pair (two of a kind) and pear (a fruit)\n6. hear (to perceive sound) and here (in this place)\n7. to (preposition indicating direction or action) and too (also, very)\n8. knight (medieval soldier) and night (period of darkness)\n9. sale (discounted price) and sail (propell...",
  "hallucination": "yes",
  "hallucination_spans": [
    "6. hear (to perceive sound) and here (in this place)\n10. here (in this place) and hear (perceive sound)"
  ]
}
```

---

## 3. Comparative Summary & Architectural Implications for `raguard`

| Feature | RAGTruth | HaluEval (`qa`, `dialogue`, `summarization`) | HaluEval (`*_samples`) | HaluEval (`general`) |
|---|---|---|---|---|
| **Primary Domain** | RAG (QA, Summary, Data2txt) | Specific Tasks (QA, Dialogue, Summary) | Same as base subsets | General ChatGPT Queries |
| **Grounding Context** | `source_info` in `source_info.jsonl` | `knowledge` or `document` | `knowledge` or `document` | **None** |
| **Response Format** | Single response per row (join via `source_id`) | **Paired** (`right_*` & `hallucinated_*`) | Single response (`answer` / `response` / `summary`) | Single response (`chatgpt_response`) |
| **Label Granularity** | Character offsets (`start`, `end`) + taxonomy | Binary (implicit via pair keys) | Binary (`hallucination`: yes/no) | Binary + raw string spans (`list[str]`) |
| **Predefined Splits** | `train` (15,090), `test` (2,700) | `data` only (10,000) | `data` only (10,000) | `data` only (4,507) |
| **Model Information** | Explicit (`gpt-4`, `llama-2`, etc.) | Implicit (LLM synthetic) | Implicit (LLM synthetic) | ChatGPT |

### Key Structural Insights
1. **Paired vs. Single Response Formats:**
   - In HaluEval, `qa`, `dialogue`, and `summarization` provide paired responses (`right_*` vs. `hallucinated_*`).
   - The `*_samples` subsets are flattened into single-response classification instances with 50/50 balanced binary labels (`"yes"` / `"no"`).
   - In contrast, RAGTruth responses are individual model outputs with natural unbalanced hallucination rates (43.1% hallucinated, 56.9% factual).
2. **Span Grounding & Character Offsets:**
   - Only RAGTruth provides exact character offsets (`start`, `end`) with fine-grained error categories (`Evident Conflict`, `Evident Baseless Info`, `Subtle Conflict`, `Subtle Baseless Info`).
   - HaluEval `general` provides hallucinated spans as unstructured strings (`list[str]`), requiring string matching against `chatgpt_response` if character offsets are needed.
3. **Retrieval Context:**
   - RAGTruth context is in a separate relational file (`source_info.jsonl`), joined by `source_id`.
   - HaluEval task subsets keep context inline: named `knowledge` for `qa` and `dialogue`, and `document` for `summarization`.
   - HaluEval `general` has no context at all, making it unsuitable for retrieval-grounded hallucination checks without external knowledge retrieval.
