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
| `source_info` | `str` or `dict` | Retrieved reference context: plain `str` for `Summary`; `dict` with `['question', 'passages']` for `QA`; `dict` of business metadata and reviews for `Data2txt` |
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
The retrieval context structure in `source_info` is polymorphic and depends strictly on `task_type`:
- **`Summary` (943 records):** `source_info` is a plain text `str` containing the full article text.
- **`QA` (989 records):** `source_info` is a `dict` with keys `['question', 'passages']`. Both values are `str`. The `passages` field is a single concatenated `str` of retrieved passages separated by double newlines (e.g. `"passage 1:... \n\npassage 2:... \n\n"`), NOT a list.
- **`Data2txt` (1,033 records):** `source_info` is a `dict` containing structured Yelp attributes and customer reviews with 9 keys: `['name', 'address', 'city', 'state', 'categories', 'hours', 'attributes', 'business_stars', 'review_info']`.

#### Example Summary Record (`source_id = "15592"`)
```json
{
  "source_id": "15592",
  "task_type": "Summary",
  "source": "CNN/DM",
  "source_info": "Seventy years ago, Anne Frank died of typhus in a Nazi concentration camp at the age of 15. Just two weeks after her supposed death on March 31, 1945, the Bergen-Belsen concentration camp where she had been imprisoned was liberated -- timing that showed how close the Jewish diarist had been to surviving the Holocaust. But new research released by the Anne Frank House shows that Anne and her older sister, Margot Frank, died at least a month earlier than previously thought. Researchers re-examined archives of the Red Cross, the International Tracing Service and the Bergen-Belsen Memorial, along with testimonies of survivors, the Anne Frank House said Tuesday. They concluded that Anne and Margot probably died in February 1945...",
  "prompt": "Summarize the following news within 116 words:\nSeventy years ago, Anne Frank died of typhus in a Nazi concentration camp at the age of 15. Just two weeks after her supposed death on March 31, 1945, the Bergen-Belsen concentration camp where she had been imprisoned was liberated -- timing that showed how close the Jewish diarist had been to surviving the Holocaust..."
}
```

#### Example QA Record (`source_id = "14292"`)
```json
{
  "source_id": "14292",
  "task_type": "QA",
  "source": "MARCO",
  "source_info": {
    "question": "butcher shop phone number",
    "passages": "passage 1:Butcher Shop - Hayward 826 B Street, Hayward CA 94541 Phone Number: (510) 889-8690\n\npassage 2:Butcher Shop - Lakefield 212 Main Street, Lakefield MN 56150 Phone Number: (507) 662-6281\n\npassage 3:Some More Phone Numbers Related To The Butcher Shop The Local Butcher Shop phone number : (510) 845-6328 Boones Butcher Shop phone number : (502) 348-3668 Ye Ole Butcher Shop phone number : (972) 423-1848 J Ms Butcher Shop phone number : (865) 483-9228\n\n"
  },
  "prompt": "Answer the question based on given passages:\nQuestion: butcher shop phone number\nPassages: passage 1:Butcher Shop - Hayward 826 B Street, Hayward CA 94541 Phone Number: (510) 889-8690\n\npassage 2:Butcher Shop - Lakefield 212 Main Street, Lakefield MN 56150 Phone Number: (507) 662-6281\n\npassage 3:Some More Phone Numbers Related To The Butcher Shop The Local Butcher Shop phone number : (510) 845-6328 Boones Butcher Shop phone number : (502) 348-3668 Ye Ole Butcher Shop phone number : (972) 423-1848 J Ms Butcher Shop phone number : (865) 483-9228\n\n"
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


---

## 4. Label Semantics per Dataset

### 4.1 RAGTruth Label Semantics
- **Reference Publication:** Cheng Niu, Yuanhao Wu, Dian Jiao, et al., *"RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models"*, Proceedings of ACL 2024.
- **Source Repository:** `https://github.com/ParticleMedia/RAGTruth`

RAGTruth categorizes hallucination spans along two orthogonal axes:
1. **Error Nature (Conflict vs. Baseless Information):**
   - **Conflict:** The generated text contradicts facts or relations stated in the reference context.
     > *Paper definition:* "A conflict hallucination occurs when the generated text contains factual statements that contradict the reference context."
   - **Baseless Information:** The generated text introduces assertions or entities that are absent from or cannot be deduced from the reference context, even if they are plausible.
     > *Paper definition:* "A baseless information hallucination occurs when the generated text introduces assertions or entities that are not mentioned in or cannot be substantiated by the reference context."
2. **Error Salience (Evident vs. Subtle):**
   - **Evident:** Glaring factual errors or clear extrapolations that are readily identifiable without intricate inference.
     > *Paper definition:* "Evident hallucinations are obvious factual errors or fabricated details that can be clearly identified with minimal effort."
   - **Subtle:** Fine-grained, nuanced discrepancies involving minor temporal shifts, subtle entity confusions, or slight misinterpretations of the source passages.
     > *Paper definition:* "Subtle hallucinations are nuanced factual deviations, subtle entity confusions, or slight misinterpretations of the provided passages."

#### Resulting 4 Label Types:
1. `Evident Conflict`: Direct, unambiguous factual contradiction against the retrieved context passages (5,324 spans, 37.3% of all spans).
2. `Subtle Conflict`: Nuanced, fine-grained contradiction against the context (201 spans, 1.4% of all spans).
3. `Evident Baseless Info`: Blatant assertion of facts, entities, or claims completely absent from the context (6,237 spans, 43.6% of all spans).
4. `Subtle Baseless Info`: Minor, nuanced details or unsupported elaborations not present in the context (2,527 spans, 17.7% of all spans).

#### Span-Level Auxiliary Flags:
- `implicit_true` (`bool`): Set to `True` when the hallucinated span, though ungrounded by the retrieved context, represents information that is true according to real-world knowledge (parametric memory intrusion).
  > *Paper definition:* "implicit_true indicates that the hallucinated statement, while not grounded in the source passages, is actually true in the real world." (Observed in 1,928 spans).
- `due_to_null` (`bool`): Set to `True` when a hallucinated detail in structured-to-text tasks is triggered because the source attribute had a missing or `null` value in the input dictionary.
  > *Paper definition:* "due_to_null indicates that the hallucinated statement was triggered by a missing or null attribute in the structured table or context." (Observed in 1,642 spans; strictly confined to the Data2txt Yelp subset).

---

### 4.2 HaluEval Label Semantics
- **Reference Publication:** Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, Ji-Rong Wen, *"HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models"*, Proceedings of EMNLP 2023, pages 6449–6464.
- **Source Repository:** `https://github.com/RUCAIBox/HaluEval`

HaluEval defines hallucination as:
> *Paper definition:* "Hallucination in our benchmark is defined as generated information that conflicts with the user query, provided context, or factual reality."

#### 1. Task-Specific Subsets (`qa`, `dialogue`, `summarization`):
These subsets present paired responses for each input:
- `right_*` (`right_answer`, `right_response`, `right_summary`):
  > *Paper definition:* "The ground-truth, factual response retrieved or validated from the benchmark dataset (HotpotQA, OpenDialKG, CNN/DailyMail)."
- `hallucinated_*` (`hallucinated_answer`, `hallucinated_response`, `hallucinated_summary`):
  > *Paper definition:* "A synthetically generated response produced by prompting ChatGPT with one-shot task instructions designed to inject realistic hallucinations (e.g., entity swapping, factual contradiction, and fabricated details) conditioned on the context."

#### 2. Sampled Subsets (`qa_samples`, `dialogue_samples`, `summarization_samples`):
These flatten the paired instances into single-response classification instances with a binary label:
- `hallucination`:
  - `"no"`: Corresponds directly to the factual `right_*` response (exactly 4,990 records, 49.9%).
  - `"yes"`: Corresponds directly to the synthetic `hallucinated_*` response (exactly 5,010 records, 50.1%).

#### 3. General Subset (`general`):
Evaluates open-domain, closed-book ChatGPT generations on 4,507 diverse user prompts:
- `hallucination`: Binary flag (`"no"` in 3,692 records / 81.92%; `"yes"` in 815 records / 18.08%).
- `hallucination_spans`: Raw text substrings representing the inaccurate assertions extracted by human annotators.
  > *Annotation rule:* Unverified / exact annotator prompt guidelines not detailed in the community repo; in 13 cases labeled `"yes"`, `hallucination_spans` is empty `[]`.

---

## 5. Gaps for RAGuard

The table below audits the availability of pipeline inputs and evaluation labels across all datasets and subsets. Each element is categorized strictly as:
- **PRESENT**: Directly available as an explicit field.
- **DERIVABLE**: Can be unambiguously computed or extracted from available fields with standard deterministic logic.
- **MISSING**: Completely absent from the dataset.

| Dataset / Subset | User Query | Retrieval Context | Model Response | Response-Level Label | Span-Level Labels |
|---|---|---|---|---|---|
| **RAGTruth (QA)** | **PRESENT** (`source_info['question']`) | **PRESENT** (`source_info['passages']`) | **PRESENT** (`response`) | **DERIVABLE** (`len(labels) > 0`) | **PRESENT** (`start`, `end`, `label_type`) |
| **RAGTruth (Summary)** | **DERIVABLE** (`prompt` task instruction) | **PRESENT** (`source_info` article text) | **PRESENT** (`response`) | **DERIVABLE** (`len(labels) > 0`) | **PRESENT** (`start`, `end`, `label_type`) |
| **RAGTruth (Data2txt)** | **MISSING** (prompt template only) | **PRESENT** (`source_info` business dict) | **PRESENT** (`response`) | **DERIVABLE** (`len(labels) > 0`) | **PRESENT** (`start`, `end`, `label_type`) |
| **HaluEval (`qa`)** | **PRESENT** (`question`) | **PRESENT** (`knowledge`) | **DERIVABLE** (`right_answer` & `hallucinated_answer`) | **DERIVABLE** (implicit binary from pair key) | **MISSING** |
| **HaluEval (`qa_samples`)** | **PRESENT** (`question`) | **PRESENT** (`knowledge`) | **PRESENT** (`answer`) | **PRESENT** (`hallucination`: `"yes"`/`"no"`) | **MISSING** |
| **HaluEval (`dialogue`)** | **DERIVABLE** (`dialogue_history` as query) | **PRESENT** (`knowledge`) | **DERIVABLE** (`right_response` & `hallucinated_response`) | **DERIVABLE** (implicit binary from pair key) | **MISSING** |
| **HaluEval (`dialogue_samples`)** | **DERIVABLE** (`dialogue_history` as query) | **PRESENT** (`knowledge`) | **PRESENT** (`response`) | **PRESENT** (`hallucination`: `"yes"`/`"no"`) | **MISSING** |
| **HaluEval (`summarization`)** | **MISSING** | **PRESENT** (`document`) | **DERIVABLE** (`right_summary` & `hallucinated_summary`) | **DERIVABLE** (implicit binary from pair key) | **MISSING** |
| **HaluEval (`summarization_samples`)** | **MISSING** | **PRESENT** (`document`) | **PRESENT** (`summary`) | **PRESENT** (`hallucination`: `"yes"`/`"no"`) | **MISSING** |
| **HaluEval (`general`)** | **PRESENT** (`user_query`) | **MISSING** | **PRESENT** (`chatgpt_response`) | **PRESENT** (`hallucination`: `"yes"`/`"no"`) | **DERIVABLE** (raw string list in `hallucination_spans`) |

### Key Gap Takeaways for RAGuard Architecture:
1. **Dynamic Retrieval Gap:** None of the public datasets provide a dynamic search index or query retrieval harness; contexts are static and fixed. To test RAGuard's action router (specifically the "Re-retrieve with reformulated query" path on Retrieval Misses), an external corpus index (such as MS MARCO or Wikipedia) must be paired with the evaluation benchmark.
2. **Span Granularity Gap in HaluEval:** HaluEval task subsets lack span-level annotations entirely. Therefore, HaluEval can only evaluate end-to-end response-level classification and score fusion, whereas fine-grained claim-level decomposition and span verification must be benchmarked on RAGTruth.
3. **Contextless General Subset:** HaluEval `general` has no retrieval context document, making NLI cross-encoder entailment and dense attribution inapplicable without an external retrieval engine. It can only be used to evaluate Signal C (Self-Consistency).

---

## 6. License and Usage Terms

### 6.1 RAGTruth
- **Source Repository:** `https://github.com/ParticleMedia/RAGTruth`
- **Repository License:** **Not found** (no `LICENSE` file is included in the official GitHub repository root as of inspection).
- **Underlying Provenance & Terms:**
  - *MS MARCO (QA):* Microsoft Research License (strictly non-commercial research use).
  - *CNN / DailyMail (Summary):* Apache 2.0 / non-commercial research.
  - *Yelp Open Dataset (Data2txt):* Governed by the Yelp Dataset User Agreement, which strictly restricts use to non-commercial academic research and prohibits redistributing raw Yelp data without attribution.
- **RAGuard Compliance:** Suitable for academic research, benchmarking, and peer-reviewed conference publications. Raw Yelp data must remain git-ignored (as enforced in `.gitignore`).

### 6.2 HaluEval
- **Original Publication:** Li et al., *"HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models"*, EMNLP 2023.
- **Original Official Repository:** `https://github.com/RUCAIBox/HaluEval` (maintained by RUCAIBox, Renmin University of China).
- **Original Repository License:** **MIT License** (per official `RUCAIBox/HaluEval` repository).
- **Hugging Face Copy Used:** `pminervini/HaluEval` (hosted by Pasquale Minervini).
  - *Is the HF copy official?* **No**. It is an unofficial community conversion to Hugging Face dataset format (Parquet/Arrow). The Hugging Face `dataset_info.json` metadata leaves the `license` field blank (`""`). However, the upstream content originates from the MIT-licensed RUCAIBox repository.
- **Underlying Provenance & Terms:**
  - *HotpotQA (`qa`):* Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
  - *OpenDialKG (`dialogue`):* Creative Commons Attribution 4.0 International (CC BY 4.0).
  - *CNN / DailyMail (`summarization`):* Apache 2.0 / non-commercial research.
- **RAGuard Compliance:** Fully compatible with open-source academic benchmarking and evaluation under MIT and CC BY-SA terms.

---

## 7. Open Questions

1. **Validation Split Generation for RAGTruth:** RAGTruth provides only `train` (15,090) and `test` (2,700). Because RAGuard requires a validation split to calibrate fusion weights $(w_1, w_2, w_3)$ and detector thresholds $(\tau_E, \tau_M, \tau_C)$ without contaminating test results, how should `val` be carved out from `train`? Partitioning must occur strictly at the `source_id` level (e.g. 2,015 train sources / 500 val sources) to avoid multi-model data leakage.
2. **Evaluation Split Standardization for HaluEval:** HaluEval provides only a single `data` split of 10,000 records per subset. A deterministic split protocol (e.g., 80% train, 10% val, 10% test) partitioned by unique context/query keys must be implemented in `eval/splits.py`.
3. **Evaluating Corrective Action Routing on Static Datasets:** RAGuard's Action Router dispatches targeted corrective actions (re-retrieval, stricter prompt regeneration, resampled temperature 0, abstention). Since RAGTruth and HaluEval are static offline corpora, how should re-retrieval be evaluated? Options include:
   - Indexing the underlying corpora (MS MARCO passages, HotpotQA documents) into BM25/FAISS to evaluate dynamic multi-hop retrieval.
   - Using a mock retrieval environment that returns gold vs. distractor passages.
4. **Claim Boundary vs. Character Span Alignment:** RAGTruth annotations provide exact character spans ($[start, end]$), whereas RAGuard decomposes responses into atomic sentence claims via spaCy. How should claim verification be mapped to character spans? We must define a threshold (e.g., character IoU $\ge 0.5$ or token containment) to designate whether a claim is considered an annotated hallucination.
5. **Linearization of Structured Data2txt Contexts:** RAGTruth Data2txt context is a serialized JSON dictionary of business attributes and reviews. Pretrained NLI cross-encoders (e.g. `roberta-large-snli_mnli_fever_anli_r1_r2_r3` or `deberta-v3-large`) are pretrained on natural prose premises, not raw JSON. Will unlinearized JSON cause domain degradation, and should a lightweight serializer/verbalizer be introduced?
6. **Redundancy Between Paired and Sampled HaluEval:** Since `qa_samples`, `dialogue_samples`, and `summarization_samples` are exact 50/50 balanced subsets of the paired datasets, benchmarking on both is redundant. The pipeline should use `*_samples` for binary classification metrics (F1, PR-AUC, ROC-AUC) and paired datasets for candidate ranking or preference tests.
