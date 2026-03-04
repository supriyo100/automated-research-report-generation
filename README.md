# 🏗️ Enterprise Agentic Report Generation System

> A production-grade, multi-agent pipeline for automated research report generation with human-in-the-loop oversight, decentralized QA, max-retry circuit breakers, and full observability — built on LangGraph, MCP, A2A, AWS EKS, and S3.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Technology Stack](#3-technology-stack)
4. [Entry Point & Security Layer](#4-entry-point--security-layer)
5. [Phase 1 — Planning & Scope Alignment](#5-phase-1--planning--scope-alignment)
6. [Phase 2 — Question Generation](#6-phase-2--question-generation)
7. [Phase 3 — Hybrid Search & Multi-Source Retrieval](#7-phase-3--hybrid-search--multi-source-retrieval)
8. [Phase 4 — Reranking & Relevance Filtering](#8-phase-4--reranking--relevance-filtering)
9. [Phase 5 — Evaluator Agent: Fact Extraction & Conflict Detection](#9-phase-5--evaluator-agent-fact-extraction--conflict-detection)
10. [Phase 6 — Validated Fact Base Construction](#10-phase-6--validated-fact-base-construction)
11. [Phase 7 — Synthesizer Agent](#11-phase-7--synthesizer-agent)
12. [Phase 8 — Writer Agent](#12-phase-8--writer-agent)
13. [Phase 9 — Decentralized QA & Finalization](#13-phase-9--decentralized-qa--finalization)
14. [Observability, Telemetry & Cost Monitoring](#14-observability-telemetry--cost-monitoring)
15. [MCP Server Integration](#15-mcp-server-integration)
16. [A2A (Agent-to-Agent) Communication Protocol](#16-a2a-agent-to-agent-communication-protocol)
17. [Retry Circuits & Tie-Breaker Logic](#17-retry-circuits--tie-breaker-logic)
18. [Infrastructure: AWS EKS Deployment](#18-infrastructure-aws-eks-deployment)
19. [Storage Layer: S3, s3fs & boto3](#19-storage-layer-s3-s3fs--boto3)
20. [LangGraph State & Graph Definition](#20-langgraph-state--graph-definition)
21. [Configuration & Environment Variables](#21-configuration--environment-variables)
22. [Installation & Local Development](#22-installation--local-development)
23. [Running the Pipeline](#23-running-the-pipeline)
24. [Testing Strategy](#24-testing-strategy)
25. [Security & Compliance](#25-security--compliance)
26. [Contributing](#26-contributing)

---
## Environment creation
uv syn
uv add venv
uv add -r requirements.txt 
uv pip install -r requirements.txt
uv pip show <langgraph>


---

## 1. Architecture Overview

This system implements a **9-phase agentic pipeline** that accepts a user-defined topic and constraints, autonomously researches it across heterogeneous sources, validates every fact, synthesizes a narrative plan, drafts a full report, and runs it through decentralized quality assurance before delivering a final, human-approved artifact.

The system is designed around five core engineering principles:

**No single point of failure.** Every phase has a retry circuit with a maximum attempt count and an explicit human escalation path when circuits open. No god agent owns the final verdict — three specialized critic agents vote and a tie-breaker resolves disputes.

**Full auditability.** Every tool call, agent decision, human input, retry attempt, and score is written to an immutable audit trail on S3. Nothing is ephemeral.

**Separation of concerns.** Each phase owns exactly one responsibility. The Planner does not search. The Writer does not evaluate. The Evaluator does not write. Agent boundaries are enforced at the LangGraph node level.

**Human-in-the-loop at critical gates.** Humans approve the scope at Phase 1, validate flagged sources at Phase 5, and give final approval at Phase 9. All other decisions are automated.

**Infrastructure-native.** The system runs on AWS EKS with autoscaling node groups per phase, S3 for all persistent state, SQS for async inter-phase messaging, and CloudWatch + OpenTelemetry for full observability.

---

## 2. Repository Structure

```
enterprise-agentic-report/
│
├── agents/                         # One module per agent
│   ├── planner/
│   │   ├── agent.py                # PlannerAgent class
│   │   ├── tools.py                # All tool call implementations
│   │   └── prompts.py              # System + user prompt templates
│   ├── question_generator/
│   ├── evaluator/
│   ├── synthesizer/
│   ├── writer/
│   ├── fact_checker/
│   ├── alignment/
│   └── format_agent/
│
├── graph/
│   ├── state.py                    # Shared LangGraph TypedDict state schema
│   ├── graph.py                    # Full LangGraph StateGraph definition
│   ├── nodes.py                    # All node functions (one per box in diagram)
│   ├── edges.py                    # Conditional edge logic
│   └── retry.py                    # RetryCircuit and CircuitBreaker classes
│
├── retrieval/
│   ├── web/
│   │   ├── google_search.py
│   │   ├── news_api.py
│   │   ├── arxiv_search.py
│   │   └── gov_data.py
│   ├── databases/
│   │   ├── vector_db.py            # FAISS / Pinecone / Weaviate abstraction
│   │   ├── bm25_search.py          # Elasticsearch / OpenSearch
│   │   ├── knowledge_base.py       # Internal KB over S3
│   │   └── structured_query.py     # SQL + GraphQL + REST
│   └── specialist/
│       ├── wolfram.py
│       ├── financial_api.py
│       ├── geospatial.py
│       └── domain_kb.py
│
├── reranking/
│   ├── cross_encoder.py            # sentence-transformers CrossEncoder
│   ├── contextual_compressor.py    # LangChain ContextualCompressionRetriever
│   ├── mmr_sampler.py              # Maximal Marginal Relevance
│   └── composite_scorer.py
│
├── validation/
│   ├── ner.py                      # spaCy / Hugging Face NER
│   ├── relation_extractor.py
│   ├── claim_identifier.py
│   ├── numeric_validator.py
│   ├── credibility/
│   │   ├── authority_scorer.py
│   │   ├── freshness_checker.py
│   │   ├── corroboration.py
│   │   └── misinfo_flagging.py
│   └── conflict/
│       ├── contradiction_detector.py
│       ├── confidence_scorer.py
│       ├── majority_vote.py
│       └── conflict_register.py
│
├── fact_base/
│   ├── builder.py                  # FactBaseBuilder orchestrator
│   ├── deduplicator.py
│   ├── citation_linker.py
│   ├── gap_analyzer.py
│   └── serializer.py               # JSON serialization → S3
│
├── qa/
│   ├── ragas_scorer.py             # RAGAS faithfulness + relevance metrics
│   ├── hallucination_detector.py
│   ├── tie_breaker.py              # Voting + weighted tie-break logic
│   └── aggregator.py
│
├── guardrails/
│   ├── input_guard.py              # PII redaction, injection detection
│   └── output_guard.py             # Toxicity + data leak check
│
├── cache/
│   └── semantic_cache.py           # Redis / pgvector semantic cache
│
├── mcp/
│   ├── server_registry.py          # MCP server discovery & tool registration
│   ├── tool_dispatcher.py          # Routes tool calls to MCP servers
│   └── schemas/                    # JSON schemas per MCP tool
│
├── a2a/
│   ├── protocol.py                 # A2A message envelope + routing
│   ├── registry.py                 # Agent capability registry
│   └── bus.py                      # SQS-backed async message bus
│
├── infra/
│   ├── eks/
│   │   ├── cluster.yaml            # EKS cluster definition
│   │   ├── node_groups.yaml        # Per-phase autoscaling node groups
│   │   └── helm/                   # Helm charts for each service
│   ├── s3/
│   │   └── bucket_policy.json
│   └── iam/
│       └── roles.yaml
│
├── observability/
│   ├── tracer.py                   # OpenTelemetry tracer setup
│   ├── metrics.py                  # Prometheus metrics definitions
│   ├── cost_tracker.py             # Token + API cost accumulator
│   └── audit_logger.py             # Immutable S3 audit log writer
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── config/
│   ├── settings.py                 # Pydantic BaseSettings
│   └── logging.yaml
│
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 3. Technology Stack

### Orchestration

| Library | Version | Role |
|---|---|---|
| `langgraph` | ≥0.2 | State machine graph, node execution, conditional edges |
| `langchain-core` | ≥0.3 | Runnable interface, tool binding, prompt templates |
| `langchain-anthropic` | latest | Claude LLM integration |
| `langchain-openai` | latest | GPT fallback LLM |

### Infrastructure

| Tool | Role |
|---|---|
| AWS EKS | Kubernetes cluster hosting all agent pods |
| `boto3` | AWS SDK — S3 reads/writes, SQS messaging, SSM parameter store |
| `s3fs` | POSIX-like filesystem interface over S3 buckets |
| `shutil` | Local temp file manipulation, copy/move before S3 upload |
| `pathlib` | Cross-platform path construction for all file operations |

### Retrieval & Search

| Library | Role |
|---|---|
| `faiss-cpu` / `faiss-gpu` | Vector similarity search |
| `pinecone-client` | Managed vector DB |
| `weaviate-client` | Hybrid vector + keyword DB |
| `elasticsearch` | BM25 keyword search |
| `sentence-transformers` | Embeddings + CrossEncoder reranking |
| `langchain-community` | Web search, Wikipedia, ArXiv loaders |

### Validation & NLP

| Library | Role |
|---|---|
| `spacy` | NER, dependency parsing |
| `transformers` | Hugging Face models for relation extraction, claim detection |
| `ragas` | Faithfulness, answer relevance, context precision scoring |

### Observability

| Tool | Role |
|---|---|
| `opentelemetry-sdk` | Distributed tracing across all agents |
| `opentelemetry-exporter-otlp` | Exports traces to Jaeger / Grafana Tempo |
| `prometheus-client` | Metrics exposition |
| `structlog` | Structured JSON logging |

### Infrastructure-as-Code

| Tool | Role |
|---|---|
| Helm 3 | Kubernetes package manager |
| AWS CDK (Python) | EKS cluster, S3 buckets, IAM roles, SQS queues |
| Docker | Container images per agent service |

---

## 4. Entry Point & Security Layer

### Input Guardrails (`guardrails/input_guard.py`)

Every request enters through a guardrail layer before any agent sees it.

**PII Redaction** uses the `presidio-analyzer` and `presidio-anonymizer` libraries to detect and redact names, emails, phone numbers, SSNs, credit card numbers, and IP addresses from user-supplied text. Redacted entities are stored in a reversible mapping keyed by session ID in AWS Secrets Manager so the final report can be de-anonymized on delivery if the user has the appropriate permissions.

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact_pii(text: str) -> tuple[str, dict]:
    results = analyzer.analyze(text=text, language="en")
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, {r.entity_type: r for r in results}
```

**Prompt Injection Detection** runs the input through a fine-tuned classifier (hosted as an MCP tool on the security MCP server) that scores the probability of injection patterns. Inputs scoring above `0.85` are rejected and logged with a `SECURITY_BLOCK` event.

### Semantic Cache (`cache/semantic_cache.py`)

Before invoking any agent, the system embeds the user's topic + constraints and queries a pgvector table (hosted on AWS RDS Aurora Postgres) for semantically similar past requests. Similarity threshold is configurable (default: `0.92` cosine similarity). Cache hits return the stored report instantly and log a `CACHE_HIT` event to the audit trail. Cache misses proceed to the Planner.

```python
import boto3
import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def check_semantic_cache(query: str, threshold: float = 0.92) -> dict | None:
    embedding = model.encode(query).tolist()
    # pgvector cosine similarity query
    conn = psycopg2.connect(get_db_dsn())
    cur = conn.cursor()
    cur.execute(
        "SELECT report_id, report_s3_key, 1 - (embedding <=> %s::vector) AS similarity "
        "FROM report_cache ORDER BY similarity DESC LIMIT 1",
        (embedding,)
    )
    row = cur.fetchone()
    if row and row[2] >= threshold:
        return load_report_from_s3(row[1])
    return None
```

---

## 5. Phase 1 — Planning & Scope Alignment

### PlannerAgent (`agents/planner/agent.py`)

The Planner is a LangGraph node that receives the sanitized user brief and produces a fully structured outline with per-section research goals, a risk register, word budgets, and milestone checkpoints. It is bound to four tools and calls them in parallel using LangChain's `RunnableParallel`.

#### Tool 1 — Load User Brief (`PT1`)

```python
from pathlib import Path
import shutil
import s3fs

fs = s3fs.S3FileSystem()

def load_user_brief(session_id: str) -> dict:
    s3_path = f"s3://my-bucket/briefs/{session_id}/brief.json"
    local_path = Path(f"/tmp/{session_id}/brief.json")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # Copy from S3 to local temp using s3fs + shutil pattern
    with fs.open(s3_path, "rb") as remote:
        with open(local_path, "wb") as local:
            shutil.copyfileobj(remote, local)
    return json.loads(local_path.read_text())
```

`pathlib.Path` is used throughout for all local file path operations. `shutil.copyfileobj` streams large files without loading them fully into memory. `s3fs` provides the S3 file handle with the same interface as a local file.

#### Tool 2 — Retrieve Past Similar Reports (`PT2`)

Queries the vector store (Pinecone) for the top-5 most semantically similar past report outlines. These are surfaced to the Planner as context so it can reuse proven section structures and avoid known dead ends.

```python
from pinecone import Pinecone

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index("report-outlines")

def retrieve_past_outlines(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [m.metadata for m in results.matches]
```

#### Tool 3 — Fetch Domain Taxonomy (`PT3`)

Calls the **Knowledge MCP server** to retrieve domain ontologies and taxonomies. The MCP tool `fetch_taxonomy` accepts a domain string and returns a structured JSON tree of concepts, relationships, and preferred terminology that constrains the Planner's vocabulary choices.

```python
# MCP tool call via tool_dispatcher
taxonomy = await mcp_dispatch(
    server="knowledge-mcp",
    tool="fetch_taxonomy",
    args={"domain": brief["domain"], "depth": 3}
)
```

#### Tool 4 — Constraint Checker (`PT4`)

Validates user-supplied constraints (max word count, target audience, output format, submission deadline) against system capabilities. Returns a `ConstraintReport` with warnings for any constraint that cannot be met deterministically.

### Plan Construction Subgraph

After tools complete, the Planner constructs four artifacts in sequence:

**Draft Outline (`PP1`)** — A hierarchical JSON structure: `{sections: [{title, word_budget, subsections: [...]}]}`. Word budgets are allocated proportionally based on the section's importance score from `PT4`.

**Research Goals (`PP2`)** — Each section receives 3–5 atomic research questions. These are passed verbatim to the Question Generator in Phase 2. Goals are tagged with `priority: high | medium | low` which drives query batching later.

**Risk Register (`PP3`)** — The Planner calls a risk-scoring MCP tool that evaluates each section for data availability risk (is this topic likely to have scarce sources?), sensitivity risk (is the topic politically or legally sensitive?), and freshness risk (does this topic require up-to-date data?). Flagged risks gate human review in Phase 5.

**Milestone Checkpoints (`PP4`)** — Defines the three human review gates: after scope approval (this phase), after source flagging (Phase 5), and before final delivery (Phase 9). Checkpoints are stored in the shared LangGraph state and referenced by conditional edge logic.

### Retry Circuit

```python
# graph/retry.py
from dataclasses import dataclass, field

@dataclass
class RetryCircuit:
    max_attempts: int = 3
    attempts: int = field(default=0, init=False)

    def can_retry(self) -> bool:
        return self.attempts < self.max_attempts

    def increment(self):
        self.attempts += 1

    @property
    def is_open(self) -> bool:
        return self.attempts >= self.max_attempts
```

When the human rejects the scope, the Planner re-runs with the feedback appended to its prompt. After 3 rejections the circuit opens and the pipeline escalates by emitting the partial draft + risk flags to a human review queue (AWS SQS `human-escalation-queue`).

---

## 6. Phase 2 — Question Generation

### QuestionGeneratorAgent (`agents/question_generator/agent.py`)

Receives the Approved Outline (sections + research goals + risk register) and decomposes it into atomic, routable search queries.

#### Tool 1 — NLP Decomposition (`QT1`)

Uses `spaCy`'s `en_core_web_trf` model to extract entities, relations, and intents from each research goal. This produces a normalized representation that makes query expansion and routing more precise.

```python
import spacy

nlp = spacy.load("en_core_web_trf")

def decompose_goal(goal: str) -> dict:
    doc = nlp(goal)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    return {"entities": entities, "noun_chunks": noun_chunks, "root_verb": doc[0].lemma_}
```

#### Tool 2 — Query Router (`QT2`)

A lightweight classifier (fine-tuned DistilBERT, served via the **Routing MCP server**) that assigns each query to one or more source types: `web`, `news`, `academic`, `gov`, `vector_db`, `bm25`, `internal_kb`, `sql`, `specialist`. This prevents, for example, a mathematical formula query from being sent to a news API.

The MCP tool `classify_query_source` returns a ranked list of source types with confidence scores. Queries are dispatched to all sources scoring above `0.4`.

#### Tool 3 — Query Expander (`QT3`)

Calls the **NLP MCP server**'s `expand_query` tool, which enriches each query with synonyms (WordNet), related entities (Wikidata), and multilingual equivalents where the topic has significant non-English literature. The expanded variants are used as parallel search queries.

#### Tool 4 — Priority Ranker (`QT4`)

Scores each expanded query by its parent section's priority (from the Research Goals), estimated source availability (from the Risk Register), and structural position in the outline. Returns three sorted batches: A (high priority), B (medium), C (low).

### Query Plan Subgraph

Batches A, B, and C are dispatched simultaneously using `asyncio.gather`. This is the first place in the pipeline where significant parallelism occurs. All three batches fire to Phase 3 concurrently, and the system waits for all to complete before proceeding to reranking.

```python
import asyncio

async def dispatch_queries(batch_a, batch_b, batch_c):
    results = await asyncio.gather(
        search_all_sources(batch_a),
        search_all_sources(batch_b),
        search_all_sources(batch_c),
    )
    return [item for sublist in results for item in sublist]
```

---

## 7. Phase 3 — Hybrid Search & Multi-Source Retrieval

This phase contains three parallel tool banks, each comprising 4 specialized tools, for a total of 12 concurrent retrieval operations per query batch.

### Web Search Tools

#### `WS1` — Web Search API

Calls Google Custom Search, Bing Web Search, and Brave Search APIs in parallel. Results are normalized into a common `SearchResult` schema: `{url, title, snippet, source_name, retrieved_at}`.

```python
import asyncio
import httpx

async def google_search(query: str, api_key: str, cx: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"q": query, "key": api_key, "cx": cx, "num": 10}
        )
        return [normalize_google(item) for item in resp.json().get("items", [])]
```

#### `WS2` — News API

Queries NewsAPI.org and GDELT Project for recent articles. Includes a recency filter configurable per-topic (default: articles from the past 90 days). Press release aggregators (PR Newswire, BusinessWire via Meltwater API) are queried for corporate topics.

#### `WS3` — Academic Search

Uses the `arxiv` Python library for preprints, `pymed` for PubMed queries, and the Semantic Scholar API for citation-enriched academic results. Academic results receive a baseline authority boost in Phase 4 scoring.

```python
import arxiv

def search_arxiv(query: str, max_results: int = 20) -> list[dict]:
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results,
                          sort_by=arxiv.SortCriterion.Relevance)
    return [{"title": r.title, "url": r.entry_id,
             "abstract": r.summary, "published": r.published} 
            for r in client.results(search)]
```

#### `WS4` — Government & Regulatory Sources

Calls the Federal Register API, EUR-Lex SPARQL endpoint, Data.gov CKAN API, and the UK National Archives API. These sources are prioritized for policy, legal, and regulatory topics identified by the Query Router.

### Database & Knowledge Tools

#### `DB1` — Vector DB Lookup

Abstracts over FAISS (local, for development), Pinecone (managed, for production), and Weaviate (hybrid, for on-prem requirements). The abstraction layer in `retrieval/databases/vector_db.py` selects the backend based on `settings.VECTOR_DB_BACKEND`.

```python
from langchain_community.vectorstores import FAISS, Pinecone, Weaviate
from langchain_core.vectorstores import VectorStore

def get_vector_store(backend: str) -> VectorStore:
    match backend:
        case "faiss":
            return FAISS.load_local("./faiss_index", embeddings)
        case "pinecone":
            return Pinecone.from_existing_index(index_name, embeddings)
        case "weaviate":
            return Weaviate(client=weaviate_client, index_name="Documents")
```

#### `DB2` — BM25 Keyword Search

Uses the `elasticsearch` Python client against a managed OpenSearch Service cluster on AWS. Queries use a `multi_match` with `best_fields` strategy across `title`, `content`, and `metadata` fields. BM25 results are essential for exact-match queries (product names, legislation numbers, person names) where semantic search may miss.

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(hosts=[settings.OPENSEARCH_ENDPOINT])

def bm25_search(query: str, index: str = "documents", size: int = 20) -> list[dict]:
    body = {
        "query": {"multi_match": {"query": query,
                                   "fields": ["title^3", "content", "metadata"],
                                   "type": "best_fields"}},
        "size": size
    }
    resp = es.search(index=index, body=body)
    return [h["_source"] | {"score": h["_score"]} for h in resp["hits"]["hits"]]
```

#### `DB3` — Internal Knowledge Base

Loads company documents, past reports, and SOPs stored as chunked embeddings in S3 and indexed in the vector DB under a `internal_kb` namespace. Access is controlled by IAM role-based policies. The `s3fs` library provides transparent streaming of document chunks without downloading full files.

```python
import s3fs
import json

fs = s3fs.S3FileSystem()

def load_kb_chunks(prefix: str = "s3://my-bucket/knowledge-base/") -> list[dict]:
    chunks = []
    for path in fs.glob(f"{prefix}**/*.jsonl"):
        with fs.open(path, "r") as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks
```

#### `DB4` — Structured DB Query

Executes SQL queries against an AWS RDS PostgreSQL instance for structured data (statistics, tables, financial figures), GraphQL queries against an Apollo Federation gateway for entity graphs, and REST API calls against registered internal services. Query templates are managed by the **Data MCP server**.

### Specialist Tools

#### `SP1` — Wolfram Alpha

Uses the Wolfram Alpha Short Answers API for mathematical derivations, statistical computations, unit conversions, and formula lookups. Results are tagged `source_type: computation` which prevents them from entering the credibility scoring pipeline (computations are treated as ground truth, not opinions).

#### `SP2` — Financial Data API

Integrates with Bloomberg B-PIPE (enterprise) or Yahoo Finance (`yfinance` library) for market data, and the SEC EDGAR XBRL API for financial filings. Financial figures extracted here are passed to the Numeric Validator in Phase 5 for unit and magnitude checks.

```python
import yfinance as yf

def get_financial_data(ticker: str, period: str = "1y") -> dict:
    stock = yf.Ticker(ticker)
    return {
        "info": stock.info,
        "history": stock.history(period=period).to_dict(),
        "financials": stock.financials.to_dict()
    }
```

#### `SP3` — Geospatial API

Calls the US Census Bureau API, OpenStreetMap Nominatim/Overpass API, and the World Bank Indicators API for geographic and demographic data. Used for reports requiring regional analysis, population statistics, or location-specific context.

#### `SP4` — Domain Expert KB

The **Expert MCP server** exposes curated ontologies and taxonomy trees from domain-specific knowledge graphs (Medical Subject Headings for healthcare topics, ICD-10 for medical coding, NAICS codes for business topics). These constrain terminology and ensure the report uses authoritative domain language.

### Result Collection & Deduplication Subgraph

All 12 retrieval tools feed into a four-step merge pipeline:

**Raw Result Aggregator (`RM1`)** — Collects all results into a single in-memory list, tagged with their source tool. Implemented as an `asyncio` collector that waits for all 12 concurrent tasks.

**URL & Content Deduplicator (`RM2`)** — First removes exact URL duplicates, then runs MinHash LSH (via the `datasketch` library) to detect near-duplicate content with Jaccard similarity > 0.85. Only the highest-authority copy of near-duplicates is retained.

```python
from datasketch import MinHash, MinHashLSH

lsh = MinHashLSH(threshold=0.85, num_perm=128)

def deduplicate_results(results: list[dict]) -> list[dict]:
    seen = {}
    for i, result in enumerate(results):
        m = MinHash(num_perm=128)
        for word in result["content"].lower().split():
            m.update(word.encode("utf8"))
        dupes = lsh.query(m)
        if not dupes:
            lsh.insert(str(i), m)
            seen[i] = result
    return list(seen.values())
```

**Chunk Splitter (`RM3`)** — Splits long documents into 512-token passages using `langchain`'s `RecursiveCharacterTextSplitter`. Each chunk inherits the parent document's metadata plus a `chunk_index` for citation reconstruction.

**Metadata Tagger (`RM4`)** — Attaches a confidence tier (`high | medium | low | unverified`) based on source type (government > academic > established news > general web > unknown). This tier is used downstream in scoring and display.

---

## 8. Phase 4 — Reranking & Relevance Filtering

### Reranking Tools (parallel)

#### `RR1` — Cross-Encoder Reranker

Uses `sentence-transformers` `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to compute true query-passage relevance scores. Unlike bi-encoder embeddings, cross-encoders process the query and passage together, producing significantly more accurate relevance scores at the cost of higher latency.

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_passages(query: str, passages: list[str]) -> list[float]:
    pairs = [(query, p) for p in passages]
    return cross_encoder.predict(pairs).tolist()
```

#### `RR2` — Contextual Compressor

Uses `langchain`'s `LLMChainExtractor` to extract only the sentence spans within each passage that are directly relevant to the query. This is critical for long documents where only a small portion answers the question — it reduces noise sent to the Evaluator.

#### `RR3` — Diversity Sampler (MMR)

Maximal Marginal Relevance (`langchain`'s `max_marginal_relevance_search`) ensures that the top-K results are not all paraphrases of the same source. It trades off relevance for diversity: given a set of high-scoring passages, it iteratively selects the next passage that is most relevant to the query but least similar to already-selected passages.

#### `RR4` — Recency Scorer

For topics flagged as time-sensitive by the Risk Register, applies a recency boost: `recency_score = 1 / (1 + days_since_published / 30)`. This decays to 0.5 at 30 days and 0.33 at 60 days. The boost is added to the composite score as a weighted component.

### Composite Scoring & Threshold Gate

**Composite Score (`SC1`):**

```
composite = (0.4 × cross_encoder_score)
           + (0.2 × authority_tier_score)   # from RM4 metadata
           + (0.2 × recency_score)           # from RR4
           + (0.2 × diversity_contribution)  # from MMR selection
```

**Threshold Filter (`SC2`):** Passages below `0.35` composite score are dropped. The threshold is configurable per topic sensitivity in `config/settings.py`.

**Top-K Selector (`SC3`):** Retains the top 50 passages per query (configurable). Passed to the Evaluator Agent in Phase 5.

---

## 9. Phase 5 — Evaluator Agent: Fact Extraction & Conflict Detection

The Evaluator Agent is the most complex node in the pipeline. It orchestrates 12 sub-tools across three subgraphs: Fact Extraction, Credibility Assessment, and Conflict Detection.

### Fact Extraction Tools

#### `FE1` — Named Entity Recognizer

Runs `spaCy`'s `en_core_web_trf` pipeline and Hugging Face's `dslim/bert-base-NER` model in ensemble. Extracts people, organizations, dates, locations, products, and legislation references. Each entity is stored in the fact base with its source passage and character offsets.

#### `FE2` — Relation Extractor

Uses a fine-tuned REBEL model (`Babelscape/rebel-large`) to extract Subject–Predicate–Object triples from passages. These triples form the backbone of the conflict detection step — two passages asserting contradictory predicates for the same subject are flagged.

```python
from transformers import pipeline

rebel_pipeline = pipeline("text2text-generation",
                           model="Babelscape/rebel-large",
                           tokenizer="Babelscape/rebel-large")

def extract_relations(text: str) -> list[tuple]:
    output = rebel_pipeline(text, return_tensors=True, return_text=False)
    return parse_rebel_output(output)
```

#### `FE3` — Claim Identifier

A fine-tuned DeBERTa classifier that distinguishes verifiable factual claims from opinions, speculation, and background context. Only verifiable claims enter the conflict detection pipeline. Opinions are retained in the fact base but tagged `claim_type: opinion` and excluded from cross-source corroboration requirements.

#### `FE4` — Numeric Validator

Checks all extracted numeric values for: dimensional consistency (percentages sum to ≤100%, growth rates are plausible), unit agreement (million vs billion, km vs miles), and cross-passage consistency (the same statistic should not differ by more than a configurable tolerance across sources). Flagged inconsistencies are added to the Conflict Register.

### Credibility Assessment Tools

#### `CR1` — Source Authority Scorer

Queries the **Credibility MCP server**, which maintains a regularly-updated database of domain authority scores, AllSides media bias ratings, and Semantic Scholar citation counts. Returns a score 0–1. Government and peer-reviewed sources score 0.9+. Unknown blogs score 0.1.

#### `CR2` — Date & Freshness Check

For each source, computes the gap between the source's publication date and the topic's expected data freshness requirement (set in the Risk Register). Sources older than the freshness threshold are penalized in the composite score and flagged for human review if they are the only source for a critical claim.

#### `CR3` — Cross-Source Corroboration

For each extracted claim, counts the number of independent sources (different domain names, different publication dates) that assert the same claim. Claims corroborated by 3+ independent sources are marked `corroboration: strong`. Claims from a single source are marked `corroboration: single_source` and flagged for reviewer attention.

#### `CR4` — Misinformation Flagging

Queries the **Misinfo MCP server**, which wraps the Duke Reporters' Lab ClaimBuster API and a private registry of known false claims. Any passage scoring above `0.7` on ClaimBuster's check-worthy score AND appearing in the false-claim registry is hard-blocked from the fact base and logged as a `MISINFO_BLOCK` audit event.

### Conflict Detection & Resolution Tools

#### `CD1` — Contradiction Detector

Compares all SPO triples with the same subject and predicate across different sources. Uses a Natural Language Inference (NLI) model (`cross-encoder/nli-deberta-v3-base`) to classify each pair as `entailment`, `neutral`, or `contradiction`. All `contradiction` pairs are logged to the Conflict Register with both source passages.

#### `CD2` — Confidence Scorer

Assigns a 0–1 confidence score to each claim, computed as: `confidence = corroboration_weight × authority_score × (1 - contradiction_penalty)`. Claims in outright contradiction receive a heavy penalty; claims with strong corroboration and high authority receive near-1.0 confidence.

#### `CD3` — Majority Vote Resolver

For contradicted claims, tallies the number of high-authority sources supporting each position. If a clear majority (>60% of sources weighted by authority score) supports one position, that position is auto-resolved with `resolution: majority_vote`. If the vote is too close, the claim is escalated to the Unresolved Conflict Register.

#### `CD4` — Unresolved Conflict Register

Serializes all unresolved conflicts to `s3://my-bucket/sessions/{session_id}/conflicts.json` using `boto3`. This file is presented to the human reviewer in the Phase 5 human gate with full source passages, authority scores, and vote tallies.

```python
import boto3
import json
from pathlib import Path

s3 = boto3.client("s3")

def save_conflict_register(session_id: str, conflicts: list[dict]):
    local_path = Path(f"/tmp/{session_id}/conflicts.json")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(conflicts, indent=2))
    s3.upload_file(
        str(local_path),
        "my-bucket",
        f"sessions/{session_id}/conflicts.json"
    )
    # Clean up temp file
    local_path.unlink()
```

### Retry Circuit

If the Evaluator determines that data is insufficient (fewer than `settings.MIN_FACTS_PER_SECTION` high-confidence facts for any section), it signals the retry circuit which routes back to the Question Generator with a refined strategy: higher-priority batches, expanded query terms, and relaxed source-type constraints. After 3 retries, the circuit opens and processing continues with a `low_confidence` flag attached to undersupported sections.

---

## 10. Phase 6 — Validated Fact Base Construction

The `FactBaseBuilder` (`fact_base/builder.py`) assembles all validated claims into a structured, citable, outline-aligned JSON fact store.

#### `FB1` — Claim Deduplicator

Merges claims where NLI scores `entailment` between two assertions from different sources (same meaning, different wording). Retains the higher-authority source as the canonical claim, appending the secondary source as a corroborating citation.

#### `FB2` — Citation Linker (`fact_base/citation_linker.py`)

Maps every claim to its exact source passage, URL, page number (for PDFs, via `pypdf`), and retrieved timestamp. Citations are stored in CSL JSON format for downstream formatting flexibility (APA, MLA, Chicago, Vancouver).

#### `FB3` — Confidence Annotator

Writes the final confidence score (computed in `CD2`) to each fact. Facts below `0.4` confidence are tagged `review_recommended: true`.

#### `FB4` — Outline Mapper

Aligns every fact to its target outline section using a combination of semantic similarity (the fact's embedding vs. the section's research goals) and keyword matching. Facts that don't clearly belong to any section are placed in a `general` bucket for the Synthesizer to allocate.

#### `FB5` — Gap Analyzer

For each outline section, checks: minimum fact count met? Sufficient high-confidence facts? At least one corroborated source? If any section fails, the Gap Analyzer emits a targeted query batch back to the Question Generator specifying exactly which sections need more coverage. This loop continues until all sections meet minimum thresholds or the retry circuit opens.

#### `FB6` — Fact Base Serializer

Writes the final validated fact base as a structured JSON document to S3:

```python
import boto3
import json
from pathlib import Path
import shutil

def serialize_fact_base(session_id: str, fact_base: dict) -> str:
    local_path = Path(f"/tmp/{session_id}/fact_base.json")
    local_path.write_text(json.dumps(fact_base, indent=2, ensure_ascii=False))
    
    s3_key = f"sessions/{session_id}/fact_base.json"
    boto3.client("s3").upload_file(str(local_path), "my-bucket", s3_key)
    
    # Archive a copy to the report history bucket
    archive_key = f"archive/{session_id}/fact_base.json"
    boto3.client("s3").copy_object(
        CopySource={"Bucket": "my-bucket", "Key": s3_key},
        Bucket="my-archive-bucket",
        Key=archive_key
    )
    shutil.rmtree(local_path.parent)  # Clean up temp directory
    return s3_key
```

---

## 11. Phase 7 — Synthesizer Agent

The Synthesizer bridges the fact base and the Writer. Its job is to produce a narrative map — a structured writing plan with ordered evidence — so that the Writer never needs to make factual decisions, only prose decisions.

#### `ST1` — Outline Section Loader

Loads the Approved Outline from S3 (written in Phase 1). For each section, loads: the section title, research goals, word budget, and any risk flags from the Risk Register.

#### `ST2` — Fact Selector

For each section, queries the fact base (filtered to that section via `FB4`'s outline mapping) and selects the top facts by confidence score, subject to the word budget. The selection algorithm ensures that at least one fact addresses each of the section's research goals.

#### `ST3` — Narrative Thread Builder

Uses an LLM call (Claude via `langchain-anthropic`) to arrange the selected facts into a logical argument chain. The prompt instructs the model to output only a structured JSON plan (claim IDs in order, transition type between each: `supports | contrasts | elaborates | concludes`) — no prose, no paraphrasing. This prevents hallucination at this stage.

#### `ST4` — Evidence Ranker

Re-orders supporting evidence within each argument step by descending authority score, so the strongest evidence appears first in the Writer's context window.

#### `ST5` — Contradiction Pre-check

Before passing the synthesis plan to the Writer, runs a final NLI pass over adjacent claims in the narrative thread to ensure no contradictions were introduced by the arrangement. Any flagged pair triggers a local rearrangement.

### Synthesis Plan Output

The Synthesizer writes a `synthesis_plan.json` to S3 with this structure:

```json
{
  "sections": [
    {
      "title": "Market Overview",
      "word_budget": 600,
      "thesis": "The global market reached $X in 2023, driven by Y.",
      "evidence_chain": [
        {"claim_id": "C042", "transition": "supports", "citation_key": "bloomberg_2023"},
        {"claim_id": "C078", "transition": "elaborates", "citation_key": "reuters_2024"}
      ]
    }
  ],
  "cross_reference_index": {"C042": {...}, "C078": {...}}
}
```

---

## 12. Phase 8 — Writer Agent

The Writer Agent produces all prose. It never has direct access to the raw source documents — it works exclusively from the Synthesis Plan. This architectural constraint ensures that all factual content is pre-validated before prose generation.

#### `WT1` — Section Drafter

Calls Claude via `langchain-anthropic` with a structured prompt that includes the section thesis, the ordered evidence chain (claim text + citation keys), the word budget, and the target audience. The model is instructed to produce prose that flows from the evidence chain without adding new facts.

#### `WT2` — Citation Injector (`agents/writer/tools.py`)

Parses the generated prose for claim references and injects formatted citations (footnotes for academic format, inline links for web format). Uses the CSL JSON citation data from `FB2`.

#### `WT3` — Word Count Enforcer

Checks each section's word count against its budget. Sections over budget are trimmed using an extractive summarization model (`facebook/bart-large-cnn`). Sections under budget trigger a targeted expansion prompt to the Section Drafter.

#### `WT4` — Style Formatter

Applies the output format template: Markdown for web delivery, LaTeX for PDF, DOCX via `python-docx` for Word format. Heading levels, table formatting, figure captions, and list styles are all applied here. Template selection is driven by `brief["output_format"]`.

#### `WT5` — Executive Summary Generator

After all body sections are drafted, generates a 250-word executive summary by prompting Claude with the complete body text and the original user brief. The summary is prepended to the report.

---

## 13. Phase 9 — Decentralized QA & Finalization

The draft is broadcast simultaneously to three independent critic agents. No single agent can pass or block the report — decisions require a majority or unanimous vote, with tie-breaker logic for 1-1-1 splits.

### Fact-Checker Agent

#### `FC1` — Claim Extractor

Uses the same claim identification model from `FE3` to parse every verifiable claim from the draft prose.

#### `FC2` — Fact Base Lookup

For each extracted claim, performs a semantic search against the Validated Fact Base to find the closest matching validated claim. Computes a semantic similarity score between the draft claim and its best match.

#### `FC3` — Hallucination Detector

Claims with no Fact Base match above similarity threshold `0.75` are flagged as potential hallucinations. Each flagged claim is annotated with: the draft text, the nearest Fact Base match (if any), and the similarity gap. These annotations are passed to the Writer in revision feedback.

#### `FC4` — Faithfulness Score

Uses `ragas` to compute a claim-level faithfulness score: what fraction of the draft's claims are grounded in the Fact Base?

```python
from ragas.metrics import faithfulness
from ragas import evaluate
from datasets import Dataset

def compute_faithfulness(questions, answers, contexts) -> float:
    data = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts
    })
    result = evaluate(data, metrics=[faithfulness])
    return result["faithfulness"]
```

### Alignment Agent

#### `AL1` — Section Coverage Checker

Verifies that every section defined in the Approved Outline has a corresponding section in the draft with substantive content (at least `min_words_per_section` words).

#### `AL2` — Goal Alignment Verifier

For each section, uses an LLM judge call to verify that the section's content answers the research goals defined in Phase 1. The judge returns a 0–1 alignment score per goal.

#### `AL3` — Constraint Validator

Checks hard constraints: total word count within ±10% of target, audience-appropriate vocabulary (Flesch-Kincaid reading level within range), no prohibited topics from the brief's constraint list, deadline metadata present.

#### `AL4` — Relevance Score

Computes RAGAS answer relevance: does the generated content actually address the user's original topic and constraints?

### Format Agent

#### `FM1` — Tone & Register Checker

Uses a style classifier to verify the register matches the target audience (`formal | semi-formal | technical | executive`). Flags sentences that are too casual for formal reports or too jargon-heavy for executive audiences.

#### `FM2` — Structure Validator

Checks heading hierarchy (no skipped levels), paragraph length (no single-sentence paragraphs, no paragraphs over 200 words), logical flow (topic sentences present, concluding sentences present), and transition quality.

#### `FM3` — Citation Completeness

Every verifiable claim must have a citation. Every citation key must resolve to a real entry in the CSL citation index. Orphaned citations (cited but not referenced in text) are flagged.

#### `FM4` — Readability Score

Computes Flesch Reading Ease, Gunning Fog Index, and SMOG Grade using `textstat`. Scores are compared against audience-appropriate benchmarks from the brief's constraint set.

### Routing & Aggregation Node (`qa/aggregator.py`)

Collects all three critic agents' verdicts (pass/fail per check + numeric scores) and assembles a `CriticVerdict` object for the tie-breaker.

### Tie-Breaker Logic (`qa/tie_breaker.py`)

```python
from dataclasses import dataclass

@dataclass
class CriticVerdict:
    fact_checker_pass: bool
    alignment_pass: bool
    format_pass: bool
    failure_annotations: dict  # per-agent, per-check failure details

def resolve_verdict(verdict: CriticVerdict) -> tuple[str, list[str]]:
    votes = [verdict.fact_checker_pass, verdict.alignment_pass, verdict.format_pass]
    passing = sum(votes)

    if passing == 3:
        return "PASS", []
    elif passing == 2:
        # Majority pass: identify the failing agent, collect its annotations only
        failing_agent = ["fact_checker", "alignment", "format"][votes.index(False)]
        feedback = verdict.failure_annotations[failing_agent]
        return "MAJORITY_PASS_WITH_WARNINGS", feedback
    elif passing == 1:
        # Majority fail: collect all failure annotations
        all_feedback = []
        for i, passed in enumerate(votes):
            if not passed:
                agent = ["fact_checker", "alignment", "format"][i]
                all_feedback.extend(verdict.failure_annotations[agent])
        return "FAIL", all_feedback
    else:
        # Unanimous fail
        all_feedback = [ann for anns in verdict.failure_annotations.values() 
                        for ann in anns]
        return "FAIL", all_feedback
```

`PASS` and `MAJORITY_PASS_WITH_WARNINGS` proceed to Output Guardrails. `FAIL` triggers the Writer retry circuit with the specific failure annotations as targeted revision instructions. After 3 failed attempts, the circuit opens and the annotated draft is escalated to a human reviewer via SQS.

### Output Guardrails (`guardrails/output_guard.py`)

Before human review, the final draft passes through:

**Toxicity Check** — `detoxify` library scores the report for toxicity, severe toxicity, obscenity, identity attacks, insults, threats, and sexual content. Any score above `0.1` triggers a review flag.

**Data Leak Check** — Runs the same `presidio-analyzer` PII scan from the input guardrail to ensure no PII was introduced during generation. Also checks for any internal document content that should not appear verbatim in the external report (using MinHash fingerprinting against the internal KB).

### Final Delivery

On human approval, the report is:

1. Saved to S3 in all requested formats via `boto3`.
2. Indexed in the semantic cache for future cache hits.
3. Audit trail finalized and sealed in S3.
4. Delivery notification sent via SNS.

```python
import boto3
from pathlib import Path
import shutil

s3 = boto3.client("s3")
sns = boto3.client("sns")

def deliver_report(session_id: str, report: dict, formats: list[str]):
    for fmt in formats:
        local_path = Path(f"/tmp/{session_id}/report.{fmt}")
        render_report(report, fmt, local_path)
        s3.upload_file(
            str(local_path),
            "my-reports-bucket",
            f"reports/{session_id}/report.{fmt}"
        )
    shutil.rmtree(Path(f"/tmp/{session_id}"))  # Clean all temp files
    sns.publish(
        TopicArn=settings.DELIVERY_SNS_TOPIC,
        Message=json.dumps({"session_id": session_id, "status": "delivered"})
    )
```

---

## 14. Observability, Telemetry & Cost Monitoring

All observability is implemented in `observability/` and injected as LangGraph callbacks.

### Distributed Tracing (`tracer.py`)

Every LangGraph node emits OpenTelemetry spans. Parent spans wrap entire phases; child spans wrap individual tool calls. Traces are exported to AWS X-Ray (via the OTLP exporter) and to Grafana Tempo for long-term storage.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.OTLP_ENDPOINT))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agentic-report-pipeline")
```

### Token & Cost Tracking (`cost_tracker.py`)

Every LLM call goes through a cost-tracking wrapper that records: model name, input tokens, output tokens, USD cost (based on published pricing), phase, node name, and session ID. Costs are accumulated per session and written to a DynamoDB table for billing and monitoring. CloudWatch alarms fire if a single session exceeds `settings.MAX_COST_PER_SESSION_USD`.

### Latency Monitoring (`metrics.py`)

Prometheus histograms track P50/P95/P99 latency for each node. Grafana dashboards visualize these with phase-level aggregations. Alert rules fire PagerDuty incidents if P95 for any phase exceeds SLA thresholds.

### Retry & Circuit-Breaker Log

Every retry attempt and circuit-breaker state change is logged as a structured event to CloudWatch Logs with: `session_id`, `phase`, `node`, `attempt_number`, `trigger_reason`, `circuit_state`. This log feeds a Grafana panel showing retry rates over time, helping identify systematically flaky phases.

### RAGAS Dashboard

Faithfulness and relevance scores from Phase 9 are written to a CloudWatch Metrics namespace (`AgenticReport/RAGAS`). A Grafana dashboard shows rolling averages, p10 scores, and distribution histograms across all completed reports.

### Immutable Audit Trail (`audit_logger.py`)

Every decision, tool call, human input, retry event, and score is appended to an append-only JSON Lines file in S3 with Object Lock (WORM — Write Once Read Many) enabled. S3 versioning is enabled on the audit bucket. The audit log is the single source of truth for compliance and post-incident analysis.

```python
import boto3
import json
from datetime import datetime, timezone

s3 = boto3.client("s3")

def audit_log(session_id: str, event_type: str, payload: dict):
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "event_type": event_type,
        **payload
    }
    s3.put_object(
        Bucket="my-audit-bucket",
        Key=f"audit/{session_id}/{datetime.now().timestamp()}.json",
        Body=json.dumps(event).encode(),
    )
```

---

## 15. MCP Server Integration

The system integrates with **six MCP servers**, each exposing a set of tools that agents call via the central `mcp/tool_dispatcher.py`.

| MCP Server | Tools Exposed | Used By |
|---|---|---|
| `knowledge-mcp` | `fetch_taxonomy`, `lookup_ontology`, `get_domain_terms` | Planner (PT3), QGen (QT3) |
| `routing-mcp` | `classify_query_source`, `route_to_datasource` | QGen (QT2) |
| `nlp-mcp` | `expand_query`, `extract_entities`, `detect_language` | QGen (QT3), Evaluator |
| `credibility-mcp` | `score_domain_authority`, `check_bias_rating`, `get_citation_count` | Evaluator (CR1) |
| `misinfo-mcp` | `check_claim_buster`, `lookup_false_claim_registry` | Evaluator (CR4) |
| `security-mcp` | `detect_prompt_injection`, `classify_pii`, `check_toxicity` | Input/Output Guards |

### MCP Tool Dispatcher (`mcp/tool_dispatcher.py`)

```python
import httpx
from mcp.client import MCPClient

clients: dict[str, MCPClient] = {}

def get_client(server_name: str) -> MCPClient:
    if server_name not in clients:
        endpoint = settings.MCP_SERVER_ENDPOINTS[server_name]
        clients[server_name] = MCPClient(endpoint)
    return clients[server_name]

async def mcp_dispatch(server: str, tool: str, args: dict) -> dict:
    client = get_client(server)
    result = await client.call_tool(tool, args)
    audit_log(current_session_id(), "MCP_TOOL_CALL", {
        "server": server, "tool": tool, "args_keys": list(args.keys())
    })
    return result
```

MCP servers are deployed as separate microservices on EKS, each in its own namespace. Inter-service communication uses AWS App Mesh (Envoy proxy) for mTLS and traffic management.

---

## 16. A2A (Agent-to-Agent) Communication Protocol

For long-running phases and the Phase 9 broadcast pattern, agents communicate via the A2A protocol implemented in `a2a/`.

### Message Envelope (`a2a/protocol.py`)

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

@dataclass
class A2AMessage:
    message_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str = ""
    sender_agent: str = ""
    recipient_agent: str = ""
    message_type: str = ""   # REQUEST | RESPONSE | ERROR | BROADCAST
    payload: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    correlation_id: str | None = None  # links responses to requests
```

### Agent Registry (`a2a/registry.py`)

All agents self-register on startup, advertising their capabilities (accepted message types and payload schemas) to a DynamoDB agent registry table. The Phase 9 Broadcast node queries this registry to discover all available critic agents dynamically — adding a new critic type requires only deploying the new agent, not changing orchestration code.

### SQS Message Bus (`a2a/bus.py`)

```python
import boto3
import json

sqs = boto3.client("sqs")

def send_message(queue_url: str, message: A2AMessage):
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message.__dict__),
        MessageGroupId=message.session_id,  # FIFO queue for ordering within session
    )

def receive_messages(queue_url: str, max_messages: int = 10) -> list[A2AMessage]:
    resp = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=max_messages,
        WaitTimeSeconds=20  # Long polling
    )
    messages = []
    for msg in resp.get("Messages", []):
        messages.append(A2AMessage(**json.loads(msg["Body"])))
        sqs.delete_message(QueueUrl=queue_url,
                           ReceiptHandle=msg["ReceiptHandle"])
    return messages
```

Each agent type has its own SQS FIFO queue. The Aggregation Node in Phase 9 polls all three critic queues with a deadline and collects responses. If a critic doesn't respond within `settings.CRITIC_TIMEOUT_SECONDS`, it is treated as a `FAIL` vote (fail-safe default).

---

## 17. Retry Circuits & Tie-Breaker Logic

### Circuit Breaker Pattern

The `RetryCircuit` class (shown in Phase 1) is used at four points in the pipeline: Planner (Phase 1), Evaluator (Phase 5), Fact Base Gap Analyzer (Phase 6), and Writer QA (Phase 9). All circuit state is stored in the LangGraph state dict so it persists across graph re-entries.

```python
# graph/state.py (excerpt)
from typing import TypedDict

class PipelineState(TypedDict):
    session_id: str
    user_brief: dict
    approved_outline: dict | None
    validated_fact_base: dict | None
    synthesis_plan: dict | None
    raw_draft: str | None
    final_report: str | None
    # Retry counters
    planner_retries: int
    evaluator_retries: int
    gap_retries: int
    writer_retries: int
    # Human gate flags
    scope_approved: bool
    sources_validated: bool
    final_approved: bool
    # Escalation flags
    planner_escalated: bool
    evaluator_escalated: bool
    writer_escalated: bool
```

### Conditional Edge Logic (`graph/edges.py`)

```python
def planner_retry_edge(state: PipelineState) -> str:
    if state["scope_approved"]:
        return "question_generator"
    elif state["planner_retries"] >= 3:
        return "planner_escalation"
    else:
        return "planner"

def writer_qa_edge(state: PipelineState) -> str:
    verdict, _ = resolve_verdict(state["last_critic_verdict"])
    if verdict in ("PASS", "MAJORITY_PASS_WITH_WARNINGS"):
        return "output_guardrails"
    elif state["writer_retries"] >= 3:
        return "writer_escalation"
    else:
        return "writer"
```

---

## 18. Infrastructure: AWS EKS Deployment

### Cluster Architecture

The EKS cluster (`infra/eks/cluster.yaml`) runs on `us-east-1` with three availability zones. It uses managed node groups with autoscaling:

| Node Group | Instance Type | Min | Max | Phase Affinity |
|---|---|---|---|---|
| `retrieval-ng` | `c5.2xlarge` | 2 | 20 | Phase 3 (parallel search) |
| `nlp-ng` | `g4dn.xlarge` (GPU) | 1 | 8 | Phase 4, 5 (reranking, NER) |
| `general-ng` | `m5.xlarge` | 3 | 15 | All other phases |
| `human-gate-ng` | `t3.medium` | 1 | 3 | Phase 1, 5, 9 human gates |

### Pod Autoscaling

Each agent is deployed as a separate Kubernetes Deployment with a Horizontal Pod Autoscaler (HPA) triggered on CPU utilization (target: 70%) and custom Prometheus metrics (queue depth for SQS-backed agents).

### Service Mesh

AWS App Mesh provides mTLS between all services, circuit breaking (Envoy-level, complementing application-level retry circuits), and traffic shaping. Each agent service has an Envoy sidecar injected automatically.

### Helm Charts

Each agent service has a Helm chart under `infra/eks/helm/`. Charts share a common `values.yaml` schema for consistency. Deployment is managed by ArgoCD in a GitOps pattern.

---

## 19. Storage Layer: S3, s3fs & boto3

### Bucket Structure

```
my-bucket/
├── briefs/{session_id}/
│   └── brief.json
├── sessions/{session_id}/
│   ├── outline.json
│   ├── query_plan.json
│   ├── raw_results/
│   ├── reranked_passages.json
│   ├── conflicts.json
│   ├── fact_base.json
│   ├── synthesis_plan.json
│   ├── draft.md
│   └── final_report.{md,pdf,docx}
└── knowledge-base/
    └── **/*.jsonl

my-audit-bucket/          # Object Lock WORM
└── audit/{session_id}/
    └── *.json

my-reports-bucket/        # Final delivery
└── reports/{session_id}/
    └── report.{md,pdf,docx}

my-archive-bucket/        # Long-term archive
└── archive/{session_id}/
```

### s3fs Usage Pattern

`s3fs` is used throughout the codebase for streaming access to S3 objects as if they were local files. This is particularly important for large document corpora in the Internal KB and for streaming result aggregation without loading everything into memory.

```python
import s3fs

fs = s3fs.S3FileSystem(
    key=settings.AWS_ACCESS_KEY_ID,
    secret=settings.AWS_SECRET_ACCESS_KEY,
    client_kwargs={"region_name": "us-east-1"}
)

# Stream a large JSONL file line by line
with fs.open("s3://my-bucket/knowledge-base/docs.jsonl", "r") as f:
    for line in f:
        process_chunk(json.loads(line))
```

### pathlib + shutil Pattern

All local temporary file operations follow a consistent pattern using `pathlib` for path construction and `shutil` for file operations:

```python
from pathlib import Path
import shutil

def with_temp_file(session_id: str, filename: str):
    """Context manager for temp files that auto-cleans on exit."""
    tmp = Path(f"/tmp/{session_id}/{filename}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield tmp
    finally:
        if tmp.exists():
            tmp.unlink()
        # Clean session temp dir if empty
        if not any(tmp.parent.iterdir()):
            shutil.rmtree(tmp.parent, ignore_errors=True)
```

---

## 20. LangGraph State & Graph Definition

```mermaid
graph TD

    %% ============================================================
    %% ENTERPRISE AGENTIC REPORT GENERATION — FULL ARCHITECTURE
    %% All subgraphs · All tool calls · Retrieval · Validation · Planning
    %% ============================================================

    %% ── Entry Point ─────────────────────────────────────────────
    User([👤 User: defines topic & constraints])
        --> InGuard[🛡️ Input Guardrails\nPII Redaction · Prompt Injection Check]
    InGuard --> CacheHit{📦 Semantic Cache Hit?}
    CacheHit -- Yes --> Cached([⚡ Return Cached Report Instantly])
    CacheHit -- No  --> Planner

    %% ════════════════════════════════════════════════════════════
    %% PHASE 1 — PLANNING & SCOPE ALIGNMENT
    %% ════════════════════════════════════════════════════════════
    subgraph P1 [Phase 1 · Planning & Scope Alignment]
        direction TB
        Planner[🧠 Planner Agent\nGenerates scope · outline · constraints]

        subgraph P1_TOOLS [Planner Tool Calls]
            direction LR
            PT1[📁 Load User Brief]
            PT2[📜 Retrieve Past Similar Reports\nfrom Vector Store]
            PT3[🗂️ Fetch Domain Taxonomy\nOntology Lookup]
            PT4[📐 Constraint Checker\nLength · Format · Audience · Deadline]
        end

        Planner --> PT1 & PT2 & PT3 & PT4

        subgraph P1_PLAN [Plan Construction]
            direction TB
            PP1[📝 Draft Outline\nSections · Subsections · Word Budget]
            PP2[🎯 Define Research Goals\nper Section]
            PP3[⚠️ Flag Known Risks\nData Gaps · Sensitive Topics]
            PP4[🗓️ Set Milestone Checkpoints\nPhase gates for Human review]
        end

        PT1 & PT2 & PT3 & PT4 --> PP1
        PP1 --> PP2 --> PP3 --> PP4

        P1Retry{🔁 Retry Circuit\nMax 3 attempts}
        PP4 --> P1Retry
        P1Retry -- Attempt 1–3 --> H1{👤 Human Approves Scope?}
        P1Retry -- Max exceeded  --> P1Fallback[⚠️ Escalate:\npartial draft + risk flags]
        H1 -- No · Feedback --> Planner
        H1 -- Yes --> Outline([✅ Approved Outline\n+ Research Goals\n+ Risk Register])
    end

    %% ════════════════════════════════════════════════════════════
    %% PHASE 2 — QUESTION GENERATION
    %% ════════════════════════════════════════════════════════════
    subgraph P2 [Phase 2 · Question Generation]
        direction TB
        QGen[🔍 Question Generator Agent\nDecomposes outline into atomic queries]

        subgraph P2_TOOLS [QGen Tool Calls]
            direction LR
            QT1[🧬 NLP Decomposition\nEntity · Relation · Intent extraction]
            QT2[🗺️ Query Router\nClassifies → Web · DB · Wiki · KB · API]
            QT3[🌐 Query Expander\nSynonyms · Related terms · Multilingual]
            QT4[📊 Priority Ranker\nScores queries by section importance]
        end

        Outline --> QGen
        QGen --> QT1 --> QT2 --> QT3 --> QT4

        subgraph P2_QPLAN [Query Plan]
            direction TB
            QP1[📋 Query Batch A\nHigh Priority — Critical sections]
            QP2[📋 Query Batch B\nMedium Priority — Supporting evidence]
            QP3[📋 Query Batch C\nLow Priority — Background context]
            QP4[🔀 Async Parallel Dispatch\nAll batches fire simultaneously]
        end

        QT4 --> QP1 & QP2 & QP3
        QP1 & QP2 & QP3 --> QP4
    end

    %% ════════════════════════════════════════════════════════════
    %% PHASE 3 — HYBRID SEARCH & RETRIEVAL
    %% ════════════════════════════════════════════════════════════
    subgraph P3 [Phase 3 · Hybrid Search & Multi-Source Retrieval]
        direction TB

        subgraph P3_WEB [Web Search Tools]
            direction LR
            WS1[🌐 Web Search API\nGoogle · Bing · Brave]
            WS2[📰 News API\nRecent articles · Press releases]
            WS3[🔬 Academic Search\nArXiv · PubMed · Semantic Scholar]
            WS4[🏛️ Gov & Regulatory\nFederal Register · EUR-Lex · Data.gov]
        end

        subgraph P3_DB [Database & Knowledge Tools]
            direction LR
            DB1[🗄️ Vector DB Lookup\nFAISS · Pinecone · Weaviate\nSemantic similarity search]
            DB2[🔤 BM25 Keyword Search\nElastic · Opensearch\nExact term matching]
            DB3[📚 Internal Knowledge Base\nCompany docs · Past reports · SOPs]
            DB4[🔗 Structured DB Query\nSQL · GraphQL · REST APIs]
        end

        subgraph P3_SPEC [Specialist Tools]
            direction LR
            SP1[🧮 Wolfram Alpha\nMath · Statistics · Formulas]
            SP2[📈 Financial Data API\nBloomberg · Yahoo Finance]
            SP3[🗺️ Geospatial API\nMaps · Census · OSM data]
            SP4[🧠 Domain Expert KB\nCurated ontologies · Taxonomies]
        end

        QP4 --> WS1 & WS2 & WS3 & WS4
        QP4 --> DB1 & DB2 & DB3 & DB4
        QP4 --> SP1 & SP2 & SP3 & SP4

        subgraph P3_MERGE [Result Collection & Deduplication]
            direction TB
            RM1[🧹 Raw Result Aggregator\nCollects all source responses]
            RM2[🔗 URL & Content Deduplicator\nRemoves exact + near-duplicates]
            RM3[📦 Chunk Splitter\nSplits docs into 512-token passages]
            RM4[🏷️ Metadata Tagger\nSource · Date · Domain · Confidence tier]
        end

        WS1 & WS2 & WS3 & WS4 --> RM1
        DB1 & DB2 & DB3 & DB4 --> RM1
        SP1 & SP2 & SP3 & SP4 --> RM1
        RM1 --> RM2 --> RM3 --> RM4
    end

    %% ════════════════════════════════════════════════════════════
    %% PHASE 4 — RERANKING & RELEVANCE FILTERING
    %% ════════════════════════════════════════════════════════════
    subgraph P4 [Phase 4 · Reranking & Relevance Filtering]
        direction TB

        subgraph P4_RERANK [Reranking Tools]
            direction LR
            RR1[📐 Cross-Encoder Reranker\nComputes query-passage similarity]
            RR2[🎯 Contextual Compressor\nExtracts only relevant sentence spans]
            RR3[📊 Diversity Sampler\nMMR — ensures source variety]
            RR4[🗓️ Recency Scorer\nBoosts recent sources for time-sensitive topics]
        end

        RM4 --> RR1 & RR2 & RR3 & RR4

        subgraph P4_SCORE [Scoring & Threshold Gate]
            direction TB
            SC1[🧮 Composite Score Calculator\nRelevance · Recency · Authority · Diversity]
            SC2[🚦 Threshold Filter\nDrop passages below min score]
            SC3[📋 Top-K Selector\nRetain top N passages per query]
        end

        RR1 & RR2 & RR3 & RR4 --> SC1 --> SC2 --> SC3
    end

    %% ════════════════════════════════════════════════════════════
    %% PHASE 5 — EVALUATION, FACT EXTRACTION & CONFLICT DETECTION
    %% ════════════════════════════════════════════════════════════
    subgraph P5 [Phase 5 · Evaluator Agent · Fact Extraction & Conflict Detection]
        direction TB
        Eval[🔬 Evaluator Agent\nOrchestrates all validation sub-tasks]

        subgraph P5_EXTRACT [Fact Extraction Tools]
            direction LR
            FE1[🏷️ Named Entity Recognizer\nPeople · Orgs · Dates · Locations]
            FE2[🔗 Relation Extractor\nSubject–Predicate–Object triples]
            FE3[📌 Claim Identifier\nIsolates verifiable assertions]
            FE4[🔢 Numeric Validator\nChecks stats · percentages · units]
        end

        subgraph P5_CRED [Credibility Assessment Tools]
            direction LR
            CR1[🏛️ Source Authority Scorer\nDomain reputation · Citations · Bias ratings]
            CR2[📅 Date & Freshness Check\nCompares source date vs topic recency needs]
            CR3[🔍 Cross-Source Corroboration\nCounts independent confirmations per claim]
            CR4[⚠️ Misinformation Flagging\nChecks against known false-claim registries]
        end

        subgraph P5_CONFLICT [Conflict Detection & Resolution]
            direction TB
            CD1[⚡ Contradiction Detector\nCompares claims across all sources]
            CD2[📊 Confidence Scorer\nAssigns confidence per claim 0–1]
            CD3[🗳️ Majority Vote Resolver\nAccepts claim if majority of sources agree]
            CD4[🚩 Unresolved Conflict Register\nLogs all claims that cannot be auto-resolved]
        end

        SC3 --> Eval
        Eval --> FE1 & FE2 & FE3 & FE4
        Eval --> CR1 & CR2 & CR3 & CR4
        FE1 & FE2 & FE3 & FE4 --> CD1
        CR1 & CR2 & CR3 & CR4 --> CD1
        CD1 --> CD2 --> CD3 --> CD4

        P5Retry{🔁 Retry Circuit\nMax 3 attempts}
        CD4 --> P5Retry
        P5Retry -- Insufficient data · Refines queries --> QGen
        P5Retry -- Max exceeded --> P5Fallback[⚠️ Proceed with low-confidence flags]

        CD4 --> SuffCheck{✅ Sufficient\nHigh-Confidence Facts?}
        SuffCheck -- No --> QGen
        SuffCheck -- Yes --> ConflictCheck{⚠️ Unresolved\nConflicts or Sensitive?}
        ConflictCheck -- Yes --> H2[👤 Human Validates\nFlagged Sources via RBAC]
        ConflictCheck -- No  --> FactBase
        H2 --> FactBase
        P5Fallback --> FactBase
    end

    %% ════════════════════════════════════════════════════════════
    %% PHASE 6 — VALIDATED FACT BASE CONSTRUCTION
    %% ════════════════════════════════════════════════════════════
    subgraph P6 [Phase 6 · Validated Fact Base Construction]
        direction TB

        subgraph P6_BUILD [Fact Base Builder Tools]
            direction LR
            FB1[🗂️ Claim Deduplicator\nMerges identical claims from multiple sources]
            FB2[🔗 Citation Linker\nMaps every claim → source passage + URL]
            FB3[📊 Confidence Annotator\nAttaches confidence score to each fact]
            FB4[🗺️ Outline Mapper\nAligns every fact to the correct outline section]
            FB5[🧩 Gap Analyzer\nIdentifies outline sections with insufficient facts]
            FB6[📋 Fact Base Serializer\nOutputs structured JSON fact store]
        end

        FactBase([📋 Raw Validated Facts]) --> FB1 --> FB2 --> FB3 --> FB4
        FB4 --> FB5
        FB5 -- Gaps found --> QGen
        FB5 -- No gaps    --> FB6
        FB6 --> ValidatedFactBase([✅ Validated Fact Base\nCited · Scored · Outline-Aligned])
    end

    %% ════════════════════════════════════════════════════════════
    %% PHASE 7 — SYNTHESIS
    %% ════════════════════════════════════════════════════════════
    subgraph P7 [Phase 7 · Synthesizer Agent]
        direction TB
        Synth[🗺️ Synthesizer Agent\nMaps facts to narrative structure]

        subgraph P7_TOOLS [Synthesis Tool Calls]
            direction LR
            ST1[📐 Outline Section Loader\nLoads section goals + word budget]
            ST2[🎯 Fact Selector\nPicks highest-confidence facts per section]
            ST3[🔗 Narrative Thread Builder\nLinks facts into logical argument chains]
            ST4[📊 Evidence Ranker\nOrders supporting evidence by strength]
            ST5[⚡ Contradiction Pre-check\nFinal scan before passing to Writer]
        end

        ValidatedFactBase --> Synth
        Synth --> ST1 & ST2 & ST3 & ST4 & ST5

        subgraph P7_PLAN [Synthesis Plan Output]
            direction TB
            SP_A[📝 Section-by-Section Narrative Map\nFact IDs · Evidence order · Argument flow]
            SP_B[📌 Key Takeaway Statements\nDraft thesis per section]
            SP_C[🔗 Cross-Reference Index\nLinked citations ready for Writer]
        end

        ST1 & ST2 & ST3 & ST4 & ST5 --> SP_A --> SP_B --> SP_C
        SP_C --> SynthPlan([✅ Synthesis Plan\nReady for Writer Agent])
    end

    %% ════════════════════════════════════════════════════════════
    %% PHASE 8 — WRITING
    %% ════════════════════════════════════════════════════════════
    subgraph P8 [Phase 8 · Writer Agent]
        direction TB

        subgraph P8_TOOLS [Writer Tool Calls]
            direction LR
            WT1[✍️ Section Drafter\nGenerates prose per section from narrative map]
            WT2[🔗 Citation Injector\nInlines footnotes + hyperlinks]
            WT3[📏 Word Count Enforcer\nTrimming · expansion per budget]
            WT4[🎨 Style Formatter\nApplies tone · register · template]
            WT5[📑 Executive Summary Generator\nAuto-drafts from completed body]
        end

        SynthPlan --> WT1 --> WT2 --> WT3 --> WT4 --> WT5
        WT5 --> RawDraft([📄 Raw Draft Report])
    end

    %% ════════════════════════════════════════════════════════════
    %% PHASE 9 — DECENTRALIZED QA
    %% ════════════════════════════════════════════════════════════
    subgraph P9 [Phase 9 · Decentralized QA · No God Agent]
        direction TB
        Broadcast{📡 Broadcast Draft\nto Specialized Critics}

        subgraph P9_FACT [Fact-Checker Agent]
            direction TB
            FC1[🔎 Claim Extractor\nParses all verifiable claims from draft]
            FC2[🗄️ Fact Base Lookup\nChecks each claim against Validated Fact Base]
            FC3[⚠️ Hallucination Detector\nFlags claims with no source backing]
            FC4[📊 Faithfulness Score\nRAGAS — claim-level precision]
        end

        subgraph P9_SCOPE [Alignment Agent]
            direction TB
            AL1[📐 Section Coverage Checker\nEvery outline section present?]
            AL2[🎯 Goal Alignment Verifier\nDoes each section meet its research goal?]
            AL3[⚖️ Constraint Validator\nWord count · audience · deadline met?]
            AL4[📊 Relevance Score\nRAGAS — answer relevance metric]
        end

        subgraph P9_FORMAT [Format Agent]
            direction TB
            FM1[🖊️ Tone & Register Checker\nFormal · Neutral · Audience-appropriate]
            FM2[📏 Structure Validator\nHeadings · Paragraphs · Flow]
            FM3[🔗 Citation Completeness\nAll claims cited? Format correct?]
            FM4[🧹 Readability Score\nFlesch · Gunning Fog index]
        end

        RawDraft --> Broadcast
        Broadcast --> FC1 --> FC2 --> FC3 --> FC4
        Broadcast --> AL1 --> AL2 --> AL3 --> AL4
        Broadcast --> FM1 --> FM2 --> FM3 --> FM4

        Agg[⚖️ Routing & Aggregation Node\nCollects all critic verdicts + scores]
        FC4 --> Agg
        AL4 --> Agg
        FM4 --> Agg

        TieBreak{🗳️ Tie-Breaker Logic}
        Agg --> TieBreak
        TieBreak -- Unanimous Pass        --> PassGate{✅ All Critics Pass?}
        TieBreak -- Majority Pass 2 of 3  --> PassGate
        TieBreak -- Tie or Majority Fail  --> WriterRetry

        PassGate -- No · Targeted failure routes --> WriterRetry{🔁 Retry Circuit\nMax 3 attempts}
        WriterRetry -- Attempt 1–3 · Specific feedback --> WT1
        WriterRetry -- Max exceeded --> P9Fallback[⚠️ Escalate:\nannotated draft to Human]

        PassGate -- Yes --> OutGuard[🛡️ Output Guardrails\nToxicity · Data Leak · PII re-check]
        OutGuard --> H3{👤 Human Final Approval?}
        H3 -- No · Minor tweaks --> WT1
        H3 -- Yes --> SaveCache[(💾 Save to Semantic Cache DB)]
        SaveCache --> Done([🏁 Final Report Delivered])
        P9Fallback --> H3
    end

    %% ════════════════════════════════════════════════════════════
    %% OBSERVABILITY LAYER
    %% ════════════════════════════════════════════════════════════
    subgraph OBS [🔭 Observability · Telemetry · Cost Monitoring — Wraps All Phases]
        direction LR
        O1[📈 Token & Cost Tracker\nper agent · per phase]
        O2[⏱️ Latency Monitor\nP50 · P95 · P99 per node]
        O3[🔁 Retry & Circuit-Breaker Log\nAll phases · attempt counts]
        O4[📊 RAGAS Dashboard\nFaithfulness · Relevance · Precision live]
        O5[🪵 Immutable Audit Trail\nAll decisions · tool calls · human inputs]
        O6[🚨 Alert Manager\nBreaches SLA or quality thresholds → PagerDuty]
    end

    %% ── Styling ─────────────────────────────────────────────────
    classDef human    fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef agent    fill:#cce5ff,stroke:#0056b3,stroke-width:2px,color:#004085;
    classDef tool     fill:#e8daef,stroke:#7d3c98,stroke-width:1.5px,color:#4a235a;
    classDef security fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
    classDef data     fill:#e2e3e5,stroke:#6c757d,stroke-width:2px,color:#383d41;
    classDef retry    fill:#f8d7da,stroke:#dc3545,stroke-width:2px,color:#721c24;
    classDef fallback fill:#f9d0c4,stroke:#c0392b,stroke-width:2.5px,color:#7b241c;
    classDef obs      fill:#eaf4fb,stroke:#2e86c1,stroke-width:1.5px,color:#1a5276;
    classDef done     fill:#d5f5e3,stroke:#1e8449,stroke-width:2px,color:#145a32;
    classDef plan     fill:#fef9e7,stroke:#d4ac0d,stroke-width:2px,color:#7d6608;

    class H1,H2,H3 human;
    class Planner,QGen,Eval,Synth,WT1,FactCheck,ScopeCheck,ToneCheck,Broadcast,Agg agent;
    class PT1,PT2,PT3,PT4,QT1,QT2,QT3,QT4,WS1,WS2,WS3,WS4 tool;
    class DB1,DB2,DB3,DB4,SP1,SP2,SP3,SP4,RR1,RR2,RR3,RR4 tool;
    class FE1,FE2,FE3,FE4,CR1,CR2,CR3,CR4,CD1,CD2,CD3,CD4 tool;
    class FB1,FB2,FB3,FB4,FB5,FB6,ST1,ST2,ST3,ST4,ST5 tool;
    class WT2,WT3,WT4,WT5,FC1,FC2,FC3,FC4,AL1,AL2,AL3,AL4 tool;
    class FM1,FM2,FM3,FM4 tool;
    class InGuard,OutGuard security;
    class Outline,ValidatedFactBase,SynthPlan,SaveCache data;
    class P1Retry,P2Retry,P5Retry,WriterRetry retry;
    class P1Fallback,P2Fallback,P5Fallback,P9Fallback fallback;
    class O1,O2,O3,O4,O5,O6 obs;
    class Done,Cached done;
    class PP1,PP2,PP3,PP4,QP1,QP2,QP3,QP4,SP_A,SP_B,SP_C plan;
```

### Full Graph (`graph/graph.py`)

```python
from langgraph.graph import StateGraph, END
from graph.state import PipelineState
from graph.nodes import *
from graph.edges import *

def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    # Register all nodes
    g.add_node("input_guardrails",      input_guardrails_node)
    g.add_node("cache_check",           cache_check_node)
    g.add_node("planner",               planner_node)
    g.add_node("planner_escalation",    planner_escalation_node)
    g.add_node("question_generator",    question_generator_node)
    g.add_node("hybrid_search",         hybrid_search_node)
    g.add_node("reranker",              reranker_node)
    g.add_node("evaluator",             evaluator_node)
    g.add_node("evaluator_escalation",  evaluator_escalation_node)
    g.add_node("fact_base_builder",     fact_base_builder_node)
    g.add_node("synthesizer",           synthesizer_node)
    g.add_node("writer",                writer_node)
    g.add_node("qa_broadcast",          qa_broadcast_node)
    g.add_node("aggregator",            aggregator_node)
    g.add_node("output_guardrails",     output_guardrails_node)
    g.add_node("writer_escalation",     writer_escalation_node)
    g.add_node("save_cache",            save_cache_node)

    # Entry point
    g.set_entry_point("input_guardrails")

    # Static edges
    g.add_edge("input_guardrails", "cache_check")
    g.add_edge("question_generator", "hybrid_search")
    g.add_edge("hybrid_search", "reranker")
    g.add_edge("reranker", "evaluator")
    g.add_edge("fact_base_builder", "synthesizer")
    g.add_edge("synthesizer", "writer")
    g.add_edge("writer", "qa_broadcast")
    g.add_edge("qa_broadcast", "aggregator")
    g.add_edge("save_cache", END)

    # Conditional edges
    g.add_conditional_edges("cache_check",      cache_check_edge)
    g.add_conditional_edges("planner",          planner_retry_edge)
    g.add_conditional_edges("evaluator",        evaluator_retry_edge)
    g.add_conditional_edges("fact_base_builder",gap_analyzer_edge)
    g.add_conditional_edges("aggregator",       writer_qa_edge)
    g.add_conditional_edges("output_guardrails",human_final_edge)

    return g.compile()
```

---

## 21. Configuration & Environment Variables

All configuration is managed via Pydantic `BaseSettings` in `config/settings.py`. Values are loaded from environment variables, with AWS SSM Parameter Store as the secrets backend in production.

```bash
# LLM
ANTHROPIC_API_KEY=
OPENAI_API_KEY=                  # fallback

# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
S3_MAIN_BUCKET=my-bucket
S3_AUDIT_BUCKET=my-audit-bucket
S3_REPORTS_BUCKET=my-reports-bucket
SQS_HUMAN_ESCALATION_QUEUE=
SNS_DELIVERY_TOPIC=

# Databases
OPENSEARCH_ENDPOINT=
VECTOR_DB_BACKEND=pinecone        # faiss | pinecone | weaviate
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=
POSTGRES_DSN=                     # semantic cache

# MCP Servers
MCP_SERVER_ENDPOINTS='{
  "knowledge-mcp": "http://knowledge-mcp-svc:8080",
  "routing-mcp":   "http://routing-mcp-svc:8080",
  "nlp-mcp":       "http://nlp-mcp-svc:8080",
  "credibility-mcp":"http://credibility-mcp-svc:8080",
  "misinfo-mcp":   "http://misinfo-mcp-svc:8080",
  "security-mcp":  "http://security-mcp-svc:8080"
}'

# Pipeline Thresholds
MIN_COMPOSITE_SCORE=0.35
MIN_FAITHFULNESS_SCORE=0.80
SEMANTIC_CACHE_THRESHOLD=0.92
MAX_RETRIES=3
CRITIC_TIMEOUT_SECONDS=120
MAX_COST_PER_SESSION_USD=5.00

# Observability
OTLP_ENDPOINT=http://otel-collector:4317
PROMETHEUS_PORT=9090
```

---

## 22. Installation & Local Development

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- AWS CLI configured
- `kubectl` + `helm` (for EKS deployment)

### Local Setup

```bash
git clone https://github.com/your-org/enterprise-agentic-report.git
cd enterprise-agentic-report

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy and fill environment variables
cp .env.example .env
# Edit .env with your credentials

# Start local dependencies (Redis, Postgres, OpenSearch, mock MCP servers)
docker-compose up -d

# Run database migrations (semantic cache schema)
python -m alembic upgrade head

# Download spaCy model
python -m spacy download en_core_web_trf

# Verify setup
python -m pytest tests/unit/ -v
```

### Running Locally with LangGraph Studio

```bash
pip install langgraph-cli
langgraph dev --graph graph.graph:build_graph
```

This starts the LangGraph Studio UI at `http://localhost:8123`, where you can visualize the full graph, step through nodes, inspect state at each checkpoint, and replay failed runs.

---

## 23. Running the Pipeline

### CLI

```bash
python -m pipeline.run \
  --topic "Global semiconductor supply chain risks 2024" \
  --constraints '{"max_words": 5000, "audience": "executive", "format": "pdf"}' \
  --session-id "my-session-001"
```

### Python API

```python
from graph.graph import build_graph
from graph.state import PipelineState

graph = build_graph()

initial_state: PipelineState = {
    "session_id": "my-session-001",
    "user_brief": {
        "topic": "Global semiconductor supply chain risks 2024",
        "max_words": 5000,
        "audience": "executive",
        "output_format": "pdf"
    },
    # All other fields default to None / 0 / False
    "planner_retries": 0,
    "evaluator_retries": 0,
    "gap_retries": 0,
    "writer_retries": 0,
    "scope_approved": False,
    "sources_validated": False,
    "final_approved": False,
    "planner_escalated": False,
    "evaluator_escalated": False,
    "writer_escalated": False,
}

for event in graph.stream(initial_state, stream_mode="updates"):
    print(event)
```

---

## 24. Testing Strategy

### Unit Tests (`tests/unit/`)

Each tool function is unit-tested in isolation with mocked external dependencies. Fixture files in `tests/fixtures/` provide sample search results, claim extracts, and fact bases for deterministic testing.

### Integration Tests (`tests/integration/`)

Test full phase pipelines against real (but sandboxed) external APIs. Integration tests are tagged `@pytest.mark.integration` and excluded from CI runs by default. Run with:

```bash
pytest tests/integration/ -v -m integration
```

### End-to-End Tests (`tests/e2e/`)

Test full pipeline runs using a curated set of "golden" topics with known expected outputs. RAGAS scores from E2E tests are tracked in a quality dashboard — regressions below thresholds block deployment.

### RAGAS Evaluation Suite

```bash
python -m tests.ragas_eval \
  --dataset tests/e2e/golden_dataset.json \
  --output results/ragas_report.json
```

---

## 25. Security & Compliance

**Network:** All inter-service communication uses mTLS via AWS App Mesh. All S3 traffic uses VPC endpoints (no public internet). All external API calls route through a NAT gateway with egress filtering.

**IAM:** Each EKS pod uses IRSA (IAM Roles for Service Accounts) with least-privilege policies. No agent can write to buckets it doesn't own. The audit bucket policy denies all `DELETE` and `PUT` over existing objects (enforcing WORM semantics alongside S3 Object Lock).

**Secrets:** API keys and database passwords are stored in AWS Secrets Manager and injected as Kubernetes secrets mounted as environment variables. No secrets appear in container images or code.

**Data Residency:** S3 replication is configured for multi-region durability. For EU deployments, a separate `eu-west-1` stack ensures data residency compliance with GDPR.

**RBAC:** Human review gates check the requesting user's IAM group membership against a role matrix before surfacing flagged sources or accepting final approval. Unauthorized approval attempts are logged as `SECURITY_BLOCK` audit events.

---

## 26. Contributing

1. Fork the repository and create a feature branch: `git checkout -b feature/your-feature`
2. Ensure all new tool implementations include: the tool function, a corresponding unit test, an entry in the MCP schema registry if the tool is MCP-backed, and observability instrumentation (OTel span wrapping).
3. Run the full test suite: `pytest tests/unit/ tests/integration/ -v`
4. Run the RAGAS evaluation suite and ensure no regression: `python -m tests.ragas_eval`
5. Submit a pull request with a description of the tool's phase, purpose, inputs, outputs, and any new dependencies.

---

*Architecture diagram source: `enterprise_agentic_workflow_full.mermaid`*