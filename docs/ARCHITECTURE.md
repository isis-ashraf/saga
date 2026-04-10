# S.A.G.A. Architecture

## High-Level Flow

```text
Books
  -> Chapter extraction
  -> Scene extraction
  -> Scene analysis + identity analysis
  -> Deterministic downstream services
  -> Search / export / dashboard review
```

## Components

### Ingestion

- [services/series_processor.py](/B:/Documents/PyCharm/graduationProject/services/series_processor.py)
- [services/epub_processor.py](/B:/Documents/PyCharm/graduationProject/services/epub_processor.py)
- [services/pdf_processor.py](/B:/Documents/PyCharm/graduationProject/services/pdf_processor.py)

Responsibility:
- normalize one or more books into chapter rows

### Scene Layer

- [analysis/scene_extractor.py](/B:/Documents/PyCharm/graduationProject/analysis/scene_extractor.py)
- [analysis/scene_analyzer.py](/B:/Documents/PyCharm/graduationProject/analysis/scene_analyzer.py)
- [analysis/identity_analyzer.py](/B:/Documents/PyCharm/graduationProject/analysis/identity_analyzer.py)

Responsibility:
- split chapters into scenes
- extract structured scene data
- extract identity/canonical/mention information in parallel

### Entity And State Layer

- [entities/entity_registry_service.py](/B:/Documents/PyCharm/graduationProject/entities/entity_registry_service.py)
- [state/state_transition_service.py](/B:/Documents/PyCharm/graduationProject/state/state_transition_service.py)
- [state/canon_state_service.py](/B:/Documents/PyCharm/graduationProject/state/canon_state_service.py)

Responsibility:
- build tracked entities
- apply state changes in reading order
- reconstruct canon state at a chosen point

### Timeline Layer

- [timeline/timeline_service.py](/B:/Documents/PyCharm/graduationProject/timeline/timeline_service.py)
- [timeline/character_timeline_service.py](/B:/Documents/PyCharm/graduationProject/timeline/character_timeline_service.py)
- [timeline/character_normalizer.py](/B:/Documents/PyCharm/graduationProject/timeline/character_normalizer.py)
- [timeline/causal_graph_service.py](/B:/Documents/PyCharm/graduationProject/timeline/causal_graph_service.py)

Responsibility:
- build ordered story events
- group events by character
- normalize character labels
- infer batched causal links and graph metrics

### Retrieval And Query Layer

- [rag/story_index_service.py](/B:/Documents/PyCharm/graduationProject/rag/story_index_service.py)
- [rag/scene_index_service.py](/B:/Documents/PyCharm/graduationProject/rag/scene_index_service.py)
- [query/story_query_service.py](/B:/Documents/PyCharm/graduationProject/query/story_query_service.py)

Responsibility:
- index outputs for search
- retrieve grounded evidence from structured outputs

### Infrastructure

- [infrastructure/llm_client.py](/B:/Documents/PyCharm/graduationProject/infrastructure/llm_client.py)

Responsibility:
- multi-provider JSON-first generation
- timeout/retry behavior
- Ollama and hosted model routing

### Dashboard

- [story_dashboard.py](/B:/Documents/PyCharm/graduationProject/story_dashboard.py)

Responsibility:
- product UI
- live execution orchestration
- export contract generation
- search and visualization

## Design Notes

- Raw extraction and resolved/grouped outputs are kept separate.
- Scene analysis and identity analysis run side by side.
- Downstream grouping is deterministic.
- Causal graph generation is batched to reduce prompt size and timeout risk.
