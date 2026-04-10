"""Main Streamlit dashboard for S.A.G.A.

This app is the primary product surface for ingesting books, running the
analysis pipeline, browsing outputs, and exporting the JSON contract.
"""

import json
import math
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from analysis.identity_analyzer import IdentityAnalyzer
from analysis.scene_analyzer import SceneAnalyzer
from analysis.scene_extractor import SceneExtractor
from entities.entity_registry_service import EntityRegistryService
from infrastructure.llm_client import LLMClient
from query.story_query_service import StoryQueryService
from rag.story_index_service import StoryIndexService
from services.series_processor import SeriesProcessor
from state.canon_state_service import CanonStateService
from state.state_transition_service import StateTransitionService
from timeline.character_normalizer import CharacterNormalizer
from timeline.character_timeline_service import CharacterTimelineService
from timeline.causal_graph_service import CausalGraphService
from timeline.timeline_service import TimelineService


UPLOAD_DIR = Path(r"B:\Documents\PyCharm\graduationProject\uploads")
DEFAULT_SCENE_TARGET_WORDS = 900
MODEL_OPTIONS = ["gpt_oss", "deepseek", "mistral", "gemini"]
LIVE_RENDER_INTERVAL_SECONDS = 2.0
EXPORT_CONTRACT_VERSION = "1.0.0"
FORBIDDEN_IDENTITY_LABELS = {
    "i",
    "me",
    "my",
    "myself",
    "he",
    "she",
    "they",
    "them",
    "him",
    "her",
    "his",
    "hers",
    "their",
    "theirs",
    "it",
    "its",
    "narrator",
    "protagonist",
    "person",
    "character",
}
GENERIC_ALIAS_LABELS = {"man", "woman", "boy", "girl", "person", "figure", "voice"}

st.set_page_config(page_title="S.A.G.A.", layout="wide")

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
else:
    logging.getLogger().setLevel(logging.INFO)


def init_state():
    defaults = {
        "book_inputs": [],
        "chapters": [],
        "scene_analyses": [],
        "resolved_scene_analyses": [],
        "entity_registry": [],
        "state_result": {"transitions": [], "latest_state": []},
        "canon_snapshot": [],
        "timeline": [],
        "character_timelines": [],
        "identity_result": {"alias_map": {}, "rejected_non_characters": [], "decisions": [], "alias_history": []},
        "story_index_result": None,
        "causal_graph_result": {"graph": {"events": [], "critical_path": [], "flexible_events": [], "causal_chains": [], "divergence_points": []}, "metrics": {}},
        "analysis_model": "gpt_oss",
        "identity_model": "deepseek",
        "target_scene_words": DEFAULT_SCENE_TARGET_WORDS,
        "book_order_rows": [],
        "pipeline_running": False,
        "latest_status": "Idle",
        "latest_scene_summary": "",
        "current_scene_ref": None,
        "processed_scene_count": 0,
        "estimated_total_scenes": 0,
        "run_started_at": 0.0,
        "elapsed_seconds": 0.0,
        "last_scene_seconds": 0.0,
        "avg_scene_seconds": 0.0,
        "last_live_render_at": 0.0,
        "post_run_refresh_pending": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_pipeline_outputs():
    st.session_state["chapters"] = []
    st.session_state["scene_analyses"] = []
    st.session_state["resolved_scene_analyses"] = []
    st.session_state["entity_registry"] = []
    st.session_state["state_result"] = {"transitions": [], "latest_state": []}
    st.session_state["canon_snapshot"] = []
    st.session_state["timeline"] = []
    st.session_state["character_timelines"] = []
    st.session_state["identity_result"] = {"alias_map": {}, "rejected_non_characters": [], "decisions": [], "alias_history": []}
    st.session_state["story_index_result"] = None
    st.session_state["causal_graph_result"] = {"graph": {"events": [], "critical_path": [], "flexible_events": [], "causal_chains": [], "divergence_points": []}, "metrics": {}}
    st.session_state["pipeline_running"] = False
    st.session_state["latest_status"] = "Idle"
    st.session_state["latest_scene_summary"] = ""
    st.session_state["current_scene_ref"] = None
    st.session_state["processed_scene_count"] = 0
    st.session_state["estimated_total_scenes"] = 0
    st.session_state["run_started_at"] = 0.0
    st.session_state["elapsed_seconds"] = 0.0
    st.session_state["last_scene_seconds"] = 0.0
    st.session_state["avg_scene_seconds"] = 0.0
    st.session_state["last_live_render_at"] = 0.0
    st.session_state["post_run_refresh_pending"] = False


def save_uploaded_books(uploaded_files) -> List[Dict]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded in uploaded_files:
        destination = UPLOAD_DIR / uploaded.name
        destination.write_bytes(uploaded.getbuffer())
        suffix = destination.suffix.lower()
        if suffix not in {".epub", ".pdf"}:
            continue
        saved.append({
            "path": str(destination),
            "type": suffix.lstrip("."),
            "title": uploaded.name,
        })
    return saved


def resolve_book_inputs() -> List[Dict]:
    edited_rows = st.session_state.get("book_order_rows") or []
    if edited_rows:
        ordered = sorted(edited_rows, key=lambda row: int(row["order"]))
        return [
            {"path": row["path"], "type": row["type"], "title": row["title"]}
            for row in ordered
        ]
    return []


def paged_items(items: List[Dict], key_prefix: str, page_size: int = 10) -> List[Dict]:
    if not items:
        return []

    total_pages = max(1, math.ceil(len(items) / page_size))
    page = st.number_input(
        f"{key_prefix} page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key=f"{key_prefix}_page",
    )
    start = (page - 1) * page_size
    end = start + page_size
    st.caption(f"Showing {start + 1}-{min(end, len(items))} of {len(items)}")
    return items[start:end]


def build_chapters(book_inputs: List[Dict], model_mode: str) -> List[Dict]:
    logging.info("Chapter build started | books=%s | model=%s", len(book_inputs), model_mode)
    processor = SeriesProcessor(
        llm_client=LLMClient(
            mode=model_mode,
            max_retries=1,
            base_delay=0.0,
        )
    )
    chapters = processor.process(book_inputs)
    logging.info("Chapter build completed | chapters=%s", len(chapters))
    return chapters


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds or 0.0))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def render_all_throttled(
    placeholders: Dict[str, st.delta_generator.DeltaGenerator],
    compact: bool,
    force: bool = False,
):
    now = time.perf_counter()
    last_render_at = float(st.session_state.get("last_live_render_at") or 0.0)
    if force or (now - last_render_at) >= LIVE_RENDER_INTERVAL_SECONDS:
        render_all(placeholders, compact=compact)
        st.session_state["last_live_render_at"] = now


def normalize_identity_key(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def article_insensitive_key(name: str) -> str:
    normalized = normalize_identity_key(name)
    for prefix in ("the ", "a ", "an "):
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return normalized


def is_forbidden_identity(name: str) -> bool:
    normalized = normalize_identity_key(name)
    return not normalized or len(normalized) <= 1 or normalized in FORBIDDEN_IDENTITY_LABELS


def is_generic_alias(name: str) -> bool:
    return normalize_identity_key(name) in GENERIC_ALIAS_LABELS


def looks_like_proper_name(name: str) -> bool:
    cleaned = (name or "").strip()
    if not cleaned:
        return False

    tokens = [token for token in cleaned.replace("-", " ").split() if token]
    if not tokens:
        return False

    alpha_tokens = []
    for token in tokens:
        letters = "".join(ch for ch in token if ch.isalpha() or ch in {"'", "-"})
        if not letters:
            return False
        alpha_tokens.append(letters)

    if len(alpha_tokens) >= 2:
        return all(token[:1].isupper() and token[1:].islower() for token in alpha_tokens if len(token) > 1)

    token = alpha_tokens[0]
    return len(token) >= 4 and token[:1].isupper() and token[1:].islower()


def canonical_lookup(alias_map: Dict[str, List[str]]) -> Dict[str, str]:
    lookup = {}
    for canonical_name, aliases in alias_map.items():
        lookup[canonical_name.lower()] = canonical_name
        for alias in aliases:
            lookup[alias.lower()] = canonical_name
    return lookup


def resolve_existing_canonical_name(name: str, alias_map: Dict[str, List[str]]) -> str:
    if not name:
        return ""

    normalized = normalize_identity_key(name)
    article_free = article_insensitive_key(name)
    candidates = []

    for canonical_name, aliases in alias_map.items():
        known_names = [canonical_name, *aliases]
        for known_name in known_names:
            if normalize_identity_key(known_name) == normalized:
                return canonical_name
            if article_insensitive_key(known_name) == article_free:
                return canonical_name
            candidates.append((canonical_name, known_name))

    token = normalized
    if " " not in token and len(token) >= 4:
        matches = set()
        for canonical_name, known_name in candidates:
            known_token = normalize_identity_key(known_name)
            if " " in known_token:
                continue
            short, long_name = sorted([token, known_token], key=len)
            if len(long_name) - len(short) >= 2 and long_name.startswith(short):
                matches.add(canonical_name)
        if len(matches) == 1:
            return next(iter(matches))

    return ""


def sanitize_alias_map(alias_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    cleaned = {}
    for canonical_name, aliases in (alias_map or {}).items():
        canonical = (canonical_name or "").strip()
        if is_forbidden_identity(canonical):
            continue

        valid_aliases = {canonical}
        for alias in aliases or []:
            cleaned_alias = (alias or "").strip()
            if not cleaned_alias:
                continue
            if is_forbidden_identity(cleaned_alias):
                continue
            valid_aliases.add(cleaned_alias)

        if valid_aliases:
            cleaned[canonical] = sorted(valid_aliases, key=str.lower)

    return cleaned


def canonicalize_name(name: str, alias_map: Dict[str, List[str]], rejected: List[str]) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        return ""
    if cleaned.lower() in {item.lower() for item in rejected}:
        return ""
    lookup = canonical_lookup(alias_map)
    if cleaned.lower() in lookup:
        return lookup[cleaned.lower()]
    resolved = resolve_existing_canonical_name(cleaned, alias_map)
    return resolved or cleaned


def build_scene_context(scene_text: str, resolved_scene_analyses: List[Dict], state_result: Dict, identity_result: Dict, window: int = 6) -> str:
    parts = []
    alias_map = identity_result.get("alias_map") or {}
    if alias_map:
        parts.append("Known canonical characters: " + ", ".join(sorted(alias_map.keys(), key=str.lower)[:20]))

    recent_summaries = []
    for scene in resolved_scene_analyses[-window:]:
        summary = (scene.get("scene_summary") or "").strip()
        if summary:
            recent_summaries.append(
                f"- Book {scene.get('book_index')} Chapter {scene.get('chapter_index')} Scene {scene.get('scene_index')}: {summary}"
            )
    if recent_summaries:
        parts.append("Recent scene summaries:")
        parts.extend(recent_summaries)

    latest_state = (state_result or {}).get("latest_state") or []
    scene_text_lower = (scene_text or "").lower()
    relevant_state = []
    for item in latest_state:
        entity_name = (item.get("entity_name") or "").strip()
        if not entity_name or entity_name.lower() not in scene_text_lower:
            continue
        attr_text = ", ".join(f"{key}={value}" for key, value in (item.get("attributes") or {}).items())
        if attr_text:
            relevant_state.append(f"- {entity_name}: {attr_text}")
    if relevant_state:
        parts.append("Relevant latest known state:")
        parts.extend(relevant_state[:8])

    return "\n".join(parts).strip()


def resolve_scene_analysis(scene_analysis: Dict, alias_map: Dict[str, List[str]], rejected: List[str]) -> Dict:
    resolved = dict(scene_analysis)
    lookup = canonical_lookup(alias_map)

    valid_character_names = set()
    for character in scene_analysis.get("canonical_characters", []):
        raw_name = (character.get("name") or "").strip()
        canonical = canonicalize_name(raw_name, alias_map, rejected)
        if canonical:
            valid_character_names.add(canonical)
        if raw_name:
            valid_character_names.add(raw_name)

    for mention in scene_analysis.get("character_mentions", []):
        canonical = canonicalize_name(mention.get("canonical_name", ""), alias_map, rejected)
        if canonical:
            valid_character_names.add(canonical)

    resolved_canonicals = []
    seen_canonicals = set()
    for character in scene_analysis.get("canonical_characters", []):
        canonical_name = canonicalize_name(character.get("name", ""), alias_map, rejected)
        if not canonical_name:
            continue
        lowered = canonical_name.lower()
        if lowered in seen_canonicals:
            continue
        seen_canonicals.add(lowered)
        names_used = []
        names_seen = set()
        for alias in character.get("names_used", []):
            cleaned = str(alias).strip()
            if not cleaned:
                continue
            lowered_alias = cleaned.lower()
            if lowered_alias in names_seen:
                continue
            names_seen.add(lowered_alias)
            names_used.append(cleaned)
        if lowered not in names_seen:
            names_used.insert(0, canonical_name)
        resolved_canonicals.append({
            **character,
            "name": canonical_name,
            "names_used": names_used,
        })
    resolved["canonical_characters"] = resolved_canonicals

    resolved_mentions = []
    for mention in scene_analysis.get("character_mentions", []):
        resolved_mentions.append({
            **mention,
            "canonical_name": canonicalize_name(mention.get("canonical_name", ""), alias_map, rejected),
        })
    resolved["character_mentions"] = resolved_mentions

    resolved_events = []
    for event in scene_analysis.get("events", []):
        characters = []
        for character in event.get("characters", []):
            canonical = canonicalize_name(character, alias_map, rejected)
            lowered = (character or "").strip().lower()
            is_known_alias = lowered in lookup
            if canonical and (canonical in valid_character_names or is_known_alias) and canonical not in characters:
                characters.append(canonical)
        resolved_events.append({**event, "characters": characters})
    resolved["events"] = resolved_events

    resolved_entities = []
    seen_entities = set()
    character_entities = [
        {"name": item["name"], "entity_type": "character"}
        for item in resolved_canonicals
    ]
    entity_source = character_entities + list(scene_analysis.get("entities_present", []))
    for entity in entity_source:
        name = canonicalize_name(entity.get("name", ""), alias_map, rejected) if entity.get("entity_type") == "character" else (entity.get("name") or "").strip()
        if not name:
            continue
        key = (name.lower(), entity.get("entity_type"))
        if key in seen_entities:
            continue
        seen_entities.add(key)
        resolved_entities.append({"name": name, "entity_type": entity.get("entity_type")})
    resolved["entities_present"] = resolved_entities

    resolved_descriptions = []
    for item in scene_analysis.get("entity_descriptions", []):
        entity_name = canonicalize_name(item.get("entity_name", ""), alias_map, rejected) if item.get("entity_type") == "character" else item.get("entity_name", "")
        if not entity_name:
            continue
        resolved_descriptions.append({**item, "entity_name": entity_name})
    resolved["entity_descriptions"] = resolved_descriptions

    resolved_state_changes = []
    for item in scene_analysis.get("state_changes", []):
        entity_name = canonicalize_name(item.get("entity_name", ""), alias_map, rejected) if item.get("entity_type") == "character" else item.get("entity_name", "")
        if not entity_name:
            continue
        resolved_state_changes.append({**item, "entity_name": entity_name})
    resolved["state_changes"] = resolved_state_changes

    resolved_relationship_changes = []
    for item in scene_analysis.get("relationship_changes", []):
        source_entity = canonicalize_name(item.get("source_entity", ""), alias_map, rejected)
        target_entity = canonicalize_name(item.get("target_entity", ""), alias_map, rejected)
        if not source_entity or not target_entity:
            continue
        resolved_relationship_changes.append({**item, "source_entity": source_entity, "target_entity": target_entity})
    resolved["relationship_changes"] = resolved_relationship_changes

    return resolved


def rebuild_resolved_scene_analyses(scene_analyses: List[Dict], identity_result: Dict) -> List[Dict]:
    alias_map = identity_result.get("alias_map", {})
    rejected = identity_result.get("rejected_non_characters", [])
    return [
        resolve_scene_analysis(scene_analysis, alias_map, rejected)
        for scene_analysis in scene_analyses
    ]


def apply_identity_updates(scene_analysis: Dict, alias_result: Dict):
    alias_map = alias_result["alias_map"]
    rejected = alias_result["rejected_non_characters"]
    decisions = alias_result["decisions"]
    alias_history = alias_result["alias_history"]

    scene_ref = {
        "book_index": scene_analysis.get("book_index"),
        "chapter_index": scene_analysis.get("chapter_index"),
        "scene_index": scene_analysis.get("scene_index"),
    }

    rejected_lower = {item.lower() for item in rejected}
    for name in scene_analysis.get("rejected_identity_candidates", []):
        if not name or not name.strip():
            continue
        if looks_like_proper_name(name):
            alias_map.setdefault(name, [name])
            known_canonicals = set(alias_map.keys())
            known_canonicals.add(name)
            decisions.append({
                "decision_type": "inline_name_promoted",
                "character": name,
                "canonical_name": name,
                "same_character": True,
                "confidence": 1.0,
                "reasoning": "Promoted from rejection list because it matches a proper-name pattern.",
                "scene_ref": scene_ref,
            })
            alias_history.append({
                "canonical_name": name,
                "alias_name": name,
                "scene_ref": scene_ref,
            })
            continue
        if name.lower() not in rejected_lower:
            rejected.append(name)
            rejected_lower.add(name.lower())
            decisions.append({
                "decision_type": "inline_rejection",
                "character": name,
                "same_character": False,
                "confidence": 1.0,
                "reasoning": "Rejected during scene analysis as clearly non-character or incidental.",
                "scene_ref": scene_ref,
            })

    known_canonicals = set(alias_map.keys())
    for character in scene_analysis.get("canonical_characters", []):
        canonical_name = (character.get("name") or "").strip()
        if not canonical_name or is_forbidden_identity(canonical_name):
            continue
        alias_map.setdefault(canonical_name, [canonical_name])
        merged = {
            alias
            for alias in {canonical_name, *character.get("names_used", [])}
            if alias and not is_forbidden_identity(alias)
        }
        alias_map[canonical_name] = sorted(merged, key=str.lower)
        known_canonicals.add(canonical_name)

    for mention in scene_analysis.get("character_mentions", []):
        alias = (mention.get("mention_text") or "").strip()
        canonical_name = (mention.get("canonical_name") or "").strip()
        if not alias or not canonical_name or not mention.get("is_consequential_character", False):
            continue
        if is_forbidden_identity(alias) or alias.lower() in rejected_lower:
            continue
        resolved_canonical = resolve_existing_canonical_name(canonical_name, alias_map) or canonical_name
        alias_map.setdefault(resolved_canonical, [resolved_canonical])
        alias_map[resolved_canonical] = sorted(
            {resolved_canonical, alias, *alias_map[resolved_canonical]},
            key=str.lower,
        )
        known_canonicals.add(resolved_canonical)

    for update in scene_analysis.get("alias_updates", []):
        alias = update["alias"].strip()
        canonical_name = update["canonical_name"].strip()
        action = update["action"]

        if not alias or not canonical_name:
            continue
        if is_forbidden_identity(alias) or is_forbidden_identity(canonical_name):
            if alias.lower() not in rejected_lower:
                rejected.append(alias)
                rejected_lower.add(alias.lower())
            continue
        if alias.lower() in rejected_lower:
            continue

        resolved_canonical = (
            resolve_existing_canonical_name(canonical_name, alias_map)
            or resolve_existing_canonical_name(alias, alias_map)
            or canonical_name
        )

        if action == "new_canonical" and resolved_canonical != canonical_name:
            action = "map_alias"

        alias_map.setdefault(resolved_canonical, [resolved_canonical])
        merged = {resolved_canonical, alias, *alias_map[resolved_canonical]}
        alias_map[resolved_canonical] = sorted(merged, key=str.lower)
        known_canonicals.add(resolved_canonical)

        decisions.append({
            "decision_type": "inline_alias_update",
            "character": alias,
            "canonical_name": resolved_canonical,
            "same_character": True,
            "confidence": 1.0,
            "reasoning": update["reasoning"],
            "scene_ref": scene_ref,
        })
        alias_history.append({
            "canonical_name": resolved_canonical,
            "alias_name": alias,
            "scene_ref": scene_ref,
        })

    alias_result["alias_map"] = sanitize_alias_map(alias_map)


def build_entity_registry(scene_analyses: List[Dict]) -> List[Dict]:
    return EntityRegistryService().build(scene_analyses)


def build_state_result(scene_analyses: List[Dict]) -> Dict:
    return StateTransitionService().build(scene_analyses)


def build_canon_snapshot(state_result: Dict, scene_ref: Tuple[int, int, int]) -> List[Dict]:
    return CanonStateService().snapshot_at(state_result.get("transitions", []), scene_ref=scene_ref)


def build_timeline(scene_analyses: List[Dict]) -> List[Dict]:
    return TimelineService().build_from_scene_analyses(scene_analyses)


def build_character_timelines(timeline: List[Dict]) -> List[Dict]:
    return CharacterTimelineService().build(timeline)


def normalize_character_timelines(character_timelines: List[Dict], identity_result: Dict) -> List[Dict]:
    normalized = CharacterNormalizer().normalize(character_timelines)
    existing_alias_map = identity_result.setdefault("alias_map", {})

    for canonical_name, aliases in normalized.get("alias_map", {}).items():
        merged = set(existing_alias_map.get(canonical_name, []))
        merged.update(aliases)
        merged.add(canonical_name)
        existing_alias_map[canonical_name] = sorted(merged, key=str.lower)

    identity_result["alias_map"] = sanitize_alias_map(existing_alias_map)
    return normalized.get("character_timelines", character_timelines)


def build_story_index(scene_analyses: List[Dict], timeline: List[Dict], character_timelines: List[Dict], entity_registry: List[Dict], canon_snapshot: List[Dict], state_result: Dict, identity_result: Dict) -> Dict:
    service = StoryIndexService()
    result = service.build(
        scene_analyses=scene_analyses,
        timeline=timeline,
        character_timelines=character_timelines,
        entity_registry=entity_registry,
        canon_snapshot=canon_snapshot,
        state_result=state_result,
        identity_result=identity_result,
        causal_graph_result=st.session_state.get("causal_graph_result") or {},
    )
    return {"service": service, "query_service": StoryQueryService(), **result}


def build_export_contract() -> Dict:
    story_index_result = st.session_state.get("story_index_result") or {}
    causal_graph_result = st.session_state.get("causal_graph_result") or {}

    return {
        "contract_version": EXPORT_CONTRACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "app": {
            "name": "S.A.G.A.",
            "pipeline_status": st.session_state.get("latest_status", "Idle"),
        },
        "configuration": {
            "analysis_model": st.session_state.get("analysis_model"),
            "identity_model": st.session_state.get("identity_model"),
            "target_scene_words": st.session_state.get("target_scene_words"),
        },
        "inputs": {
            "books": st.session_state.get("book_inputs") or [],
        },
        "outputs": {
            "chapters": st.session_state.get("chapters") or [],
            "scene_analyses": st.session_state.get("scene_analyses") or [],
            "resolved_scene_analyses": st.session_state.get("resolved_scene_analyses") or [],
            "entity_registry": st.session_state.get("entity_registry") or [],
            "state_result": st.session_state.get("state_result") or {"transitions": [], "latest_state": []},
            "canon_snapshot": st.session_state.get("canon_snapshot") or [],
            "timeline": st.session_state.get("timeline") or [],
            "character_timelines": st.session_state.get("character_timelines") or [],
            "identity_result": st.session_state.get("identity_result") or {"alias_map": {}, "rejected_non_characters": [], "decisions": [], "alias_history": []},
            "causal_graph_result": causal_graph_result,
            "story_index_summary": {
                "document_count": story_index_result.get("document_count", 0),
            },
        },
        "runtime": {
            "elapsed_seconds": st.session_state.get("elapsed_seconds", 0.0),
            "last_scene_seconds": st.session_state.get("last_scene_seconds", 0.0),
            "avg_scene_seconds": st.session_state.get("avg_scene_seconds", 0.0),
            "processed_scene_count": st.session_state.get("processed_scene_count", 0),
            "estimated_total_scenes": st.session_state.get("estimated_total_scenes", 0),
        },
    }


def export_contract_json() -> str:
    return json.dumps(build_export_contract(), ensure_ascii=False, indent=2)


def has_exportable_outputs() -> bool:
    return bool(
        st.session_state.get("chapters")
        or st.session_state.get("scene_analyses")
        or st.session_state.get("timeline")
        or st.session_state.get("entity_registry")
    )


def build_causal_graph(timeline: List[Dict], scene_analyses: List[Dict], model_mode: str) -> Dict:
    logging.info(
        "Causal graph build started | timeline_rows=%s | scenes=%s | model=%s",
        len(timeline),
        len(scene_analyses),
        model_mode,
    )
    service = CausalGraphService(
        llm_client=LLMClient(mode=model_mode, max_retries=2, base_delay=0.0, timeout=120),
        batch_size=20,
    )
    result = service.build(timeline, scene_analyses)
    graph = result.get("graph", {})
    logging.info(
        "Causal graph build completed | events=%s | warning=%s | error=%s",
        len(graph.get("events", [])),
        graph.get("warning", ""),
        graph.get("error", ""),
    )
    return result


def merge_scene_outputs(scene: Dict, content_result: Dict, identity_result: Dict, elapsed_seconds: float) -> Dict:
    merged = dict(content_result)
    merged["canonical_characters"] = identity_result.get("canonical_characters", [])
    merged["character_mentions"] = identity_result.get("character_mentions", [])
    merged["alias_updates"] = identity_result.get("alias_updates", [])
    merged["rejected_identity_candidates"] = identity_result.get("rejected_identity_candidates", [])
    merged["analysis_duration_seconds"] = round(elapsed_seconds, 2)
    merged.setdefault("book_index", scene.get("book_index"))
    merged.setdefault("chapter_index", scene.get("chapter_index"))
    merged.setdefault("scene_index", scene.get("scene_index"))
    merged.setdefault("length", scene.get("length"))
    merged.setdefault("text", scene.get("text", ""))
    return merged


def is_overflow_error(result: Dict) -> bool:
    error_blob = " ".join([str(result.get("error", "")), str(result.get("last_error", ""))]).lower()
    return any(keyword in error_blob for keyword in ["context", "token", "overflow", "length", "too long", "prompt"])


def next_smaller_scene_target(target_words: int) -> int | None:
    if target_words == 0:
        return DEFAULT_SCENE_TARGET_WORDS
    smaller = int(target_words * 0.75)
    if smaller >= target_words:
        smaller = target_words - 100
    if smaller < 180:
        return None
    return smaller


def analyze_scene_with_fallback(
    scene: Dict,
    target_scene_words: int,
    analysis_model: str,
    identity_model: str,
    alias_result: Dict,
    state_result: Dict,
    resolved_scene_analyses: List[Dict],
) -> Tuple[List[Dict], List[int]]:
    current_target = target_scene_words
    attempted_targets = []
    content_client = LLMClient(mode=analysis_model)
    identity_client = LLMClient(mode=identity_model)
    working_scenes = [scene]

    while current_target is not None:
        attempted_targets.append(current_target)
        analyzer = SceneAnalyzer(llm_client=content_client)
        identity_analyzer = IdentityAnalyzer(llm_client=identity_client)
        analyzed = []
        overflow_triggered = False

        for current_scene in working_scenes:
            scene_context = build_scene_context(
                current_scene.get("text", ""),
                resolved_scene_analyses,
                state_result,
                alias_result,
            )
            started_at = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as executor:
                content_future = executor.submit(
                    analyzer.analyze,
                    current_scene,
                    alias_map=alias_result["alias_map"],
                    rejected_identities=alias_result["rejected_non_characters"],
                    scene_context=scene_context,
                )
                identity_future = executor.submit(
                    identity_analyzer.analyze,
                    current_scene,
                    alias_map=alias_result["alias_map"],
                    rejected_identities=alias_result["rejected_non_characters"],
                    scene_context=scene_context,
                )
                content_result = content_future.result()
                identity_result = identity_future.result()
            elapsed_seconds = time.perf_counter() - started_at
            result = merge_scene_outputs(current_scene, content_result, identity_result, elapsed_seconds)
            if result.get("error") and is_overflow_error(result):
                overflow_triggered = True
                break
            analyzed.append(result)

        if not overflow_triggered:
            return analyzed, attempted_targets

        current_target = next_smaller_scene_target(current_target)
        if current_target is not None:
            extractor = SceneExtractor.from_target_words(current_target)
            next_working_scenes = []
            for item in working_scenes:
                next_working_scenes.extend(extractor.split_scene(item, current_target))
            working_scenes = next_working_scenes

    fallback_error = {
        "book_index": scene["book_index"],
        "chapter_index": scene["chapter_index"],
        "scene_index": 1,
        "length": len(scene.get("text", "").split()),
        "text": scene.get("text", ""),
        "scene_summary": "",
        "events": [],
        "entities_present": [],
        "entity_descriptions": [],
        "state_changes": [],
        "relationship_changes": [],
        "location": {},
        "time_signals": [],
        "canonical_characters": [],
        "character_mentions": [],
        "alias_updates": [],
        "rejected_identity_candidates": [],
        "error": "context_overflow_unresolved",
        "last_error": "",
        "fallback_targets": attempted_targets,
    }
    return [fallback_error], attempted_targets


def render_books(container, book_inputs: List[Dict]):
    with container.container():
        st.subheader("Selected Books")
        rows = [
            {
                "order": index,
                "title": item.get("title") or Path(item["path"]).name,
                "type": item.get("type", ""),
                "path": item["path"],
            }
            for index, item in enumerate(book_inputs, start=1)
        ]
        st.dataframe(rows, width="stretch")


def render_chapters(container, chapters: List[Dict], compact: bool):
    with container.container():
        st.subheader("Chapters")
        st.write(f"Rows: {len(chapters)}")
        if compact:
            preview = chapters[-3:]
            if preview:
                st.caption("Latest chapters")
                for chapter in preview:
                    st.write(f"Book {chapter['book_index']} | Chapter {chapter['chapter_index']} | {chapter['chapter_title']}")
            return
        items = paged_items(chapters, "chapters", page_size=5)
        for chapter in items:
            with st.expander(f"Book {chapter['book_index']} | Chapter {chapter['chapter_index']} | {chapter['chapter_title']}"):
                st.text(f"Source: {chapter['source_file']}")
                st.code(chapter["content"])


def render_scenes(container, scene_analyses: List[Dict], compact: bool):
    with container.container():
        st.subheader("Scenes")
        st.write(f"Scenes: {len(scene_analyses)}")
        items = scene_analyses[-3:] if compact else paged_items(scene_analyses, "scenes", page_size=6)
        for scene in items:
            chapter_label = f"Chapter {scene['chapter_index']}"
            if scene.get("end_chapter_index") and scene.get("end_chapter_index") != scene["chapter_index"]:
                chapter_label = f"Chapters {scene['chapter_index']}-{scene['end_chapter_index']}"
            with st.expander(f"Book {scene['book_index']} | {chapter_label} | Scene {scene['scene_index']}"):
                if scene.get("fallback_targets"):
                    st.write(f"Fallback target words tried: {scene.get('fallback_targets')}")
                if scene.get("error"):
                    st.warning(f"Analysis error: {scene['error']} | {scene.get('last_error', '')}")
                st.write(f"Summary: {scene.get('scene_summary') or 'None'}")
                st.write(f"Canonical characters: {scene.get('canonical_characters') or []}")
                st.write(f"Character mentions: {scene.get('character_mentions') or []}")
                st.write(f"Events: {scene.get('events') or []}")
                st.write(f"Alias updates: {scene.get('alias_updates') or []}")
                st.write(f"Rejected identities: {scene.get('rejected_identity_candidates') or []}")
                st.code(scene["text"])


def render_entity_registry(container, entity_registry: List[Dict], compact: bool):
    with container.container():
        st.subheader("Entity Registry")
        st.write(f"Entities: {len(entity_registry)}")
        if not entity_registry:
            st.info("No entity registry entries yet.")
            return
        if compact:
            st.caption("Latest entities")
            for item in entity_registry[-5:]:
                st.write(f"{item['name']} | {item['entity_type']} | mentions={item['mention_count']}")
            return
        items = paged_items(entity_registry, "entity_registry", page_size=8)
        for item in items:
            with st.expander(f"{item['name']} | {item['entity_type']} | mentions={item['mention_count']}"):
                st.write(item)


def render_state_result(container, state_result: Dict, compact: bool):
    with container.container():
        transitions = state_result.get("transitions", [])
        latest_state = state_result.get("latest_state", [])
        st.subheader("State Transitions")
        st.write(f"Transitions: {len(transitions)}")
        if not transitions:
            st.info("No state changes yet.")
        else:
            if compact:
                for item in transitions[-5:]:
                    st.write(f"State {item['state_index']} | {item['entity_name']} | {item['attribute']} -> {item['new_state']}")
            else:
                items = paged_items(transitions, "state_transitions", page_size=8)
                for item in items:
                    with st.expander(f"State {item['state_index']} | {item['entity_name']} | {item['attribute']} -> {item['new_state']}"):
                        st.write(item)
        st.subheader("Latest Known State")
        if not latest_state:
            st.caption("No latest state yet.")
        elif compact:
            for item in latest_state[-5:]:
                st.write(f"{item['entity_name']} | {item['entity_type']}")
        else:
            items = paged_items(latest_state, "latest_state", page_size=8)
            for item in items:
                with st.expander(f"{item['entity_name']} | {item['entity_type']}"):
                    st.write(item["attributes"])


def render_canon_snapshot(container, canon_snapshot: List[Dict], compact: bool):
    with container.container():
        st.subheader("Canon Snapshot")
        st.write(f"Entities in snapshot: {len(canon_snapshot)}")
        if not canon_snapshot:
            st.info("No canon state available up to the current point yet.")
            return
        if compact:
            for item in canon_snapshot[-5:]:
                st.write(f"{item['entity_name']} | {item['entity_type']}")
            return
        items = paged_items(canon_snapshot, "canon_snapshot", page_size=8)
        for item in items:
            with st.expander(f"{item['entity_name']} | {item['entity_type']}"):
                st.write(item["attributes"])


def render_timeline(container, timeline: List[Dict], compact: bool):
    with container.container():
        st.subheader("Timeline")
        st.write(f"Timeline rows: {len(timeline)}")
        if not timeline:
            st.info("No timeline rows yet.")
            return
        if compact:
            for row in timeline[-5:]:
                st.write(f"Time {row['time_index']} | Book {row['book_index']} | Chapter {row['chapter_index']} | Scene {row['scene_index']} | {row['summary']}")
            return
        items = paged_items(timeline, "timeline", page_size=12)
        for row in items:
            with st.expander(f"Time {row['time_index']} | Book {row['book_index']} | Chapter {row['chapter_index']} | Scene {row['scene_index']}"):
                st.write(row)


def render_character_timelines(container, character_timelines: List[Dict], compact: bool):
    with container.container():
        st.subheader("Character Timelines")
        st.write(f"Characters: {len(character_timelines)}")
        if not character_timelines:
            st.info("No character timelines yet.")
            return
        if compact:
            for item in character_timelines[-5:]:
                st.write(f"{item['character']} | {len(item['events'])} events")
            return
        items = paged_items(character_timelines, "character_timelines", page_size=8)
        for item in items:
            with st.expander(f"{item['character']} | {len(item['events'])} events"):
                st.write(item["events"])


def render_alias_map(container, identity_result: Dict, compact: bool):
    with container.container():
        alias_map = identity_result.get("alias_map", {})
        st.subheader("Alias Map")
        st.write(f"Canonical characters with aliases: {len(alias_map)}")
        if not alias_map:
            st.info("No alias decisions yet.")
        else:
            items = [{"canonical_name": name, "aliases": aliases} for name, aliases in alias_map.items()]
            if compact:
                for item in items[-5:]:
                    st.write(f"{item['canonical_name']} | {len(item['aliases'])} aliases")
            else:
                items = paged_items(items, "alias_map", page_size=8)
                for item in items:
                    with st.expander(f"{item['canonical_name']} | {len(item['aliases'])} aliases"):
                        st.write(item["aliases"])
        st.subheader("Rejected Non-Characters")
        rejected = identity_result.get("rejected_non_characters") or []
        if compact:
            st.write(rejected[-5:])
        else:
            st.write(rejected)
        st.subheader("Alias Resolution History")
        history = identity_result.get("alias_history") or []
        st.write(history[-5:] if compact else history)


def render_identity_decisions(container, identity_result: Dict, compact: bool):
    with container.container():
        decisions = identity_result.get("decisions", [])
        st.subheader("Identity Decisions")
        st.write(f"Decisions: {len(decisions)}")
        if not decisions:
            st.info("No identity decisions yet.")
            return
        if compact:
            for item in decisions[-5:]:
                st.write(f"{item.get('decision_type')} | {item.get('character')} | {item.get('canonical_name', '')}")
            return
        items = paged_items(decisions, "identity_decisions", page_size=8)
        for item in items:
            with st.expander(f"{item.get('decision_type')} | {item.get('character')} | {item.get('canonical_name', '')}"):
                st.write(item)


def render_story_search(container, story_index_result: Dict, compact: bool):
    with container.container():
        st.subheader("Story Search")
        if not story_index_result:
            st.info("Story index has not been built yet.")
            return
        if compact or st.session_state.get("pipeline_running"):
            st.info("Search controls will appear after the live run completes. Indexed document count is updating in real time.")
            st.caption(f"Indexed documents so far: {story_index_result.get('document_count', 0)}")
            return
        query = st.text_input(
            "Search the indexed story outputs",
            value="Feyre was going under the mountain to save Tamlin",
            key="story_search_query",
        )
        min_similarity = st.slider(
            "Story search minimum similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            key="story_search_min_similarity",
        )
        max_results = st.slider(
            "Story search max results",
            min_value=1,
            max_value=12,
            value=8,
            step=1,
            key="story_search_max_results",
        )
        results = story_index_result["query_service"].search(story_index_result["service"], query, min_similarity=min_similarity, max_results=max_results)
        st.caption(f"Indexed documents: {story_index_result.get('document_count', 0)} | Matches: {len(results)}")
        for item in results:
            meta = item["metadata"]
            title = f"{item['item_type']} | score={item['score']:.3f}"
            if meta.get("book_index") is not None:
                title += f" | Book {meta.get('book_index')} Chapter {meta.get('chapter_index')} Scene {meta.get('scene_index')}"
            with st.expander(title):
                st.write(f"Summary: {item['summary']}")
                st.write(f"Scene reference: {item['scene_ref']}")
                st.write(f"Metadata: {meta}")
                st.code(item["text"])


def render_causal_graph(container, causal_graph_result: Dict, compact: bool):
    with container.container():
        graph = (causal_graph_result or {}).get("graph", {})
        events = graph.get("events", [])
        st.subheader("Causal Graph")
        st.write(f"Causal events: {len(events)}")
        if graph.get("error"):
            st.warning(f"Causal graph error: {graph.get('error')} | {graph.get('last_error', '')}")
        if not events:
            st.info("No causal graph events yet.")
            return
        if compact:
            for item in events[-5:]:
                st.write(f"{item['id']} | time {item.get('time_index')} | {item.get('event_type')}")
            return
        items = paged_items(events, "causal_graph_events", page_size=12)
        for item in items:
            with st.expander(f"{item['id']} | time {item.get('time_index')} | {item.get('event_type')}"):
                st.write(item)


def render_causal_metrics(container, causal_graph_result: Dict):
    with container.container():
        metrics = (causal_graph_result or {}).get("metrics", {})
        graph = (causal_graph_result or {}).get("graph", {})
        st.subheader("Causal Metrics")
        if not metrics:
            st.info("No causal metrics yet.")
            return
        cols = st.columns(6)
        cols[0].metric("Events", metrics.get("total_events", 0))
        cols[1].metric("Links", metrics.get("total_links", 0))
        cols[2].metric("Avg Links/Event", metrics.get("avg_links_per_event", 0))
        cols[3].metric("Critical Path", metrics.get("critical_path_length", 0))
        cols[4].metric("Chains", metrics.get("causal_chain_count", 0))
        cols[5].metric("Divergence", metrics.get("divergence_count", 0))

        st.subheader("Critical Path")
        st.write(graph.get("critical_path") or [])
        st.subheader("Flexible Events")
        st.write(graph.get("flexible_events") or [])
        st.subheader("Causal Chains")
        st.write(graph.get("causal_chains") or [])
        st.subheader("Divergence Points")
        st.write(graph.get("divergence_points") or [])


def render_status(container, compact: bool):
    with container.container():
        st.subheader("Run Status")
        st.markdown(f"**{st.session_state.get('latest_status', 'Idle')}**")

        metric_cols = st.columns(6)
        metric_cols[0].metric("Chapters", len(st.session_state.get("chapters") or []))
        processed = int(st.session_state.get("processed_scene_count") or 0)
        total = int(st.session_state.get("estimated_total_scenes") or 0)
        scene_label = f"{processed} / {total}" if total else str(processed)
        metric_cols[1].metric("Scenes", scene_label)
        metric_cols[2].metric("Aliases", len((st.session_state.get("identity_result") or {}).get("alias_map", {})))
        metric_cols[3].metric("Indexed Docs", int((st.session_state.get("story_index_result") or {}).get("document_count", 0)))
        metric_cols[4].metric("Elapsed", format_duration(float(st.session_state.get("elapsed_seconds") or 0.0)))
        metric_cols[5].metric("Last Scene", format_duration(float(st.session_state.get("last_scene_seconds") or 0.0)))

        info_cols = st.columns(2)
        with info_cols[0]:
            st.caption("Current Scene")
            st.write(st.session_state.get("current_scene_ref") or "Not started")
        with info_cols[1]:
            st.caption("Latest Scene Summary")
            st.write(st.session_state.get("latest_scene_summary") or "No scene analyzed yet")

        st.caption(f"Average scene analysis time: {format_duration(float(st.session_state.get('avg_scene_seconds') or 0.0))}")

        if compact:
            st.caption("Live mode: deterministic downstream modules are rebuilt after each analyzed scene.")


def render_all(placeholders: Dict[str, st.delta_generator.DeltaGenerator], compact: bool):
    render_status(placeholders["status"], compact)
    render_books(placeholders["books"], st.session_state.get("book_inputs") or [])
    render_chapters(placeholders["chapters"], st.session_state.get("chapters") or [], compact)
    render_scenes(placeholders["scenes"], st.session_state.get("scene_analyses") or [], compact)
    render_entity_registry(placeholders["entities"], st.session_state.get("entity_registry") or [], compact)
    render_state_result(placeholders["state"], st.session_state.get("state_result") or {"transitions": [], "latest_state": []}, compact)
    render_canon_snapshot(placeholders["snapshot"], st.session_state.get("canon_snapshot") or [], compact)
    render_timeline(placeholders["timeline"], st.session_state.get("timeline") or [], compact)
    render_character_timelines(placeholders["characters"], st.session_state.get("character_timelines") or [], compact)
    render_alias_map(placeholders["aliases"], st.session_state.get("identity_result") or {"alias_map": {}, "rejected_non_characters": [], "alias_history": []}, compact)
    render_identity_decisions(placeholders["decisions"], st.session_state.get("identity_result") or {"decisions": []}, compact)
    render_causal_graph(placeholders["causal_graph"], st.session_state.get("causal_graph_result") or {"graph": {"events": []}}, compact)
    render_causal_metrics(placeholders["causal_metrics"], st.session_state.get("causal_graph_result") or {"metrics": {}})
    render_story_search(placeholders["search"], st.session_state.get("story_index_result"), compact)


init_state()
if st.session_state.get("post_run_refresh_pending"):
    st.session_state["post_run_refresh_pending"] = False
st.title("S.A.G.A.")
st.caption("Story Analysis, Generation, and Archives")
st.caption("Run the full one-pass pipeline with live downstream updates after each analyzed scene.")

status_tab, books_tab, chapters_tab, scenes_tab, entities_tab, state_tab, snapshot_tab, timeline_tab, characters_tab, aliases_tab, decisions_tab, causal_graph_tab, causal_metrics_tab, search_tab = st.tabs(
    ["Status", "Books", "Chapters", "Scenes", "Entity Registry", "State Transitions", "Canon Snapshot", "Timeline", "Character Timelines", "Alias Map", "Identity Decisions", "Causal Graph", "Causal Metrics", "Story Search"]
)
placeholders = {
    "status": status_tab.empty(),
    "books": books_tab.empty(),
    "chapters": chapters_tab.empty(),
    "scenes": scenes_tab.empty(),
    "entities": entities_tab.empty(),
    "state": state_tab.empty(),
    "snapshot": snapshot_tab.empty(),
    "timeline": timeline_tab.empty(),
    "characters": characters_tab.empty(),
    "aliases": aliases_tab.empty(),
    "decisions": decisions_tab.empty(),
    "causal_graph": causal_graph_tab.empty(),
    "causal_metrics": causal_metrics_tab.empty(),
    "search": search_tab.empty(),
}

with st.sidebar:
    st.header("Controls")
    if st.session_state.get("book_order_editor") is None and "book_order_editor" in st.session_state:
        del st.session_state["book_order_editor"]
    uploaded_files = st.file_uploader("Upload one or more books", type=["epub", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        uploaded_books = save_uploaded_books(uploaded_files)
        if uploaded_books:
            order_rows = [
                {"order": index, "title": item.get("title") or Path(item["path"]).name, "type": item["type"], "path": item["path"]}
                for index, item in enumerate(uploaded_books, start=1)
            ]
            edited_rows = st.data_editor(order_rows, num_rows="fixed", width="stretch", key="book_order_editor")
            st.session_state["book_order_rows"] = edited_rows
        else:
            st.session_state["book_order_rows"] = []
            st.error("The uploaded files are not supported. Please upload EPUB or PDF books.")
    else:
        st.session_state["book_order_rows"] = []
        st.info("Upload one or more EPUB or PDF books to begin.")

    st.selectbox("Scene analysis model", MODEL_OPTIONS, key="analysis_model")
    st.selectbox("Identity model", MODEL_OPTIONS, key="identity_model")
    st.slider("Target scene size (words)", min_value=0, max_value=5000, key="target_scene_words")
    if st.session_state["target_scene_words"] == 0:
        st.caption("Scene size 0 means one full chapter per scene.")
    else:
        st.caption("Chunks can span chapter boundaries when the target size is larger than a single chapter.")

    run_clicked = st.button("Run Pipeline", width="stretch")
    reset_clicked = st.button("Reset Results", width="stretch")

    export_ready = has_exportable_outputs() and not st.session_state.get("pipeline_running")
    st.download_button(
        label="Export JSON Contract",
        data=export_contract_json() if has_exportable_outputs() else "{}",
        file_name="saga_contract.json",
        mime="application/json",
        width="stretch",
        key="sidebar_export_json_contract",
        disabled=not export_ready,
    )
    if not has_exportable_outputs():
        st.caption("Run the pipeline to enable JSON export.")
    elif st.session_state.get("pipeline_running"):
        st.caption("JSON export will be enabled when the current run finishes.")

    if reset_clicked:
        reset_pipeline_outputs()

if reset_clicked:
    render_all(placeholders, compact=False)
else:
    st.session_state["book_inputs"] = resolve_book_inputs()
    render_all(placeholders, compact=False)

if run_clicked:
    reset_pipeline_outputs()
    st.session_state["pipeline_running"] = True
    st.session_state["book_inputs"] = resolve_book_inputs()
    if not st.session_state["book_inputs"]:
        st.session_state["pipeline_running"] = False
        st.session_state["latest_status"] = "No books selected."
        st.error("No books selected. Upload one or more EPUB or PDF files before running the pipeline.")
        render_all(placeholders, compact=False)
        st.stop()
    st.session_state["run_started_at"] = time.perf_counter()
    st.session_state["latest_status"] = "Loading chapters..."
    render_all(placeholders, compact=True)
    st.session_state["last_live_render_at"] = time.perf_counter()

    chapters = build_chapters(st.session_state["book_inputs"], st.session_state["analysis_model"])
    st.session_state["chapters"] = chapters
    base_extractor = SceneExtractor.from_target_words(st.session_state["target_scene_words"])
    planned_scenes = base_extractor.extract_many(chapters, allow_cross_chapter=True)
    st.session_state["estimated_total_scenes"] = len(planned_scenes)
    render_all(placeholders, compact=True)
    st.session_state["last_live_render_at"] = time.perf_counter()

    progress = st.sidebar.progress(0.0)
    total_scenes = len(planned_scenes)

    for scene_position, planned_scene in enumerate(planned_scenes, start=1):
        chapter_label = f"Chapter {planned_scene['chapter_index']}"
        if planned_scene.get("end_chapter_index") and planned_scene["end_chapter_index"] != planned_scene["chapter_index"]:
            chapter_label = f"Chapters {planned_scene['chapter_index']}-{planned_scene['end_chapter_index']}"
        st.session_state["latest_status"] = f"Processing scene {scene_position}/{total_scenes}: {chapter_label}"
        logging.info(
            "Scene analysis started | position=%s/%s | book=%s | chapter=%s | end_chapter=%s | scene=%s | target_words=%s",
            scene_position,
            total_scenes,
            planned_scene["book_index"],
            planned_scene["chapter_index"],
            planned_scene.get("end_chapter_index", planned_scene["chapter_index"]),
            planned_scene["scene_index"],
            st.session_state["target_scene_words"],
        )
        analyzed_scenes, attempted_targets = analyze_scene_with_fallback(
            planned_scene,
            st.session_state["target_scene_words"],
            st.session_state["analysis_model"],
            st.session_state["identity_model"],
            st.session_state["identity_result"],
            st.session_state["state_result"],
            st.session_state["resolved_scene_analyses"],
        )

        for scene_analysis in analyzed_scenes:
            scene_analysis["fallback_targets"] = attempted_targets
            apply_identity_updates(scene_analysis, st.session_state["identity_result"])
            st.session_state["scene_analyses"].append(scene_analysis)
            st.session_state["resolved_scene_analyses"] = rebuild_resolved_scene_analyses(
                st.session_state["scene_analyses"],
                st.session_state["identity_result"],
            )

            st.session_state["entity_registry"] = build_entity_registry(st.session_state["resolved_scene_analyses"])
            st.session_state["state_result"] = build_state_result(st.session_state["resolved_scene_analyses"])
            st.session_state["timeline"] = build_timeline(st.session_state["resolved_scene_analyses"])
            st.session_state["character_timelines"] = build_character_timelines(st.session_state["timeline"])
            st.session_state["character_timelines"] = normalize_character_timelines(
                st.session_state["character_timelines"],
                st.session_state["identity_result"],
            )
            st.session_state["processed_scene_count"] = len(st.session_state["scene_analyses"])
            st.session_state["last_scene_seconds"] = float(scene_analysis.get("analysis_duration_seconds") or 0.0)
            processed = max(st.session_state["processed_scene_count"], 1)
            total_elapsed = time.perf_counter() - float(st.session_state.get("run_started_at") or 0.0)
            st.session_state["elapsed_seconds"] = round(total_elapsed, 2)
            st.session_state["avg_scene_seconds"] = round(total_elapsed / processed, 2)
            current_chapter_label = f"Chapter {scene_analysis['chapter_index']}"
            if scene_analysis.get("end_chapter_index") and scene_analysis["end_chapter_index"] != scene_analysis["chapter_index"]:
                current_chapter_label = f"Chapters {scene_analysis['chapter_index']}-{scene_analysis['end_chapter_index']}"
            st.session_state["current_scene_ref"] = f"Book {scene_analysis['book_index']} | {current_chapter_label} | Scene {scene_analysis['scene_index']}"
            st.session_state["latest_scene_summary"] = scene_analysis.get("scene_summary") or "No summary"
            st.session_state["canon_snapshot"] = build_canon_snapshot(
                st.session_state["state_result"],
                (scene_analysis["book_index"], scene_analysis["chapter_index"], scene_analysis["scene_index"]),
            )
            st.session_state["story_index_result"] = build_story_index(
                st.session_state["resolved_scene_analyses"],
                st.session_state["timeline"],
                st.session_state["character_timelines"],
                st.session_state["entity_registry"],
                st.session_state["canon_snapshot"],
                st.session_state["state_result"],
                st.session_state["identity_result"],
            )
            render_all_throttled(placeholders, compact=True)

        logging.info(
            "Scene analysis completed | book=%s | chapter=%s | end_chapter=%s | produced=%s | attempted_targets=%s",
            planned_scene["book_index"],
            planned_scene["chapter_index"],
            planned_scene.get("end_chapter_index", planned_scene["chapter_index"]),
            len(analyzed_scenes),
            attempted_targets,
        )
        if len(analyzed_scenes) != 1:
            st.session_state["estimated_total_scenes"] += len(analyzed_scenes) - 1
        progress.progress(scene_position / total_scenes if total_scenes else 1.0)
        render_all_throttled(placeholders, compact=True, force=True)

    st.session_state["latest_status"] = "Building causal graph..."
    render_all(placeholders, compact=True)
    st.session_state["last_live_render_at"] = time.perf_counter()
    st.session_state["causal_graph_result"] = build_causal_graph(
        st.session_state["timeline"],
        st.session_state["resolved_scene_analyses"],
        st.session_state["analysis_model"],
    )
    st.session_state["story_index_result"] = build_story_index(
        st.session_state["resolved_scene_analyses"],
        st.session_state["timeline"],
        st.session_state["character_timelines"],
        st.session_state["entity_registry"],
        st.session_state["canon_snapshot"],
        st.session_state["state_result"],
        st.session_state["identity_result"],
    )
    progress.progress(1.0)
    st.session_state["pipeline_running"] = False
    total_elapsed = time.perf_counter() - float(st.session_state.get("run_started_at") or 0.0)
    st.session_state["elapsed_seconds"] = round(total_elapsed, 2)
    graph = (st.session_state.get("causal_graph_result") or {}).get("graph", {})
    if graph.get("error"):
        st.session_state["latest_status"] = f"Pipeline completed with causal-graph issue: {graph.get('error')}"
    elif graph.get("warning"):
        st.session_state["latest_status"] = f"Pipeline completed with warning: {graph.get('warning')}"
    else:
        st.session_state["latest_status"] = "Pipeline completed."
    render_all(placeholders, compact=False)
    st.session_state["post_run_refresh_pending"] = True
    st.rerun()
