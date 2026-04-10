from analysis.scene_analyzer import SceneAnalyzer
from infrastructure.llm_client import LLMClient


class StubLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_json(self, prompt: str, strict: bool = False, validator=None):
        self.calls += 1
        response = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if validator and isinstance(response, dict) and "error" not in response and not validator(response):
            return {"error": "validation_failed", "raw_output": response}
        return response


def build_sample_scene():
    return {
        "book_index": 1,
        "chapter_index": 1,
        "scene_index": 1,
        "length": 220,
        "text": (
            "Feyre stalks a doe through the snow. A wolf appears in the thicket. "
            "Feyre kills the wolf with an ash arrow before sunset."
        ),
    }


def test_scene_analyzer_normalizes_and_retries():
    invalid = {
        "scene_summary": "missing required arrays"
    }
    valid = {
        "scene_summary": "Feyre hunts in the winter forest and kills a wolf.",
        "canonical_characters": [
            {
                "name": "Feyre",
                "role": "huntress",
                "is_new_character": True,
                "names_used": ["Feyre", "the huntress"],
            }
        ],
        "character_mentions": [
            {
                "mention_text": "the huntress",
                "mention_type": "title",
                "canonical_name": "Feyre",
                "is_consequential_character": True,
            }
        ],
        "events": [
            {
                "description": "Feyre tracks a doe through the snowy forest.",
                "characters": ["Feyre"],
                "type": "movement",
            },
            {
                "description": "Feyre kills the wolf with an ash arrow.",
                "characters": ["Feyre", "Wolf"],
                "type": "action",
            },
        ],
        "entities_present": [
            {"name": "Feyre", "entity_type": "character"},
            {"name": "Wolf", "entity_type": "creature"},
            {"name": "Feyre", "entity_type": "character"},
        ],
        "entity_descriptions": [
            {
                "entity_name": "Feyre",
                "entity_type": "character",
                "description": "moving quietly through the snow",
                "description_type": "appearance_note",
            }
        ],
        "state_changes": [
            {
                "entity_name": "Wolf",
                "entity_type": "creature",
                "attribute": "status",
                "previous_state": "alive",
                "new_state": "dead",
                "change_type": "physical_state",
                "evidence": "Feyre kills the wolf with an ash arrow.",
            }
        ],
        "relationship_changes": [
            {
                "source_entity": "Feyre",
                "target_entity": "Wolf",
                "relationship": "kills",
                "change": "becomes dead due to the attack",
                "evidence": "Feyre kills the wolf with an ash arrow.",
            }
        ],
        "location": {
            "name": "the forest",
            "entity_type": "location",
            "description": "a snowy winter hunting ground",
        },
        "time_signals": ["before sunset", "winter"],
        "alias_updates": [],
        "rejected_identity_candidates": ["doe"],
    }

    stub = StubLLMClient([invalid, valid])
    analyzer = SceneAnalyzer(llm_client=stub, max_attempts=2)
    result = analyzer.analyze(build_sample_scene())

    assert stub.calls == 2
    assert result["scene_summary"].startswith("Feyre hunts")
    assert result["canonical_characters"][0]["name"] == "Feyre"
    assert result["character_mentions"][0]["canonical_name"] == "Feyre"
    assert len(result["events"]) == 2
    assert result["events"][0]["event_id"] == "evt_1"
    assert len(result["entities_present"]) == 2
    assert result["state_changes"][0]["new_state"] == "dead"
    assert result["location"]["entity_type"] == "location"
    assert result["time_signals"] == ["before sunset", "winter"]
    assert result["rejected_identity_candidates"] == ["doe"]


def test_llm_client_modes():
    deepseek_client = LLMClient(mode="deepseek")
    gpt_oss_client = LLMClient(mode="gpt_oss")
    local_alias_client = LLMClient(mode="local")

    assert deepseek_client.mode == "deepseek"
    assert gpt_oss_client.mode == "gpt_oss"
    assert local_alias_client.mode == "deepseek"
