from typing import Dict, List, Tuple


class EntityRegistryService:
    """
    Aggregates entity mentions, descriptions, and state changes across analyzed scenes.
    """

    def build(self, analyzed_scenes: List[Dict]) -> List[Dict]:
        registry: Dict[Tuple[str, str], Dict] = {}

        for scene in sorted(analyzed_scenes, key=self._scene_key):
            scene_ref = {
                "book_index": scene.get("book_index"),
                "chapter_index": scene.get("chapter_index"),
                "scene_index": scene.get("scene_index"),
            }

            for entity in scene.get("entities_present", []):
                key = (entity["name"].strip().lower(), entity["entity_type"])
                entry = registry.setdefault(key, self._new_entry(entity["name"], entity["entity_type"], scene_ref))
                entry["mention_count"] += 1
                entry["mentions"].append(dict(scene_ref))

            location = scene.get("location") or {}
            if location.get("name") and location.get("entity_type") == "location":
                key = (location["name"].strip().lower(), "location")
                entry = registry.setdefault(key, self._new_entry(location["name"], "location", scene_ref))
                entry["mention_count"] += 1
                entry["mentions"].append(dict(scene_ref))
                if location.get("description"):
                    entry["descriptions"].append({
                        "description": location["description"],
                        "description_type": "appearance_note",
                        **scene_ref,
                    })

            for description in scene.get("entity_descriptions", []):
                key = (description["entity_name"].strip().lower(), description["entity_type"])
                entry = registry.setdefault(
                    key,
                    self._new_entry(description["entity_name"], description["entity_type"], scene_ref),
                )
                entry["descriptions"].append({
                    "description": description["description"],
                    "description_type": description["description_type"],
                    **scene_ref,
                })

            for change in scene.get("state_changes", []):
                key = (change["entity_name"].strip().lower(), change["entity_type"])
                entry = registry.setdefault(
                    key,
                    self._new_entry(change["entity_name"], change["entity_type"], scene_ref),
                )
                entry["state_changes"].append({
                    "attribute": change["attribute"],
                    "previous_state": change["previous_state"],
                    "new_state": change["new_state"],
                    "change_type": change["change_type"],
                    "evidence": change["evidence"],
                    **scene_ref,
                })

        return sorted(registry.values(), key=lambda item: (item["entity_type"], item["name"].lower()))

    def _new_entry(self, name: str, entity_type: str, scene_ref: Dict) -> Dict:
        return {
            "name": name,
            "entity_type": entity_type,
            "first_seen": dict(scene_ref),
            "mention_count": 0,
            "mentions": [],
            "descriptions": [],
            "state_changes": [],
        }

    def _scene_key(self, scene: Dict) -> Tuple[int, int, int]:
        return (
            scene.get("book_index", 0),
            scene.get("chapter_index", 0),
            scene.get("scene_index", 0),
        )
