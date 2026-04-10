"""Deterministic chapter-to-scene chunking utilities."""

import re
from typing import Dict, List


class SceneExtractor:
    """
    Splits chapter text into ordered scene-sized chunks without using an LLM.
    """

    SIZE_PRESETS = {
        0: None,
        1: {"target_min_words": 180, "target_max_words": 280, "min_scene_words": 90},
        2: {"target_min_words": 320, "target_max_words": 480, "min_scene_words": 140},
        3: {"target_min_words": 500, "target_max_words": 800, "min_scene_words": 180},
        4: {"target_min_words": 800, "target_max_words": 1200, "min_scene_words": 260},
    }

    def __init__(
        self,
        target_min_words: int = 500,
        target_max_words: int = 800,
        min_scene_words: int = 180,
        scene_size_level: int = 3,
    ):
        self.target_min_words = target_min_words
        self.target_max_words = target_max_words
        self.min_scene_words = min_scene_words
        self.scene_size_level = scene_size_level

    @classmethod
    def from_size_level(cls, scene_size_level: int) -> "SceneExtractor":
        preset = cls.SIZE_PRESETS.get(scene_size_level)
        if preset is None:
            return cls(scene_size_level=0)

        return cls(
            target_min_words=preset["target_min_words"],
            target_max_words=preset["target_max_words"],
            min_scene_words=preset["min_scene_words"],
            scene_size_level=scene_size_level,
        )

    def extract(self, chapter: Dict) -> List[Dict]:
        paragraphs = self._split_paragraphs(chapter.get("content", ""))
        if not paragraphs:
            return []

        if self.scene_size_level == 0:
            chapter_text = "\n\n".join(paragraphs).strip()
            chapter_length = self._word_count(chapter_text)
            if not chapter_length:
                return []

            return [{
                "book_index": chapter["book_index"],
                "chapter_index": chapter["chapter_index"],
                "scene_index": 1,
                "text": chapter_text,
                "length": chapter_length,
                "scene_size_level": self.scene_size_level,
            }]

        chunks = self._build_chunks(paragraphs)
        chunks = self._merge_small_chunks(chunks)

        scenes = []

        for scene_index, text in enumerate(chunks, start=1):
            length = self._word_count(text)
            if not length:
                continue

            scenes.append({
                "book_index": chapter["book_index"],
                "chapter_index": chapter["chapter_index"],
                "scene_index": scene_index,
                "text": text,
                "length": length,
                "scene_size_level": self.scene_size_level,
            })

        return scenes

    def extract_many(self, chapters: List[Dict]) -> List[Dict]:
        scenes = []

        for chapter in chapters:
            scenes.extend(self.extract(chapter))

        return scenes

    def _split_paragraphs(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        parts = re.split(r"\n+", text)
        paragraphs = [self._clean(paragraph) for paragraph in parts if self._clean(paragraph)]

        merged = []
        for paragraph in paragraphs:
            if merged and self._word_count(paragraph) < 12:
                merged[-1] = f"{merged[-1]} {paragraph}".strip()
            else:
                merged.append(paragraph)

        return merged

    def _build_chunks(self, paragraphs: List[str]) -> List[str]:
        chunks = []
        current = []
        current_words = 0

        for paragraph in paragraphs:
            paragraph_words = self._word_count(paragraph)
            projected_words = current_words + paragraph_words

            if current and current_words >= self.target_min_words and projected_words > self.target_max_words:
                chunks.append("\n\n".join(current))
                current = [paragraph]
                current_words = paragraph_words
                continue

            current.append(paragraph)
            current_words = projected_words

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        if not chunks:
            return []

        merged = []

        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue

            if self._word_count(chunk) < self.min_scene_words:
                merged[-1] = f"{merged[-1]}\n\n{chunk}".strip()
            else:
                merged.append(chunk)

        if len(merged) > 1 and self._word_count(merged[0]) < self.min_scene_words:
            merged[1] = f"{merged[0]}\n\n{merged[1]}".strip()
            merged = merged[1:]

        return merged

    def _word_count(self, text: str) -> int:
        return len((text or "").split())

    def _clean(self, text: str) -> str:
        text = re.sub(r"\r\n|\r", "\n", text or "")
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()
