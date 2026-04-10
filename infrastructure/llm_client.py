"""Shared JSON-first LLM client used across the production pipeline.

The client hides provider-specific request details and centralizes retry,
timeout, and logging behavior for dashboard and service code.
"""

import json
import logging
import os
import re
import time
from typing import Callable, Optional

import requests
from google import genai
from mistralai import Mistral


class LLMClient:
    """
    Multi-provider JSON-first LLM client.

    Supported modes:
    - deepseek: Ollama-backed DeepSeek model
    - gpt_oss: Ollama-backed GPT-OSS model
    - mistral: hosted Mistral API
    - gemini: hosted Gemini API

    The legacy "local" mode is still accepted as an alias for "deepseek"
    so existing modules keep working while the new architecture is rolled out.
    """

    MODE_LOCAL_ALIAS = "local"
    MODE_DEEPSEEK = "deepseek"
    MODE_GPT_OSS = "gpt_oss"
    MODE_MISTRAL = "mistral"
    MODE_GEMINI = "gemini"

    def __init__(
        self,
        mode: str = MODE_GPT_OSS,
        mistral_model: str = "mistral-large-2512",
        gemini_model: str = "gemini-2.0-flash",
        deepseek_model: str = "deepseek-v3.1:671b-cloud",
        gpt_oss_model: str = "gpt-oss:120b-cloud",
        max_retries: int = 5,
        base_delay: float = 1.0,
        timeout: int = 180,
    ):
        requested_mode = (mode or self.MODE_GPT_OSS).lower()
        self.mode = self._normalize_mode(requested_mode)

        self.mistral_model = mistral_model
        self.gemini_model_name = gemini_model
        self.deepseek_model = deepseek_model
        self.gpt_oss_model = gpt_oss_model

        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if self.mode == self.MODE_MISTRAL:
            if not self.mistral_api_key:
                raise ValueError("MISTRAL_API_KEY not set")
            self.mistral_client = Mistral(api_key=self.mistral_api_key)

        if self.mode == self.MODE_GEMINI:
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set")
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)

        self.ollama_url = "http://localhost:11434/api/generate"
        self.base_delay = max(0.0, float(base_delay))
        self.max_retries = max(1, int(max_retries))
        self.timeout = max(1, int(timeout))
        self.json_failures = 0

        if requested_mode == self.MODE_LOCAL_ALIAS:
            logging.warning("LLMClient mode='local' is deprecated; use 'deepseek' or 'gpt_oss'.")

    def generate_json(self, prompt: str, strict: bool = False, validator: Optional[Callable] = None) -> dict:
        start_time = time.time()

        if strict:
            prompt = self._apply_strict_mode(prompt)

        logging.info("LLM Request | Mode: %s | Prompt chars: %s", self.mode, len(prompt))

        if self.mode in {self.MODE_DEEPSEEK, self.MODE_GPT_OSS}:
            result = self._retry_wrapper(self._generate_json_ollama, prompt)
        elif self.mode == self.MODE_MISTRAL:
            result = self._retry_wrapper(self._generate_json_mistral, prompt)
            if "error" in result:
                logging.warning("Mistral failed; falling back to DeepSeek Ollama mode")
                result = self._retry_wrapper(
                    lambda current_prompt: self._generate_json_ollama(current_prompt, model_name=self.deepseek_model),
                    prompt,
                )
        elif self.mode == self.MODE_GEMINI:
            result = self._retry_wrapper(self._generate_json_gemini, prompt)
            if "error" in result:
                logging.warning("Gemini failed; falling back to DeepSeek Ollama mode")
                result = self._retry_wrapper(
                    lambda current_prompt: self._generate_json_ollama(current_prompt, model_name=self.deepseek_model),
                    prompt,
                )
        else:
            return {"error": "invalid_mode"}

        duration = round(time.time() - start_time, 2)
        logging.info("LLM Response Time: %ss", duration)

        if validator and isinstance(result, dict) and "error" not in result:
            if not validator(result):
                logging.warning("Response failed validation")
                return {"error": "validation_failed", "raw_output": result}

        return result

    def _normalize_mode(self, mode: str) -> str:
        if mode == self.MODE_LOCAL_ALIAS:
            return self.MODE_DEEPSEEK

        valid_modes = {
            self.MODE_DEEPSEEK,
            self.MODE_GPT_OSS,
            self.MODE_MISTRAL,
            self.MODE_GEMINI,
        }
        if mode not in valid_modes:
            raise ValueError(f"Unsupported mode: {mode}")
        return mode

    def _retry_wrapper(self, func, prompt: str) -> dict:
        last_error = "unknown_error"
        for attempt in range(self.max_retries):
            try:
                if self.base_delay:
                    time.sleep(self.base_delay)
                result = func(prompt)

                if isinstance(result, dict) and "error" not in result:
                    return result

                error = result.get("error", "unknown_error") if isinstance(result, dict) else "unknown_error"
                last_error = error
                raise RuntimeError(error)
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                retry_after = self._retry_after_seconds(exc.response)
                if status_code == 429:
                    wait = retry_after if retry_after is not None else min(8 * (attempt + 1), 45)
                    last_error = f"HTTP 429 rate_limited"
                    if attempt >= self.max_retries - 1:
                        logging.warning("Attempt %s hit rate limit (429); no retries remaining", attempt + 1)
                        break
                    logging.warning("Attempt %s hit rate limit (429); retry in %ss", attempt + 1, wait)
                    time.sleep(wait)
                    continue
                last_error = str(exc)
                wait = 2 ** attempt
                if attempt >= self.max_retries - 1:
                    logging.warning("Attempt %s failed: %s; no retries remaining", attempt + 1, exc)
                    break
                logging.warning("Attempt %s failed: %s; retry in %ss", attempt + 1, exc, wait)
                time.sleep(wait)
            except Exception as exc:
                last_error = str(exc)
                wait = 2 ** attempt
                if attempt >= self.max_retries - 1:
                    logging.warning("Attempt %s failed: %s; no retries remaining", attempt + 1, exc)
                    break
                logging.warning("Attempt %s failed: %s; retry in %ss", attempt + 1, exc, wait)
                time.sleep(wait)

        logging.error("All retries failed")
        return {"error": "max_retries_exceeded", "last_error": last_error}

    def _generate_json_mistral(self, prompt: str) -> dict:
        response = self.mistral_client.chat.complete(
            model=self.mistral_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        return self._safe_parse_json(content)

    def _generate_json_gemini(self, prompt: str) -> dict:
        response = self.gemini_client.models.generate_content(
            model=self.gemini_model_name,
            contents=self._apply_strict_mode(prompt),
        )
        content = (response.text or "").strip()
        return self._safe_parse_json(content)

    def _generate_json_ollama(self, prompt: str, model_name: Optional[str] = None) -> dict:
        response = requests.post(
            self.ollama_url,
            json={
                "model": model_name or self._ollama_model_for_mode(),
                "prompt": prompt,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        content = result.get("response", "").strip()
        return self._safe_parse_json(content)

    def _ollama_model_for_mode(self) -> str:
        if self.mode == self.MODE_GPT_OSS:
            return self.gpt_oss_model
        return self.deepseek_model

    def _apply_strict_mode(self, prompt: str) -> str:
        return (
            "Return ONLY valid JSON.\n"
            "NO markdown.\n"
            "NO explanations.\n"
            "NO extra text.\n\n"
            f"{prompt}"
        )

    def _safe_parse_json(self, content: str) -> dict:
        if not content:
            self.json_failures += 1
            return {"error": "empty_response"}

        try:
            return json.loads(content)
        except Exception:
            pass

        cleaned = content.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass

        try:
            return json.loads(cleaned)
        except Exception:
            pass

        logging.warning("JSON parse failed")
        self.json_failures += 1
        return {
            "error": "parse_failed",
            "raw_output": content,
        }

    def _retry_after_seconds(self, response) -> Optional[int]:
        if response is None:
            return None
        header_value = response.headers.get("Retry-After")
        if not header_value:
            return None
        try:
            return max(1, int(header_value))
        except (TypeError, ValueError):
            return None
