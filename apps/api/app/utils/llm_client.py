"""Evijnar Health AI adapter for intelligent data mapping.

This service replaces the previous Claude integration with a deterministic,
domain-specific AI layer that uses:
- local medical knowledge tables
- heuristic normalization rules
- Redis-backed response caching
- optional remote knowledge-base lookups when configured

The ingestion pipeline keeps the same async shape, but the implementation is
fully under Evijnar's control and does not depend on a third-party LLM API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from copy import deepcopy
from typing import Optional, Dict, Any
import aioredis

from app.config import settings

logger = logging.getLogger("evijnar.ai")


class LLMCache:
    """Redis-backed cache for AI responses."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis_url if hasattr(settings, 'redis_url') else "redis://localhost:6379"
        self.redis: Optional[aioredis.Redis] = None
        self.enabled = self.redis_url is not None

    async def connect(self):
        """Initialize Redis connection."""
        if not self.enabled:
            return

        try:
            self.redis = await aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Connected to AI cache")
        except Exception as exc:
            logger.warning(f"Failed to connect to Redis: {exc}. Continuing without cache.")
            self.enabled = False

    async def disconnect(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

    def _get_cache_key(self, prompt: str, model: str) -> str:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        return f"ai:v1:{model}:{prompt_hash}"

    async def get(self, prompt: str, model: str) -> Optional[dict[str, Any]]:
        if not self.enabled or not self.redis:
            return None

        try:
            key = self._get_cache_key(prompt, model)
            cached = await self.redis.get(key)
            if cached:
                logger.debug(f"Cache hit for: {key[:20]}...")
                return json.loads(cached)
        except Exception as exc:
            logger.warning(f"Cache retrieval error: {exc}")

        return None

    async def set(self, prompt: str, model: str, response: dict[str, Any], ttl: int = 86400):
        if not self.enabled or not self.redis:
            return

        try:
            key = self._get_cache_key(prompt, model)
            await self.redis.setex(key, ttl, json.dumps(response))
            logger.debug(f"Cached response: {key[:20]}...")
        except Exception as exc:
            logger.warning(f"Cache write error: {exc}")


class EvijnarHealthAI:
    """Async AI adapter for the Evijnar ingestion pipeline."""

    def __init__(self, cache: Optional[LLMCache] = None):
        self.cache = cache or LLMCache()
        self.model = "evijnar-health-ai-v1"
        self.max_retries = 2
        self.retry_delay = 0.5
        self.usage_stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cached": 0,
            "estimated_cost_usd": 0.0,
        }
        self.kb_url = getattr(settings, "evijnar_ai_kb_url", None)
        self._knowledge_base: dict[str, Any] = {
            "procedures": {
                "27447": {
                    "icd10_code": "M17.11",
                    "icd10_description": "Unilateral primary osteoarthritis, right knee",
                    "clinical_category": "Orthopedics",
                    "complexity_score": 4,
                    "uhi_code": "SURG-1001",
                    "ehds_identifier": "EHDS-ORTHO-27447",
                    "us_median_cost_usd": 45200.0,
                },
                "47562": {
                    "icd10_code": "K80.20",
                    "icd10_description": "Calculus of gallbladder without cholecystitis",
                    "clinical_category": "General Surgery",
                    "complexity_score": 3,
                    "uhi_code": "SURG-2102",
                    "ehds_identifier": "EHDS-GI-47562",
                    "us_median_cost_usd": 18950.0,
                },
                "58571": {
                    "icd10_code": "D25.9",
                    "icd10_description": "Leiomyoma of uterus, unspecified",
                    "clinical_category": "Gynecology",
                    "complexity_score": 4,
                    "uhi_code": "SURG-3307",
                    "ehds_identifier": "EHDS-GYN-58571",
                    "us_median_cost_usd": 27800.0,
                },
            },
            "hospitals": {
                "specialty_keywords": [
                    "clinic",
                    "specialty",
                    "center of excellence",
                    "orthopedic",
                    "cardiac",
                    "oncology",
                    "surgery",
                ],
                "diagnostic_keywords": ["diagnostic", "imaging", "radiology", "scan", "lab"],
                "nursing_keywords": ["rehab", "nursing", "long-term care", "elder", "home"],
            },
        }

    async def initialize(self):
        await self.cache.connect()
        await self._load_remote_knowledge_base()

    async def shutdown(self):
        await self.cache.disconnect()

    async def _load_remote_knowledge_base(self):
        if not self.kb_url:
            return

        try:
            import urllib.request

            def fetch() -> dict[str, Any]:
                with urllib.request.urlopen(self.kb_url, timeout=5) as response:
                    return json.loads(response.read().decode("utf-8"))

            remote = await asyncio.to_thread(fetch)
            if isinstance(remote, dict):
                self._knowledge_base.update(remote)
                logger.info("Loaded remote Evijnar Health AI knowledge base")
        except Exception as exc:
            logger.warning(f"Remote knowledge base unavailable: {exc}. Using local knowledge base.")

    async def call_eh_ai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = "json",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        cache_ttl: int = 86400,
    ) -> dict[str, Any]:
        try:
            cached_response = await self.cache.get(prompt, self.model)
            if cached_response:
                self.usage_stats["total_cached"] += 1
                return cached_response

            parsed = self._route_prompt(prompt, system_prompt or "")
            if response_format == "text":
                result: dict[str, Any] = {"text": json.dumps(parsed, ensure_ascii=False)}
            else:
                result = parsed

            self._update_usage(prompt, result)
            await self.cache.set(prompt, self.model, result, cache_ttl)
            return result
        except Exception as exc:
            logger.error(f"Evijnar Health AI error: {exc}")
            raise

    def _update_usage(self, prompt: str, response: dict[str, Any]) -> None:
        approx_tokens = max(len(prompt) // 4, 1) + max(len(json.dumps(response)) // 4, 1)
        self.usage_stats["total_calls"] += 1
        self.usage_stats["total_tokens"] += approx_tokens
        self.usage_stats["estimated_cost_usd"] += 0.0

    def _route_prompt(self, prompt: str, system_prompt: str) -> dict[str, Any]:
        combined = f"{system_prompt}\n{prompt}".lower()
        if "normalize the following hospital information" in combined:
            return self._map_hospital(prompt)
        if "map the following clinical procedure/service to standard coding" in combined:
            return self._map_procedure(prompt)
        if "map this us cpt code to international medical coding standards" in combined:
            return self._map_normalizer(prompt)
        return {"status": "ok", "message": "Evijnar Health AI processed the request."}

    def _extract_line(self, prompt: str, label: str) -> str:
        match = re.search(rf"^{re.escape(label)}:\s*(.*)$", prompt, re.MULTILINE)
        return match.group(1).strip() if match else ""

    def _map_hospital(self, prompt: str) -> dict[str, Any]:
        name = self._extract_line(prompt, "Name") or "Unnamed Hospital"
        alternate_name = self._extract_line(prompt, "Alternate Name")
        description = self._extract_line(prompt, "Description")
        city = self._extract_line(prompt, "City")
        state = self._extract_line(prompt, "State/Province")
        country = self._extract_line(prompt, "Country")
        phone = self._extract_line(prompt, "Phone")
        email = self._extract_line(prompt, "Email")
        hospital_type_raw = self._extract_line(prompt, "Hospital Type (raw)")
        accreditations = self._extract_line(prompt, "Accreditations")

        normalized_name = alternate_name if alternate_name and alternate_name != "N/A" else name
        normalized_name = normalized_name.replace("  ", " ").strip()
        hospital_type = self._classify_hospital_type(
            normalized_name,
            description,
            hospital_type_raw,
            accreditations,
        )

        postal_code = self._extract_postal_code(prompt)
        confidence = 0.92 if hospital_type != "GENERAL_HOSPITAL" else 0.84

        return {
            "normalized_name": normalized_name,
            "hospital_type": hospital_type,
            "postal_code": postal_code or None,
            "confidence": confidence,
            "reasoning": f"Normalized with Evijnar Health AI based on source metadata for {city}, {state}, {country}",
            "contact": {
                "phone": phone if phone != "N/A" else None,
                "email": email if email != "N/A" else None,
            },
        }

    def _classify_hospital_type(self, name: str, description: str, hospital_type_raw: str, accreditations: str) -> str:
        text = f"{name} {description} {hospital_type_raw} {accreditations}".lower()
        if any(keyword in text for keyword in self._knowledge_base["hospitals"]["diagnostic_keywords"]):
            return "DIAGNOSTIC_CENTER"
        if any(keyword in text for keyword in self._knowledge_base["hospitals"]["nursing_keywords"]):
            return "NURSING_HOME"
        if any(keyword in text for keyword in self._knowledge_base["hospitals"]["specialty_keywords"]):
            return "SPECIALTY_CENTER"
        return "GENERAL_HOSPITAL"

    def _extract_postal_code(self, prompt: str) -> str:
        match = re.search(r"\b\d{4,6}\b", prompt)
        return match.group(0) if match else ""

    def _map_procedure(self, prompt: str) -> dict[str, Any]:
        description = self._extract_line(prompt, "Description")
        code = self._extract_line(prompt, "Code (if provided)")
        price_text = self._extract_line(prompt, "Price")
        success_rate_text = self._extract_line(prompt, "Success Rate")
        complication_text = self._extract_line(prompt, "Complication Rate")

        cpt_code = self._normalize_code(code)
        mapping = deepcopy(self._knowledge_base["procedures"].get(cpt_code, {}))

        if not mapping:
            mapping = self._infer_procedure(description, cpt_code, price_text)

        if success_rate_text and success_rate_text != "Not available%":
            success_rate = self._extract_numeric(success_rate_text, default=95.0)
        else:
            success_rate = 95.0

        complication_rate = self._extract_numeric(complication_text, default=2.0)
        if mapping.get("complexity_score", 5) >= 7:
            complexity = mapping.get("complexity_score", 7)
        else:
            complexity = mapping.get("complexity_score", self._infer_complexity(description, cpt_code))

        return {
            "icd10_code": mapping.get("icd10_code", "Z00.00"),
            "icd10_description": mapping.get("icd10_description", description or "Procedure mapping"),
            "clinical_category": mapping.get("clinical_category", self._infer_category(description)),
            "complexity_score": int(complexity),
            "uhi_code": mapping.get("uhi_code"),
            "ehds_identifier": mapping.get("ehds_identifier"),
            "post_op_monitoring": "High" if complexity >= 7 else "Standard",
            "confidence": 0.94 if cpt_code in self._knowledge_base["procedures"] else 0.78,
            "success_rate": success_rate,
            "complication_rate": complication_rate,
        }

    def _map_normalizer(self, prompt: str) -> dict[str, Any]:
        cpt_code = self._extract_line(prompt, "CPT Code")
        cpt_description = self._extract_line(prompt, "CPT Description")
        mapping = deepcopy(self._knowledge_base["procedures"].get(cpt_code, {}))
        if not mapping:
            mapping = self._infer_procedure(cpt_description, cpt_code, "")

        return {
            "cpt_code": cpt_code,
            "cpt_description": cpt_description,
            "icd10_code": mapping.get("icd10_code", "Z00.00"),
            "icd10_description": mapping.get("icd10_description", cpt_description or "Procedure mapping"),
            "uhi_code": mapping.get("uhi_code"),
            "ehds_identifier": mapping.get("ehds_identifier"),
            "clinical_category": mapping.get("clinical_category", self._infer_category(cpt_description)),
            "complexity_score": int(mapping.get("complexity_score", self._infer_complexity(cpt_description, cpt_code))),
            "us_median_cost_usd": float(mapping.get("us_median_cost_usd", self._estimate_cost(cpt_code, cpt_description))),
            "source": "Evijnar Health AI medical knowledge base",
        }

    def _normalize_code(self, code: str) -> str:
        digits = re.sub(r"\D", "", code or "")
        return digits[:5]

    def _infer_procedure(self, description: str, cpt_code: str, price_text: str) -> dict[str, Any]:
        text = description.lower()
        if any(keyword in text for keyword in ["knee", "orthopedic", "arthroplasty"]):
            return {
                "icd10_code": "M17.11",
                "icd10_description": "Unilateral primary osteoarthritis, right knee",
                "clinical_category": "Orthopedics",
                "complexity_score": 4,
                "uhi_code": "SURG-1001",
                "ehds_identifier": "EHDS-ORTHO-27447",
                "us_median_cost_usd": self._estimate_cost(cpt_code, description),
            }
        if any(keyword in text for keyword in ["gallbladder", "cholecyst", "laparoscopy"]):
            return {
                "icd10_code": "K80.20",
                "icd10_description": "Calculus of gallbladder without cholecystitis",
                "clinical_category": "General Surgery",
                "complexity_score": 3,
                "uhi_code": "SURG-2102",
                "ehds_identifier": "EHDS-GI-47562",
                "us_median_cost_usd": self._estimate_cost(cpt_code, description),
            }
        if any(keyword in text for keyword in ["hysterectomy", "gynecology", "uterus"]):
            return {
                "icd10_code": "D25.9",
                "icd10_description": "Leiomyoma of uterus, unspecified",
                "clinical_category": "Gynecology",
                "complexity_score": 4,
                "uhi_code": "SURG-3307",
                "ehds_identifier": "EHDS-GYN-58571",
                "us_median_cost_usd": self._estimate_cost(cpt_code, description),
            }

        return {
            "icd10_code": "Z00.00",
            "icd10_description": description or "General procedure mapping",
            "clinical_category": self._infer_category(description),
            "complexity_score": self._infer_complexity(description, cpt_code),
            "uhi_code": f"UHI-{cpt_code or 'GEN'}",
            "ehds_identifier": f"EHDS-{cpt_code or 'GEN'}",
            "us_median_cost_usd": self._estimate_cost(cpt_code, description),
        }

    def _infer_category(self, description: str) -> str:
        text = description.lower()
        if any(keyword in text for keyword in ["knee", "joint", "hip", "orthopedic"]):
            return "Orthopedic Surgery"
        if any(keyword in text for keyword in ["heart", "cardio", "vascular"]):
            return "Cardiology"
        if any(keyword in text for keyword in ["brain", "neuro", "spine"]):
            return "Neurosurgery"
        if any(keyword in text for keyword in ["scan", "x-ray", "imaging", "ct", "mri", "lab"]):
            return "Diagnostic Imaging"
        if any(keyword in text for keyword in ["woman", "uterus", "gyn", "obstetric", "hysterectomy"]):
            return "Gynecology"
        return "General"

    def _infer_complexity(self, description: str, cpt_code: str) -> int:
        text = description.lower()
        if any(keyword in text for keyword in ["neurosurgery", "cardiothoracic", "transplant"]):
            return 9
        if any(keyword in text for keyword in ["replacement", "arthroplasty", "hysterectomy", "laparoscopy", "surgery"]):
            return 5
        if any(keyword in text for keyword in ["scan", "x-ray", "consult", "lab"]):
            return 2
        if cpt_code.startswith(("2", "4", "5")):
            return 4
        return 3

    def _estimate_cost(self, cpt_code: str, description: str) -> float:
        if cpt_code in self._knowledge_base["procedures"]:
            return float(self._knowledge_base["procedures"][cpt_code]["us_median_cost_usd"])

        text = description.lower()
        if any(keyword in text for keyword in ["knee", "replacement", "arthroplasty"]):
            return 45200.0
        if any(keyword in text for keyword in ["gallbladder", "cholecyst"]):
            return 18950.0
        if any(keyword in text for keyword in ["hysterectomy", "uterus"]):
            return 27800.0
        if any(keyword in text for keyword in ["scan", "imaging"]):
            return 1200.0
        return 5000.0

    def _extract_numeric(self, value: str, default: float) -> float:
        match = re.search(r"-?\d+(?:\.\d+)?", value or "")
        if not match:
            return default
        return float(match.group(0))

    def get_usage_stats(self) -> dict[str, Any]:
        return self.usage_stats.copy()

    def reset_usage_stats(self):
        self.usage_stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cached": 0,
            "estimated_cost_usd": 0.0,
        }


_eh_ai_client: Optional[EvijnarHealthAI] = None


async def get_evijnar_health_ai_client() -> EvijnarHealthAI:
    """Get or create the global Evijnar Health AI client."""
    global _eh_ai_client

    if _eh_ai_client is None:
        _eh_ai_client = EvijnarHealthAI()
        await _eh_ai_client.initialize()

    return _eh_ai_client


async def shutdown_evijnar_health_ai_client():
    """Shutdown the global Evijnar Health AI client."""
    global _eh_ai_client

    if _eh_ai_client:
        await _eh_ai_client.shutdown()
        _eh_ai_client = None
