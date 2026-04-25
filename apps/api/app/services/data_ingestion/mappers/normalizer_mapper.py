# apps/api/app/services/data_ingestion/mappers/normalizer_mapper.py
"""
Price Normalizer mapper for CPT ↔ ICD-10 ↔ UHI code standardization.
Critical for enabling price comparisons across US, Europe, and India.
"""

import logging
from typing import Optional

from ..models import NormalizedPriceNormalizerData, IngestSource
from ..errors import MapperError, LLMParsingError
from ....utils.llm_client import get_evijnar_health_ai_client

logger = logging.getLogger("evijnar.ingest.normalizer_mapper")


class NormalizerMapper:
    """
    Maps medical procedure codes across international standards.
    CPT (USA) ↔ ICD-10 (Global) ↔ UHI (India) ↔ EHDS (Europe)
    """

    SYSTEM_PROMPT = """You are a medical coding standards expert with deep knowledge of:
- CPT (Current Procedural Terminology) - USA
- ICD-10 (International Classification of Diseases) - Global
- UHI (Unified Health Interface) - India
- EHDS (European Health Data Space) - Europe

Your task is to map procedure codes across these systems accurately.

Respond with valid JSON always.
"""

    # Known mappings cache (seed data from Prisma seed.ts)
    KNOWN_MAPPINGS = {
        "27447": {
            "icd10": "M17.11",
            "uhi": "SURG-1001",
            "description": "Total knee replacement with prosthesis",
            "category": "Orthopedic Surgery",
        },
        "70450": {
            "icd10": "R51.9",
            "uhi": "DIAG-0051",
            "description": "CT head/brain without contrast",
            "category": "Diagnostic Imaging",
        },
        "99213": {
            "icd10": "Z90.0",
            "uhi": "CON-0001",
            "description": "Office visit, established patient, low complexity",
            "category": "Consultation",
        },
    }

    def __init__(self):
        self.llm_client = None

    async def initialize(self):
        """Initialize AI client"""
        self.llm_client = await get_evijnar_health_ai_client()

    async def map_cpt_code(
        self,
        cpt_code: str,
        cpt_description: str,
        source: IngestSource = IngestSource.HHS_TRANSPARENCY,
    ) -> NormalizedPriceNormalizerData:
        """
        Map CPT code to standardized codes and create PriceNormalizer entry.

        Args:
            cpt_code: 5-digit CPT code
            cpt_description: CPT procedure description
            source: Data source

        Returns:
            NormalizedPriceNormalizerData ready for database insertion

        Raises:
            MapperError: If mapping fails
        """
        try:
            if not self.llm_client:
                await self.initialize()

            # Check known mappings first (avoid redundant LLM calls)
            if cpt_code in self.KNOWN_MAPPINGS:
                logger.debug(f"Using known mapping for CPT {cpt_code}")
                known = self.KNOWN_MAPPINGS[cpt_code]
                return NormalizedPriceNormalizerData(
                    cpt_code=cpt_code,
                    cpt_description=cpt_description,
                    icd10_code=known["icd10"],
                    icd10_description=known["description"],
                    uhi_code=known.get("uhi"),
                    clinical_category=known["category"],
                    complexity_score=self._estimate_complexity(cpt_code),
                    us_median_cost_usd=self._get_median_cost(cpt_code),
                )

            # Call Evijnar Health AI for unknown codes
            prompt = self._build_prompt(cpt_code, cpt_description)
            logger.debug(f"Mapping CPT code {cpt_code} via Evijnar Health AI")

            response = await self.llm_client.call_eh_ai(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                response_format="json",
                temperature=0.2,  # Low temp for medical accuracy
                max_tokens=512,
                cache_ttl=31536000,  # Cache forever (medical codes don't change)
            )

            normalized = self._parse_ai_response(response, cpt_code, cpt_description)
            logger.info(f"Mapped CPT {cpt_code} to ICD-10 {normalized.icd10_code}")

            return normalized

        except Exception as e:
            logger.error(f"Error mapping CPT code {cpt_code}: {str(e)}")
            raise MapperError(f"Failed to map CPT code: {str(e)}", details={"cpt_code": cpt_code})

    def _build_prompt(self, cpt_code: str, cpt_description: str) -> str:
        """Build Evijnar Health AI prompt for code mapping"""
        return f"""Map this US CPT code to international medical coding standards:

CPT Code: {cpt_code}
CPT Description: {cpt_description}

Please provide:
1. ICD-10 diagnosis code (format: ABC.DE)
2. ICD-10 full description
3. UHI code for India (if available)
4. EHDS identifier for Europe (if available)
5. Clinical category
6. Complexity score (1-10)
7. Estimated US median procedure cost in USD (based on typical Medicare rates)

Respond with JSON:
{{
    "icd10_code": "M17.11",
    "icd10_description": "Primary osteoarthritis, right knee",
    "uhi_code": "SURG-1001",
    "ehds_identifier": "ORTHO-001",
    "clinical_category": "Orthopedic Surgery",
    "complexity_score": 8,
    "us_median_cost_usd": 35000,
    "source": "Medicare fee schedule 2024-2025"
}}"""

    def _parse_ai_response(self, response: dict, cpt_code: str, cpt_description: str) -> NormalizedPriceNormalizerData:
        """Parse AI response into NormalizedPriceNormalizerData"""
        try:
            icd10_code = response.get("icd10_code")
            if not icd10_code:
                raise ValueError("Missing ICD-10 code in response")

            normalized = NormalizedPriceNormalizerData(
                cpt_code=cpt_code,
                cpt_description=cpt_description,
                icd10_code=icd10_code,
                icd10_description=response.get("icd10_description", ""),
                uhi_code=response.get("uhi_code"),
                ehds_identifier=response.get("ehds_identifier"),
                clinical_category=response.get("clinical_category", "General"),
                complexity_score=response.get("complexity_score", 5),
                us_median_cost_usd=response.get("us_median_cost_usd", 5000),
            )

            return normalized

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse AI mapping response: {str(e)}")
            raise LLMParsingError(f"Failed to parse price normalizer mapping: {str(e)}")

    def _estimate_complexity(self, cpt_code: str) -> int:
        """Estimate complexity from CPT code (heuristic)"""
        code = int(cpt_code)
        if 10000 <= code < 20000:  # E/M codes
            return 2
        elif 20000 <= code < 30000:  # Surgery codes
            return 7
        elif 30000 <= code < 40000:  # Diagnostic codes
            return 3
        else:
            return 5  # Default

    def _get_median_cost(self, cpt_code: str) -> float:
        """Get estimated US median cost from known costs"""
        costs = {
            "27447": 35000,  # Knee replacement
            "70450": 800,    # CT scan
            "99213": 120,    # Office visit
            "99214": 180,    # Office visit moderate
        }
        return costs.get(cpt_code, 5000)  # Default to $5000
