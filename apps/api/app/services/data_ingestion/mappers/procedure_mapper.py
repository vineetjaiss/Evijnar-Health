# apps/api/app/services/data_ingestion/mappers/procedure_mapper.py
"""
Procedure mapper using Evijnar Health AI for intelligent mapping to ICD-10 and complexity scoring.
"""

import logging
from typing import List

from ..models import RawProcedureData, NormalizedProcedureData, IngestSource
from ..errors import MapperError, LLMError, LLMParsingError
from ....utils.llm_client import get_evijnar_health_ai_client

logger = logging.getLogger("evijnar.ingest.procedure_mapper")


class ProcedureMapper:
    """Maps procedure data to standardized clinical codes using Evijnar Health AI"""

    SYSTEM_PROMPT = """You are a medical coding expert familiar with ICD-10, CPT, and UHI coding systems.
Your task is to standardize procedure/service descriptions into clinical codes.

Complexity scoring (1-10):
1-3: Simple diagnostic or preventive (blood test, X-ray, consultation)
4-6: Standard procedure (cholecystectomy, appendectomy, routine surgery)
7-9: Complex procedure (major cardiothoracic, neurosurgery)
10: Highly complex (multiple organ surgery, emergency salvage procedures)

Always respond with valid JSON.
"""

    def __init__(self):
        self.llm_client = None

    async def initialize(self):
        """Initialize AI client"""
        self.llm_client = await get_evijnar_health_ai_client()

    async def map_procedures(self, raw_procedures: List[RawProcedureData]) -> List[NormalizedProcedureData]:
        """
        Map list of raw procedures to normalized format.

        Args:
            raw_procedures: List of raw procedure data

        Returns:
            List of normalized procedure data

        Raises:
            MapperError: If mapping fails
        """
        normalized_procedures = []

        for idx, raw_proc in enumerate(raw_procedures):
            try:
                normalized = await self.map_procedure(raw_proc)
                normalized_procedures.append(normalized)
            except Exception as e:
                logger.warning(f"Failed to map procedure {idx}: {raw_proc.description}. Error: {str(e)}")
                continue

        logger.info(f"Mapped {len(normalized_procedures)} of {len(raw_procedures)} procedures")
        return normalized_procedures

    async def map_procedure(self, raw_proc: RawProcedureData, source: IngestSource = IngestSource.HHS_TRANSPARENCY) -> NormalizedProcedureData:
        """
        Map single procedure to normalized format using Evijnar Health AI.

        Args:
            raw_proc: Raw procedure data
            source: Source of data

        Returns:
            NormalizedProcedureData

        Raises:
            MapperError: If mapping fails
        """
        try:
            if not self.llm_client:
                await self.initialize()

            prompt = self._build_prompt(raw_proc)

            logger.debug(f"Mapping procedure: {raw_proc.description}")
            response = await self.llm_client.call_eh_ai(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                response_format="json",
                temperature=0.3,
                max_tokens=512,
                cache_ttl=604800,  # Cache for 7 days
            )

            normalized = self._parse_ai_response(response, raw_proc, source)
            logger.debug(f"Mapped procedure to ICD-10: {normalized.icd10_code}, Complexity: {normalized.complexity_score}")

            return normalized

        except Exception as e:
            logger.error(f"Error mapping procedure {raw_proc.description}: {str(e)}")
            raise MapperError(f"Failed to map procedure: {str(e)}", details={"procedure": raw_proc.description})

    def _build_prompt(self, raw: RawProcedureData) -> str:
        """Build Evijnar Health AI prompt for procedure mapping"""
        return f"""Map the following clinical procedure/service to standard coding:

Description: {raw.description}
Code (if provided): {raw.code or 'Not provided'}
Price: {raw.price or 'Not provided'} {raw.currency}
Success Rate: {raw.success_rate or 'Not available'}%
Complication Rate: {raw.complication_rate or 'Not available'}%

Please provide:
1. ICD-10 diagnosis code (format: ABC.DE, e.g., M17.11)
2. Clinical category (e.g., Orthopedic Surgery, Cardiology, Oncology)
3. Complexity score (1-10, where 10 is most complex)
4. If available, map to UHI code for India

Respond with JSON:
{{
    "icd10_code": "M17.11",
    "icd10_description": "Primary osteoarthritis, right knee",
    "clinical_category": "Orthopedic Surgery",
    "complexity_score": 8,
    "uhi_code": "SURG-1001",
    "post_op_monitoring": "High",
    "confidence": 0.92
}}"""

    def _parse_ai_response(self, response: dict, raw: RawProcedureData, source: IngestSource) -> NormalizedProcedureData:
        """Parse AI response into NormalizedProcedureData"""
        try:
            icd10_code = response.get("icd10_code")
            if not icd10_code:
                raise ValueError("Missing ICD-10 code in AI response")

            complexity = response.get("complexity_score", 5)
            if not isinstance(complexity, int) or complexity < 1 or complexity > 10:
                complexity = 5  # Default to medium complexity

            normalized = NormalizedProcedureData(
                cpt_code=raw.code if raw.code and len(raw.code) == 5 else None,
                icd10_code=icd10_code,
                uhi_code=response.get("uhi_code"),
                ehds_identifier=response.get("ehds_identifier"),
                clinical_category=response.get("clinical_category", "General"),
                complexity_score=complexity,
                base_price=raw.price or 0,
                currency_code=raw.currency,
                success_rate=raw.success_rate,
                complication_rate=raw.complication_rate,
                data_source=source,
            )

            return normalized

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse AI response for procedure mapping: {str(e)}")
            raise LLMParsingError(f"Failed to parse procedure mapping response: {str(e)}")
