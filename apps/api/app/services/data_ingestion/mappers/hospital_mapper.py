# apps/api/app/services/data_ingestion/mappers/hospital_mapper.py
"""
Hospital entity mapper using Evijnar Health AI for intelligent normalization.
Maps messy hospital descriptions to standardized GlobalHospital format.
"""

import logging
from typing import Optional

from ..models import RawHospitalData, NormalizedHospitalData, IngestSource
from ..errors import MapperError, LLMError, LLMParsingError, ValidationError
from ....utils.llm_client import get_evijnar_health_ai_client

logger = logging.getLogger("evijnar.ingest.hospital_mapper")


class HospitalMapper:
    """Maps hospital data using Evijnar Health AI for intelligent normalization"""

    SYSTEM_PROMPT = """You are a healthcare data standardization expert.
Your task is to normalize hospital information from various international sources.

Hospital types:
- SPECIALTY_CENTER: Centers of Excellence specializing in specific procedures
- GENERAL_HOSPITAL: Multi-service general purpose hospitals
- DIAGNOSTIC_CENTER: Imaging and diagnostic facilities
- NURSING_HOME: Long-term care and rehabilitation

Always respond with valid JSON.
"""

    def __init__(self):
        self.llm_client = None

    async def initialize(self):
        """Initialize AI client"""
        self.llm_client = await get_evijnar_health_ai_client()

    async def map_hospital(self, raw_hospital: RawHospitalData) -> NormalizedHospitalData:
        """
        Map raw hospital data to normalized format using Evijnar Health AI.

        Args:
            raw_hospital: Raw hospital data from source

        Returns:
            NormalizedHospitalData ready for database

        Raises:
            MapperError: If mapping fails
        """
        try:
            if not self.llm_client:
                await self.initialize()

            # Build prompt for Evijnar Health AI
            prompt = self._build_prompt(raw_hospital)

            # Call Evijnar Health AI
            logger.debug(f"Mapping hospital: {raw_hospital.name}")
            response = await self.llm_client.call_eh_ai(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                response_format="json",
                temperature=0.2,  # Low temp for consistent classification
                max_tokens=512,
                cache_ttl=86400,  # Cache for 24h
            )

            # Parse and validate response
            normalized = self._parse_ai_response(response, raw_hospital)

            logger.info(f"Successfully mapped hospital: {normalized.name} (type: {normalized.hospital_type})")
            return normalized

        except LLMError as e:
            logger.error(f"LLM error mapping hospital {raw_hospital.name}: {str(e)}")
            raise MapperError(f"Failed to map hospital: {str(e)}", details={"hospital": raw_hospital.name})
        except Exception as e:
            logger.error(f"Unexpected error mapping hospital {raw_hospital.name}: {str(e)}")
            raise MapperError(f"Error mapping hospital: {str(e)}", details={"hospital": raw_hospital.name})

    def _build_prompt(self, raw: RawHospitalData) -> str:
        """Build Evijnar Health AI prompt for hospital normalization"""
        return f"""Normalize the following hospital information:

Name: {raw.name}
Alternate Name: {raw.name_alternate or 'N/A'}
Description: {raw.description or 'N/A'}
City: {raw.city}
State/Province: {raw.state_or_province}
Country: {raw.country_code}
Phone: {raw.phone or 'N/A'}
Email: {raw.email or 'N/A'}
Hospital Type (raw): {raw.hospital_type_raw or 'N/A'}
Accreditations: JCI={raw.jci_accredited}, NABH={raw.nabh_accredited}, Other={', '.join(raw.other_accreditations) if raw.other_accreditations else 'None'}

Please provide:
1. Cleaned/normalized hospital name
2. Classification (SPECIALTY_CENTER, GENERAL_HOSPITAL, DIAGNOSTIC_CENTER, or NURSING_HOME)
3. Postal code (if available)
4. Notes on classification reasoning

Respond with JSON format:
{{
    "normalized_name": "cleaned hospital name",
    "hospital_type": "SPECIALTY_CENTER",
    "postal_code": "postal code or null",
    "confidence": 0.95,
    "reasoning": "explanation of classification"
}}"""

    def _parse_ai_response(self, response: dict, raw: RawHospitalData) -> NormalizedHospitalData:
        """Parse AI response into NormalizedHospitalData"""
        try:
            # Extract fields from AI response
            normalized_name = response.get("normalized_name", raw.name)
            hospital_type = response.get("hospital_type", "GENERAL_HOSPITAL")

            # Validate hospital type
            valid_types = ["SPECIALTY_CENTER", "GENERAL_HOSPITAL", "DIAGNOSTIC_CENTER", "NURSING_HOME"]
            if hospital_type not in valid_types:
                logger.warning(f"Invalid hospital type from Evijnar Health AI: {hospital_type}, defaulting to GENERAL_HOSPITAL")
                hospital_type = "GENERAL_HOSPITAL"

            # Create normalized data object
            normalized = NormalizedHospitalData(
                name=normalized_name or raw.name,
                hospital_type=hospital_type,
                country_code=raw.country_code,
                state_province=raw.state_or_province,
                city=raw.city,
                postal_code=response.get("postal_code") or raw.postal_code,
                phone_primary=raw.phone,
                email=raw.email,
                website_url=raw.website,
                jci_accredited=raw.jci_accredited or False,
                nabh_accredited=raw.nabh_accredited or False,
                avg_quality_score=raw.quality_score or 0,
                complication_rate=raw.complication_rate,
                readmission_rate=raw.readmission_rate,
                patient_reviews_count=raw.reviews_count or 0,
                price_data_source=raw.source,
                price_data_verified_at=raw.data_verified_date,
                source_id=raw.source_id,
            )

            return normalized

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse AI response: {str(e)}, Response: {response}")
            raise LLMParsingError(f"Failed to parse AI response: {str(e)}", details={"response": response})
