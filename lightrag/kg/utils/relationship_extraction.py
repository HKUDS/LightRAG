"""
Relationship extraction utilities for the Knowledge Graph component.
This module provides functions for extracting relationships from text using LLMs
and processing them for storage in the Neo4j database.
"""

import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

from ...utils import logger
from .relationship_registry import (
    RelationshipTypeRegistry,
)
from .threshold_manager import ThresholdManager


class RelationshipExtractor:
    """
    A class for extracting and processing relationships from text.

    This class handles the extraction of relationships from text using LLMs,
    and processes them for storage in the Neo4j database. It also provides
    methods for enriching relationships with additional metadata.
    """

    def __init__(
        self,
        llm: Any,  # Using Any to avoid circular import with BaseLLM
        relationship_registry: Optional[RelationshipTypeRegistry] = None,
        threshold_manager: Optional[ThresholdManager] = None,
    ):
        """
        Initialize the relationship extractor.

        Args:
            llm: The language model to use for extraction
            relationship_registry: Registry of valid relationship types
            threshold_manager: Manager for relationship thresholds
        """
        self.llm = llm
        self.relationship_registry = relationship_registry or RelationshipTypeRegistry()
        self.threshold_manager = threshold_manager or ThresholdManager()

    async def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relationships from text using the LLM.

        Args:
            text: The text to extract relationships from

        Returns:
            List of extracted relationships
        """
        logger.debug(
            f"Starting relationship extraction from text of length {len(text)} characters"
        )

        # Get valid relationship types from the registry
        relationship_types = ", ".join(
            self.relationship_registry.get_all_relationship_types()
        )
        logger.debug(
            f"Using {len(self.relationship_registry.get_all_relationship_types())} relationship types from registry"
        )

        # Prepare the prompt
        prompt_template = """
        Extract relationships between entities in the following text.

        For each relationship, identify:
        1. Source entity
        2. Target entity
        3. Relationship type (choose from the list of valid relationships)
        4. Relationship weight (0.0 to 1.0)
        5. Brief description of the relationship
        6. Keywords relevant to this relationship

        Valid relationship types include:
        {relationship_types}

        Text: {text}

        Output as JSON in the following format:
        [
          {{
            "source": "entity_name",
            "target": "entity_name",
            "relationship_type": "one of the valid relationship types",
            "weight": float between A0 and 1,
            "description": "brief description of the relationship",
            "keywords": ["keyword1", "keyword2"]
          }}
        ]
        """

        # Format the prompt
        formatted_prompt = prompt_template.format(
            relationship_types=relationship_types, text=text
        )

        logger.debug(
            f"Formatted extraction prompt with {len(formatted_prompt)} characters"
        )

        # Call the LLM to extract relationships
        try:
            logger.debug("Calling LLM for relationship extraction")
            response = await self.llm.generate(formatted_prompt)
            logger.debug(f"Received LLM response of length {len(response)} characters")

            # Parse the JSON response
            try:
                # First try to parse the entire response as JSON
                relationships = json.loads(response)

                # Validate the response
                if not isinstance(relationships, list):
                    logger.warning(
                        "LLM returned non-list response for relationship extraction"
                    )
                    relationships = []
                else:
                    logger.info(
                        f"Successfully parsed {len(relationships)} relationships from LLM response"
                    )
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse LLM response as direct JSON, trying pattern extraction"
                )
                # If that fails, try to extract a JSON array from the text
                # This handles cases where the LLM adds extra text before/after the JSON
                json_pattern = r'\[\s*{\s*"source".*}\s*\]'
                json_match = re.search(json_pattern, response, re.DOTALL)

                if json_match:
                    try:
                        relationships = json.loads(json_match.group(0))
                        if not isinstance(relationships, list):
                            relationships = []
                        else:
                            logger.info(
                                f"Successfully extracted {len(relationships)} relationships using pattern matching"
                            )
                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse JSON from LLM response pattern match"
                        )
                        relationships = []
                else:
                    logger.warning("No valid JSON found in LLM response")
                    relationships = []

            # Process and validate the extracted relationships
            logger.debug(f"Processing {len(relationships)} extracted relationships")
            processed_relationships = self._process_relationships(relationships)

            # Log extraction statistics
            processed_count = len(processed_relationships)
            if len(relationships) > 0:
                success_rate = (processed_count / len(relationships)) * 100
                logger.info(
                    f"Relationship extraction completed: {processed_count}/{len(relationships)} relationships processed successfully ({success_rate:.1f}% success rate)"
                )
            else:
                logger.info(
                    "Relationship extraction completed: No relationships found in text"
                )

            # Log relationship types found
            if processed_relationships:
                rel_types = {}
                for rel in processed_relationships:
                    rel_type = rel.get("relationship_type", "unknown")
                    rel_types[rel_type] = rel_types.get(rel_type, 0) + 1

                type_summary = ", ".join(
                    [f"{rt}: {count}" for rt, count in rel_types.items()]
                )
                logger.debug(f"Relationship types extracted: {type_summary}")

            return processed_relationships

        except Exception as e:
            logger.error(f"Error extracting relationships: {str(e)}")
            logger.error(
                f"Text length: {len(text)}, Available relationship types: {len(self.relationship_registry.get_all_relationship_types())}"
            )
            return []

    def _process_relationships(
        self, relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process and validate the extracted relationships.

        Args:
            relationships: The list of extracted relationships

        Returns:
            The processed relationships
        """
        logger.debug(
            f"Processing {len(relationships)} raw relationships for validation and standardization"
        )

        processed = []
        validation_stats = {
            "missing_fields": 0,
            "invalid_weights": 0,
            "weight_corrections": 0,
            "threshold_applications": 0,
            "keyword_fixes": 0,
        }

        for i, rel in enumerate(relationships):
            # Check for required fields
            if not all(k in rel for k in ["source", "target", "relationship_type"]):
                logger.warning(
                    f"Skipping relationship {i + 1} with missing required fields: {rel}"
                )
                validation_stats["missing_fields"] += 1
                continue

            # Get relationship type and check if it's valid
            rel_type = rel["relationship_type"]

            # Standardize the relationship type using the registry
            neo4j_type = self.relationship_registry.get_neo4j_type(rel_type)
            logger.debug(
                f"Standardized relationship type '{rel_type}' to Neo4j type '{neo4j_type}'"
            )

            # Process the weight field
            weight = rel.get("weight", 0.5)
            original_weight = weight

            # Convert string weights to float
            if isinstance(weight, str):
                try:
                    weight = float(weight)
                    logger.debug(
                        f"Converted string weight '{original_weight}' to float {weight}"
                    )
                except ValueError:
                    logger.warning(f"Invalid weight format: {weight}, using default")
                    validation_stats["invalid_weights"] += 1
                    weight = 0.5

            # Ensure weight is within 0-1 range
            if weight < 0 or weight > 1:
                if weight <= 10:  # Likely on a 0-10 scale
                    weight = weight / 10.0
                    logger.debug(
                        f"Converted 0-10 scale weight {original_weight} to 0-1 scale: {weight}"
                    )
                    validation_stats["weight_corrections"] += 1
                else:
                    logger.warning(
                        f"Weight {original_weight} out of range, using default 0.5"
                    )
                    validation_stats["invalid_weights"] += 1
                    weight = 0.5  # Default for invalid weights

            # Apply threshold manager to ensure minimum weight
            min_threshold = self.threshold_manager.get_threshold(rel_type)
            if weight < min_threshold:
                logger.debug(
                    f"Applied minimum threshold for {rel_type}: {weight} -> {min_threshold}"
                )
                weight = min_threshold
                validation_stats["threshold_applications"] += 1

            # Process keywords
            keywords = rel.get("keywords", [])
            original_keywords = keywords

            if isinstance(keywords, str):
                # Split comma-separated keywords
                keywords = [k.strip() for k in keywords.split(",")]
                logger.debug(
                    f"Split keyword string '{original_keywords}' into list: {keywords}"
                )
                validation_stats["keyword_fixes"] += 1
            elif not isinstance(keywords, list):
                keywords = []
                logger.debug(
                    f"Converted non-list keywords {type(original_keywords)} to empty list"
                )
                validation_stats["keyword_fixes"] += 1

            # Default keywords if empty
            if not keywords:
                keywords = [rel["source"], rel["target"], rel_type]
                logger.debug(f"Generated default keywords for relationship: {keywords}")
                validation_stats["keyword_fixes"] += 1

            # Create a processed relationship
            processed_rel = {
                "source": rel["source"],
                "target": rel["target"],
                "relationship_type": rel_type,
                "neo4j_type": neo4j_type,
                "weight": weight,
                "description": rel.get(
                    "description",
                    f"Relationship between {rel['source']} and {rel['target']}",
                ),
                "keywords": keywords,
                "confidence": rel.get(
                    "confidence", weight
                ),  # Default to weight if not provided
                "extraction_timestamp": datetime.now().isoformat(),
                "extraction_source": "llm",
            }

            processed.append(processed_rel)

        # Log processing statistics
        logger.info(
            f"Relationship processing completed: {len(processed)}/{len(relationships)} relationships validated successfully"
        )

        if any(validation_stats.values()):
            stats_summary = ", ".join(
                [f"{k}: {v}" for k, v in validation_stats.items() if v > 0]
            )
            logger.debug(f"Validation corrections applied: {stats_summary}")

        return processed

    async def calculate_relationship_weight(
        self, source: str, target: str, relationship_type: str, context: str
    ) -> float:
        """
        Calculate relationship weight using LLM analysis.

        Args:
            source: Source entity name
            target: Target entity name
            relationship_type: Relationship type
            context: Context text where relationship was found

        Returns:
            Float weight between 0.0 and 1.0
        """
        prompt = f"""
        Analyze the strength of the relationship between '{source}' and '{target}'
        of type '{relationship_type}' based on the following context.

        Context: {context}

        On a scale of 0.0 to 1.0, how strong is this relationship? Consider:
        - How explicitly is the relationship stated? (implicit = lower, explicit = higher)
        - How central is this relationship to the context? (peripheral = lower, central = higher)
        - How certain are you of this relationship? (uncertain = lower, certain = higher)

        Provide a single decimal number between 0.0 and 1.0 as your answer.
        """

        try:
            # Call LLM
            response = await self.llm.generate(prompt)

            # Extract and validate weight
            response = response.strip()

            # Try to extract a float value from the response
            match = re.search(r"(\d+\.\d+|\d+)", response)
            if match:
                weight_str = match.group(1)
                try:
                    weight = float(weight_str)
                    # Ensure weight is between 0 and 1
                    if weight > 1.0:
                        weight = weight / 10.0 if weight <= 10.0 else 1.0
                    return max(0.0, min(1.0, weight))
                except ValueError:
                    logger.warning(f"Invalid weight format from LLM: {weight_str}")
                    return 0.5  # Default weight
            else:
                logger.warning(f"No numeric weight found in LLM response: {response}")
                return 0.5  # Default weight
        except Exception as e:
            logger.error(f"Error calculating relationship weight: {str(e)}")
            return 0.5  # Default weight on error

    async def enrich_relationship_metadata(
        self, relationship: Dict[str, Any], context: str = None
    ) -> Dict[str, Any]:
        """
        Enrich a relationship with additional metadata using LLM.

        Args:
            relationship: The relationship to enrich
            context: Optional context text for the relationship

        Returns:
            The enriched relationship
        """
        # Skip enrichment if essential fields are missing
        if not all(
            k in relationship for k in ["source", "target", "relationship_type"]
        ):
            return relationship

        source = relationship["source"]
        target = relationship["target"]
        rel_type = relationship["relationship_type"]

        # If no context is provided, create a simple description
        if not context:
            context = f"A relationship where {source} {rel_type} {target}."

        # Create prompt for LLM to enrich the relationship
        prompt = f"""
        Analyze the relationship between '{source}' and '{target}' of type '{rel_type}'
        based on this context:

        {context}

        Please provide the following information about this relationship in JSON format:
        1. A detailed description of how {source} {rel_type} {target}
        2. A confidence score between 0.0 and 1.0 indicating how certain you are about this relationship
        3. Up to 5 keywords that characterize this relationship
        4. Any potential business value or strategic insight from this relationship

        Return your analysis as a valid JSON object with these fields:
        "description", "confidence", "keywords", "business_value"
        """

        try:
            # Call LLM
            response = await self.llm.generate(prompt)

            # Parse the JSON response
            try:
                metadata = json.loads(response)

                # Update relationship with enriched metadata
                if "description" in metadata and metadata["description"]:
                    relationship["description"] = metadata["description"]

                if "confidence" in metadata:
                    try:
                        confidence = float(metadata["confidence"])
                        relationship["confidence"] = max(0.0, min(1.0, confidence))
                    except (ValueError, TypeError):
                        pass  # Keep existing confidence

                if "keywords" in metadata and isinstance(metadata["keywords"], list):
                    relationship["keywords"] = metadata["keywords"]

                if "business_value" in metadata:
                    relationship["business_value"] = metadata["business_value"]

            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLM enrichment response")

            # Mark as enriched
            relationship["enriched"] = True
            relationship["enrichment_timestamp"] = datetime.now().isoformat()

            return relationship

        except Exception as e:
            logger.error(f"Error enriching relationship metadata: {str(e)}")
            return relationship

    async def batch_process_relationships(
        self,
        relationships: List[Dict[str, Any]],
        context: str = None,
        calculate_weights: bool = True,
        enrich_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of relationships, adding weights and enriching metadata.

        Args:
            relationships: List of relationships to process
            context: Optional context text for the relationships
            calculate_weights: Whether to calculate weights using LLM
            enrich_metadata: Whether to enrich metadata using LLM

        Returns:
            The processed relationships
        """
        # Process in small batches for better quality
        processed = []
        batch_size = 5  # Small batch size for better quality

        for i in range(0, len(relationships), batch_size):
            batch = relationships[i : i + batch_size]

            # Process each relationship in the batch
            tasks = []

            # Calculate weights if requested
            if calculate_weights and context:
                for rel in batch:
                    task = self.calculate_relationship_weight(
                        rel["source"], rel["target"], rel["relationship_type"], context
                    )
                    tasks.append(task)

                # Wait for all weight calculations to complete
                weights = await asyncio.gather(*tasks)

                # Apply weights to relationships
                for j, rel in enumerate(batch):
                    rel["weight"] = weights[j]

            # Enrich metadata if requested
            if enrich_metadata:
                tasks = []
                for rel in batch:
                    task = self.enrich_relationship_metadata(rel, context)
                    tasks.append(task)

                # Wait for all enrichment tasks to complete
                enriched = await asyncio.gather(*tasks)

                # Process and add to results
                processed.extend(enriched)
            else:
                processed.extend(batch)

        return processed
