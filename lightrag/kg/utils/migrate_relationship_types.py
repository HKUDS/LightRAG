"""
Migration script to convert existing Neo4j relationships to the new typed relationship system.
This script will convert relationship data from using a generic 'RELATED' type with a 'rel_type'
property to using proper Neo4j relationship types.
"""

import time
import asyncio
import logging
from typing import Dict, Any

from ...utils import logger
from .relationship_registry import (
    RelationshipTypeRegistry,
)


async def migrate_neo4j_relationships(
    driver,
    database: str,
    dry_run: bool = True,
    batch_size: int = 100,
    max_relationships: int = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Migrate Neo4j relationships from generic RELATED type to proper relationship types.

    Args:
        driver: Neo4j AsyncDriver instance
        database: Neo4j database name
        dry_run: If True, only report what would be done without making changes
        batch_size: Number of relationships to process in each batch
        max_relationships: Maximum number of relationships to process, None for all
        verbose: Whether to log detailed information

    Returns:
        Dict with migration statistics
    """
    start_time = time.time()
    registry = RelationshipTypeRegistry()

    # Set up logging
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Statistics dictionary
    stats = {
        "total_relationships": 0,
        "migrated_relationships": 0,
        "relationship_types": {},
        "errors": 0,
        "execution_time": 0,
    }

    try:
        # First count all relationships to migrate
        async with driver.session(database=database) as session:
            count_query = """
            MATCH ()-[r:RELATED]->()
            WHERE EXISTS(r.rel_type)
            RETURN COUNT(r) AS count
            """
            count_result = await session.run(count_query)
            count_record = await count_result.single()
            total_relationships = count_record["count"] if count_record else 0
            await count_result.consume()

            stats["total_relationships"] = total_relationships
            logger.info(f"Found {total_relationships} relationships to migrate")

            if max_relationships:
                logger.info(f"Limiting migration to {max_relationships} relationships")
                total_relationships = min(total_relationships, max_relationships)

            if total_relationships == 0:
                logger.info("No relationships to migrate")
                stats["execution_time"] = time.time() - start_time
                return stats

            # Get relationship type counts
            type_query = """
            MATCH ()-[r:RELATED]->()
            WHERE EXISTS(r.rel_type)
            RETURN r.rel_type AS rel_type, COUNT(r) AS count
            ORDER BY count DESC
            """
            type_result = await session.run(type_query)

            async for record in type_result:
                rel_type = record["rel_type"]
                count = record["count"]
                neo4j_type = registry.get_neo4j_type(rel_type)

                stats["relationship_types"][rel_type] = {
                    "count": count,
                    "neo4j_type": neo4j_type,
                }

            await type_result.consume()

            # Process relationships in batches
            offset = 0
            processed = 0

            while offset < total_relationships:
                current_batch_size = min(batch_size, total_relationships - offset)

                # Get a batch of relationships to migrate
                batch_query = """
                MATCH (src)-[r:RELATED]->(tgt)
                WHERE EXISTS(r.rel_type)
                RETURN ID(r) AS rel_id, src.entity_id AS source, tgt.entity_id AS target,
                       r.rel_type AS rel_type, properties(r) AS properties
                SKIP $offset
                LIMIT $limit
                """
                batch_result = await session.run(
                    batch_query, offset=offset, limit=current_batch_size
                )

                # Collect relationships for migration
                relationships = []
                async for record in batch_result:
                    relationships.append(
                        {
                            "rel_id": record["rel_id"],
                            "source": record["source"],
                            "target": record["target"],
                            "rel_type": record["rel_type"],
                            "properties": record["properties"],
                            "neo4j_type": registry.get_neo4j_type(record["rel_type"]),
                        }
                    )

                await batch_result.consume()

                # Process the batch
                if not dry_run:
                    for rel in relationships:
                        rel_id = rel["rel_id"]
                        source = rel["source"]
                        target = rel["target"]
                        rel_type = rel["rel_type"]
                        neo4j_type = rel["neo4j_type"]
                        properties = rel["properties"]

                        try:
                            # Store the original relationship type
                            if "original_type" not in properties:
                                properties["original_type"] = rel_type

                            # Create the new typed relationship
                            create_query = f"""
                            MATCH (src:base {{entity_id: $source}}), (tgt:base {{entity_id: $target}})
                            CREATE (src)-[r:{neo4j_type}]->(tgt)
                            SET r = $properties
                            RETURN r
                            """

                            result = await session.run(
                                create_query,
                                source=source,
                                target=target,
                                properties=properties,
                            )
                            await result.consume()

                            # Delete the old relationship
                            delete_query = """
                            MATCH ()-[r]-() WHERE ID(r) = $rel_id
                            DELETE r
                            """

                            result = await session.run(delete_query, rel_id=rel_id)
                            await result.consume()

                            stats["migrated_relationships"] += 1

                            if verbose:
                                logger.debug(
                                    f"Migrated relationship: {source} -[{rel_type}]-> {target} "
                                    f"to {neo4j_type}"
                                )

                        except Exception as e:
                            stats["errors"] += 1
                            logger.error(
                                f"Error migrating relationship {rel_id}: {str(e)}"
                            )
                else:
                    # In dry run mode, just log what would be done
                    if verbose:
                        for rel in relationships:
                            logger.debug(
                                f"Would migrate: {rel['source']} -[{rel['rel_type']}]-> {rel['target']} "
                                f"to {rel['neo4j_type']}"
                            )

                    # Count as migrated for statistics
                    stats["migrated_relationships"] += len(relationships)

                # Update progress
                processed += len(relationships)
                offset += current_batch_size

                # Log progress
                progress = (processed / total_relationships) * 100
                logger.info(
                    f"Progress: {processed}/{total_relationships} ({progress:.1f}%)"
                )

                # Small delay to avoid hammering the database
                await asyncio.sleep(0.1)

            # Calculate execution time
            stats["execution_time"] = time.time() - start_time

            # Log completion
            if dry_run:
                logger.info(
                    f"Dry run completed. Would migrate {stats['migrated_relationships']} relationships "
                    f"in {stats['execution_time']:.2f} seconds"
                )
            else:
                logger.info(
                    f"Migration completed. Migrated {stats['migrated_relationships']} relationships "
                    f"in {stats['execution_time']:.2f} seconds"
                )

            return stats

    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        stats["errors"] += 1
        stats["execution_time"] = time.time() - start_time
        return stats


async def verify_migration(
    driver, database: str, verbose: bool = False
) -> Dict[str, Any]:
    """
    Verify the migration by counting relationships by type.

    Args:
        driver: Neo4j AsyncDriver instance
        database: Neo4j database name
        verbose: Whether to log detailed information

    Returns:
        Dict with verification statistics
    """
    stats = {
        "total_relationships": 0,
        "relationship_types": {},
        "legacy_relationships": 0,
    }

    try:
        async with driver.session(database=database) as session:
            # Count relationships by type
            type_query = """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, COUNT(r) AS count
            ORDER BY count DESC
            """
            type_result = await session.run(type_query)

            async for record in type_result:
                rel_type = record["rel_type"]
                count = record["count"]

                stats["relationship_types"][rel_type] = count
                stats["total_relationships"] += count

                if rel_type == "RELATED":
                    stats["legacy_relationships"] = count

            await type_result.consume()

            # Count relationships that still have rel_type property
            legacy_query = """
            MATCH ()-[r]->()
            WHERE EXISTS(r.rel_type)
            RETURN COUNT(r) AS count
            """
            legacy_result = await session.run(legacy_query)
            record = await legacy_result.single()
            stats["relationships_with_rel_type"] = record["count"] if record else 0
            await legacy_result.consume()

            # Calculate percentage migrated
            if stats["total_relationships"] > 0:
                non_legacy = (
                    stats["total_relationships"] - stats["legacy_relationships"]
                )
                stats["percent_migrated"] = (
                    non_legacy / stats["total_relationships"]
                ) * 100
            else:
                stats["percent_migrated"] = 0

            # Log results
            if verbose:
                logger.info(f"Total relationships: {stats['total_relationships']}")
                logger.info(
                    f"Legacy RELATED relationships: {stats['legacy_relationships']}"
                )
                logger.info(f"Percent migrated: {stats['percent_migrated']:.1f}%")

                for rel_type, count in stats["relationship_types"].items():
                    if rel_type != "RELATED":
                        logger.info(f"  {rel_type}: {count}")

            return stats

    except Exception as e:
        logger.error(f"Error verifying migration: {str(e)}")
        return stats


async def rollback_migration(
    driver,
    database: str,
    dry_run: bool = True,
    batch_size: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Rollback the migration by converting typed relationships back to RELATED type.

    Args:
        driver: Neo4j AsyncDriver instance
        database: Neo4j database name
        dry_run: If True, only report what would be done without making changes
        batch_size: Number of relationships to process in each batch
        verbose: Whether to log detailed information

    Returns:
        Dict with rollback statistics
    """
    start_time = time.time()
    registry = RelationshipTypeRegistry()

    # Statistics dictionary
    stats = {
        "total_relationships": 0,
        "rollback_relationships": 0,
        "relationship_types": {},
        "errors": 0,
        "execution_time": 0,
    }

    try:
        # First count all relationships to rollback
        async with driver.session(database=database) as session:
            # Get all non-RELATED relationship types
            type_query = """
            MATCH ()-[r]->()
            WHERE type(r) <> 'RELATED'
            RETURN type(r) AS rel_type, COUNT(r) AS count
            ORDER BY count DESC
            """
            type_result = await session.run(type_query)

            total_relationships = 0

            async for record in type_result:
                rel_type = record["rel_type"]
                count = record["count"]

                stats["relationship_types"][rel_type] = count
                total_relationships += count

                if verbose:
                    logger.debug(f"Found {count} relationships of type {rel_type}")

            await type_result.consume()

            stats["total_relationships"] = total_relationships
            logger.info(f"Found {total_relationships} typed relationships to rollback")

            if total_relationships == 0:
                logger.info("No relationships to rollback")
                stats["execution_time"] = time.time() - start_time
                return stats

            # Process each relationship type
            for rel_type, count in stats["relationship_types"].items():
                logger.info(f"Processing {count} relationships of type {rel_type}")

                offset = 0
                while offset < count:
                    current_batch_size = min(batch_size, count - offset)

                    # Get a batch of relationships to rollback
                    batch_query = f"""
                    MATCH (src)-[r:{rel_type}]->(tgt)
                    RETURN ID(r) AS rel_id, src.entity_id AS source, tgt.entity_id AS target,
                           properties(r) AS properties
                    SKIP $offset
                    LIMIT $limit
                    """
                    batch_result = await session.run(
                        batch_query, offset=offset, limit=current_batch_size
                    )

                    # Collect relationships for rollback
                    relationships = []
                    async for record in batch_result:
                        rel_id = record["rel_id"]
                        source = record["source"]
                        target = record["target"]
                        properties = record["properties"]

                        # Determine original relationship type
                        original_type = properties.get("original_type")
                        rel_type_prop = properties.get("rel_type")

                        # Use the best available relationship type
                        relationship_type = (
                            original_type or rel_type_prop or rel_type.lower()
                        )

                        relationships.append(
                            {
                                "rel_id": rel_id,
                                "source": source,
                                "target": target,
                                "properties": properties,
                                "relationship_type": relationship_type,
                            }
                        )

                    await batch_result.consume()

                    # Process the batch
                    if not dry_run:
                        for rel in relationships:
                            rel_id = rel["rel_id"]
                            source = rel["source"]
                            target = rel["target"]
                            properties = rel["properties"]
                            relationship_type = rel["relationship_type"]

                            try:
                                # Ensure rel_type property exists
                                properties["rel_type"] = relationship_type

                                # Create the new RELATED relationship
                                create_query = """
                                MATCH (src:base {entity_id: $source}), (tgt:base {entity_id: $target})
                                CREATE (src)-[r:RELATED]->(tgt)
                                SET r = $properties
                                RETURN r
                                """

                                result = await session.run(
                                    create_query,
                                    source=source,
                                    target=target,
                                    properties=properties,
                                )
                                await result.consume()

                                # Delete the old relationship
                                delete_query = """
                                MATCH ()-[r]-() WHERE ID(r) = $rel_id
                                DELETE r
                                """

                                result = await session.run(delete_query, rel_id=rel_id)
                                await result.consume()

                                stats["rollback_relationships"] += 1

                                if verbose:
                                    logger.debug(
                                        f"Rolled back relationship: {source} -[{rel_type}]-> {target} "
                                        f"to RELATED with rel_type={relationship_type}"
                                    )

                            except Exception as e:
                                stats["errors"] += 1
                                logger.error(
                                    f"Error rolling back relationship {rel_id}: {str(e)}"
                                )
                    else:
                        # In dry run mode, just log what would be done
                        if verbose:
                            for rel in relationships:
                                logger.debug(
                                    f"Would rollback: {rel['source']} -[{rel_type}]-> {rel['target']} "
                                    f"to RELATED with rel_type={rel['relationship_type']}"
                                )

                        # Count as rolled back for statistics
                        stats["rollback_relationships"] += len(relationships)

                    # Update offset
                    offset += current_batch_size

                    # Log progress
                    progress = (offset / count) * 100
                    logger.info(
                        f"Progress for {rel_type}: {offset}/{count} ({progress:.1f}%)"
                    )

                    # Small delay to avoid hammering the database
                    await asyncio.sleep(0.1)

            # Calculate execution time
            stats["execution_time"] = time.time() - start_time

            # Log completion
            if dry_run:
                logger.info(
                    f"Dry run completed. Would rollback {stats['rollback_relationships']} relationships "
                    f"in {stats['execution_time']:.2f} seconds"
                )
            else:
                logger.info(
                    f"Rollback completed. Rolled back {stats['rollback_relationships']} relationships "
                    f"in {stats['execution_time']:.2f} seconds"
                )

            return stats

    except Exception as e:
        logger.error(f"Error during rollback: {str(e)}")
        stats["errors"] += 1
        stats["execution_time"] = time.time() - start_time
        return stats
