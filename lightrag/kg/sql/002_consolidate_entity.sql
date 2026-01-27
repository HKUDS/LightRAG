-- lightrag_consolidate_entity: Merge two entities server-side.
-- Merges descriptions, redirects edges, deletes the old node,
-- and synchronizes KV tables (full_relations, relation_chunks).
-- All in a single transaction, eliminating multiple round-trips.
--
-- Parameters:
--   p_workspace: workspace identifier
--   p_old_name: entity name to be merged (will be deleted)
--   p_canonical_name: target canonical name (will be kept)
--
-- Returns: JSONB with status and details

CREATE OR REPLACE FUNCTION lightrag_consolidate_entity(
    p_workspace VARCHAR,
    p_old_name VARCHAR,
    p_canonical_name VARCHAR
) RETURNS JSONB AS $$
DECLARE
    v_old_node JSONB;
    v_canonical_node JSONB;
    v_old_desc TEXT;
    v_canonical_desc TEXT;
    v_merged_desc TEXT;
    v_edges_redirected INT := 0;
    v_edges_deleted INT := 0;
    rec RECORD;
BEGIN
    -- Get old node
    SELECT properties INTO v_old_node
    FROM lightrag_graph_nodes
    WHERE workspace = p_workspace AND node_id = p_old_name;

    IF v_old_node IS NULL THEN
        RETURN jsonb_build_object(
            'status', 'skipped',
            'reason', 'old_node_not_found',
            'old_name', p_old_name,
            'canonical_name', p_canonical_name
        );
    END IF;

    -- Get canonical node
    SELECT properties INTO v_canonical_node
    FROM lightrag_graph_nodes
    WHERE workspace = p_workspace AND node_id = p_canonical_name;

    -- If canonical doesn't exist, just rename the old node
    IF v_canonical_node IS NULL THEN
        -- Update node_id (rename)
        UPDATE lightrag_graph_nodes
        SET node_id = p_canonical_name, updated_at = CURRENT_TIMESTAMP
        WHERE workspace = p_workspace AND node_id = p_old_name;

        -- Update edges referencing old name
        UPDATE lightrag_graph_edges
        SET source_id = p_canonical_name, updated_at = CURRENT_TIMESTAMP
        WHERE workspace = p_workspace AND source_id = p_old_name;

        UPDATE lightrag_graph_edges
        SET target_id = p_canonical_name, updated_at = CURRENT_TIMESTAMP
        WHERE workspace = p_workspace AND target_id = p_old_name;

        -- Sync KV tables for rename
        PERFORM _lightrag_sync_relation_kv(p_workspace, p_old_name, p_canonical_name);

        RETURN jsonb_build_object(
            'status', 'renamed',
            'old_name', p_old_name,
            'new_name', p_canonical_name
        );
    END IF;

    -- Both nodes exist: merge descriptions
    v_old_desc := COALESCE(v_old_node->>'description', '');
    v_canonical_desc := COALESCE(v_canonical_node->>'description', '');

    IF v_old_desc != '' AND v_canonical_desc != '' THEN
        v_merged_desc := v_canonical_desc || E'\n' || v_old_desc;
    ELSIF v_old_desc != '' THEN
        v_merged_desc := v_old_desc;
    ELSE
        v_merged_desc := v_canonical_desc;
    END IF;

    -- Merge source_ids
    UPDATE lightrag_graph_nodes
    SET properties = jsonb_set(
            jsonb_set(properties, '{description}', to_jsonb(v_merged_desc)),
            '{source_id}',
            to_jsonb(
                COALESCE(v_canonical_node->>'source_id', '') ||
                CASE WHEN COALESCE(v_old_node->>'source_id', '') != ''
                     THEN E'\x1e' || (v_old_node->>'source_id')
                     ELSE '' END
            )
        ),
        updated_at = CURRENT_TIMESTAMP
    WHERE workspace = p_workspace AND node_id = p_canonical_name;

    -- Redirect edges from old to canonical
    FOR rec IN
        SELECT source_id, target_id FROM lightrag_graph_edges
        WHERE workspace = p_workspace
          AND (source_id = p_old_name OR target_id = p_old_name)
    LOOP
        DECLARE
            v_new_source VARCHAR;
            v_new_target VARCHAR;
            v_existing INT;
        BEGIN
            v_new_source := CASE WHEN rec.source_id = p_old_name THEN p_canonical_name ELSE rec.source_id END;
            v_new_target := CASE WHEN rec.target_id = p_old_name THEN p_canonical_name ELSE rec.target_id END;

            -- Skip self-loops
            IF v_new_source = v_new_target THEN
                v_edges_deleted := v_edges_deleted + 1;
                CONTINUE;
            END IF;

            -- Check if redirected edge already exists
            SELECT 1 INTO v_existing
            FROM lightrag_graph_edges
            WHERE workspace = p_workspace
              AND ((source_id = v_new_source AND target_id = v_new_target)
                   OR (source_id = v_new_target AND target_id = v_new_source));

            IF v_existing IS NOT NULL THEN
                -- Edge already exists at target, just delete old one
                v_edges_deleted := v_edges_deleted + 1;
            ELSE
                -- Redirect edge
                UPDATE lightrag_graph_edges
                SET source_id = v_new_source, target_id = v_new_target, updated_at = CURRENT_TIMESTAMP
                WHERE workspace = p_workspace
                  AND source_id = rec.source_id AND target_id = rec.target_id;
                v_edges_redirected := v_edges_redirected + 1;
            END IF;
        END;
    END LOOP;

    -- Delete old edges still pointing to old node
    DELETE FROM lightrag_graph_edges
    WHERE workspace = p_workspace
      AND (source_id = p_old_name OR target_id = p_old_name);

    -- Delete old node
    DELETE FROM lightrag_graph_nodes
    WHERE workspace = p_workspace AND node_id = p_old_name;

    -- Sync KV tables for merge
    PERFORM _lightrag_sync_relation_kv(p_workspace, p_old_name, p_canonical_name);

    RETURN jsonb_build_object(
        'status', 'merged',
        'old_name', p_old_name,
        'canonical_name', p_canonical_name,
        'edges_redirected', v_edges_redirected,
        'edges_deleted', v_edges_deleted
    );
END;
$$ LANGUAGE plpgsql;


-- _lightrag_sync_relation_kv: Internal helper to sync KV tables after entity rename/merge.
-- Updates LIGHTRAG_FULL_RELATIONS (relation_pairs) and LIGHTRAG_RELATION_CHUNKS (keys).
-- Called by lightrag_consolidate_entity for both rename and merge paths.

CREATE OR REPLACE FUNCTION _lightrag_sync_relation_kv(
    p_workspace VARCHAR,
    p_old_name VARCHAR,
    p_canonical_name VARCHAR
) RETURNS VOID AS $$
DECLARE
    v_parts TEXT[];
    v_new_key TEXT;
    v_existing_chunks JSONB;
    rec RECORD;
BEGIN
    -- Step A: Update relation_pairs in LIGHTRAG_FULL_RELATIONS
    -- Replace old_name with canonical_name in all relation pair arrays
    UPDATE LIGHTRAG_FULL_RELATIONS
    SET relation_pairs = (
            SELECT COALESCE(jsonb_agg(
                CASE
                    WHEN elem->>0 = p_old_name AND elem->>1 = p_old_name
                        THEN jsonb_build_array(p_canonical_name, p_canonical_name)
                    WHEN elem->>0 = p_old_name
                        THEN jsonb_build_array(p_canonical_name, elem->>1)
                    WHEN elem->>1 = p_old_name
                        THEN jsonb_build_array(elem->>0, p_canonical_name)
                    ELSE elem
                END
            ), '[]'::jsonb)
            FROM jsonb_array_elements(relation_pairs) AS elem
        ),
        update_time = CURRENT_TIMESTAMP
    WHERE workspace = p_workspace
      AND EXISTS (
        SELECT 1 FROM jsonb_array_elements(relation_pairs) AS elem
        WHERE elem->>0 = p_old_name OR elem->>1 = p_old_name
      );

    -- Step B: Rename keys in LIGHTRAG_RELATION_CHUNKS
    -- Keys use format: sorted(src, tgt) joined by '<SEP>'
    FOR rec IN
        SELECT id, chunk_ids FROM LIGHTRAG_RELATION_CHUNKS
        WHERE workspace = p_workspace
          AND id LIKE '%' || p_old_name || '%'
    LOOP
        v_parts := string_to_array(rec.id, '<SEP>');
        IF array_length(v_parts, 1) = 2 THEN
            -- Replace old_name in parts
            IF v_parts[1] = p_old_name THEN v_parts[1] := p_canonical_name; END IF;
            IF v_parts[2] = p_old_name THEN v_parts[2] := p_canonical_name; END IF;

            -- Self-relation after rename → delete
            IF v_parts[1] = v_parts[2] THEN
                DELETE FROM LIGHTRAG_RELATION_CHUNKS
                WHERE workspace = p_workspace AND id = rec.id;
                CONTINUE;
            END IF;

            -- Re-sort key (keys are always sorted alphabetically)
            IF v_parts[1] > v_parts[2] THEN
                v_new_key := v_parts[2] || '<SEP>' || v_parts[1];
            ELSE
                v_new_key := v_parts[1] || '<SEP>' || v_parts[2];
            END IF;

            -- Skip if key unchanged
            IF v_new_key = rec.id THEN CONTINUE; END IF;

            -- Check if target key already exists
            SELECT chunk_ids INTO v_existing_chunks
            FROM LIGHTRAG_RELATION_CHUNKS
            WHERE workspace = p_workspace AND id = v_new_key;

            IF v_existing_chunks IS NOT NULL THEN
                -- Merge chunk_ids (deduplicate via DISTINCT)
                UPDATE LIGHTRAG_RELATION_CHUNKS
                SET chunk_ids = (
                        SELECT jsonb_agg(DISTINCT val)
                        FROM (
                            SELECT val FROM jsonb_array_elements(v_existing_chunks) AS val
                            UNION
                            SELECT val FROM jsonb_array_elements(rec.chunk_ids) AS val
                        ) sub
                    ),
                    update_time = CURRENT_TIMESTAMP
                WHERE workspace = p_workspace AND id = v_new_key;

                DELETE FROM LIGHTRAG_RELATION_CHUNKS
                WHERE workspace = p_workspace AND id = rec.id;
            ELSE
                -- Simply rename the key
                UPDATE LIGHTRAG_RELATION_CHUNKS
                SET id = v_new_key, update_time = CURRENT_TIMESTAMP
                WHERE workspace = p_workspace AND id = rec.id;
            END IF;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
