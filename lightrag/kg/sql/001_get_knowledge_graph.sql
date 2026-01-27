-- lightrag_get_knowledge_graph: Server-side BFS for knowledge graph retrieval.
-- Eliminates N+1 round-trips by performing BFS entirely in PostgreSQL.
--
-- Parameters:
--   p_workspace: workspace identifier
--   p_node_label: label to match nodes ('*' for all)
--   p_max_depth: maximum BFS depth
--   p_max_nodes: maximum nodes to return
--
-- Returns: JSONB with {nodes, edges, is_truncated}

CREATE OR REPLACE FUNCTION lightrag_get_knowledge_graph(
    p_workspace VARCHAR,
    p_node_label VARCHAR,
    p_max_depth INT,
    p_max_nodes INT
) RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
    v_nodes JSONB := '[]'::jsonb;
    v_edges JSONB := '[]'::jsonb;
    v_is_truncated BOOLEAN := FALSE;
    v_frontier TEXT[];
    v_next_frontier TEXT[];
    v_visited TEXT[] := '{}';
    v_depth INT := 0;
    v_node_count INT := 0;
    rec RECORD;
BEGIN
    -- Find starting nodes
    IF p_node_label = '*' THEN
        SELECT array_agg(node_id), count(*)
        INTO v_frontier, v_node_count
        FROM (
            SELECT node_id FROM lightrag_graph_nodes
            WHERE workspace = p_workspace
            LIMIT p_max_nodes
        ) sub;
    ELSE
        SELECT array_agg(node_id), count(*)
        INTO v_frontier, v_node_count
        FROM (
            SELECT node_id FROM lightrag_graph_nodes
            WHERE workspace = p_workspace AND node_id ILIKE '%' || p_node_label || '%'
            LIMIT p_max_nodes
        ) sub;
    END IF;

    IF v_frontier IS NULL THEN
        RETURN jsonb_build_object('nodes', '[]'::jsonb, 'edges', '[]'::jsonb, 'is_truncated', FALSE);
    END IF;

    IF v_node_count >= p_max_nodes THEN
        v_is_truncated := TRUE;
    END IF;

    v_visited := v_frontier;

    -- Add starting nodes to result
    SELECT jsonb_agg(
        jsonb_build_object(
            'id', n.node_id,
            'labels', jsonb_build_array(COALESCE(n.properties->>'entity_type', 'entity')),
            'properties', n.properties
        )
    )
    INTO v_nodes
    FROM lightrag_graph_nodes n
    WHERE n.workspace = p_workspace AND n.node_id = ANY(v_frontier);

    IF v_nodes IS NULL THEN
        v_nodes := '[]'::jsonb;
    END IF;

    -- BFS traversal
    WHILE v_depth < p_max_depth AND array_length(v_frontier, 1) > 0 AND NOT v_is_truncated LOOP
        v_next_frontier := '{}';

        -- Get all edges from current frontier
        FOR rec IN
            SELECT DISTINCT e.source_id, e.target_id, e.properties
            FROM lightrag_graph_edges e
            WHERE e.workspace = p_workspace
              AND (e.source_id = ANY(v_frontier) OR e.target_id = ANY(v_frontier))
        LOOP
            -- Add edge to results (deduplicated by checking both directions)
            v_edges := v_edges || jsonb_build_object(
                'id', rec.source_id || '-' || rec.target_id,
                'type', COALESCE(rec.properties->>'relationship', 'related_to'),
                'source', rec.source_id,
                'target', rec.target_id,
                'properties', rec.properties
            );

            -- Check neighbors
            IF NOT rec.target_id = ANY(v_visited) THEN
                IF array_length(v_visited, 1) + 1 > p_max_nodes THEN
                    v_is_truncated := TRUE;
                    EXIT;
                END IF;
                v_visited := v_visited || rec.target_id;
                v_next_frontier := v_next_frontier || rec.target_id;
            END IF;

            IF NOT rec.source_id = ANY(v_visited) THEN
                IF array_length(v_visited, 1) + 1 > p_max_nodes THEN
                    v_is_truncated := TRUE;
                    EXIT;
                END IF;
                v_visited := v_visited || rec.source_id;
                v_next_frontier := v_next_frontier || rec.source_id;
            END IF;
        END LOOP;

        -- Add new frontier nodes to result
        IF array_length(v_next_frontier, 1) > 0 THEN
            SELECT v_nodes || COALESCE(jsonb_agg(
                jsonb_build_object(
                    'id', n.node_id,
                    'labels', jsonb_build_array(COALESCE(n.properties->>'entity_type', 'entity')),
                    'properties', n.properties
                )
            ), '[]'::jsonb)
            INTO v_nodes
            FROM lightrag_graph_nodes n
            WHERE n.workspace = p_workspace AND n.node_id = ANY(v_next_frontier);
        END IF;

        v_frontier := v_next_frontier;
        v_depth := v_depth + 1;
    END LOOP;

    RETURN jsonb_build_object(
        'nodes', COALESCE(v_nodes, '[]'::jsonb),
        'edges', COALESCE(v_edges, '[]'::jsonb),
        'is_truncated', v_is_truncated
    );
END;
$$ LANGUAGE plpgsql;
