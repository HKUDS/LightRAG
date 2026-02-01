-- lightrag_get_knowledge_graph: Server-side BFS for knowledge graph retrieval.
-- Eliminates N+1 round-trips by performing BFS entirely in PostgreSQL.
--
-- Algorithm: Multi-component BFS - starts from highest-degree node, then jumps
-- to next unvisited highest-degree node when a component is exhausted.
-- This handles disconnected graph components while maintaining connectivity within each.
--
-- Parameters:
--   p_workspace: workspace identifier
--   p_node_label: label to match nodes ('*' for all)
--   p_max_depth: maximum BFS depth (per component)
--   p_max_nodes: maximum nodes to return
--   p_min_degree: minimum node degree to include (default 1, set 0 for all nodes)
--
-- Returns: JSONB with {nodes, edges, is_truncated}

CREATE OR REPLACE FUNCTION lightrag_get_knowledge_graph(
    p_workspace VARCHAR,
    p_node_label VARCHAR,
    p_max_depth INT,
    p_max_nodes INT,
    p_min_degree INT DEFAULT 1
) RETURNS JSONB AS $$
DECLARE
    v_nodes JSONB := '[]'::jsonb;
    v_edges JSONB := '[]'::jsonb;
    v_is_truncated BOOLEAN := FALSE;
    v_frontier TEXT[];
    v_next_frontier TEXT[];
    v_visited TEXT[] := '{}';
    v_depth INT := 0;
    v_node_degrees JSONB;
    v_seed_node TEXT;
    v_total_eligible INT;
    v_eligible_nodes TEXT[];
    rec RECORD;
BEGIN
    -- Precompute all node degrees for filtering during BFS
    -- This is used to exclude peripheral nodes (degree < p_min_degree)
    SELECT jsonb_object_agg(node_id, degree)
    INTO v_node_degrees
    FROM (
        SELECT node_id, COUNT(*) AS degree FROM (
            SELECT source_id AS node_id FROM lightrag_graph_edges WHERE workspace = p_workspace
            UNION ALL
            SELECT target_id AS node_id FROM lightrag_graph_edges WHERE workspace = p_workspace
        ) e GROUP BY node_id
    ) deg;

    -- Get all eligible nodes ordered by degree (for multi-component traversal)
    IF p_node_label = '*' THEN
        SELECT array_agg(node_id ORDER BY degree DESC), COUNT(*)
        INTO v_eligible_nodes, v_total_eligible
        FROM (
            SELECT n.node_id, d.degree
            FROM lightrag_graph_nodes n
            INNER JOIN (
                SELECT node_id, COUNT(*) AS degree FROM (
                    SELECT source_id AS node_id FROM lightrag_graph_edges WHERE workspace = p_workspace
                    UNION ALL
                    SELECT target_id AS node_id FROM lightrag_graph_edges WHERE workspace = p_workspace
                ) e GROUP BY node_id
            ) d ON n.node_id = d.node_id
            WHERE n.workspace = p_workspace
              AND d.degree >= p_min_degree
        ) sub;
    ELSE
        SELECT array_agg(node_id ORDER BY degree DESC), COUNT(*)
        INTO v_eligible_nodes, v_total_eligible
        FROM (
            SELECT n.node_id, COALESCE(d.degree, 0) AS degree
            FROM lightrag_graph_nodes n
            LEFT JOIN (
                SELECT node_id, COUNT(*) AS degree FROM (
                    SELECT source_id AS node_id FROM lightrag_graph_edges WHERE workspace = p_workspace
                    UNION ALL
                    SELECT target_id AS node_id FROM lightrag_graph_edges WHERE workspace = p_workspace
                ) e GROUP BY node_id
            ) d ON n.node_id = d.node_id
            WHERE n.workspace = p_workspace
              AND n.node_id ILIKE '%' || p_node_label || '%'
              AND COALESCE(d.degree, 0) >= p_min_degree
        ) sub;
    END IF;

    -- No eligible nodes - return empty result
    IF v_eligible_nodes IS NULL OR array_length(v_eligible_nodes, 1) IS NULL THEN
        RETURN jsonb_build_object('nodes', '[]'::jsonb, 'edges', '[]'::jsonb, 'is_truncated', FALSE);
    END IF;

    -- Multi-component BFS: explore each disconnected component until max_nodes reached
    -- When BFS exhausts a component, jump to next unvisited highest-degree node
    WHILE array_length(v_visited, 1) IS NULL OR array_length(v_visited, 1) < p_max_nodes LOOP
        -- Find next seed: highest-degree unvisited eligible node
        v_seed_node := NULL;
        FOR i IN 1..array_length(v_eligible_nodes, 1) LOOP
            IF NOT v_eligible_nodes[i] = ANY(COALESCE(v_visited, '{}')) THEN
                v_seed_node := v_eligible_nodes[i];
                EXIT;
            END IF;
        END LOOP;

        -- No more unvisited nodes - we've explored all components
        IF v_seed_node IS NULL THEN
            EXIT;
        END IF;

        -- Add seed to visited
        IF v_visited IS NULL OR array_length(v_visited, 1) IS NULL THEN
            v_visited := ARRAY[v_seed_node];
        ELSE
            IF array_length(v_visited, 1) + 1 > p_max_nodes THEN
                v_is_truncated := TRUE;
                EXIT;
            END IF;
            v_visited := v_visited || v_seed_node;
        END IF;

        -- BFS from this seed within current component
        v_frontier := ARRAY[v_seed_node];
        v_depth := 0;

        WHILE v_depth < p_max_depth AND array_length(v_frontier, 1) > 0 AND NOT v_is_truncated LOOP
            v_next_frontier := '{}';

            -- Traverse edges from current frontier to discover neighbors
            FOR rec IN
                SELECT DISTINCT e.source_id, e.target_id
                FROM lightrag_graph_edges e
                WHERE e.workspace = p_workspace
                  AND (e.source_id = ANY(v_frontier) OR e.target_id = ANY(v_frontier))
            LOOP
                -- Discover unvisited neighbors (only if they meet minimum degree requirement)
                IF NOT rec.target_id = ANY(v_visited) THEN
                    IF COALESCE((v_node_degrees->>rec.target_id)::int, 0) >= p_min_degree THEN
                        IF array_length(v_visited, 1) + 1 > p_max_nodes THEN
                            v_is_truncated := TRUE;
                            EXIT;
                        END IF;
                        v_visited := v_visited || rec.target_id;
                        v_next_frontier := v_next_frontier || rec.target_id;
                    END IF;
                END IF;

                IF NOT rec.source_id = ANY(v_visited) THEN
                    IF COALESCE((v_node_degrees->>rec.source_id)::int, 0) >= p_min_degree THEN
                        IF array_length(v_visited, 1) + 1 > p_max_nodes THEN
                            v_is_truncated := TRUE;
                            EXIT;
                        END IF;
                        v_visited := v_visited || rec.source_id;
                        v_next_frontier := v_next_frontier || rec.source_id;
                    END IF;
                END IF;
            END LOOP;

            v_frontier := v_next_frontier;
            v_depth := v_depth + 1;
        END LOOP;

        -- If truncated, stop exploring more components
        IF v_is_truncated THEN
            EXIT;
        END IF;
    END LOOP;

    -- Set truncated if we couldn't include all eligible nodes
    IF v_total_eligible > COALESCE(array_length(v_visited, 1), 0) THEN
        v_is_truncated := TRUE;
    END IF;

    -- Build nodes JSONB from all visited node IDs
    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'id', n.node_id,
            'labels', jsonb_build_array(COALESCE(n.properties->>'entity_type', 'entity')),
            'properties', n.properties
        )
    ), '[]'::jsonb)
    INTO v_nodes
    FROM lightrag_graph_nodes n
    WHERE n.workspace = p_workspace AND n.node_id = ANY(v_visited);

    -- Fetch ALL edges where BOTH endpoints are in the visited set.
    -- This guarantees no orphan edges (edges referencing nodes not in the result),
    -- which would crash frontend graph renderers like d3.js force layout.
    SELECT COALESCE(jsonb_agg(
        jsonb_build_object(
            'id', e.source_id || '-' || e.target_id,
            'type', COALESCE(e.properties->>'relationship', 'related_to'),
            'source', e.source_id,
            'target', e.target_id,
            'properties', e.properties
        )
    ), '[]'::jsonb)
    INTO v_edges
    FROM lightrag_graph_edges e
    WHERE e.workspace = p_workspace
      AND e.source_id = ANY(v_visited)
      AND e.target_id = ANY(v_visited);

    RETURN jsonb_build_object(
        'nodes', v_nodes,
        'edges', v_edges,
        'is_truncated', v_is_truncated
    );
END;
$$ LANGUAGE plpgsql;
