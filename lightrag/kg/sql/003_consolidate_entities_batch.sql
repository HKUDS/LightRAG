-- lightrag_consolidate_entities_batch: Batch entity consolidation server-side.
-- Processes multiple entity merges in a single transaction.
--
-- Parameters:
--   p_workspace: workspace identifier
--   p_consolidations: JSONB array of {"old": "old_name", "canonical": "canonical_name"}
--
-- Returns: JSONB with {processed, results: [...]}

CREATE OR REPLACE FUNCTION lightrag_consolidate_entities_batch(
    p_workspace VARCHAR,
    p_consolidations JSONB
) RETURNS JSONB AS $$
DECLARE
    v_item JSONB;
    v_old_name VARCHAR;
    v_canonical_name VARCHAR;
    v_result JSONB;
    v_results JSONB := '[]'::jsonb;
    v_processed INT := 0;
BEGIN
    FOR v_item IN SELECT * FROM jsonb_array_elements(p_consolidations)
    LOOP
        v_old_name := v_item->>'old';
        v_canonical_name := v_item->>'canonical';

        -- Call the single-entity consolidation function
        v_result := lightrag_consolidate_entity(p_workspace, v_old_name, v_canonical_name);
        v_results := v_results || v_result;
        v_processed := v_processed + 1;
    END LOOP;

    RETURN jsonb_build_object(
        'status', 'ok',
        'processed', v_processed,
        'results', v_results
    );
EXCEPTION
    WHEN OTHERS THEN
        RETURN jsonb_build_object(
            'status', 'error',
            'message', SQLERRM,
            'processed', v_processed
        );
END;
$$ LANGUAGE plpgsql;
