// src/pages/SanitizeData.tsx
import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:9621';

export default function SanitizeData() {
  const [entities, setEntities] = useState<string[]>([]);
  const [filterText, setFilterText] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedEntities, setSelectedEntities] = useState<string[]>([]);
  const [firstEntity, setFirstEntity] = useState<string | null>(null);
  const [targetEntity, setTargetEntity] = useState('');
  const [entityType, setEntityType] = useState('');
  const [descriptionStrategy, setDescriptionStrategy] = useState('join_unique');
  const [sourceIdStrategy, setSourceIdStrategy] = useState('join_unique');

  // Dropdown suggestions = currently selected entities
  const targetOptions = [...selectedEntities].sort((a, b) => a.localeCompare(b));

  // Store fetched details: entityName → { desc, type, sourceId, filePath, relatedEntities, relationships }
  const [entityDetails, setEntityDetails] = useState<Record<string, any>>({});

  // Loading state (optional but nice UX)
  const [loadingDetails, setLoadingDetails] = useState<string[]>([]);

  const listContainerRef = useRef<HTMLDivElement>(null);
  const [rowsPerPage, setRowsPerPage] = useState(20); // initial guess

  // Modal state for editing relationships
  const [editRelationshipsModalOpen, setEditRelationshipsModalOpen] = useState(false);
  const [editingEntityForRel, setEditingEntityForRel] = useState<string | null>(null);

  // Temporary edits for relationships while modal is open
  const [relationshipEdits, setRelationshipEdits] = useState<Record<string, any>>({});  

  // Unique entity types from selected entities
  const [uniqueEntityTypes, setUniqueEntityTypes] = useState<string[]>([]);

  // Modal state for selecting entity type
  const [selectTypeModalOpen, setSelectTypeModalOpen] = useState(false);

  // State for the "Select Type" Modal
  const [allEntityTypes, setAllEntityTypes] = useState<string[]>([]);
  const [selectedModalType, setSelectedModalType] = useState<string>('');
  const [loadingTypes, setLoadingTypes] = useState(false);
  const [modalFilterText, setModalFilterText] = useState('');
  const [typeSelectionContext, setTypeSelectionContext] = useState<'main' | 'create' | 'edit'>('main');
  const modalInputRef = useRef<HTMLInputElement>(null);
  const typeItemRefs = useRef<HTMLDivElement[]>([]);

  const [typesLoading, setTypesLoading] = useState(true);  
  const [filterMode, setFilterMode] = useState<'none' | 'selected' | 'type' | 'orphan'>('none');
  const [typeFilteredEntities, setTypeFilteredEntities] = useState<string[]>([]);
  const [entityTypeMap, setEntityTypeMap] = useState<Record<string, string>>({});

  const [orphanFilteredEntities, setOrphanFilteredEntities] = useState<string[]>([]);
  const [entityOrphanMap, setEntityOrphanMap] = useState<Record<string, boolean>>({});

  // Create Entity Modal state
  const [createEntityModalOpen, setCreateEntityModalOpen] = useState(false);
  const [createEntityName, setCreateEntityName] = useState('');
  const [createEntityDescription, setCreateEntityDescription] = useState('');
  const [createEntityType, setCreateEntityType] = useState('');
  const [createEntitySourceId, setCreateEntitySourceId] = useState('');
  const [createError, setCreateError] = useState<string | null>(null);  // For error messages

    // Edit Entity Modal state (replaces old description-only)
  const [editEntityModalOpen, setEditEntityModalOpen] = useState(false);
  const [editEntityOriginalName, setEditEntityOriginalName] = useState<string | null>(null); // For rename detection
  const [editEntityName, setEditEntityName] = useState('');
  const [editEntityDescription, setEditEntityDescription] = useState('');
  const [editEntityType, setEditEntityType] = useState('');
  const [editEntitySourceId, setEditEntitySourceId] = useState('');
  const [editError, setEditError] = useState<string | null>(null);

  // Create Relationship Modal state
  const [createRelModalOpen, setCreateRelModalOpen] = useState(false);
  const [createRelDescription, setCreateRelDescription] = useState('');
  const [createRelKeywords, setCreateRelKeywords] = useState('');
  const [createRelWeight, setCreateRelWeight] = useState(1.0);
  const [createRelError, setCreateRelError] = useState<string | null>(null);  

  const createNameRef = useRef<HTMLInputElement>(null);
  const createSourceRef = useRef<HTMLInputElement>(null);

  const filteredModalTypes = allEntityTypes.filter((type) =>
    type.toLowerCase().includes(modalFilterText.toLowerCase())
  );

  // Build entityTypeMap and entityOrphanMap with single fetch per entity
  const fetchEntityDetails = async (entityList: string[]) => {
    try {
      const typeMap: Record<string, string> = {};
      const orphanMap: Record<string, boolean> = {};
      await Promise.all(
        entityList.map(async (name: string) => {
          try {
            const detailRes = await axios.get(
              `${API_BASE}/graphs?label=${encodeURIComponent(name)}&max_depth=1&max_nodes=2`
            );
            // Find main node by id (robust to order)
            const mainNode = detailRes.data.nodes?.find((node: any) => node.id === name);
            const type = mainNode?.properties?.entity_type || '';
            typeMap[name] = type;

            // Detect orphan from the same response
            const isOrphan = (detailRes.data.nodes?.length || 0) <= 1 && (detailRes.data.edges?.length || 0) === 0;
            orphanMap[name] = isOrphan;
          } catch (err) {
            console.error(`Error fetching details for ${name}:`, err);
          }
        })
      );
      setEntityTypeMap(typeMap);
      setEntityOrphanMap(orphanMap);
      // console.log('Types loaded:', Object.keys(typeMap).length); // Debug
      // console.log('Orphans loaded:', Object.values(orphanMap).filter(Boolean).length); // Debug
    } catch (err) {
      console.error('Failed to fetch entity details:', err);
    } finally {
      setTypesLoading(false);
    }
  };

  // For loading the Select Type modal window
  const fetchAllTypes = async () => {
    setLoadingTypes(true);
    try {
      // 1. Get all entity names
      const listRes = await axios.get(`${API_BASE}/graph/label/list`);
      const entityNames = listRes.data as string[];
      
      // 2. Fetch types for each name (in parallel)
      // Note: If you have thousands of entities, we may need to chunk this later.
      const typeSet = new Set<string>();
      
      await Promise.all(
        entityNames.map(async (name) => {
          try {
            const detailRes = await axios.get(
              `${API_BASE}/graphs?label=${encodeURIComponent(name)}&max_depth=1&max_nodes=1`
            );
            const type = detailRes.data.nodes?.[0]?.properties?.entity_type;
            if (type) typeSet.add(type);
          } catch (err) {
            console.error(`Error fetching type for ${name}:`, err);
          }
        })
      );

      setAllEntityTypes(Array.from(typeSet).sort());
    } catch (err) {
      console.error('Failed to fetch full type list:', err);
    } finally {
      setLoadingTypes(false);
    }
  };

  useEffect(() => {
    if (selectTypeModalOpen) {
      fetchAllTypes();
    }
  }, [selectTypeModalOpen]);

  // Fetch entities
  useEffect(() => {
    const fetchEntities = async () => {
      try {
        const response = await axios.get(`${API_BASE}/graph/label/list`);
        const sorted = (response.data as string[]).sort((a, b) =>
          a.toLowerCase().localeCompare(b.toLowerCase())
        );
        setEntities(sorted);
        fetchEntityDetails(sorted);  // ← Pass sorted here
      } catch (err) {
        console.error('Failed to load entities:', err);
      }
    };
    fetchEntities();
  }, []);

  // Calculate visible rows dynamically
  useEffect(() => {
    const updateRowsPerPage = () => {
      if (!listContainerRef.current) return;

      const container = listContainerRef.current;
      const containerHeight = container.clientHeight;
      const firstRow = container.querySelector('div.grid');
      const rowHeight = firstRow ? firstRow.getBoundingClientRect().height : 36; // fallback

      const headerHeight = 42;
      const availableHeight = containerHeight - headerHeight;
      const calculated = Math.max(5, Math.floor(availableHeight / rowHeight));

      setRowsPerPage(calculated);
    };

    updateRowsPerPage();
    window.addEventListener('resize', updateRowsPerPage);
    return () => window.removeEventListener('resize', updateRowsPerPage);
  }, [entities, filterText]);

  const filteredEntities = entities.filter((e) =>
    e.toLowerCase().includes(filterText.toLowerCase())
  );

  const totalPages = Math.max(1, Math.ceil(filteredEntities.length / rowsPerPage));
  const startIndex = (currentPage - 1) * rowsPerPage;
  const paginatedEntities = filteredEntities.slice(startIndex, startIndex + rowsPerPage);

  // Fetch details for all selected entities when "Show Desc" is turned on
  useEffect(() => {
    if (selectedEntities.length > 0) {
      selectedEntities.forEach((entityName) => {
        fetchEntityDetail(entityName);
      });
    }
  }, [selectedEntities]);  // ← only this dependency now

  // Reset page when filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [filterText]);

  // Listen for Esc key to cancel any open modal
  useEffect(() => {
    const handleEscKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (editRelationshipsModalOpen) {
          setEditRelationshipsModalOpen(false);
        } else if (selectTypeModalOpen) { 
          setSelectTypeModalOpen(false);
        } else if (createEntityModalOpen) { 
          setCreateEntityModalOpen(false);
        } else if (editEntityModalOpen) {
          setEditEntityModalOpen(false);
        } else if (createRelModalOpen) {
          setCreateRelModalOpen(false);
        }
      }
    };
    document.addEventListener('keydown', handleEscKey);
    return () => document.removeEventListener('keydown', handleEscKey);
  }, [createRelModalOpen, createEntityModalOpen, editEntityModalOpen, editRelationshipsModalOpen, selectTypeModalOpen]);

  // Update unique entity types from selected entities' details
  useEffect(() => {
    const types = new Set<string>();

    selectedEntities.forEach((name) => {
      const type = entityDetails[name]?.type;
      if (type) {
        types.add(type);
      }
    });
    setUniqueEntityTypes(Array.from(types).sort());
  }, [selectedEntities, entityDetails]);

  useEffect(() => {
    if (selectTypeModalOpen) {
      setModalFilterText(''); // Reset the search box
      fetchAllTypes();
    }
  }, [selectTypeModalOpen]);

  useEffect(() => {
    if (filterMode === 'type' && entityType) {
      // console.log('Re-filtering for type:', entityType); // Optional debug
      const entitiesOfType = entities.filter((name) => entityTypeMap[name] === entityType).sort((a, b) =>
        a.toLowerCase().localeCompare(b.toLowerCase())
      );
      setTypeFilteredEntities(entitiesOfType);
    } else if (filterMode === 'type' && !entityType) {
      setTypeFilteredEntities([]); // Clear if type is empty
    }
  }, [entityType, filterMode, entities, entityTypeMap]);

  useEffect(() => {
    typeItemRefs.current = [];
  }, [filteredModalTypes]);

  useEffect(() => {
    if (filteredModalTypes.length > 0 && !selectedModalType) {
      setSelectedModalType(filteredModalTypes[0]);
    }
  }, [filteredModalTypes, selectedModalType]);

  useEffect(() => {
    if (createEntityModalOpen) {
      createNameRef.current?.focus();
    }
  }, [createEntityModalOpen]);  

  // Pagination handlers
  const goToFirst = () => setCurrentPage(1);
  const goToPrev = () => setCurrentPage((p) => Math.max(1, p - 1));
  const goToNext = () => setCurrentPage((p) => Math.min(totalPages, p + 1));
  const goToLast = () => setCurrentPage(totalPages);

  const handlePageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (val === '') return;
    const num = parseInt(val, 10);
    if (!isNaN(num)) {
      setCurrentPage(Math.max(1, Math.min(totalPages, num)));
    }
  };

  const handleShowSelectedOnly = () => {
    if (selectedEntities.length === 0) {
      alert('Please select at least one entity first (check the boxes on the left).');
      return;
    }
    
    setFilterMode('selected');
    setCurrentPage(1);
    setFilterText('');
  };

  const handleShowAllOfType = () => {
    if (typesLoading) {
      alert('Entity types are still loading. Please wait a moment and try again.');
      return;
    }
    if (!entityType) {
      alert('Please select or enter an entity type first.');
      return;
    }
    
    // console.log('Showing all of type:', entityType);
    const entitiesOfType = entities.filter((name) => entityTypeMap[name] === entityType).sort((a, b) =>
      a.toLowerCase().localeCompare(b.toLowerCase())
    );
    // console.log('Filtered count:', entitiesOfType.length);
    // console.log('entityTypeMap sample:', Object.entries(entityTypeMap).slice(0, 5));
    
    setTypeFilteredEntities(entitiesOfType);
    setFilterMode('type');
    setCurrentPage(1);
    setFilterText('');
  };

  const handleShowOrphans = () => {
    if (typesLoading) {  // Reuses the same loading state (since orphans load with types)
      alert('Entity details are still loading. Please wait a moment and try again.');
      return;
    }
    
    console.log('Showing orphans'); // Debug
    const orphans = entities.filter((name) => entityOrphanMap[name] === true).sort((a, b) =>
      a.toLowerCase().localeCompare(b.toLowerCase())
    );
    console.log('Orphan count:', orphans.length); // Debug
    
    setOrphanFilteredEntities(orphans);
    setFilterMode('orphan');
    setCurrentPage(1);
    setFilterText('');
  };

  const handleClearSelected = () => {
    setSelectedEntities([]);     // uncheck all checkboxes
    setFirstEntity(null);        // deselect "Keep First" radio
  };

  const handleCreateEntity = async () => {
    if (!createEntityName.trim()) {
      setCreateError('Entity name is required.');
      return;
    }

    setCreateError(null);

    try {
      const entityData: Record<string, any> = {
        description: createEntityDescription,
        entity_type: createEntityType,
      };
      if (createEntitySourceId.trim()) {
        entityData.source_id = createEntitySourceId;
      }

      const response = await axios.post(`${API_BASE}/graph/entity/create`, {
        entity_name: createEntityName,
        entity_data: entityData,
      });

      if (response.status === 200) {
        // Success: Close modal, refresh entities
        setCreateEntityModalOpen(false);
        // Refresh entity list (re-fetch)
        const listRes = await axios.get(`${API_BASE}/graph/label/list`);
        const sorted = (listRes.data as string[]).sort((a, b) =>
          a.toLowerCase().localeCompare(b.toLowerCase())
        );
        setEntities(sorted);
        // Also refresh types/orphans map if needed (call fetchEntityTypes again)
        // For simplicity, reload page or add a full refresh here if necessary

        fetchSingleEntityDetails(createEntityName);

        alert(response.data.message);  // Show success message
      }
    } catch (err: any) {
      console.error('Failed to create entity:', err);
      let errorMsg = 'Failed to create entity. Check console for details.';
      if (err.response?.data?.detail) {
        errorMsg = err.response.data.detail;  // e.g., "Entity 'Walsh' already exists"
      } else if (err.response?.data?.message) {
        errorMsg = err.response.data.message;  // Fallback if API uses "message"
      } else if (err.message) {
        errorMsg = err.message;  // Broader fallback (e.g., network errors)
      }
      setCreateError(errorMsg);  // Show in modal
    }
  };

  const handleSaveEntity = async () => {
    if (!editEntityName.trim()) {
      setEditError('Entity name is required.');
      return;
    }
    // Prevent duplicate name frontend (but backend will confirm)
    if (editEntityName !== editEntityOriginalName && entities.includes(editEntityName)) {
      setEditError(`Entity name "${editEntityName}" already exists.`);
      return;
    }
    setEditError(null);

    try {
      const updatedData: Record<string, any> = {
        description: editEntityDescription,
        entity_type: editEntityType,
        source_id: editEntitySourceId || '',
      };

      const allowRename = editEntityName !== editEntityOriginalName;
      const allowMerge = false;  // Explicitly false to prevent merge on conflict

      if (allowRename) {
        updatedData.entity_name = editEntityName;  // ← Put new name in updated_data
      }

      const payload = {
        entity_name: editEntityOriginalName,
        updated_data: updatedData,
        allow_rename: allowRename,
        allow_merge: allowMerge,
      };

      console.log('Sending edit payload:', JSON.stringify(payload, null, 2));  // Debug

      const response = await axios.post(`${API_BASE}/graph/entity/edit`, payload);

      console.log('Edit response:', response.data);  // Debug

      if (response.status === 200) {
        setEditEntityModalOpen(false);
        setEditEntityOriginalName(null);

        if (allowRename && editEntityOriginalName) {
          // Update selectedEntities with new name
          setSelectedEntities((prev) =>
            prev.map((n) => (n === editEntityOriginalName ? editEntityName : n))
          );

          // Update firstEntity if it was the old name
          if (firstEntity === editEntityOriginalName) {
            setFirstEntity(editEntityName);
          }

          // Migrate entityDetails to new key
          setEntityDetails((prev) => {
            if (prev[editEntityOriginalName]) {
              const newDetails = { ...prev };
              newDetails[editEntityName] = { ...prev[editEntityOriginalName] };
              delete newDetails[editEntityOriginalName];
              return newDetails;
            }
            return prev;
          });
        }

        // Full refresh
        const listRes = await axios.get(`${API_BASE}/graph/label/list`);
        const sorted = (listRes.data as string[]).sort((a, b) =>
          a.toLowerCase().localeCompare(b.toLowerCase())
        );
        setEntities(sorted);
        fetchEntityDetails(sorted);

        if (allowRename && editEntityOriginalName) {
          // Update orphanFilteredEntities if in orphan mode
          if (filterMode === 'orphan') {
            setOrphanFilteredEntities((prev) =>
              prev.map((n) => (n === editEntityOriginalName ? editEntityName : n))
            );
          }

          // Re-fetch single details (already there, but ensure)
          fetchSingleEntityDetails(editEntityName);
        }

        // Re-fetch details for the renamed entity (ensures panel sync)
        fetchEntityDetail(editEntityName, true);  // Force refresh

        // Re-fetch details for final entity
        const finalName = response.data.operation_summary?.final_entity || editEntityName;
        fetchEntityDetail(finalName, true);

        alert('Entity updated successfully!');
      } else {
        setEditError('Update failed with status: ' + response.status);
      }
    } catch (err: any) {
      console.error('Failed to edit entity:', err);
      let errorMsg = 'Failed to update entity.';
      if (err.response?.data?.detail) {
        errorMsg = err.response.data.detail;
      } else if (err.response?.data?.message) {
        errorMsg = err.response.data.message;
      } else if (err.message) {
        errorMsg = err.message;
      }
      setEditError(errorMsg);
    }
  };

  const handleDeleteEntities = async () => {
    if (selectedEntities.length === 0 || filterMode !== 'selected') return;  // Safety check

    if (!confirm(`Are you sure you want to delete ${selectedEntities.length} entity/entities? This cannot be undone.`)) {
      return;
    }

    try {
      let successCount = 0;
      let errorMessages: string[] = [];

      for (const entityName of selectedEntities) {
        try {
          const payload = { entity_name: entityName };
          const response = await axios.delete(`${API_BASE}/documents/delete_entity`, { data: payload });

          if (response.status === 200) {
            successCount++;
          } else {
            errorMessages.push(`Failed to delete ${entityName} (status: ${response.status})`);
          }
        } catch (err: any) {
          console.error(`Error deleting ${entityName}:`, err);
          let msg = `Failed to delete ${entityName}.`;
          if (err.response?.status === 404) {
            msg = `${entityName} not found.`;
          } else if (err.response?.data?.detail) {
            msg = err.response.data.detail;
          }
          errorMessages.push(msg);
        }
      }

      // Full refresh after deletes
      const listRes = await axios.get(`${API_BASE}/graph/label/list`);
      const sorted = (listRes.data as string[]).sort((a, b) =>
        a.toLowerCase().localeCompare(b.toLowerCase())
      );
      setEntities(sorted);
      fetchEntityDetails(sorted);

      // Clear selections and firstEntity
      setSelectedEntities([]);
      setFirstEntity(null);

      // Show summary
      if (successCount === selectedEntities.length) {
        alert('All selected entities deleted successfully!');
      } else if (successCount > 0) {
        alert(`Deleted ${successCount} entity/entities successfully. Errors: ${errorMessages.join(', ')}`);
      } else {
        alert(`Failed to delete any entities. Errors: ${errorMessages.join(', ')}`);
      }
    } catch (err) {
      console.error('Unexpected error during delete:', err);
      alert('An unexpected error occurred during delete.');
    }
  };

  const handleCreateRelationship = async () => {
    if (!createRelDescription.trim()) {
      setCreateRelError('Relationship description is required.');
      return;
    }
    setCreateRelError(null);

    // Derive source and target
    if (selectedEntities.length !== 2 || !targetEntity || !selectedEntities.includes(targetEntity)) {
      setCreateRelError('Invalid selection or target.');
      return;
    }
    const sourceEntity = selectedEntities.find((n) => n !== targetEntity) || '';

    try {
      const relationData: Record<string, any> = {
        description: createRelDescription,
        keywords: createRelKeywords,
        weight: createRelWeight,
      };

      const payload = {
        source_entity: sourceEntity,
        target_entity: targetEntity,
        relation_data: relationData,
      };

      console.log('Create rel payload:', JSON.stringify(payload, null, 2));  // Debug

      const response = await axios.post(`${API_BASE}/graph/relation/create`, payload);

      console.log('Create rel response:', response.data);  // Debug

      if (response.status === 200) {
        setCreateRelModalOpen(false);

        // Refresh details for affected entities
        fetchEntityDetail(sourceEntity, true);
        fetchEntityDetail(targetEntity, true);

        // Optional: Full refresh if needed (e.g., for orphans if relations change)
        // const listRes = await axios.get(`${API_BASE}/graph/label/list`);
        // const sorted = (listRes.data as string[]).sort(...);
        // setEntities(sorted);
        // fetchEntityDetails(sorted);

        alert('Relationship created successfully!');
      }
    } catch (err: any) {
      console.error('Failed to create relationship:', err);
      let errorMsg = 'Failed to create relationship.';
      if (err.response?.status === 400) {
        errorMsg = err.response?.data?.detail || 'Invalid request—check if entities exist or duplicate relationship.';
      } else if (err.response?.data?.detail) {
        errorMsg = err.response.data.detail;
      } else if (err.response?.data?.message) {
        errorMsg = err.response.data.message;
      } else if (err.message) {
        errorMsg = err.message;
      }
      setCreateRelError(errorMsg);
    }
  };

  const handleMergeEntities = async () => {
    if (selectedEntities.length < 2 || filterMode !== 'selected') return;  // Need at least 2 for merge

    if (!targetEntity || !selectedEntities.includes(targetEntity)) {
      alert('Please select a target entity from the dropdown first.');
      return;
    }

    const entitiesToChange = selectedEntities.filter((n) => n !== targetEntity);

    if (entitiesToChange.length === 0) {
      alert('No source entities to merge (select at least one besides the target).');
      return;
    }

    if (!confirm(`Are you sure you want to merge ${entitiesToChange.length} entity/entities into "${targetEntity}"? Sources will be deleted. This cannot be undone.`)) {
      return;
    }

    try {
      const payload = {
        entities_to_change: entitiesToChange,
        entity_to_change_into: targetEntity,
      };

      console.log('Merge payload:', JSON.stringify(payload, null, 2));  // Debug

      const response = await axios.post(`${API_BASE}/graph/entities/merge`, payload);

      console.log('Merge response:', response.data);  // Debug

      if (response.status === 200) {
        // Full refresh after merge
        const listRes = await axios.get(`${API_BASE}/graph/label/list`);
        const sorted = (listRes.data as string[]).sort((a, b) =>
          a.toLowerCase().localeCompare(b.toLowerCase())
        );
        setEntities(sorted);
        fetchEntityDetails(sorted);

        // Refresh target details (shows transferred relations)
        fetchEntityDetail(targetEntity, true);

        // Clear selections and firstEntity
        setSelectedEntities([]);
        setFirstEntity(null);
        setTargetEntity('');  // Optional: clear target

        alert(response.data.message || 'Entities merged successfully!');
      }
    } catch (err: any) {
      console.error('Failed to merge entities:', err);
      let errorMsg = 'Failed to merge entities.';
      if (err.response?.status === 400) {
        errorMsg = err.response?.data?.detail || 'Invalid request—check if target exists or sources are valid.';
      } else if (err.response?.data?.detail) {
        errorMsg = err.response.data.detail;
      } else if (err.response?.data?.message) {
        errorMsg = err.response.data.message;
      } else if (err.message) {
        errorMsg = err.message;
      }
      alert(errorMsg);
    }
  };

  const fetchSingleEntityDetails = async (name: string) => {
    try {
      const detailRes = await axios.get(
        `${API_BASE}/graphs?label=${encodeURIComponent(name)}&max_depth=1&max_nodes=2`
      );
      // Find main node by id (robust to order)
      const mainNode = detailRes.data.nodes?.find((node: any) => node.id === name);
      const type = mainNode?.properties?.entity_type || '';
      
      const isOrphan = (detailRes.data.nodes?.length || 0) <= 1 && (detailRes.data.edges?.length || 0) === 0;
      
      setEntityTypeMap(prev => ({ ...prev, [name]: type }));
      setEntityOrphanMap(prev => ({ ...prev, [name]: isOrphan }));
    } catch (err) {
      console.error(`Error fetching single entity details for ${name}:`, err);
    }
  };

  const fetchEntityDetail = async (entityName: string, force = false) => {

    // console.log(`fetchEntityDetail called for: "${entityName}"`);

    // Skip if we already have it
    //if (entityDetails[entityName]) return;
    if (entityDetails[entityName] && !force) {
      // console.log(`Already have details for "${entityName}" - skipping`);
      return;
    }

    // console.log(`Fetching details for "${entityName}"...`);

    setLoadingDetails((prev) => [...prev, entityName]);

    try {
      console.log("Making axios request...");
      const encodedName = encodeURIComponent(entityName);
      const url = `${API_BASE}/graphs?label=${encodedName}&max_depth=1&max_nodes=20000`;
      console.log("Request URL:", url);

    const response = await axios.get(url);
    console.log("Response received:", response.status, response.data);

      const data = response.data;

      // Parse the response (based on your Python code structure)
      let mainDesc = "No description found.";
      let mainType = "";
      let mainSourceId = "";
      let mainFilePath = "";

      const related: any[] = [];
      const edges: any[] = [];

      // Process nodes
      (data.nodes || []).forEach((node: any) => {
        const props = node.properties || {};
        if (node.id === entityName) {
          mainDesc = props.description || mainDesc;
          mainType = props.entity_type || mainType;
          mainSourceId = props.source_id || "";
          mainFilePath = props.file_path || "";
        } else {
          related.push({
            name: node.id,
            type: props.entity_type || "",
            description: props.description || "No description",
          });
        }
      });

      // Process edges/relationships
      (data.edges || []).forEach((edge: any) => {
        const props = edge.properties || {};
        edges.push({
          from: edge.source,
          to: edge.target,
          relation: props.description || "",
          weight: props.weight || 1.0,
          keywords: props.keywords || "",
        });
      });

      // Store the parsed data
      setEntityDetails((prev) => ({
        ...prev,
        [entityName]: {
          type: mainType,
          description: mainDesc,
          sourceId: mainSourceId,
          filePath: mainFilePath,
          relatedEntities: related,
          relationships: edges,
        },
      }));
    } catch (err) {
      console.error(`Error fetching "${entityName}":`, err);  
      // Optional: store error state
    } finally {
      setLoadingDetails((prev) => prev.filter((n) => n !== entityName));
    }
  };

  const openEditEntityModal = (entityName: string) => {
    setEditEntityOriginalName(entityName);
    setEditEntityName(entityName);
    setEditEntityDescription(entityDetails[entityName]?.description || '');
    setEditEntityType(entityDetails[entityName]?.type || '');
    setEditEntitySourceId(entityDetails[entityName]?.sourceId || '');
    setEditError(null);
    setEditEntityModalOpen(true);
  };

  const openEditRelationshipsModal = (entityName: string) => {
    setEditingEntityForRel(entityName);

    // Initialize editable copies of relationships
    const initialEdits: Record<string, any> = {};
    entityDetails[entityName]?.relationships?.forEach((rel: any) => {
      const key = `${rel.from}-${rel.to}`; // simple unique key
      initialEdits[key] = { ...rel }; // shallow copy
    });

    setRelationshipEdits(initialEdits);

    setEditRelationshipsModalOpen(true);
  };

  const triggerGraphRefresh = async () => {
  try {
    const response = await axios.post(`${API_BASE}/graph/refresh-data`);
    if (response.status === 200) {
      // console.log("Graph data refresh triggered successfully");
      // Optional: show toast/alert later
    }
  } catch (err) {
    console.error("Failed to trigger graph refresh:", err);
    alert("Changes saved, but failed to refresh graph view. Please restart server or try again.");
  }
  };

  const saveAllRelationshipChanges = async () => {
    if (!editingEntityForRel) return;

    try {
      let successCount = 0;

      for (const [key, editedRel] of Object.entries(relationshipEdits)) {
        const originalRel = entityDetails[editingEntityForRel].relationships.find(
          (r: any) => `${r.from}-${r.to}` === key
        );

        if (!originalRel) continue;

        // Only send if something changed
        if (
          editedRel.relation !== originalRel.relation ||
          editedRel.weight !== originalRel.weight ||
          editedRel.keywords !== originalRel.keywords
        ) {
          const payload = {
            source_id: editedRel.from,
            target_id: editedRel.to,
            updated_data: {
              description: editedRel.relation,
              keywords: editedRel.keywords,
              weight: editedRel.weight,
            },
          };

          const res = await axios.post(`${API_BASE}/graph/relation/edit`, payload);
          if (res.status === 200) successCount++;
        }
      }

      if (successCount > 0) {
        // Refresh local cache (optional but nice)
        await fetchEntityDetail(editingEntityForRel, true);

        // Commented out because it isn't needed to see updates without refreshing the webpage.
        // Trigger full graph refresh so other parts of UI see changes
        // await triggerGraphRefresh();

        alert(`Saved ${successCount} relationship change(s) successfully!`);
      } else {
        alert("No changes detected.");
      }      

      setEditRelationshipsModalOpen(false);
    } catch (err) {
      console.error("Failed to save relationship changes:", err);
      alert("Error saving relationships. Check console.");
    }
  };

  const deleteRelationship = async (from: string, to: string) => {
    if (!confirm(`Are you sure you want to delete the relationship from ${from} to ${to}? This cannot be undone.`)) {
      return;
    }

    try {
      // Change to DELETE method + correct parameter names
      await axios.delete(`${API_BASE}/documents/delete_relation`, {
        data: {  // Use 'data' for body in DELETE (axios requires this for non-GET methods)
          source_entity: from,
          target_entity: to,
        },
      });

      // console.log(`Deleted relationship: ${from} → ${to}`);

      // Remove from temp edits
      setRelationshipEdits((prev) => {
        const newEdits = { ...prev };
        delete newEdits[`${from}-${to}`];
        return newEdits;
      });

      // Re-fetch entity details to update local cache
      if (editingEntityForRel) {
        await fetchEntityDetail(editingEntityForRel, true); // Force refresh
      }

      // Trigger full graph refresh
      // await triggerGraphRefresh(); // Commented out because it has no effect

      alert("Relationship deleted successfully!");
    } catch (err) {
      console.error("Failed to delete relationship:", err);
      alert("Error deleting relationship. Check console.");
    }
  };  

  const displayEntities = filterMode === 'selected'
    ? [...selectedEntities].sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()))
    : filterMode === 'type'
    ? typeFilteredEntities
    : filterMode === 'orphan'
    ? orphanFilteredEntities
    : paginatedEntities; 

  return (
    <div className="h-full flex flex-col">
      {/* Top row - minimum height to ensure controls are visible */}
      <div className="h-auto flex border-b border-gray-300">  
        {/* Upper Left */}
        <div className={`w-1/4 border-r border-gray-300 p-2.5 flex flex-col gap-2.5 ${filterMode !== 'none' ? 'bg-indigo-50' : ''}`}>
          {filterMode !== 'none' && (
            <div className="text-xs text-indigo-700 bg-indigo-50 p-2 rounded mb-2">
              {filterMode === 'type'
                ? `Showing only entities of type: ${entityType} (${displayEntities.length})`
                : filterMode === 'orphan'
                ? `Showing only orphan entities (${displayEntities.length})`
                : `Showing only selected entities (${displayEntities.length})`}
            </div>
          )}

          {filterMode === 'none' && (
            <>
              <input
                type="text"
                placeholder="Filter entities..."
                className="w-full px-3 py-1 border border-gray-300 rounded text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
              />
              <div className="flex flex-wrap gap-1">
                <button 
                  onClick={handleShowAllOfType}
                  disabled={typesLoading}
                  className={`px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs ${typesLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  All Of Type
                </button>
                <button 
                  onClick={handleShowOrphans}  // ← Added onClick and disabled
                  disabled={typesLoading}
                  className={`px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs ${typesLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  Orphans
                </button>

                {/* Pagination (unchanged) */}
                <div className="flex flex-wrap gap-1 items-center">
                  <button
                    onClick={goToFirst}
                    disabled={currentPage === 1}
                    className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    First
                  </button>
                  <button
                    onClick={goToPrev}
                    disabled={currentPage === 1}
                    className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Prev
                  </button>

                  <div className="flex items-center gap-1 bg-gray-50 border border-gray-300 rounded px-1.5 py-0.5 text-xs">
                    Pg
                    <input
                      type="number"
                      min={1}
                      max={Math.ceil(filteredEntities.length / rowsPerPage) || 1}
                      value={currentPage}
                      onChange={handlePageInputChange}
                      className="w-10 text-center border border-gray-400 rounded text-xs focus:outline-none focus:ring-1 focus:ring-blue-500 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    />
                    /{Math.ceil(filteredEntities.length / rowsPerPage) || 1}
                  </div>

                  <button
                    onClick={goToNext}
                    disabled={currentPage >= Math.ceil(filteredEntities.length / rowsPerPage)}
                    className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next
                  </button>
                  <button
                    onClick={goToLast}
                    disabled={currentPage >= Math.ceil(filteredEntities.length / rowsPerPage)}
                    className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Last
                  </button>
                </div>
              </div>
            </>
          )}

          <div className="flex flex-wrap gap-1">
            {filterMode === 'none' ? (
              <button
                onClick={handleClearSelected}
                className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs"
              >
                Clear Sel.
              </button>
            ) : (
              <div className="w-[78px]" />   // ← invisible placeholder
            )}

            {filterMode === 'type' ? (
              <button
                className="px-2 py-0.5 border rounded text-xs transition-colors bg-indigo-600 text-white border-indigo-700"
                disabled
              >
                All Of Type
              </button>
            ) : filterMode === 'orphan' ? (
              <button
                className="px-2 py-0.5 border rounded text-xs transition-colors bg-indigo-600 text-white border-indigo-700"
                disabled
              >
                Orphans
              </button>
            ) : (
              <button
                onClick={handleShowSelectedOnly}
                className={`px-2 py-0.5 border rounded text-xs transition-colors ${
                  filterMode === 'selected'
                    ? 'bg-indigo-600 text-white border-indigo-700'
                    : 'bg-indigo-50 hover:bg-indigo-100 border-indigo-200 text-indigo-700'
                }`}
                disabled={filterMode === 'selected'}
              >
                Show Sel. Only
              </button>
            )}

            <button
              onClick={() => {
                setFilterMode('none');
                setFilterText('');
                setCurrentPage(1);
                setEntityType('');  // Optional: clears type as in previous guidance
              }}
              className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs"
            >
              Show All
            </button>
            <button
              onClick={() => {
                setFilterMode('none');
                setSelectedEntities([]);
                setFirstEntity(null);
                setFilterText('');
                setCurrentPage(1);
                setEntityType('');  // Optional: clears type as in previous guidance
              }}
              className="px-2 py-0.5 bg-red-50 hover:bg-red-100 border border-red-200 rounded text-xs text-red-700"
            >
              Reset All
            </button>
          </div>
        </div>

        {/* Upper Right - should now always be visible */}
        <div className="w-3/4 p-2.5 flex flex-col gap-2.5">
          <div className="flex flex-wrap items-end gap-2.5">
            {/* Moved: Select Type button + input */}
            <div className="min-w-[200px]">
              <button 
                onClick={() => {
                  setTypeSelectionContext('main');
                  setSelectTypeModalOpen(true);
                }}
                className="block w-full px-3 py-0.5 bg-gray-200 hover:bg-gray-300 border border-gray-300 border-b-0 rounded-t-md text-xs font-medium text-gray-800 text-left cursor-pointer shadow-sm"
              >
                Select Type
              </button>
              <div className="relative">
                <input
                  type="text"
                  list="entity-type-options"
                  className="w-full px-3 py-1.5 border border-gray-300 rounded-b-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                  value={entityType}
                  onChange={(e) => setEntityType(e.target.value)}
                  placeholder="Type or filter..."
                  autoComplete="off"
                />
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
                <datalist id="entity-type-options">
                  {uniqueEntityTypes.map((type) => (
                    <option key={type} value={type} />
                  ))}
                </datalist>
              </div>
            </div>

            {/* Original position: Target Entity */}
            <div className="flex-1 min-w-[220px]">
              <label className="block text-xs font-medium text-gray-700 mb-0.5">
                Target Entity
              </label>
              <div className="relative">
                <input
                  type="text"
                  list="target-entity-options"
                  className="w-full px-3 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                  value={targetEntity}
                  onChange={(e) => setTargetEntity(e.target.value)}
                  placeholder="Type or select target..."
                  autoComplete="off"
                />
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
                <datalist id="target-entity-options">
                  {targetOptions.map((option) => (
                    <option key={option} value={option} />
                  ))}
                </datalist>
              </div>
            </div>

            {/* Desc Strategy (unchanged, if still present) */}
            {/* Source ID Strat. (unchanged, if still present) */}
          </div>

          <div className="flex gap-2">
            <button 
              onClick={() => {
                setCreateEntityModalOpen(true);
                setCreateError(null);  // Clear any previous errors
                setCreateEntityName('');  // Reset fields
                setCreateEntityDescription('');
                setCreateEntityType('');
                setCreateEntitySourceId('');
              }}
              className="px-3.5 py-1.5 bg-purple-600 text-white rounded hover:bg-purple-700 text-sm"
            >
              Create Entity
            </button>            
            <button 
              className="px-3.5 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm disabled:opacity-50"
              disabled={selectedEntities.length < 1 || filterMode !== 'selected'}
              onClick={handleMergeEntities}  // ← Add this
              title={
                filterMode !== 'selected' 
                  ? "Enter 'Show Sel. Only' mode first\nto act on selected entities" 
                  : selectedEntities.length < 1 
                  ? "Select at least one entity first\n(check the boxes on the left)" 
                  : undefined
              }
            >
              Merge Entities
            </button>
            <button 
              className="px-3.5 py-1.5 bg-green-600 text-white rounded hover:bg-green-700 text-sm disabled:opacity-50"
              disabled={selectedEntities.length !== 2 || filterMode !== 'selected'}
              onClick={() => {
                if (!targetEntity || !selectedEntities.includes(targetEntity)) {
                  alert('Please select a target entity from the dropdown first.');
                  return;
                }
                setCreateRelModalOpen(true);
                setCreateRelError(null);
                setCreateRelDescription('');
                setCreateRelKeywords('');
                setCreateRelWeight(1.0);
              }}
              title={
                filterMode !== 'selected' 
                  ? "Enter 'Show Sel. Only' mode first\nto act on selected entities" 
                  : selectedEntities.length !== 2 
                  ? "Select exactly two entities first\n(check the boxes on the left)" 
                  : undefined
              }
            >
              Create Rel.
            </button>
            <button 
              className="px-3.5 py-1.5 bg-red-600 text-white rounded hover:bg-red-700 text-sm disabled:opacity-50"
              disabled={selectedEntities.length < 1 || filterMode !== 'selected'}
              onClick={handleDeleteEntities}
              title={
                filterMode !== 'selected' 
                  ? "Enter 'Show Sel. Only' mode first\nto act on selected entities" 
                  : selectedEntities.length < 1 
                  ? "Select at least one entity\nto enable this button" 
                  : undefined
              }
            >
              Delete Entity
            </button>
          </div>
        </div>
      </div>

      {/* Bottom row */}
      <div className="flex-1 flex overflow-hidden">
        {/* Lower Left */}
        <div className="w-1/4 border-r border-gray-300 flex flex-row">
          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="grid grid-cols-[40px_1fr] gap-1 px-2 py-1.5 bg-gray-100 border-b border-gray-300 text-xs font-medium text-center">
              <div className="text-left pl-1.5">Select<br/>Entities</div>
              <div className="text-left pl-2">Entity Name</div>
            </div>

            <div ref={listContainerRef} className="flex-1 overflow-y-auto bg-white">
              {displayEntities.map((entityName) => (
                <div
                  key={entityName}
                  className="grid grid-cols-[40px_1fr] items-center px-2 py-1.5 border-b border-gray-100 hover:bg-gray-50 text-sm"  // ← Changed to [40px_1fr]
                >
                  <div className="flex justify-center">
                    <input
                      type="checkbox"
                      checked={selectedEntities.includes(entityName)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedEntities([...selectedEntities, entityName]);
                        } else {
                          setSelectedEntities(selectedEntities.filter((e) => e !== entityName));
                          if (firstEntity === entityName) {
                            setFirstEntity(null);
                          }
                        }
                      }}
                      className="h-4 w-4 text-blue-600 rounded"
                    />
                  </div>
                  <div className="truncate pl-2">{entityName}</div>
                </div>
              ))}

              {displayEntities.length === 0 && (
                <div className="p-4 text-center text-gray-500 text-sm">
                  {selectedEntities.length > 0
                    ? "No selected entities to show"
                    : "No entities match current filter"}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Lower Right – always shows details for current selection (no "Show Desc" needed) */}
        <div className="flex-1 p-3 flex flex-col">
          <div className="flex-1 overflow-y-auto bg-white border border-gray-200 rounded p-3">
            {selectedEntities.length === 0 ? (
              <div className="flex items-center justify-center h-full text-gray-500 text-sm">
                Select one or more entities on the left to view their details
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {selectedEntities.map((name) => (
                  <div key={name} className="border border-gray-200 rounded p-3 bg-gray-50 text-sm">
                    <div className="font-medium mb-2 flex justify-between items-center">
                      <span>{name}</span>
                      <div className="flex gap-2">

                        <button
                          onClick={() => openEditEntityModal(name)}
                          className="text-xs text-blue-600 hover:underline"
                        >
                          Edit Entity
                        </button>

                        <button
                          onClick={() => openEditRelationshipsModal(name)}
                          className="text-xs text-green-600 hover:underline"
                        >
                          Edit/Delete Relationships
                        </button>
                      </div>
                    </div>

                    {loadingDetails.includes(name) ? (
                      <div className="text-gray-500 italic py-4 text-center">Loading details...</div>
                    ) : entityDetails[name] ? (
                      <div className="space-y-2 text-gray-700">
                        {/* Type */}
                        <div>
                          <strong>Type:</strong> {entityDetails[name].type || "No type found."}
                        </div>

                        {/* Related Entities count */}
                        <div>
                          <strong>Related Entities:</strong> {entityDetails[name].relatedEntities?.length || 0}
                        </div>

                        {/* Description */}
                        <div>
                          <strong>Description:</strong>
                          <div className="pl-4 mt-1">
                            {entityDetails[name].description
                              ?.split('<SEP>')
                              .map((part: string, i: number) => (
                                <p key={i} className="mb-1">
                                  {part.trim() || "No description found."}
                                </p>
                              )) || "No description found."}
                          </div>
                        </div>

                        {/* Source ID */}
                        <div>
                          <strong>Source ID:</strong>
                          <div className="pl-4 mt-1">
                            {entityDetails[name].sourceId
                              ?.split('<SEP>')
                              .map((id: string, i: number) => (
                                <p key={i} className="mb-1">
                                  {id.trim() || ""}
                                </p>
                              )) || ""}
                          </div>
                        </div>

                        {/* File Path */}
                        <div>
                          <strong>File Path:</strong>
                          <div className="pl-4 mt-1 text-gray-600 break-all">
                            {entityDetails[name].filePath
                              ?.split('<SEP>')
                              .map((path: string, i: number) => (
                                <p key={i} className="mb-1">
                                  {path.trim() || ""}
                                </p>
                              )) || "No file path"}
                          </div>
                        </div>

                        {/* Related Entities list */}
                        {entityDetails[name].relatedEntities?.length > 0 && (
                          <div className="mt-3">
                            {entityDetails[name].relatedEntities.map((rel: any, idx: number) => (
                              <div key={idx} className="mb-2">
                                <strong>Related Entity {idx + 1}: {rel.name}</strong>
                                <div className="pl-4">
                                  (Type: {rel.type || ""})
                                </div>
                                <div className="pl-4 mt-1">
                                  Description:
                                  {rel.description
                                    ?.split('<SEP>')
                                    .map((part: string, j: number) => (
                                      <p key={j} className="ml-2 mb-1">
                                        {part.trim()}
                                      </p>
                                    )) || "No description found."}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Relationships list */}
                        {entityDetails[name].relationships?.length > 0 && (
                          <div className="mt-4">
                            <span className="font-medium text-gray-700 block mb-2"><strong>Relationships:</strong></span>
                            <div className="pl-4 mt-1 space-y-4 border-l-2 border-gray-200">
                              {entityDetails[name].relationships.map((rel: any, idx: number) => (
                                <div key={idx} className="text-gray-700">
                                  <div className="font-medium">
                                    From: {rel.from}
                                    <br /> To: {rel.to}
                                  </div>
                                  <div className="mt-1">
                                    <strong>Relation:</strong>
                                    <div className="pl-4 mt-0.5">
                                      {rel.relation
                                        ?.split('<SEP>')
                                        .map((part: string, j: number) => (
                                          <p key={j} className="mb-1 last:mb-0">
                                            {part.trim() || "No relation description provided."}
                                          </p>
                                        )) || "No relation description provided."}
                                    </div>
                                  </div>
                                  <div className="text-gray-600 mt-1">
                                    <strong>Weight:</strong> {rel.weight || 1.0}
                                    {rel.keywords && (
                                      <span className="ml-4">
                                        <strong>Keywords:</strong>{' '}
                                        {rel.keywords
                                          .split(',')
                                          .map((kw: string, j: number) => (
                                            <span key={j} className="mr-1">
                                              {kw.trim()}
                                            </span>
                                          ))}
                                      </span>
                                    )}

                                    <hr style={{
                                      height: '5px',         // Sets the thickness of the line
                                      backgroundColor: 'black', // Sets the color of the line (or just use 'border')
                                      border: 'none',        // Removes default browser border/styling
                                      width: '100%',         // Makes it full width (optional, defaults to 100%)
                                      margin: '20px 0'       // Adds some vertical spacing (optional)
                                    }} />

                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-red-600 py-4 text-center">Failed to load details</div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Edit Entity Modal */}
      {editEntityModalOpen && editEntityOriginalName && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl mx-4 p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">
              Edit Entity: {editEntityOriginalName}
            </h2>
            {editError && (
              <div className="mb-4 p-3 bg-red-100 text-red-700 rounded text-sm">
                {editError}
              </div>
            )}
            <div className="space-y-4">
              {/* Entity Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Entity Name (required, unique)
                </label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={editEntityName}
                  onChange={(e) => setEditEntityName(e.target.value)}
                  placeholder="e.g., Tesla"
                />
              </div>
              {/* Description */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  className="w-full h-32 p-3 border border-gray-300 rounded resize-y focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                  value={editEntityDescription}
                  onChange={(e) => setEditEntityDescription(e.target.value)}
                  placeholder="e.g., Electric vehicle manufacturer (use <SEP> for paragraphs)"
                />
              </div>
              {/* Entity Type */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Entity Type
                </label>
                <div className="flex items-stretch">
                  <button 
                    onClick={() => {
                      setTypeSelectionContext('edit');
                      setSelectTypeModalOpen(true);
                    }}
                    className="px-3 py-2 bg-gray-200 hover:bg-gray-300 border border-gray-300 border-r-0 rounded-l-md text-sm font-medium text-gray-800 cursor-pointer shadow-sm"
                  >
                    Select Type
                  </button>
                  <input
                    type="text"
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-r-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    value={editEntityType}
                    onChange={(e) => setEditEntityType(e.target.value)}
                    placeholder="Type or select (e.g., ORGANIZATION)"
                  />
                </div>
              </div>
              {/* Source ID */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Source ID (optional)
                </label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={editEntitySourceId}
                  onChange={(e) => setEditEntitySourceId(e.target.value)}
                  placeholder="e.g., chunk-123"
                />
              </div>
            </div>
            <div className="mt-6 flex justify-end gap-3">
              <button
                onClick={() => setEditEntityModalOpen(false)}
                className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded text-gray-800 text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveEntity}
                disabled={!editEntityName.trim()}
                className={`px-4 py-2 rounded text-sm ${
                  editEntityName.trim()
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Relationships Modal */}
      {editRelationshipsModalOpen && editingEntityForRel && entityDetails[editingEntityForRel] && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 overflow-y-auto">
          <div className="bg-white rounded-lg shadow-2xl w-full max-w-4xl mx-4 my-8 p-6">
            <h2 className="text-xl font-semibold mb-4">
              Edit/Delete Relationships for: {editingEntityForRel}
            </h2>

            {entityDetails[editingEntityForRel].relationships?.length === 0 ? (
              <div className="text-gray-500 py-6 text-center">
                No relationships found for this entity.
              </div>
            ) : (
              <div className="space-y-6 max-h-[60vh] overflow-y-auto pr-2">
                {entityDetails[editingEntityForRel].relationships.map((rel: any, idx: number) => (
                  <div key={idx} className="border border-gray-200 rounded p-4 bg-gray-50">
                    <div className="font-medium mb-3">
                      {rel.from} → {rel.to}
                    </div>

                    {/* Relation Description */}
                    <div className="mb-3">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Relation Description
                      </label>
                      <textarea
                        className="w-full p-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        rows={3}
                        value={relationshipEdits[`${rel.from}-${rel.to}`]?.relation || rel.relation || ''}
                        onChange={(e) => {
                          const key = `${rel.from}-${rel.to}`;
                          setRelationshipEdits((prev) => ({
                            ...prev,
                            [key]: {
                              ...prev[key],
                              relation: e.target.value,
                            },
                          }));
                        }}
                      />
                    </div>

                    {/* Weight */}
                    <div className="mb-3">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Weight
                      </label>
                      <input
                        type="number"
                        step="1"
                        min="1"
                        max="10000"
                        className="w-24 p-2 border border-gray-300 rounded text-sm"
                        value={relationshipEdits[`${rel.from}-${rel.to}`]?.weight ?? rel.weight ?? 1}
                        onChange={(e) => {
                          const key = `${rel.from}-${rel.to}`;
                          setRelationshipEdits((prev) => ({
                            ...prev,
                            [key]: {
                              ...prev[key],
                              weight: parseFloat(e.target.value) || 1.0,
                            },
                          }));
                        }}
                      />
                    </div>

                    {/* Keywords */}
                    <div className="mb-3">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Keywords (comma-separated)
                      </label>
                      <input
                        type="text"
                        className="w-full p-2 border border-gray-300 rounded text-sm"
                        value={relationshipEdits[`${rel.from}-${rel.to}`]?.keywords || rel.keywords || ''}
                        onChange={(e) => {
                          const key = `${rel.from}-${rel.to}`;
                          setRelationshipEdits((prev) => ({
                            ...prev,
                            [key]: {
                              ...prev[key],
                              keywords: e.target.value,
                            },
                          }));
                        }}
                      />
                    </div>

                    {/* Delete button */}
                    <button
                      onClick={() => deleteRelationship(rel.from, rel.to)}
                      className="mt-2 px-3 py-1.5 bg-red-100 hover:bg-red-200 text-red-700 rounded text-sm"
                    >
                      Delete This Relationship
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Modal footer */}
            <div className="mt-6 flex justify-end gap-3">
              <button
                onClick={() => setEditRelationshipsModalOpen(false)}
                className="px-5 py-2 bg-gray-200 hover:bg-gray-300 rounded text-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={saveAllRelationshipChanges}
                className="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded"
              >
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Select Entity Type Modal */}
      {selectTypeModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-60">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4 p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">
              Select Entity Type
            </h2>
            <div className="mb-3">
              <input
                type="text"
                placeholder="Search types..."
                className="w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={modalFilterText}
                onChange={(e) => setModalFilterText(e.target.value)}
                autoFocus
                ref={modalInputRef}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && selectedModalType && filteredModalTypes.includes(selectedModalType)) {
                    e.preventDefault();
                    if (typeSelectionContext === 'main') {
                      setEntityType(selectedModalType);
                    } else if (typeSelectionContext === 'create') {
                      setCreateEntityType(selectedModalType);
                    } else if (typeSelectionContext === 'edit') {
                      setEditEntityType(selectedModalType);
                    }
                    setSelectTypeModalOpen(false);
                  }
                }}
              />
            </div>
            <div
              className="border border-gray-200 rounded-md h-64 overflow-y-auto mb-4 bg-gray-50"
              role="listbox"
              aria-label="Entity Types"
            >
              {loadingTypes ? (
                <div className="flex flex-col items-center justify-center h-full text-gray-500">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-2"></div>
                  <p className="text-xs">Scanning index for unique types...</p>
                </div>
              ) : filteredModalTypes.length > 0 ? (
                filteredModalTypes.map((type, index) => (
                  <div
                    key={type}
                    ref={(el) => { typeItemRefs.current[index] = el!; }}
                    onClick={() => setSelectedModalType(type)}
                    onDoubleClick={() => {
                      if (typeSelectionContext === 'main') {
                        setEntityType(type);
                      } else if (typeSelectionContext === 'create') {
                        setCreateEntityType(type);
                        createSourceRef.current?.focus();
                      } else if (typeSelectionContext === 'edit') {
                        setEditEntityType(type);
                      }
                      setSelectTypeModalOpen(false);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        if (typeSelectionContext === 'main') {
                          setEntityType(type);
                        } else if (typeSelectionContext === 'create') {
                          setCreateEntityType(type);
                        } else if (typeSelectionContext === 'edit') {
                          setEditEntityType(type);
                        }
                        setSelectTypeModalOpen(false);
                      }
                    }}
                    className={`px-4 py-2 cursor-pointer border-b border-gray-100 last:border-0 text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-blue-50 ${
                      selectedModalType === type
                        ? 'bg-blue-100 text-blue-800 font-semibold'
                        : 'hover:bg-gray-100 text-gray-700'
                    }`}
                    tabIndex={0}
                    role="option"
                    aria-selected={selectedModalType === type}
                  >
                    {type}
                  </div>
                ))
              ) : (
                <div className="p-4 text-center text-gray-400 italic text-sm">
                  {modalFilterText ? "No matching types found." : "No entity types found."}
                </div>
              )}
            </div>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => {
                  setSelectTypeModalOpen(false);
                  setSelectedModalType('');
                }}
                className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded text-gray-800 text-sm transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (typeSelectionContext === 'main') {
                    setEntityType(selectedModalType);
                  } else if (typeSelectionContext === 'create') {
                    setCreateEntityType(selectedModalType);
                    // Add focus here after setting type
                    createSourceRef.current?.focus();
                  } else if (typeSelectionContext === 'edit') {
                    setEditEntityType(selectedModalType);
                  }
                  setSelectTypeModalOpen(false);
                }}
                disabled={!selectedModalType}
                className={`px-4 py-2 rounded text-sm transition-colors ${
                  selectedModalType
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                Select
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create Entity Modal */}
      {createEntityModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl mx-4 p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">
              Create New Entity
            </h2>

            {createError && (
              <div className="mb-4 p-3 bg-red-100 text-red-700 rounded text-sm">
                {createError}
              </div>
            )}

            <div className="space-y-4">
              {/* Entity Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Entity Name (required, unique)
                </label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={createEntityName}
                  onChange={(e) => setCreateEntityName(e.target.value)}
                  placeholder="e.g., Tesla"
                  ref={createNameRef}
                />
              </div>

              {/* Description */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  className="w-full h-32 p-3 border border-gray-300 rounded resize-y focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                  value={createEntityDescription}
                  onChange={(e) => setCreateEntityDescription(e.target.value)}
                  placeholder="e.g., Electric vehicle manufacturer (use <SEP> for paragraphs)"
                />
              </div>

              {/* Entity Type */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Entity Type
                </label>
                <div className="flex items-stretch">
                  <button 
                    onClick={() => {
                      setTypeSelectionContext('create');
                      setSelectTypeModalOpen(true);
                    }}
                    className="px-3 py-2 bg-gray-200 hover:bg-gray-300 border border-gray-300 border-r-0 rounded-l-md text-sm font-medium text-gray-800 cursor-pointer shadow-sm"
                  >
                    Select Type
                  </button>
                  <input
                    type="text"
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-r-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    value={createEntityType}
                    onChange={(e) => setCreateEntityType(e.target.value)}
                    placeholder="Type or select (e.g., ORGANIZATION)"
                  />
                </div>
              </div>

              {/* Source ID */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Source ID (optional)
                </label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={createEntitySourceId}
                  onChange={(e) => setCreateEntitySourceId(e.target.value)}
                  placeholder="e.g., chunk-123"
                  ref={createSourceRef}
                />
              </div>
            </div>

            <div className="mt-6 flex justify-end gap-3">
              <button
                onClick={() => setCreateEntityModalOpen(false)}
                className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded text-gray-800 text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateEntity}
                disabled={!createEntityName.trim()}  // Disable if no name
                className={`px-4 py-2 rounded text-sm ${
                  createEntityName.trim()
                    ? 'bg-blue-600 hover:bg-blue-700 text-white'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Create Relationship Modal */}
      {createRelModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl mx-4 p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">
              Create Relationship
            </h2>
            {createRelError && (
              <div className="mb-4 p-3 bg-red-100 text-red-700 rounded text-sm">
                {createRelError}
              </div>
            )}
            <div className="space-y-4">
              {/* Source and Target (non-editable) */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Source Entity
                  </label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded text-sm bg-gray-100 cursor-not-allowed"
                    value={selectedEntities.find((n) => n !== targetEntity) || ''}
                    disabled
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Target Entity
                  </label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 border border-gray-300 rounded text-sm bg-gray-100 cursor-not-allowed"
                    value={targetEntity}
                    disabled
                  />
                </div>
              </div>
              {/* Description */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Relationship Description (required)
                </label>
                <textarea
                  className="w-full h-24 p-3 border border-gray-300 rounded resize-y focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                  value={createRelDescription}
                  onChange={(e) => setCreateRelDescription(e.target.value)}
                  placeholder="e.g., Elon Musk is the CEO of Tesla"
                />
              </div>
              {/* Keywords */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Keywords (comma-separated)
                </label>
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={createRelKeywords}
                  onChange={(e) => setCreateRelKeywords(e.target.value)}
                  placeholder="e.g., CEO, founder"
                />
              </div>
              {/* Weight */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Weight (default 1.0)
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  className="w-24 px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={createRelWeight}
                  onChange={(e) => setCreateRelWeight(parseFloat(e.target.value) || 1.0)}
                />
              </div>
            </div>
            <div className="mt-6 flex justify-end gap-3">
              <button
                onClick={() => setCreateRelModalOpen(false)}
                className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded text-gray-800 text-sm"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateRelationship}
                disabled={!createRelDescription.trim()}
                className={`px-4 py-2 rounded text-sm ${
                  createRelDescription.trim()
                    ? 'bg-green-600 hover:bg-green-700 text-white'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}




    </div>
  );
}