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
  const [showDescriptions, setShowDescriptions] = useState(false);
  const [showSelectedOnlyMode, setShowSelectedOnlyMode] = useState(false);

  // Dropdown suggestions = currently selected entities
  const targetOptions = [...selectedEntities].sort((a, b) => a.localeCompare(b));

  // Store fetched details: entityName → { desc, type, sourceId, filePath, relatedEntities, relationships }
  const [entityDetails, setEntityDetails] = useState<Record<string, any>>({});

  // Loading state (optional but nice UX)
  const [loadingDetails, setLoadingDetails] = useState<string[]>([]);


  const listContainerRef = useRef<HTMLDivElement>(null);
  const [rowsPerPage, setRowsPerPage] = useState(20); // initial guess

  // Fetch entities
  useEffect(() => {
    const fetchEntities = async () => {
      try {
        const response = await axios.get(`${API_BASE}/graph/label/list`);
        const sorted = (response.data as string[]).sort((a, b) =>
          a.toLowerCase().localeCompare(b.toLowerCase())
        );
        setEntities(sorted);
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
    if (showDescriptions && selectedEntities.length > 0) {
      selectedEntities.forEach((entityName) => {
        fetchEntityDetail(entityName);
      });
    }
  }, [showDescriptions, selectedEntities]);  // Run when toggle changes or selection changes

  // Reset page when filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [filterText]);

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
    if (selectedEntities.length === 0) return;
    
    setShowSelectedOnlyMode(true);
    setCurrentPage(1);
    // Optional: clear filter when entering selected-only mode
    setFilterText('');
  };

  // Reset All
  const handleResetAll = () => {
    setShowSelectedOnlyMode(false);      // exit selected-only mode
    setSelectedEntities([]);             // clear all checkboxes
    setFirstEntity(null);                // clear the radio "Keep First" selection
    setFilterText('');                   // remove any filter
    setCurrentPage(1);                   // go back to first page
  };

  const handleClearSelected = () => {
    setSelectedEntities([]);     // uncheck all checkboxes
    setFirstEntity(null);        // deselect "Keep First" radio
    // Nothing else — no change to filter, page, or showSelectedOnlyMode
  };

  const fetchEntityDetail = async (entityName: string) => {

    console.log(`fetchEntityDetail called for: "${entityName}"`);

    // Skip if we already have it
    //if (entityDetails[entityName]) return;
    if (entityDetails[entityName]) {
      console.log(`Already have details for "${entityName}" - skipping`);
      return;
    }

    console.log(`Fetching details for "${entityName}"...`);

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







  const displayEntities = showSelectedOnlyMode 
   ? selectedEntities 
   : paginatedEntities;  

  return (
    <div className="h-full flex flex-col">
      {/* Top row - minimum height to ensure controls are visible */}
      <div className="h-auto flex border-b border-gray-300">  
        {/* Upper Left */}


        {/*<div className="w-1/4 border-r border-gray-300 p-2.5 flex flex-col gap-2.5">*/}



        <div className={`w-1/4 border-r border-gray-300 p-2.5 flex flex-col gap-2.5 ${
          showSelectedOnlyMode ? 'bg-indigo-50' : ''
        }`}>

          {showSelectedOnlyMode && (
            <div className="text-xs text-indigo-700 bg-indigo-50 p-2 rounded mb-2">
              Showing only selected entities ({selectedEntities.length})
            </div>
          )}

          {!showSelectedOnlyMode && (
              <>
                <input
                  type="text"
                  placeholder="Filter entities..."
                  className="w-full px-3 py-1 border border-gray-300 rounded text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                  value={filterText}
                  onChange={(e) => setFilterText(e.target.value)}
                />
                <div className="flex flex-wrap gap-1">
                  <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs">
                    All Of Type
                  </button>
                  <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs">
                    Orphans
                  </button>

                  {/* Pagination */}
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
            
            {!showSelectedOnlyMode ? (
              <button
                onClick={handleClearSelected}
                className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs"
              >
                Clear Sel.
              </button>
            ) : (
              <div className="w-[78px]" />   // ← invisible placeholder with same approximate width
            )}

            <button
              onClick={handleShowSelectedOnly}
              className={`px-2 py-0.5 border rounded text-xs transition-colors ${
                showSelectedOnlyMode
                  ? 'bg-indigo-600 text-white border-indigo-700'
                  : 'bg-indigo-50 hover:bg-indigo-100 border-indigo-200 text-indigo-700'
              }`}
              disabled={selectedEntities.length === 0}
            >
              Show Sel. Only
            </button>
            <button
              onClick={() => {
                setShowSelectedOnlyMode(false);    // ← exit selected-only mode
                setFilterText('');
                setCurrentPage(1);
              }}
              className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs"
            >
              Show All
            </button>
            <button
              onClick={() => {
                setShowSelectedOnlyMode(false);     // Exit "Show Sel. Only" mode
                setSelectedEntities([]);            // Clear all checkboxes
                setFirstEntity(null);               // Clear the "Keep First" radio selection
                setFilterText('');                  // Remove any active filter
                setCurrentPage(1);                  // Reset to first page
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
                  autoComplete="off"  // prevents browser suggestions from interfering
                />
                
                {/* Dropdown arrow icon (visual hint only) */}
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>

                {/* The actual dropdown suggestions */}
                <datalist id="target-entity-options">
                  {targetOptions.map((option) => (
                    <option key={option} value={option} />
                  ))}
                </datalist>
              </div>
            </div>

            <div className="min-w-[200px]">
              <button className="block w-full px-3 py-0.5 bg-gray-200 hover:bg-gray-300 border border-gray-300 border-b-0 rounded-t-md text-xs font-medium text-gray-800 text-left cursor-pointer shadow-sm">
                Select Type
              </button>
              <div className="relative">
                <input
                  type="text"
                  className="w-full px-3 py-1.5 border border-gray-300 rounded-b-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                  value={entityType}
                  onChange={(e) => setEntityType(e.target.value)}
                  placeholder="Type or filter..."
                />
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="min-w-[140px]">
              <label className="block text-xs font-medium text-gray-700 mb-0.5">
                Desc Strategy
              </label>
              <select
                className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm"
                value={descriptionStrategy}
                onChange={(e) => setDescriptionStrategy(e.target.value)}
              >
                <option value="join_unique">Join Unique</option>
                <option value="concatenate">Concatenate</option>
                <option value="keep_first">Keep First</option>
              </select>
            </div>

            <div className="min-w-[140px]">
              <label className="block text-xs font-medium text-gray-700 mb-0.5">
                Source ID Strat.
              </label>
              <select
                className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm"
                value={sourceIdStrategy}
                onChange={(e) => setSourceIdStrategy(e.target.value)}
              >
                <option value="join_unique">Join Unique</option>
                <option value="concatenate">Concatenate</option>
                <option value="keep_first">Keep First</option>
              </select>
            </div>
          </div>

          <div className="flex gap-2">
            <button className="px-3.5 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm disabled:opacity-50">
              Merge Entities
            </button>
            <button className="px-3.5 py-1.5 bg-green-600 text-white rounded hover:bg-green-700 text-sm disabled:opacity-50">
              Create Rel.
            </button>
            <button className="px-3.5 py-1.5 bg-red-600 text-white rounded hover:bg-red-700 text-sm disabled:opacity-50">
              Delete
            </button>
          </div>
        </div>
      </div>

      {/* Bottom row */}
      <div className="flex-1 flex overflow-hidden">
        {/* Lower Left */}
        <div className="w-1/4 border-r border-gray-300 flex flex-row">
          <div className="flex flex-col w-10">
            <div className="h-[42px] bg-gray-100 border-b border-gray-300 flex items-center justify-center">
              <span className="text-xs font-medium text-gray-700 leading-tight text-center">
                Show<br />Desc
              </span>
            </div>

            <button
              onClick={() => setShowDescriptions(!showDescriptions)}
              className="flex-1 bg-gradient-to-b from-gray-100 to-gray-300 hover:from-gray-200 hover:to-gray-400 border-r border-gray-400 flex flex-col items-center justify-center text-xs font-medium text-gray-800 shadow-md rounded-r-lg active:bg-gray-400 transition-all"
            >
              <span>Show</span>
              <span>Desc</span>
            </button>
          </div>

          <div className="flex-1 flex flex-col overflow-hidden">
            <div className="grid grid-cols-[40px_40px_1fr] gap-1 px-2 py-1.5 bg-gray-100 border-b border-gray-300 text-xs font-medium text-center">
              <div className="text-left pl-1.5">Keep<br/>First</div>
              <div className="text-left pl-1.5">Select<br/>Entities</div>
              <div className="text-left pl-2">Entity Name</div>
            </div>

            <div ref={listContainerRef} className="flex-1 overflow-y-auto bg-white">
              {displayEntities.map((entityName) => (
                <div
                  key={entityName}
                  className="grid grid-cols-[40px_40px_1fr] items-center px-2 py-1.5 border-b border-gray-100 hover:bg-gray-50 text-sm"
                >
                  <div className="flex justify-center">
                    <input
                      type="radio"
                      name="first-entity"
                      checked={firstEntity === entityName}
                      onChange={() => setFirstEntity(entityName)}
                      className="h-4 w-4 text-blue-600"
                    />
                  </div>
                  <div className="flex justify-center">
                    <input
                      type="checkbox"
                      checked={selectedEntities.includes(entityName)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedEntities([...selectedEntities, entityName]);
                        } else {
                          setSelectedEntities(selectedEntities.filter((e) => e !== entityName));
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

        {/* Lower Right */}
        <div className="flex-1 p-3 flex flex-col">
          <div className="flex-1 overflow-y-auto bg-white border border-gray-200 rounded p-3">
            {showDescriptions ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {selectedEntities.slice(0, 9).map((name) => (
                  <div key={name} className="border border-gray-200 rounded p-3 bg-gray-50 text-sm">
                    <div className="font-medium mb-1 flex justify-between items-center">
                      <span>{name}</span>
                      <div className="flex gap-2">
                        <button className="text-xs text-blue-600 hover:underline">
                          Edit Description
                        </button>
                        <button className="text-xs text-green-600 hover:underline">
                          Edit Relationships
                        </button>
                      </div>
                    </div>

                    {loadingDetails.includes(name) ? (
                      <div className="text-gray-500 italic">Loading details...</div>
                    ) : entityDetails[name] ? (
                      <>
                        <div className="text-gray-700 mb-1">
                          <strong>Type:</strong> {entityDetails[name].type || "Unknown"}
                        </div>
                        <div className="text-gray-600 mb-2">
                          <strong>Description:</strong><br />
                          {entityDetails[name].description || "No description available"}
                        </div>
                        {/* We'll add Related Entities, Source ID, Relationships, etc. in next step */}
                      </>
                    ) : (
                      <div className="text-red-600">Failed to load details</div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-600 mb-1.5 line-clamp-3">
                {loadingDetails.includes(name) 
                  ? "Loading details..." 
                  : entityDetails[name]?.description || "No description available"}
              </div>    
            )}
          </div>
        </div>
      </div>
    </div>
  );
}