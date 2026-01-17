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

  // Calculate how many rows actually fit
  useEffect(() => {
    const updateRowsPerPage = () => {
      if (!listContainerRef.current) return;

      const container = listContainerRef.current;
      const containerHeight = container.clientHeight;

      // Measure one row height (we take the first row if available)
      const firstRow = container.querySelector('div.grid');
      const rowHeight = firstRow ? firstRow.getBoundingClientRect().height : 36; // fallback ~36px

      // Safety margin for headers, borders, etc.
      const headerHeight = 42; // known header height

      const availableHeight = containerHeight - headerHeight;
      const calculatedRows = Math.max(5, Math.floor(availableHeight / rowHeight));

      setRowsPerPage(calculatedRows);
    };

    updateRowsPerPage();

    // Re-calculate on window resize
    window.addEventListener('resize', updateRowsPerPage);
    return () => window.removeEventListener('resize', updateRowsPerPage);
  }, [entities, filterText]); // re-run when list content changes

  const filteredEntities = entities.filter((e) =>
    e.toLowerCase().includes(filterText.toLowerCase())
  );

  const totalPages = Math.max(1, Math.ceil(filteredEntities.length / rowsPerPage));
  const startIndex = (currentPage - 1) * rowsPerPage;
  const paginatedEntities = filteredEntities.slice(startIndex, startIndex + rowsPerPage);

  // Reset page when filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [filterText]);

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

  return (
    <div className="h-full flex flex-col">
      {/* Top row */}
      <div className="h-auto flex border-b border-gray-300">
        {/* Upper Left */}
        <div className="w-1/4 border-r border-gray-300 p-2.5 flex flex-col gap-2.5">
          <input
            type="text"
            placeholder="Filter entities..."
            className="w-full px-3 py-1 border border-gray-300 rounded text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
          />

          {/* Selection preset buttons */}
          <div className="flex flex-wrap gap-1">
            <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs">
              All Of Type
            </button>
            <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs">
              Orphans
            </button>
            <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs">
              Clear Sel.
            </button>
            <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs">
              Show Sel. Only
            </button>
            <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs">
              Show All
            </button>
            <button className="px-2 py-0.5 bg-red-50 hover:bg-red-100 border border-red-200 rounded text-xs text-red-700">
              Reset All
            </button>
          </div>

          {/* Pagination controls */}
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
                max={totalPages}
                value={currentPage}
                onChange={handlePageInputChange}
                className="w-10 text-center border border-gray-400 rounded text-xs focus:outline-none focus:ring-1 focus:ring-blue-500 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
              />
              /{totalPages}
            </div>

            <button
              onClick={goToNext}
              disabled={currentPage >= totalPages}
              className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
            <button
              onClick={goToLast}
              disabled={currentPage >= totalPages}
              className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Last
            </button>
          </div>
        </div>

        {/* Upper Right - unchanged */}
        {/* ... same as previous version ... */}
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
              {paginatedEntities.map((entityName) => (
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

              {paginatedEntities.length === 0 && filteredEntities.length > 0 && (
                <div className="p-4 text-center text-gray-500 text-sm">
                  No entities on this page — try another page
                </div>
              )}

              {filteredEntities.length === 0 && (
                <div className="p-4 text-center text-gray-500 text-sm">
                  No entities match current filter
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
                      <button className="text-xs text-blue-600 hover:underline">
                        Edit
                      </button>
                    </div>
                    <div className="text-gray-600 mb-1.5 line-clamp-3">
                      Loading description...
                    </div>
                    <div className="text-xs text-gray-500">
                      Type: ? • Rel: ? • Src: ?
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500 text-sm">
                Select entities on the left • Click vertical button to compare
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}