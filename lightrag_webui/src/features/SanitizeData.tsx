// src/pages/SanitizeData.tsx
import React, { useState } from 'react';

export default function SanitizeData() {
  // Placeholder states (we'll use real data later)
  const [filterText, setFilterText] = useState('');
  const [selectedEntities, setSelectedEntities] = useState<string[]>([]);
  const [firstEntity, setFirstEntity] = useState<string | null>(null);
  const [targetEntity, setTargetEntity] = useState('');
  const [entityType, setEntityType] = useState('');
  const [descriptionStrategy, setDescriptionStrategy] = useState('join_unique');
  const [sourceIdStrategy, setSourceIdStrategy] = useState('join_unique');
  const [showDescriptions, setShowDescriptions] = useState(false);

  // Pagination placeholders (will be dynamic later)
  const currentPage = 2;
  const totalPages = 5;

  return (
    <div className="h-full flex flex-col">
      {/* Top row - controls + merge panel */}
      <div className="h-1/2 flex border-b border-gray-300">
        {/* Upper Left - Filter & Selection Controls */}
        <div className="w-1/4 border-r border-gray-300 p-4 flex flex-col">
          <h2 className="text-lg font-semibold mb-3">Filter & Selection</h2>

          <div className="mb-4">
            <input
              type="text"
              placeholder="Filter entities..."
              className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
            />
          </div>

          <div className="flex flex-wrap gap-2 mb-4">
            <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-sm">
              Select All Of Type
            </button>
            <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-sm">
              Select Orphans
            </button>
            <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-sm">
              Clear Selected
            </button>
          </div>

          <div className="flex flex-wrap gap-2 mb-4">
            <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-sm">
              Show Selected Only
            </button>
            <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-sm">
              Show All
            </button>
            <button className="px-3 py-1.5 bg-red-50 hover:bg-red-100 border border-red-200 rounded text-sm text-red-700">
              Reset All
            </button>
          </div>

          {/* Pagination controls */}
          <div className="flex flex-wrap gap-2 mb-3">
            <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-sm font-medium">
              First
            </button>
            <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-sm font-medium">
              Previous
            </button>
            <div className="px-3 py-1.5 bg-gray-50 border border-gray-300 rounded text-sm flex items-center">
              Page {currentPage} of {totalPages}
            </div>
            <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-sm font-medium">
              Next
            </button>
            <button className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-sm font-medium">
              Last
            </button>
          </div>

          {/* Go to page */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-700 whitespace-nowrap">Go to page:</span>
            <input
              type="number"
              min="1"
              className="w-16 px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="1"
            />
            <button className="px-3 py-1.5 bg-blue-600 text-white rounded hover:bg-blue-700 text-sm">
              Go
            </button>
          </div>
        </div>

        {/* Upper Right - Merge / Edit Controls (no title) */}
        <div className="w-3/4 p-4 flex flex-col">
          <div className="flex flex-wrap items-end gap-4 mb-4">
            {/* Target Entity - Editable Combo */}
            <div className="flex-1 min-w-[260px]">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Target Entity
              </label>
              <div className="relative">
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={targetEntity}
                  onChange={(e) => setTargetEntity(e.target.value)}
                  placeholder="Enter or select target entity..."
                />
                {/* Dropdown arrow (visual only for now) */}
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>
            </div>

            {/* New: Select Entity Type button + editable Entity Type */}
            <div className="flex items-end gap-2 min-w-[220px]">
              <button className="px-3 py-2 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-sm font-medium">
                Select Entity Type
              </button>
              <div className="flex-1">
                <input
                  type="text"
                  className="w-full px-3 py-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={entityType}
                  onChange={(e) => setEntityType(e.target.value)}
                  placeholder="Entity Type..."
                />
              </div>
            </div>

            {/* Strategies */}
            <div className="min-w-[180px]">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description Strategy
              </label>
              <select
                className="w-full px-3 py-2 border border-gray-300 rounded"
                value={descriptionStrategy}
                onChange={(e) => setDescriptionStrategy(e.target.value)}
              >
                <option value="join_unique">Join Unique</option>
                <option value="concatenate">Concatenate</option>
                <option value="keep_first">Keep First</option>
              </select>
            </div>

            <div className="min-w-[180px]">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Source ID Strategy
              </label>
              <select
                className="w-full px-3 py-2 border border-gray-300 rounded"
                value={sourceIdStrategy}
                onChange={(e) => setSourceIdStrategy(e.target.value)}
              >
                <option value="join_unique">Join Unique</option>
                <option value="concatenate">Concatenate</option>
                <option value="keep_first">Keep First</option>
              </select>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed">
              Merge Entities
            </button>
            <button className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed">
              Create Relationship
            </button>
            <button className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed">
              Delete Entities
            </button>
          </div>
        </div>
      </div>

      {/* Bottom row - entity list + details */}
      <div className="flex-1 flex">
        {/* Lower Left - Entity List */}
        <div className="w-1/4 border-r border-gray-300 p-4 flex flex-col overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Entities</h2>
            <button
              className="text-sm px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded border border-gray-300"
              onClick={() => setShowDescriptions(!showDescriptions)}
            >
              {showDescriptions ? 'Hide Desc' : 'Show Desc'}
            </button>
          </div>

          {/* Column Headers */}
          <div className="grid grid-cols-[auto_auto_1fr] gap-2 px-3 py-2 bg-gray-100 border-b border-gray-300 font-medium text-sm text-center">
            <div className="text-left leading-tight">
              Keep<br />First
            </div>
            <div className="text-left leading-tight">
              Select<br />Entities
            </div>
            <div className="text-left">Entity Name</div>
          </div>

          <div className="flex-1 overflow-y-auto border border-gray-200 rounded bg-white mt-1">
            {/* Placeholder entity rows */}
            {Array.from({ length: 20 }).map((_, i) => {
              const entityName = `Entity_${i + 1}_Example ${i % 3 === 0 ? '(Person)' : ''}`;
              return (
                <div
                  key={i}
                  className="grid grid-cols-[auto_auto_1fr] items-center px-3 py-2 border-b border-gray-100 hover:bg-gray-50"
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
                      className="h-4 w-4 text-blue-600 rounded"
                      checked={selectedEntities.includes(entityName)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedEntities([...selectedEntities, entityName]);
                        } else {
                          setSelectedEntities(selectedEntities.filter((e) => e !== entityName));
                        }
                      }}
                    />
                  </div>
                  <div className="text-sm truncate ml-2">{entityName}</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Lower Right - Entity Details / Relationships */}
        <div className="flex-1 p-4 flex flex-col overflow-hidden">
          <h2 className="text-lg font-semibold mb-3">
            Entity Details {showDescriptions ? '& Relationships' : ''}
          </h2>

          <div className="flex-1 overflow-y-auto bg-white border border-gray-200 rounded p-4">
            {showDescriptions ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Placeholder detail cards */}
                {Array.from({ length: 6 }).map((_, i) => (
                  <div key={i} className="border border-gray-200 rounded p-4 bg-gray-50">
                    <div className="font-medium mb-2 flex justify-between items-center">
                      <span>Entity_Name_{i + 1}</span>
                      <button className="text-xs text-blue-600 hover:underline">
                        Edit Description
                      </button>
                    </div>
                    <div className="text-sm text-gray-600 mb-3">
                      This is a placeholder description that would normally contain the actual entity description from the knowledge graph...
                    </div>
                    <div className="text-xs text-gray-500">
                      Type: Concept • Relations: 4 • Source: doc_001.pdf
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                Select entities on the left to view details
                <br />
                or click "Show Desc" to see comparison view
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}