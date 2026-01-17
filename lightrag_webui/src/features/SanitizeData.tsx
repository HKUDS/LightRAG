// src/pages/SanitizeData.tsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = 'http://localhost:9621'; // ← change if your server runs elsewhere

export default function SanitizeData() {
  const [entities, setEntities] = useState<string[]>([]);
  const [filterText, setFilterText] = useState('');
  const [selectedEntities, setSelectedEntities] = useState<string[]>([]);
  const [firstEntity, setFirstEntity] = useState<string | null>(null);
  const [targetEntity, setTargetEntity] = useState('');
  const [entityType, setEntityType] = useState('');
  const [descriptionStrategy, setDescriptionStrategy] = useState('join_unique');
  const [sourceIdStrategy, setSourceIdStrategy] = useState('join_unique');
  const [showDescriptions, setShowDescriptions] = useState(false);

  // Fetch entities on mount
  useEffect(() => {
    const fetchEntities = async () => {
      try {
        const response = await axios.get(`${API_BASE}/graph/label/list`);
        const sorted = (response.data as string[]).sort((a, b) =>
          a.toLowerCase().localeCompare(b.toLowerCase())
        );
        setEntities(sorted);
        console.log(`Loaded ${sorted.length} entities`);
      } catch (err) {
        console.error('Failed to load entities:', err);
        // Optional: show error toast/message later
      }
    };

    fetchEntities();
  }, []);

  // Simple client-side filter
  const filteredEntities = entities.filter((e) =>
    e.toLowerCase().includes(filterText.toLowerCase())
  );

  return (
    <div className="h-full flex flex-col">
      {/* Top row – auto height */}
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

          <div className="flex flex-wrap gap-1 items-center">
            <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs font-medium">
              First
            </button>
            <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs font-medium">
              Prev
            </button>
            <div className="px-2 py-0.5 bg-gray-50 border border-gray-300 rounded text-xs whitespace-nowrap">
              Pg 1/{Math.ceil(filteredEntities.length / 35) || 1}
            </div>
            <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs font-medium">
              Next
            </button>
            <button className="px-2 py-0.5 bg-gray-100 hover:bg-gray-200 border border-gray-300 rounded text-xs font-medium">
              Last
            </button>
          </div>
        </div>

        {/* Upper Right */}
        <div className="w-3/4 p-2.5 flex flex-col gap-2.5">
          <div className="flex flex-wrap items-end gap-2.5">
            <div className="flex-1 min-w-[220px]">
              <label className="block text-xs font-medium text-gray-700 mb-0.5">
                Target Entity
              </label>
              <div className="relative">
                <input
                  type="text"
                  className="w-full px-3 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                  value={targetEntity}
                  onChange={(e) => setTargetEntity(e.target.value)}
                  placeholder="Enter or select target..."
                />
                <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                  <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="min-w-[200px]">
              <button
                className="block w-full px-3 py-0.5 bg-gray-200 hover:bg-gray-300 border border-gray-300 border-b-0 rounded-t-md text-xs font-medium text-gray-800 text-left cursor-pointer shadow-sm"
              >
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
      <div className="flex-1 flex">
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

          <div className="flex-1 flex flex-col">
            <div className="grid grid-cols-[40px_40px_1fr] gap-1 px-2 py-1.5 bg-gray-100 border-b border-gray-300 text-xs font-medium text-center">
              <div className="text-left pl-1.5">Keep<br/>First</div>
              <div className="text-left pl-1.5">Select<br/>Entities</div>
              <div className="text-left pl-2">Entity Name</div>
            </div>

            <div className="flex-1 overflow-y-auto bg-white">
              {filteredEntities.slice(0, 35).map((entityName) => (   // ← simple first-page limit
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
            </div>
          </div>
        </div>

        {/* Lower Right */}
        <div className="flex-1 p-3 flex flex-col">
          <div className="flex-1 overflow-y-auto bg-white border border-gray-200 rounded p-3">
            {showDescriptions ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {/* We'll show real details here later */}
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