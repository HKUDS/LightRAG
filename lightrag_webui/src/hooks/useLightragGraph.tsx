import Graph, { DirectedGraph } from 'graphology'
import { useCallback, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { randomColor, errorMessage } from '@/lib/utils'
import * as Constants from '@/lib/constants'
import { useGraphStore, RawGraph, RawNodeType, RawEdgeType } from '@/stores/graph'
import { toast } from 'sonner'
import { queryGraphs } from '@/api/lightrag'
import { useBackendState } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { useTabVisibility } from '@/contexts/useTabVisibility'

import seedrandom from 'seedrandom'

const validateGraph = (graph: RawGraph) => {
  if (!graph) {
    return false
  }
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    return false
  }

  for (const node of graph.nodes) {
    if (!node.id || !node.labels || !node.properties) {
      return false
    }
  }

  for (const edge of graph.edges) {
    if (!edge.id || !edge.source || !edge.target) {
      return false
    }
  }

  for (const edge of graph.edges) {
    const source = graph.getNode(edge.source)
    const target = graph.getNode(edge.target)
    if (source == undefined || target == undefined) {
      return false
    }
  }

  return true
}

export type NodeType = {
  x: number
  y: number
  label: string
  size: number
  color: string
  highlighted?: boolean
}
export type EdgeType = { label: string }

const fetchGraph = async (label: string, maxDepth: number, minDegree: number) => {
  let rawData: any = null

  try {
    rawData = await queryGraphs(label, maxDepth, minDegree)
  } catch (e) {
    useBackendState.getState().setErrorMessage(errorMessage(e), 'Query Graphs Error!')
    return null
  }

  let rawGraph = null

  if (rawData) {
    const nodeIdMap: Record<string, number> = {}
    const edgeIdMap: Record<string, number> = {}

    for (let i = 0; i < rawData.nodes.length; i++) {
      const node = rawData.nodes[i]
      nodeIdMap[node.id] = i

      // const seed = node.labels.length > 0 ? node.labels[0] : node.id
      seedrandom(node.id, { global: true })
      node.color = randomColor()
      node.x = Math.random()
      node.y = Math.random()
      node.degree = 0
      node.size = 10
    }

    for (let i = 0; i < rawData.edges.length; i++) {
      const edge = rawData.edges[i]
      edgeIdMap[edge.id] = i

      const source = nodeIdMap[edge.source]
      const target = nodeIdMap[edge.target]
      if (source !== undefined && source !== undefined) {
        const sourceNode = rawData.nodes[source]
        const targetNode = rawData.nodes[target]
        if (!sourceNode) {
          console.error(`Source node ${edge.source} is undefined`)
          continue
        }
        if (!targetNode) {
          console.error(`Target node ${edge.target} is undefined`)
          continue
        }
        sourceNode.degree += 1
        targetNode.degree += 1
      }
    }

    // generate node size
    let minDegree = Number.MAX_SAFE_INTEGER
    let maxDegree = 0

    for (const node of rawData.nodes) {
      minDegree = Math.min(minDegree, node.degree)
      maxDegree = Math.max(maxDegree, node.degree)
    }
    const range = maxDegree - minDegree
    if (range > 0) {
      const scale = Constants.maxNodeSize - Constants.minNodeSize
      for (const node of rawData.nodes) {
        node.size = Math.round(
          Constants.minNodeSize + scale * Math.pow((node.degree - minDegree) / range, 0.5)
        )
      }
    }

    rawGraph = new RawGraph()
    rawGraph.nodes = rawData.nodes
    rawGraph.edges = rawData.edges
    rawGraph.nodeIdMap = nodeIdMap
    rawGraph.edgeIdMap = edgeIdMap

    if (!validateGraph(rawGraph)) {
      rawGraph = null
      console.error('Invalid graph data')
    }
    console.log('Graph data loaded')
  }

  // console.debug({ data: JSON.parse(JSON.stringify(rawData)) })
  return rawGraph
}

// Create a new graph instance with the raw graph data
const createSigmaGraph = (rawGraph: RawGraph | null) => {
  // Always create a new graph instance
  const graph = new DirectedGraph()

  // Add nodes from raw graph data
  for (const rawNode of rawGraph?.nodes ?? []) {
    // Ensure we have fresh random positions for nodes
    seedrandom(rawNode.id + Date.now().toString(), { global: true })
    const x = Math.random()
    const y = Math.random()

    graph.addNode(rawNode.id, {
      label: rawNode.labels.join(', '),
      color: rawNode.color,
      x: x,
      y: y,
      size: rawNode.size,
      // for node-border
      borderColor: Constants.nodeBorderColor,
      borderSize: 0.2
    })
  }

  // Add edges from raw graph data
  for (const rawEdge of rawGraph?.edges ?? []) {
    rawEdge.dynamicId = graph.addDirectedEdge(rawEdge.source, rawEdge.target, {
      label: rawEdge.type || undefined
    })
  }

  return graph
}

const useLightrangeGraph = () => {
  const { t } = useTranslation()
  const queryLabel = useSettingsStore.use.queryLabel()
  const rawGraph = useGraphStore.use.rawGraph()
  const sigmaGraph = useGraphStore.use.sigmaGraph()
  const maxQueryDepth = useSettingsStore.use.graphQueryMaxDepth()
  const minDegree = useSettingsStore.use.graphMinDegree()
  const isFetching = useGraphStore.use.isFetching()
  const nodeToExpand = useGraphStore.use.nodeToExpand()
  const nodeToPrune = useGraphStore.use.nodeToPrune()

  // Get tab visibility
  const { isTabVisible } = useTabVisibility()
  const isGraphTabVisible = isTabVisible('knowledge-graph')

  // Track previous parameters to detect actual changes
  const prevParamsRef = useRef({ queryLabel, maxQueryDepth, minDegree })

  // Use ref to track if data has been loaded and initial load
  const dataLoadedRef = useRef(false)
  const initialLoadRef = useRef(false)

  // Check if parameters have changed
  const paramsChanged =
    prevParamsRef.current.queryLabel !== queryLabel ||
    prevParamsRef.current.maxQueryDepth !== maxQueryDepth ||
    prevParamsRef.current.minDegree !== minDegree

  const getNode = useCallback(
    (nodeId: string) => {
      return rawGraph?.getNode(nodeId) || null
    },
    [rawGraph]
  )

  const getEdge = useCallback(
    (edgeId: string, dynamicId: boolean = true) => {
      return rawGraph?.getEdge(edgeId, dynamicId) || null
    },
    [rawGraph]
  )

  // Track if a fetch is in progress to prevent multiple simultaneous fetches
  const fetchInProgressRef = useRef(false)

  // Data fetching logic - simplified but preserving TAB visibility check
  useEffect(() => {
    // Skip if fetch is already in progress
    if (fetchInProgressRef.current) {
      return
    }

    // If there's no query label, reset the graph
    if (!queryLabel) {
      if (rawGraph !== null || sigmaGraph !== null) {
        const state = useGraphStore.getState()
        state.reset()
        state.setGraphDataFetchAttempted(false)
        state.setLabelsFetchAttempted(false)
      }
      dataLoadedRef.current = false
      initialLoadRef.current = false
      return
    }

    // Check if parameters have changed
    if (!isFetching && !fetchInProgressRef.current &&
        (paramsChanged || !useGraphStore.getState().graphDataFetchAttempted)) {

      // Only fetch data if the Graph tab is visible
      if (!isGraphTabVisible) {
        console.log('Graph tab not visible, skipping data fetch');
        return;
      }

      // Set flags
      fetchInProgressRef.current = true
      useGraphStore.getState().setGraphDataFetchAttempted(true)

      const state = useGraphStore.getState()
      state.setIsFetching(true)
      state.setShouldRender(false) // Disable rendering during data loading

      // Clear selection and highlighted nodes before fetching new graph
      state.clearSelection()
      if (state.sigmaGraph) {
        state.sigmaGraph.forEachNode((node) => {
          state.sigmaGraph?.setNodeAttribute(node, 'highlighted', false)
        })
      }

      // Update parameter reference
      prevParamsRef.current = { queryLabel, maxQueryDepth, minDegree }

      console.log('Fetching graph data...')

      // Use a local copy of the parameters
      const currentQueryLabel = queryLabel
      const currentMaxQueryDepth = maxQueryDepth
      const currentMinDegree = minDegree

      // Fetch graph data
      fetchGraph(currentQueryLabel, currentMaxQueryDepth, currentMinDegree).then((data) => {
        const state = useGraphStore.getState()

        // Reset state
        state.reset()

        // Create and set new graph directly
        const newSigmaGraph = createSigmaGraph(data)
        data?.buildDynamicMap()

        // Set new graph data
        state.setSigmaGraph(newSigmaGraph)
        state.setRawGraph(data)

        // No longer need to extract labels from graph data

        // Update flags
        dataLoadedRef.current = true
        initialLoadRef.current = true
        fetchInProgressRef.current = false

        // Reset camera view
        state.setMoveToSelectedNode(true)

        // Enable rendering if the tab is visible
        state.setShouldRender(isGraphTabVisible)
        state.setIsFetching(false)
      }).catch((error) => {
        console.error('Error fetching graph data:', error)

        // Reset state on error
        const state = useGraphStore.getState()
        state.setIsFetching(false)
        state.setShouldRender(isGraphTabVisible)
        dataLoadedRef.current = false
        fetchInProgressRef.current = false
        state.setGraphDataFetchAttempted(false)
      })
    }
  }, [queryLabel, maxQueryDepth, minDegree, isFetching, paramsChanged, isGraphTabVisible, rawGraph, sigmaGraph])

  // Update rendering state and handle tab visibility changes
  useEffect(() => {
    // When tab becomes visible
    if (isGraphTabVisible) {
      // If we have data, enable rendering
      if (rawGraph) {
        useGraphStore.getState().setShouldRender(true)
      }

      // We no longer reset the fetch attempted flag here to prevent continuous API calls
    } else {
      // When tab becomes invisible, disable rendering
      useGraphStore.getState().setShouldRender(false)
    }
  }, [isGraphTabVisible, rawGraph])

  // Handle node expansion
  useEffect(() => {
    const handleNodeExpand = async (nodeId: string | null) => {
      if (!nodeId || !sigmaGraph || !rawGraph) return;
      
      try {
        // Set fetching state
        useGraphStore.getState().setIsFetching(true);
        
        // Get the node to expand
        const nodeToExpand = rawGraph.getNode(nodeId);
        if (!nodeToExpand) {
          console.error('Node not found:', nodeId);
          useGraphStore.getState().setIsFetching(false);
          return;
        }
        
        // Get the label of the node to expand
        const label = nodeToExpand.labels[0];
        if (!label) {
          console.error('Node has no label:', nodeId);
          useGraphStore.getState().setIsFetching(false);
          return;
        }
        
        // Fetch the extended subgraph with depth 2
        const extendedGraph = await queryGraphs(label, 2, 0);
        
        if (!extendedGraph || !extendedGraph.nodes || !extendedGraph.edges) {
          console.error('Failed to fetch extended graph');
          useGraphStore.getState().setIsFetching(false);
          return;
        }
        
        // Process nodes to add required properties for RawNodeType
        const processedNodes: RawNodeType[] = [];
        for (const node of extendedGraph.nodes) {
          // Generate random color values
          const r = Math.floor(Math.random() * 256);
          const g = Math.floor(Math.random() * 256);
          const b = Math.floor(Math.random() * 256);
          const color = `rgb(${r}, ${g}, ${b})`;
          
          // Create a properly typed RawNodeType
          processedNodes.push({
            id: node.id,
            labels: node.labels,
            properties: node.properties,
            size: 10, // Default size
            x: Math.random(), // Random position
            y: Math.random(), // Random position
            color: color, // Random color
            degree: 0 // Initial degree
          });
        }
        
        // Process edges to add required properties for RawEdgeType
        const processedEdges: RawEdgeType[] = [];
        for (const edge of extendedGraph.edges) {
          // Create a properly typed RawEdgeType
          processedEdges.push({
            id: edge.id,
            source: edge.source,
            target: edge.target,
            type: edge.type,
            properties: edge.properties,
            dynamicId: '' // Will be set when adding to sigma graph
          });
        }
        
        // Store current node positions
        const nodePositions: Record<string, {x: number, y: number}> = {};
        sigmaGraph.forEachNode((node) => {
          nodePositions[node] = {
            x: sigmaGraph.getNodeAttribute(node, 'x'),
            y: sigmaGraph.getNodeAttribute(node, 'y')
          };
        });
        
        // Get existing node IDs
        const existingNodeIds = new Set(sigmaGraph.nodes());
        
        // Check if there are any new nodes that can be connected to the selected node
        let hasConnectableNewNodes = false;
        for (const newNode of processedNodes) {
          // Skip if node already exists
          if (existingNodeIds.has(newNode.id)) {
            continue;
          }
          
          // Check if this node is connected to the selected node
          const isConnected = processedEdges.some(
            edge => (edge.source === nodeId && edge.target === newNode.id) || 
                   (edge.target === nodeId && edge.source === newNode.id)
          );
          
          if (isConnected) {
            hasConnectableNewNodes = true;
            break;
          }
        }
        
        // If no new connectable nodes found, show toast and return
        if (!hasConnectableNewNodes) {
          toast.info(t('graphPanel.propertiesView.node.noNewNodes'));
          useGraphStore.getState().setIsFetching(false);
          return;
        }
        
        // Get degree range from existing graph for size calculations
        let minDegree = Number.MAX_SAFE_INTEGER;
        let maxDegree = 0;
        sigmaGraph.forEachNode(node => {
          const degree = sigmaGraph.degree(node);
          minDegree = Math.min(minDegree, degree);
          maxDegree = Math.max(maxDegree, degree);
        });
        
        // Calculate size formula parameters
        const range = maxDegree - minDegree || 1; // Avoid division by zero
        const scale = Constants.maxNodeSize - Constants.minNodeSize;
        
        // Add new nodes from the processed nodes
        for (const newNode of processedNodes) {
          // Skip if node already exists
          if (existingNodeIds.has(newNode.id)) {
            continue;
          }
          
          // Check if this node is connected to the selected node
          const isConnected = processedEdges.some(
            edge => (edge.source === nodeId && edge.target === newNode.id) || 
                   (edge.target === nodeId && edge.source === newNode.id)
          );
          
          if (isConnected) {
            // Calculate node degree (number of connected edges)
            const nodeDegree = processedEdges.filter(edge => 
              edge.source === newNode.id || edge.target === newNode.id
            ).length;
            
            // Calculate node size using the same formula as in fetchGraph
            const nodeSize = Math.round(
              Constants.minNodeSize + scale * Math.pow((nodeDegree - minDegree) / range, 0.5)
            );
            
            // Add the new node to the graph with calculated size
            sigmaGraph.addNode(newNode.id, {
              label: newNode.labels.join(', '),
              color: newNode.color,
              x: nodePositions[nodeId].x + (Math.random() - 0.5) * 0.5,
              y: nodePositions[nodeId].y + (Math.random() - 0.5) * 0.5,
              size: nodeSize,
              borderColor: '#000',
              borderSize: 0.2
            });
            
            // Add the node to the raw graph
            if (!rawGraph.getNode(newNode.id)) {
              // Update the node size to match the calculated size
              newNode.size = nodeSize;
              // Add to nodes array
              rawGraph.nodes.push(newNode);
              // Update nodeIdMap
              rawGraph.nodeIdMap[newNode.id] = rawGraph.nodes.length - 1;
            }
          }
        }
        
        // Add new edges
        for (const newEdge of processedEdges) {
          // Only add edges where both source and target exist in the graph
          if (sigmaGraph.hasNode(newEdge.source) && sigmaGraph.hasNode(newEdge.target)) {
            // Skip if edge already exists
            if (sigmaGraph.hasEdge(newEdge.source, newEdge.target)) {
              continue;
            }
            
            // Add the edge to the sigma graph
            newEdge.dynamicId = sigmaGraph.addDirectedEdge(newEdge.source, newEdge.target, {
              label: newEdge.type || undefined
            });
            
            // Add the edge to the raw graph
            if (!rawGraph.getEdge(newEdge.id, false)) {
              // Add to edges array
              rawGraph.edges.push(newEdge);
              // Update edgeIdMap
              rawGraph.edgeIdMap[newEdge.id] = rawGraph.edges.length - 1;
              // Update dynamic edge map
              rawGraph.edgeDynamicIdMap[newEdge.dynamicId] = rawGraph.edges.length - 1;
            }
          }
        }
        
        // Update the dynamic edge map
        rawGraph.buildDynamicMap();
        
        // Restore positions for existing nodes
        Object.entries(nodePositions).forEach(([id, position]) => {
          if (sigmaGraph.hasNode(id)) {
            sigmaGraph.setNodeAttribute(id, 'x', position.x);
            sigmaGraph.setNodeAttribute(id, 'y', position.y);
          }
        });
        
        // Update the size of the expanded node based on its new edge count
        if (sigmaGraph.hasNode(nodeId)) {
          // Get the new degree of the expanded node
          const expandedNodeDegree = sigmaGraph.degree(nodeId);
          
          // Calculate new size for the expanded node using the same parameters
          const newSize = Math.round(
            Constants.minNodeSize + scale * Math.pow((expandedNodeDegree - minDegree) / range, 0.5)
          );
          
          // Update the size in sigma graph
          sigmaGraph.setNodeAttribute(nodeId, 'size', newSize);
          
          // Update the size in raw graph
          const expandedNodeIndex = rawGraph.nodeIdMap[nodeId];
          if (expandedNodeIndex !== undefined) {
            rawGraph.nodes[expandedNodeIndex].size = newSize;
          }
        }
        
        // Refresh the layout and store the node ID to reselect after refresh
        const nodeIdToSelect = nodeId;
        useGraphStore.getState().refreshLayout();
        
        // Use setTimeout to reselect the node after the layout refresh is complete
        setTimeout(() => {
          if (nodeIdToSelect) {
            useGraphStore.getState().setSelectedNode(nodeIdToSelect, true);
          }
        }, 2000); // Wait a bit longer than the refreshLayout timeout (which is 10ms)
        
      } catch (error) {
        console.error('Error expanding node:', error);
      } finally {
        // Reset fetching state and node to expand
        useGraphStore.getState().setIsFetching(false);
      }
    };
    
    // If there's a node to expand, handle it
    if (nodeToExpand) {
      handleNodeExpand(nodeToExpand);
      // Reset the nodeToExpand state after handling
      window.setTimeout(() => {
        useGraphStore.getState().triggerNodeExpand(null);
      }, 0);
    }
  }, [nodeToExpand, sigmaGraph, rawGraph]);
  
  // Helper function to get all nodes that will be deleted
  const getNodesThatWillBeDeleted = useCallback((nodeId: string, graph: DirectedGraph) => {
    const nodesToDelete = new Set<string>([nodeId]);
    
    // Find all nodes that would become isolated after deletion
    graph.forEachNode((node) => {
      if (node === nodeId) return; // Skip the node being deleted
      
      // Get all neighbors of this node
      const neighbors = graph.neighbors(node);
      
      // If this node has only one neighbor and that neighbor is the node being deleted,
      // this node will become isolated, so we should delete it too
      if (neighbors.length === 1 && neighbors[0] === nodeId) {
        nodesToDelete.add(node);
      }
    });
    
    return nodesToDelete;
  }, []);

  // Handle node pruning
  useEffect(() => {
    const handleNodePrune = (nodeId: string | null) => {
      if (!nodeId || !sigmaGraph || !rawGraph) return;
      
      try {
        // Check if the node exists
        if (!sigmaGraph.hasNode(nodeId)) {
          console.error('Node not found:', nodeId);
          return;
        }
        
        // Get all nodes that will be deleted (including isolated nodes)
        const nodesToDelete = getNodesThatWillBeDeleted(nodeId, sigmaGraph);
        
        // Check if we would delete all nodes in the graph
        if (nodesToDelete.size === sigmaGraph.nodes().length) {
          toast.error(t('graphPanel.propertiesView.node.deleteAllNodesError'));
          return;
        }
        
        // If the node is selected or focused, clear selection
        const state = useGraphStore.getState();
        if (state.selectedNode === nodeId || state.focusedNode === nodeId) {
          state.clearSelection();
        }
        
        // Process all nodes that need to be deleted
        for (const nodeToDelete of nodesToDelete) {
          // Remove the node from the sigma graph (this will also remove connected edges)
          sigmaGraph.dropNode(nodeToDelete);
          
          // Remove the node from the raw graph
          const nodeIndex = rawGraph.nodeIdMap[nodeToDelete];
          if (nodeIndex !== undefined) {
            // Find all edges connected to this node
            const edgesToRemove = rawGraph.edges.filter(
              edge => edge.source === nodeToDelete || edge.target === nodeToDelete
            );
            
            // Remove edges from raw graph
            for (const edge of edgesToRemove) {
              const edgeIndex = rawGraph.edgeIdMap[edge.id];
              if (edgeIndex !== undefined) {
                // Remove from edges array
                rawGraph.edges.splice(edgeIndex, 1);
                // Update edgeIdMap for all edges after this one
                for (const [id, idx] of Object.entries(rawGraph.edgeIdMap)) {
                  if (idx > edgeIndex) {
                    rawGraph.edgeIdMap[id] = idx - 1;
                  }
                }
                // Remove from edgeIdMap
                delete rawGraph.edgeIdMap[edge.id];
                // Remove from edgeDynamicIdMap
                delete rawGraph.edgeDynamicIdMap[edge.dynamicId];
              }
            }
            
            // Remove node from nodes array
            rawGraph.nodes.splice(nodeIndex, 1);
            
            // Update nodeIdMap for all nodes after this one
            for (const [id, idx] of Object.entries(rawGraph.nodeIdMap)) {
              if (idx > nodeIndex) {
                rawGraph.nodeIdMap[id] = idx - 1;
              }
            }
            
            // Remove from nodeIdMap
            delete rawGraph.nodeIdMap[nodeToDelete];
          }
        }
        
        // Rebuild the dynamic edge map
        rawGraph.buildDynamicMap();
        
        // Show notification if we deleted more than just the selected node
        if (nodesToDelete.size > 1) {
          toast.info(t('graphPanel.propertiesView.node.nodesRemoved', { count: nodesToDelete.size }));
        }
        
        // Force a refresh of the graph layout
        useGraphStore.getState().refreshLayout();
        
      } catch (error) {
        console.error('Error pruning node:', error);
      }
    };
    
    // If there's a node to prune, handle it
    if (nodeToPrune) {
      handleNodePrune(nodeToPrune);
      // Reset the nodeToPrune state after handling
      window.setTimeout(() => {
        useGraphStore.getState().triggerNodePrune(null);
      }, 0);
    }
  }, [nodeToPrune, sigmaGraph, rawGraph, getNodesThatWillBeDeleted, t]);

  const lightrageGraph = useCallback(() => {
    // If we already have a graph instance, return it
    if (sigmaGraph) {
      return sigmaGraph as Graph<NodeType, EdgeType>
    }

    // If no graph exists yet, create a new one and store it
    console.log('Creating new Sigma graph instance')
    const graph = new DirectedGraph()
    useGraphStore.getState().setSigmaGraph(graph)
    return graph as Graph<NodeType, EdgeType>
  }, [sigmaGraph])

  return { lightrageGraph, getNode, getEdge }
}

export default useLightrangeGraph
