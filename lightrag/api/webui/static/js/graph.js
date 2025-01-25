// js/graph.js
function openGraphModal(label) {
    const modal = document.getElementById("graph-modal");
    const graphTitle = document.getElementById("graph-title");

    if (!modal || !graphTitle) {
        console.error("Key element not found");
        return;
    }

    graphTitle.textContent = `Knowledge Graph - ${label}`;
    modal.style.display = "flex";

    renderGraph(label);
}

function closeGraphModal() {
    const modal = document.getElementById("graph-modal");
    modal.style.display = "none";
    clearGraph();
}

function clearGraph() {
    const svg = document.getElementById("graph-svg");
    svg.innerHTML = "";
}


async function getGraph(label) {
    try {
        const response = await fetch(`/graphs?label=${label}`);
        const rawData = await response.json();
        console.log({data: JSON.parse(JSON.stringify(rawData))});

        const nodes = rawData.nodes

        nodes.forEach(node => {
            node.id = Date.now().toString(36) + Math.random().toString(36).substring(2); // 使用 crypto.randomUUID() 生成唯一 UUID
        });

        //  Strictly verify edge data
        const edges = (rawData.edges || []).map(edge => {
            const sourceNode = nodes.find(n => n.labels.includes(edge.source));
            const targetNode = nodes.find(n => n.labels.includes(edge.target)
                )
            ;
            if (!sourceNode || !targetNode) {
                console.warn("NOT VALID EDGE:", edge);
                return null;
            }
            return {
                source: sourceNode,
                target: targetNode,
                type: edge.type || ""
            };
        }).filter(edge => edge !== null);

        return {nodes, edges};
    } catch (error) {
        console.error("Loading graph failed:", error);
        return {nodes: [], edges: []};
    }
}

async function renderGraph(label) {
    const data = await getGraph(label);


    if (!data.nodes || data.nodes.length === 0) {
        d3.select("#graph-svg")
            .html(`<text x="50%" y="50%" text-anchor="middle">No valid nodes</text>`);
        return;
    }


    const svg = d3.select("#graph-svg");
    const width = svg.node().clientWidth;
    const height = svg.node().clientHeight;

    svg.selectAll("*").remove();

    //  Create a force oriented diagram layout
    const simulation = d3.forceSimulation(data.nodes)
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2));

    //  Add a connection (if there are valid edges)
    if (data.edges.length > 0) {
        simulation.force("link",
            d3.forceLink(data.edges)
                .id(d => d.id)
                .distance(100)
        );
    }

    //  Draw nodes
    const nodes = svg.selectAll(".node")
        .data(data.nodes)
        .enter()
        .append("circle")
        .attr("class", "node")
        .attr("r", 10)
        .call(d3.drag()
            .on("start", dragStarted)
            .on("drag", dragged)
            .on("end", dragEnded)
        );


    svg.append("defs")
        .append("marker")
        .attr("id", "arrow-out")
        .attr("viewBox", "0 0 10 10")
        .attr("refX", 8)
        .attr("refY", 5)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,0 L10,5 L0,10 Z")
        .attr("fill", "#999");

    //  Draw edges (with arrows)
    const links = svg.selectAll(".link")
        .data(data.edges)
        .enter()
        .append("line")
        .attr("class", "link")
        .attr("marker-end", "url(#arrow-out)"); //  Always draw arrows on the target side

    //  Edge style configuration
    links
        .attr("stroke", "#999")
        .attr("stroke-width", 2)
        .attr("stroke-opacity", 0.8);

    //  Draw label (with background box)
    const labels = svg.selectAll(".label")
        .data(data.nodes)
        .enter()
        .append("text")
        .attr("class", "label")
        .text(d => d.labels[0] || "")
        .attr("text-anchor", "start")
        .attr("dy", "0.3em")
        .attr("fill", "#333");

    //  Update Location
    simulation.on("tick", () => {
        links
            .attr("x1", d => {
                //  Calculate the direction vector from the source node to the target node
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance === 0) return d.source.x; // 避免除以零 Avoid dividing by zero
                // Adjust the starting point coordinates (source node edge) based on radius 10
                return d.source.x + (dx / distance) * 10;
            })
            .attr("y1", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance === 0) return d.source.y;
                return d.source.y + (dy / distance) * 10;
            })
            .attr("x2", d => {
                // Adjust the endpoint coordinates (target node edge) based on a radius of 10
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance === 0) return d.target.x;
                return d.target.x - (dx / distance) * 10;
            })
            .attr("y2", d => {
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance === 0) return d.target.y;
                return d.target.y - (dy / distance) * 10;
            });

        // Update the position of nodes and labels (keep unchanged)
        nodes
            .attr("cx", d => d.x)
            .attr("cy", d => d.y);

        labels
            .attr("x", d => d.x + 12)
            .attr("y", d => d.y + 4);
    });

    // Drag and drop logic
    function dragStarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
        simulation.alpha(0.3).restart();
    }

    function dragEnded(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}
