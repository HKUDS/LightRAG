# LightRAG 3D Graph Viewer

An interactive 3D graph visualization tool included in the LightRAG package for visualizing and analyzing RAG (Retrieval-Augmented Generation) graphs and other graph structures.

![image](https://github.com/user-attachments/assets/b0d86184-99fc-468c-96ed-c611f14292bf)

## Installation

### Quick Install
```bash
pip install lightrag-hku[tools]  # Install with visualization tool only
# or
pip install lightrag-hku[api,tools]  # Install with both API and visualization tools
```

## Launch the Viewer
```bash
lightrag-viewer
```

## Features

- **3D Interactive Visualization**: High-performance 3D graphics rendering using ModernGL
- **Multiple Layout Algorithms**: Support for various graph layouts
  - Spring layout
  - Circular layout
  - Shell layout
  - Random layout
- **Community Detection**: Automatic detection and visualization of graph community structures
- **Interactive Controls**:
  - WASD + QE keys for camera movement
  - Right mouse drag for view angle control
  - Node selection and highlighting
  - Adjustable node size and edge width
  - Configurable label display
  - Quick navigation between node connections

## Tech Stack

- **imgui_bundle**: User interface
- **ModernGL**: OpenGL graphics rendering
- **NetworkX**: Graph data structures and algorithms
- **NumPy**: Numerical computations
- **community**: Community detection

## Interactive Controls

### Camera Movement
- W: Move forward
- S: Move backward
- A: Move left
- D: Move right
- Q: Move up
- E: Move down

### View Control
- Hold right mouse button and drag to rotate view

### Node Interaction
- Hover mouse to highlight nodes
- Click to select nodes

## Visualization Settings

Adjustable via UI control panel:
- Layout type
- Node size
- Edge width
- Label visibility
- Label size
- Background color

## Customization Options

- **Node Scaling**: Adjust node size via `node_scale` parameter
- **Edge Width**: Modify edge width using `edge_width` parameter
- **Label Display**: Toggle label visibility with `show_labels`
- **Label Size**: Adjust label size using `label_size`
- **Label Color**: Set label color through `label_color`
- **View Distance**: Control maximum label display distance with `label_culling_distance`

## System Requirements

- Python 3.9+
- Graphics card with OpenGL 3.3+ support
- Supported Operating Systems: Windows/Linux/MacOS

## Troubleshooting

### Common Issues

1. **Command Not Found**
   ```bash
   # Make sure you installed with the 'tools' option
   pip install lightrag-hku[tools]

   # Verify installation
   pip list | grep lightrag-hku
   ```

2. **ModernGL Initialization Failed**
   ```bash
   # Check OpenGL version
   glxinfo | grep "OpenGL version"

   # Update graphics drivers if needed
   ```

3. **Font Loading Issues**
   - The required fonts are included in the package
   - If issues persist, check your graphics drivers

## Usage with LightRAG

The viewer is particularly useful for:
- Visualizing RAG knowledge graphs
- Analyzing document relationships
- Exploring semantic connections
- Debugging retrieval patterns

## Performance Optimizations

- Efficient graphics rendering using ModernGL
- View distance culling for label display optimization
- Community detection algorithms for optimized visualization of large-scale graphs

## Support

- GitHub Issues: [LightRAG Repository](https://github.com/HKUDS/LightRAG)
- Documentation: [LightRAG Docs](https://URL-to-docs)

## License

This tool is part of LightRAG and is distributed under the MIT License. See `LICENSE` for more information.

Note: This visualization tool is an optional component of the LightRAG package. Install with the [tools] option to access the viewer functionality.
