# 3D GraphML Viewer

An interactive 3D graph visualization tool based on Dear ImGui and ModernGL.

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

## Usage

1. **Launch the Program**:
   ```bash
   python -m pip install -r requirements.txt
   python graph_visualizer.py
   ```

2. **Load Font**:
   - Place the font file `font.ttf` in the `assets` directory
   - Or modify the `CUSTOM_FONT` constant to use a different font file

3. **Load Graph File**:
   - Click the "Load GraphML" button in the interface
   - Select a graph file in GraphML format

4. **Interactive Controls**:
   - **Camera Movement**:
     - W: Move forward
     - S: Move backward
     - A: Move left
     - D: Move right
     - Q: Move up
     - E: Move down
   - **View Control**:
     - Hold right mouse button and drag to rotate view
   - **Node Interaction**:
     - Hover mouse to highlight nodes
     - Click to select nodes

5. **Visualization Settings**:
   - Adjustable via UI control panel:
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

## Performance Optimizations

- Efficient graphics rendering using ModernGL
- View distance culling for label display optimization
- Community detection algorithms for optimized visualization of large-scale graphs

## System Requirements

- Python 3.10+
- Graphics card with OpenGL 3.3+ support
- Supported Operating Systems: Windows/Linux/MacOS
