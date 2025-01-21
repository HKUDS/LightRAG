"""
3D GraphML Viewer
Author: LoLLMs
Description: An interactive 3D GraphML viewer using PyQt5 and pyqtgraph
Version: 2.2
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import pipmaster as pm
# Install all required dependencies
REQUIRED_PACKAGES = [
    "PyQt5",
    "pyqtgraph",
    "numpy",
    "PyOpenGL",
    "PyOpenGL_accelerate",
    "networkx",
    "matplotlib",
    "python-louvain",
    "ascii_colors"
]

from ascii_colors import ASCIIColors, trace_exception

def setup_dependencies():
    """
    Ensure all required packages are installed
    """
    for package in REQUIRED_PACKAGES:
        if not pm.is_installed(package):
            print(f"Installing {package}...")
            pm.install(package)

# Install dependencies
setup_dependencies()

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QMessageBox, QSpinBox, QComboBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QSplitter, QDockWidget,
    QTextEdit
)
from PyQt5.QtCore import Qt
import pyqtgraph.opengl as gl


class Point:
    """Simple point class to handle coordinates"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class NodeState:
    """Data class for node visual state"""
    NORMAL_SCALE = 1.0
    HOVER_SCALE = 1.2
    SELECTED_SCALE = 1.3
    
    NORMAL_OPACITY = 0.8
    HOVER_OPACITY = 1.0
    SELECTED_OPACITY = 1.0
    
    # Increase base node size (was 0.05)
    BASE_SIZE = 0.2
    
    SELECTED_COLOR = (1.0, 1.0, 0.0, 1.0)
    HOVER_COLOR = (1.0, 0.8, 0.0, 1.0)


class Node3D:
    """Class representing a 3D node in the graph"""
    def __init__(self, position: np.ndarray, color: Tuple[float, float, float, float], 
                 label: str, node_type: str, size: float):
        self.position = position
        self.base_color = color
        self.color = color
        self.label = label
        self.node_type = node_type
        self.size = size
        self.mesh_item = None
        self.label_item = None
        self.is_highlighted = False
        self.is_selected = False

    def highlight(self):
        """Highlight the node"""
        if not self.is_highlighted and not self.is_selected:
            self.color = NodeState.HOVER_COLOR
            self.update_appearance(NodeState.HOVER_SCALE)
            self.is_highlighted = True

    def unhighlight(self):
        """Remove highlight from node"""
        if self.is_highlighted and not self.is_selected:
            self.color = self.base_color
            self.update_appearance(NodeState.NORMAL_SCALE)
            self.is_highlighted = False

    def select(self):
        """Select the node"""
        self.is_selected = True
        self.color = NodeState.SELECTED_COLOR
        self.update_appearance(NodeState.SELECTED_SCALE)

    def deselect(self):
        """Deselect the node"""
        self.is_selected = False
        self.color = self.base_color
        self.update_appearance(NodeState.NORMAL_SCALE)

    def update_appearance(self, scale: float = 1.0):
        """Update node visual appearance"""
        if self.mesh_item:
            self.mesh_item.setData(
                color=np.array([self.color]),
                size=np.array([self.size * scale * 5])
            )

class NodeDetailsWidget(QWidget):
    """Widget to display node details"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Properties text edit
        self.properties = QTextEdit()
        self.properties.setReadOnly(True)
        layout.addWidget(QLabel("Properties:"))
        layout.addWidget(self.properties)
        
        # Connections table
        self.connections = QTableWidget()
        self.connections.setColumnCount(3)
        self.connections.setHorizontalHeaderLabels(
            ["Connected Node", "Relationship", "Direction"]
        )
        layout.addWidget(QLabel("Connections:"))
        layout.addWidget(self.connections)

    def update_node_info(self, node_data: Dict, connections: Dict):
        """Update the display with node information"""
        # Update properties
        properties_text = "Node Properties:\n"
        for key, value in node_data.items():
            properties_text += f"{key}: {value}\n"
        self.properties.setText(properties_text)
        
        # Update connections
        self.connections.setRowCount(len(connections))
        for idx, (neighbor, edge_data) in enumerate(connections.items()):
            self.connections.setItem(idx, 0, QTableWidgetItem(str(neighbor)))
            self.connections.setItem(
                idx, 1, 
                QTableWidgetItem(edge_data.get('relationship', 'unknown'))
            )
            self.connections.setItem(idx, 2, QTableWidgetItem("outgoing"))

class GraphMLViewer3D(QMainWindow):
    """Main window class for 3D GraphML visualization"""
    def __init__(self):
        super().__init__()
        
        self.graph: Optional[nx.Graph] = None
        self.nodes: Dict[str, Node3D] = {}
        self.edges: List[gl.GLLinePlotItem] = []
        self.edge_labels: List[gl.GLTextItem] = []
        self.selected_node = None
        self.communities = None
        self.community_colors = None
        
        self.mouse_pos_last = None
        self.mouse_buttons_pressed = set()
        self.distance = 20  # Initial camera distance
        self.center = np.array([0, 0, 0])  # View center point
        self.elevation = 30  # Initial camera elevation
        self.azimuth = 45  # Initial camera azimuth


        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("3D GraphML Viewer")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.main_splitter)
        
        # Create left panel for 3D view
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Create controls
        self.create_toolbar(left_layout)
        
        # Create 3D view
        self.view = gl.GLViewWidget()
        self.view.setMouseTracking(True)
        
        # Connect mouse events
        self.view.mousePressEvent = self.on_mouse_press
        self.view.mouseMoveEvent = self.on_mouse_move
        left_layout.addWidget(self.view)
        
        self.main_splitter.addWidget(left_widget)
        
        # Create details widget
        self.details = NodeDetailsWidget()
        details_dock = QDockWidget("Node Details", self)
        details_dock.setWidget(self.details)
        self.addDockWidget(Qt.RightDockWidgetArea, details_dock)
        
        # Add status bar
        self.statusBar().showMessage("Ready")
        
        # Add initial grid
        grid = gl.GLGridItem()
        grid.setSize(x=20, y=20, z=20)
        grid.setSpacing(x=1, y=1, z=1)
        self.view.addItem(grid)

        # Set initial camera position
        self.view.setCameraPosition(
            distance=self.distance,
            elevation=self.elevation,
            azimuth=self.azimuth
        )
        
        # Connect all mouse events
        self.view.mousePressEvent = self.on_mouse_press
        self.view.mouseReleaseEvent = self.on_mouse_release
        self.view.mouseMoveEvent = self.on_mouse_move
        self.view.wheelEvent = self.on_mouse_wheel

    def calculate_node_sizes(self) -> Dict[str, float]:
        """Calculate node sizes based on number of connections"""
        if not self.graph:
            return {}
            
        # Get degree (number of connections) for each node
        degrees = dict(self.graph.degree())
        
        # Calculate size scaling
        max_degree = max(degrees.values())
        min_degree = min(degrees.values())
        
        # Normalize sizes between 0.5 and 2.0
        sizes = {}
        for node, degree in degrees.items():
            if max_degree == min_degree:
                sizes[node] = 1.0
            else:
                # Normalize and scale size
                normalized = (degree - min_degree) / (max_degree - min_degree)
                sizes[node] = 0.5 + normalized * 1.5
                
        return sizes


    def create_toolbar(self, layout: QVBoxLayout):
        """Create the toolbar with controls"""
        toolbar = QHBoxLayout()
        
        # Load button
        load_btn = QPushButton("Load GraphML")
        load_btn.clicked.connect(self.load_graphml)
        toolbar.addWidget(load_btn)
        
        # Reset view button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(lambda: self.view.setCameraPosition(distance=20))
        toolbar.addWidget(reset_btn)
        
        # Layout selector
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Spring", "Circular", "Shell", "Random"])
        self.layout_combo.currentTextChanged.connect(self.refresh_layout)
        toolbar.addWidget(QLabel("Layout:"))
        toolbar.addWidget(self.layout_combo)
        
        # Node size control
        self.node_size = QSpinBox()
        self.node_size.setRange(1, 100)
        self.node_size.setValue(20)
        self.node_size.valueChanged.connect(self.refresh_layout)
        toolbar.addWidget(QLabel("Node Size:"))
        toolbar.addWidget(self.node_size)
        
        # Show labels checkbox
        self.show_labels = QCheckBox("Show Labels")
        self.show_labels.setChecked(True)
        self.show_labels.stateChanged.connect(self.refresh_layout)
        toolbar.addWidget(self.show_labels)
        
        layout.addLayout(toolbar)
        # Reset view button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)  # Use the new reset_view method
        toolbar.addWidget(reset_btn)



    def load_graphml(self) -> None:
        """Load and visualize a GraphML file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open GraphML file", "", "GraphML files (*.graphml)"
            )
            
            if file_path:
                self.graph = nx.read_graphml(Path(file_path))
                self.refresh_layout()
                self.statusBar().showMessage(f"Loaded: {file_path}")
        except Exception as e:
            trace_exception(e)
            QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    def calculate_layout(self) -> Dict[str, np.ndarray]:
        """Calculate node positions based on selected layout"""
        layout_type = self.layout_combo.currentText().lower()
        
        # Detect communities for coloring
        self.communities = community.best_partition(self.graph)
        num_communities = len(set(self.communities.values()))
        self.community_colors = plt.cm.rainbow(np.linspace(0, 1, num_communities))
        
        if layout_type == "spring":
            pos = nx.spring_layout(
                self.graph,
                dim=3,
                k=2.0,
                iterations=100,
                weight=None
            )
        elif layout_type == "circular":
            pos_2d = nx.circular_layout(self.graph)
            pos = {node: np.array([x, y, 0.0]) for node, (x, y) in pos_2d.items()}
        elif layout_type == "shell":
            comm_lists = [[] for _ in range(num_communities)]
            for node, comm in self.communities.items():
                comm_lists[comm].append(node)
            pos_2d = nx.shell_layout(self.graph, comm_lists)
            pos = {node: np.array([x, y, 0.0]) for node, (x, y) in pos_2d.items()}
        else:  # random
            pos = {node: np.random.rand(3) * 2 - 1 for node in self.graph.nodes()}
        
        # Scale positions
        positions = np.array(list(pos.values()))
        if len(positions) > 0:
            scale = 10.0 / max(1.0, np.max(np.abs(positions)))
            return {node: coords * scale for node, coords in pos.items()}
        return pos

    def get_node_color(self, node_id: str) -> Tuple[float, float, float, float]:
        """Get RGBA color based on community"""
        if hasattr(self, 'communities') and node_id in self.communities:
            comm_id = self.communities[node_id]
            color = self.community_colors[comm_id]
            return tuple(color)
        return (0.5, 0.5, 0.5, 0.8)


    def create_node(self, node_id: str, position: np.ndarray, node_type: str) -> Node3D:
        """Create a 3D node with interaction capabilities"""
        color = self.get_node_color(node_id)
        
        # Get size multiplier based on connections
        size_multiplier = self.node_sizes.get(node_id, 1.0)
        size = NodeState.BASE_SIZE * self.node_size.value() / 50.0 * size_multiplier
        
        node = Node3D(position, color, str(node_id), node_type, size)
        
        node.mesh_item = gl.GLScatterPlotItem(
            pos=np.array([position]),
            size=np.array([size * 8]),
            color=np.array([color]),
            pxMode=False
        )
        
        # Enable picking and set node ID
        node.mesh_item.setGLOptions('translucent')
        node.mesh_item.node_id = node_id
        
        if self.show_labels.isChecked():
            node.label_item = gl.GLTextItem(
                pos=position,
                text=str(node_id),
                color=(1, 1, 1, 1),
            )
        
        return node


    

    def mapToView(self, pos) -> Point:
        """Convert screen coordinates to world coordinates"""
        # Get the viewport size
        width = self.view.width()
        height = self.view.height()
        
        # Normalize coordinates
        x = (pos.x() / width - 0.5) * 20  # Scale factor of 20 matches the grid size
        y = -(pos.y() / height - 0.5) * 20
        
        return Point(x, y)

    def on_mouse_move(self, event):
        """Handle mouse movement for pan, rotate and hover"""
        if self.mouse_pos_last is None:
            self.mouse_pos_last = event.pos()
            return
            
        pos = event.pos()
        dx = pos.x() - self.mouse_pos_last.x()
        dy = pos.y() - self.mouse_pos_last.y()
        
        # Handle right button drag for panning
        if Qt.RightButton in self.mouse_buttons_pressed:
            # Scale the pan amount based on view distance
            scale = self.distance / 1000.0
            
            # Calculate pan in view coordinates
            right = np.cross([0, 0, 1], self.view.cameraPosition())
            right = right / np.linalg.norm(right)
            up = np.cross(self.view.cameraPosition(), right)
            up = up / np.linalg.norm(up)
            
            pan = -right * dx * scale + up * dy * scale
            self.center += pan
            self.view.pan(dx, dy, 0)
        
        # Handle middle button drag for rotation
        elif Qt.MiddleButton in self.mouse_buttons_pressed:
            self.azimuth += dx * 0.5  # Adjust rotation speed as needed
            self.elevation -= dy * 0.5
            
            # Clamp elevation to prevent gimbal lock
            self.elevation = np.clip(self.elevation, -89, 89)
            
            self.view.setCameraPosition(
                distance=self.distance,
                elevation=self.elevation,
                azimuth=self.azimuth
            )
        
        # Handle hover events when no buttons are pressed
        elif not self.mouse_buttons_pressed:
            # Get the mouse position in world coordinates
            mouse_pos = self.mapToView(pos)
            
            # Check for hover
            min_dist = float('inf')
            hovered_node = None
            
            for node_id, node in self.nodes.items():
                # Calculate distance to mouse in world coordinates
                dx = mouse_pos.x - node.position[0]
                dy = mouse_pos.y - node.position[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < min_dist and dist < 0.5:  # Adjust threshold as needed
                    min_dist = dist
                    hovered_node = node_id
            
            # Update hover states
            for node_id, node in self.nodes.items():
                if node_id == hovered_node:
                    node.highlight()
                    self.statusBar().showMessage(f"Node: {node_id} ({node.node_type})")
                else:
                    if not node.is_selected:
                        node.unhighlight()
            self.mouse_pos_last = pos                      

    def on_mouse_press(self, event):
        """Handle mouse press events"""
        self.mouse_pos_last = event.pos()
        self.mouse_buttons_pressed.add(event.button())
        
        # Handle left click for node selection
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            mouse_pos = self.mapToView(pos)            
            
            # Find closest node
            min_dist = float('inf')
            clicked_node = None
            
            for node_id, node in self.nodes.items():
                dx = mouse_pos.x - node.position[0]
                dy = mouse_pos.y - node.position[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < min_dist and dist < 0.5:  # Adjust threshold as needed
                    min_dist = dist
                    clicked_node = node_id
            
            # Handle selection
            if clicked_node:
                if self.selected_node and self.selected_node in self.nodes:
                    self.nodes[self.selected_node].deselect()
                
                self.nodes[clicked_node].select()
                self.selected_node = clicked_node
                
                if self.graph:
                    self.details.update_node_info(
                        self.graph.nodes[clicked_node],
                        self.graph[clicked_node]
                    )
    def on_mouse_release(self, event):
        """Handle mouse release events"""
        self.mouse_buttons_pressed.discard(event.button())
        self.mouse_pos_last = None


    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        delta = event.angleDelta().y()
        
        # Adjust zoom speed based on current distance
        zoom_speed = self.distance / 100.0
        
        # Update distance with limits
        self.distance -= delta * zoom_speed
        self.distance = np.clip(self.distance, 1.0, 100.0)
        
        self.view.setCameraPosition(
            distance=self.distance,
            elevation=self.elevation,
            azimuth=self.azimuth
        )
    
    def reset_view(self):
        """Reset camera to default position"""
        self.distance = 20
        self.elevation = 30
        self.azimuth = 45
        self.center = np.array([0, 0, 0])
        
        self.view.setCameraPosition(
            distance=self.distance,
            elevation=self.elevation,
            azimuth=self.azimuth
        )


    def create_edge(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                   color: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.2)
                   ) -> gl.GLLinePlotItem:
        """Create a 3D edge between nodes"""
        return gl.GLLinePlotItem(
            pos=np.array([start_pos, end_pos]),
            color=color,
            width=1,
            antialias=True,
            mode='lines'
        )

    def handle_node_hover(self, event: Any, node_id: str) -> None:
        """Handle node hover events"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            if event.isEnter():
                node.highlight()
                self.statusBar().showMessage(f"Node: {node_id} ({node.node_type})")
            elif event.isExit():
                node.unhighlight()
                self.statusBar().showMessage("")

    def handle_node_click(self, event: Any, node_id: str) -> None:
        """Handle node click events"""
        if event.button() != Qt.LeftButton or node_id not in self.nodes:
            return
            
        if self.selected_node and self.selected_node in self.nodes:
            self.nodes[self.selected_node].deselect()
        
        node = self.nodes[node_id]
        node.select()
        self.selected_node = node_id
        
        if self.graph:
            self.details.update_node_info(
                self.graph.nodes[node_id],
                self.graph[node_id]
            )

    def refresh_layout(self) -> None:
        """Refresh the graph visualization"""
        if not self.graph:
            return
            
        self.positions = self.calculate_layout()
        self.node_sizes = self.calculate_node_sizes()
            
        self.view.clear()
        self.nodes.clear()
        self.edges.clear()
        self.edge_labels.clear()
        
        grid = gl.GLGridItem()
        grid.setSize(x=20, y=20, z=20)
        grid.setSpacing(x=1, y=1, z=1)
        self.view.addItem(grid)
        
        positions = self.calculate_layout()
        
        for node_id in self.graph.nodes():
            node_type = self.graph.nodes[node_id].get('type', 'default')
            node = self.create_node(node_id, positions[node_id], node_type)
            
            self.view.addItem(node.mesh_item)
            if node.label_item:
                self.view.addItem(node.label_item)
                
            self.nodes[node_id] = node
        
        for source, target in self.graph.edges():
            edge = self.create_edge(positions[source], positions[target])
            self.view.addItem(edge)
            self.edges.append(edge)
            
            if self.show_labels.isChecked():
                mid_point = (positions[source] + positions[target]) / 2
                relationship = self.graph.edges[source, target].get('relationship', '')
                if relationship:
                    label = gl.GLTextItem(
                        pos=mid_point,
                        text=relationship,
                        color=(0.8, 0.8, 0.8, 0.8),
                    )
                    self.view.addItem(label)
                    self.edge_labels.append(label)

def main():
    """Application entry point"""
    import sys
    
    app = QApplication(sys.argv)
    viewer = GraphMLViewer3D()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()