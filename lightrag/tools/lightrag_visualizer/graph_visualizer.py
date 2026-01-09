from typing import Optional, Tuple, Dict, List
import numpy as np
import networkx as nx
import pipmaster as pm

# Added automatic libraries install using pipmaster
if not pm.is_installed("moderngl"):
    pm.install("moderngl")
if not pm.is_installed("imgui_bundle"):
    pm.install("imgui_bundle")
if not pm.is_installed("pyglm"):
    pm.install("pyglm")
if not pm.is_installed("python-louvain"):
    pm.install("python-louvain")

import moderngl
from imgui_bundle import imgui, immapp, hello_imgui
import community
import glm
import tkinter as tk
from tkinter import filedialog
import traceback
import colorsys
import os

CUSTOM_FONT = "font.ttf"

DEFAULT_FONT_ENG = "Geist-Regular.ttf"
DEFAULT_FONT_CHI = "SmileySans-Oblique.ttf"


class Node3D:
    """Class representing a 3D node in the graph"""

    def __init__(
        self, position: glm.vec3, color: glm.vec3, label: str, size: float, idx: int
    ):
        self.position = position
        self.color = color
        self.label = label
        self.size = size
        self.idx = idx


class GraphViewer:
    """Main class for 3D graph visualization"""

    def __init__(self):
        self.glctx = None  # ModernGL context
        self.graph: Optional[nx.Graph] = None
        self.nodes: List[Node3D] = []
        self.id_node_map: Dict[str, Node3D] = {}
        self.communities = None
        self.community_colors = None

        # Window dimensions
        self.window_width = 1280
        self.window_height = 720

        # Camera parameters
        self.position = glm.vec3(0.0, -10.0, 0.0)  # Initial camera position
        self.front = glm.vec3(0.0, 1.0, 0.0)  # Direction camera is facing
        self.up = glm.vec3(0.0, 0.0, 1.0)  # Up vector
        self.yaw = 90.0  # Horizontal rotation (around Z axis)
        self.pitch = 0.0  # Vertical rotation
        self.move_speed = 0.05
        self.mouse_sensitivity = 0.15

        # Graph visualization settings
        self.layout_type = "Spring"
        self.node_scale = 0.2
        self.edge_width = 0.5
        self.show_labels = True
        self.label_size = 2
        self.label_color = (1.0, 1.0, 1.0, 1.0)
        self.label_culling_distance = 10.0
        self.available_layouts = ("Spring", "Circular", "Shell", "Random")
        self.background_color = (0.05, 0.05, 0.05, 1.0)

        # Mouse interaction
        self.last_mouse_pos = None
        self.mouse_pressed = False
        self.mouse_button = -1
        self.first_mouse = True

        # File dialog state
        self.show_load_error = False
        self.error_message = ""

        # Selection state
        self.selected_node: Optional[Node3D] = None
        self.highlighted_node: Optional[Node3D] = None

        # Node id map
        self.node_id_fbo = None
        self.node_id_texture = None
        self.node_id_depth = None
        self.node_id_texture_np: np.ndarray = None

        # Static data
        self.sphere_data = create_sphere()

        # Initialization flag
        self.initialized = False

    def setup(self):
        self.setup_render_context()
        self.setup_shaders()
        self.setup_buffers()
        self.initialized = True

    def handle_keyboard_input(self):
        """Handle WASD keyboard input for camera movement"""
        io = imgui.get_io()

        if io.want_capture_keyboard:
            return

        # Calculate camera vectors
        right = glm.normalize(glm.cross(self.front, self.up))

        # Get movement direction from WASD keys
        if imgui.is_key_down(imgui.Key.w):  # Forward
            self.position += self.front * self.move_speed * 0.1
        if imgui.is_key_down(imgui.Key.s):  # Backward
            self.position -= self.front * self.move_speed * 0.1
        if imgui.is_key_down(imgui.Key.a):  # Left
            self.position -= right * self.move_speed * 0.1
        if imgui.is_key_down(imgui.Key.d):  # Right
            self.position += right * self.move_speed * 0.1
        if imgui.is_key_down(imgui.Key.q):  # Up
            self.position += self.up * self.move_speed * 0.1
        if imgui.is_key_down(imgui.Key.e):  # Down
            self.position -= self.up * self.move_speed * 0.1

    def handle_mouse_interaction(self):
        """Handle mouse interaction for camera control and node selection"""
        if (
            imgui.is_any_item_active()
            or imgui.is_any_item_hovered()
            or imgui.is_any_item_focused()
        ):
            return

        io = imgui.get_io()
        mouse_pos = (io.mouse_pos.x, io.mouse_pos.y)
        if (
            mouse_pos[0] < 0
            or mouse_pos[1] < 0
            or mouse_pos[0] >= self.window_width
            or mouse_pos[1] >= self.window_height
        ):
            return

        # Handle first mouse input
        if self.first_mouse:
            self.last_mouse_pos = mouse_pos
            self.first_mouse = False
            return

        # Handle mouse movement for camera rotation
        if self.mouse_pressed and self.mouse_button == 1:  # Right mouse button
            dx = self.last_mouse_pos[0] - mouse_pos[0]
            dy = self.last_mouse_pos[1] - mouse_pos[1]  # Reversed for intuitive control

            dx *= self.mouse_sensitivity
            dy *= self.mouse_sensitivity

            self.yaw += dx
            self.pitch += dy

            # Limit pitch to avoid flipping
            self.pitch = np.clip(self.pitch, -89.0, 89.0)

            # Update front vector
            self.front = glm.normalize(
                glm.vec3(
                    np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
                    np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
                    np.sin(np.radians(self.pitch)),
                )
            )

        if not imgui.is_window_hovered():
            return

        if io.mouse_wheel != 0:
            self.move_speed += io.mouse_wheel * 0.05
            self.move_speed = np.max([self.move_speed, 0.01])

        # Handle mouse press/release
        for button in range(3):
            if imgui.is_mouse_clicked(button):
                self.mouse_pressed = True
                self.mouse_button = button
                if button == 0 and self.highlighted_node:  # Left click for selection
                    self.selected_node = self.highlighted_node

            if imgui.is_mouse_released(button) and self.mouse_button == button:
                self.mouse_pressed = False
                self.mouse_button = -1

        # Handle node hovering
        if not self.mouse_pressed:
            hovered = self.find_node_at((int(mouse_pos[0]), int(mouse_pos[1])))
            self.highlighted_node = hovered

        # Update last mouse position
        self.last_mouse_pos = mouse_pos

    def update_layout(self):
        """Update the graph layout"""
        pos = nx.spring_layout(
            self.graph,
            dim=3,
            pos={
                node_id: list(node.position)
                for node_id, node in self.id_node_map.items()
            },
            k=2.0,
            iterations=100,
            weight=None,
        )

        # Update node positions
        for node_id, position in pos.items():
            self.id_node_map[node_id].position = glm.vec3(position)
        self.update_buffers()

    def render_node_details(self):
        """Render node details window"""
        if self.selected_node and imgui.begin("Node Details"):
            imgui.text(f"ID: {self.selected_node.label}")

            if self.graph:
                node_data = self.graph.nodes[self.selected_node.label]
                imgui.text(f"Type: {node_data.get('type', 'default')}")

                degree = self.graph.degree[self.selected_node.label]
                imgui.text(f"Degree: {degree}")

                for key, value in node_data.items():
                    if key != "type":
                        imgui.text(f"{key}: {value}")
                        if value and imgui.is_item_hovered():
                            imgui.set_tooltip(str(value))

                imgui.separator()

                connections = self.graph[self.selected_node.label]
                if connections:
                    imgui.text("Connections:")
                    keys = next(iter(connections.values())).keys()

                    # Add extra columns for inference info
                    table_columns = ["Node"] + list(keys) + ["Type", "Inferred"]

                    if imgui.begin_table(
                        "Connections",
                        len(table_columns),
                        imgui.TableFlags_.borders
                        | imgui.TableFlags_.row_bg
                        | imgui.TableFlags_.resizable
                        | imgui.TableFlags_.hideable,
                    ):
                        for col_name in table_columns:
                            imgui.table_setup_column(col_name)
                        imgui.table_headers_row()

                        for neighbor, edge_data in connections.items():
                            imgui.table_next_row()

                            # Node name (clickable)
                            imgui.table_set_column_index(0)
                            if imgui.selectable(str(neighbor), True)[0]:
                                # Select neighbor node
                                self.selected_node = self.id_node_map[neighbor]
                                self.position = self.selected_node.position - self.front

                            # Original edge data
                            for idx, key in enumerate(keys):
                                imgui.table_set_column_index(idx + 1)
                                value = str(edge_data.get(key, ""))
                                imgui.text(value)
                                if value and imgui.is_item_hovered():
                                    imgui.set_tooltip(value)

                            # Relationship type
                            imgui.table_set_column_index(len(keys) + 1)
                            rel_type = edge_data.get("relationship_type", "")
                            if rel_type == "competitor":
                                imgui.text_colored((1.0, 0.27, 0.23, 1.0), "🥊 Competitor")
                            elif rel_type == "partnership":
                                imgui.text_colored((0.2, 0.78, 0.35, 1.0), "🤝 Partner")
                            elif rel_type == "supply_chain":
                                imgui.text_colored((0.35, 0.78, 0.98, 1.0), "📦 Supply Chain")
                            else:
                                imgui.text(rel_type if rel_type else "-")

                            # Inferred status
                            imgui.table_set_column_index(len(keys) + 2)
                            is_inferred = str(edge_data.get("inferred", "false")).lower() == "true"
                            if is_inferred:
                                confidence = edge_data.get("confidence", "")
                                conf_text = f"Yes ({confidence})" if confidence else "Yes"
                                imgui.text_colored((0.8, 0.8, 0.2, 1.0), conf_text)
                                if imgui.is_item_hovered():
                                    method = edge_data.get("inference_method", "")
                                    tooltip = f"Inferred relationship"
                                    if method:
                                        tooltip += f"\nMethod: {method}"
                                    if confidence:
                                        tooltip += f"\nConfidence: {confidence}"
                                    imgui.set_tooltip(tooltip)
                            else:
                                imgui.text("No")
                                if imgui.is_item_hovered():
                                    imgui.set_tooltip("Explicitly extracted")

                        imgui.end_table()

            imgui.end()

    def setup_render_context(self):
        """Initialize ModernGL context"""
        self.glctx = moderngl.create_context()
        self.glctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.glctx.clear_color = self.background_color

    def setup_shaders(self):
        """Setup vertex and fragment shaders for node and edge rendering"""
        # Node shader program
        self.node_prog = self.glctx.program(
            vertex_shader="""
                #version 330

                uniform mat4 mvp;
                uniform vec3 camera;
                uniform int selected_node;
                uniform int highlighted_node;
                uniform float scale;

                in vec3 in_position;
                in vec3 in_instance_position;
                in vec3 in_instance_color;
                in float in_instance_size;

                out vec3 frag_color;
                out vec3 frag_normal;
                out vec3 frag_view_dir;

                void main() {
                    vec3 pos = in_position * in_instance_size * scale + in_instance_position;
                    gl_Position = mvp * vec4(pos, 1.0);

                    frag_normal = normalize(in_position);
                    frag_view_dir = normalize(camera - pos);

                    if (selected_node == gl_InstanceID) {
                        frag_color = vec3(1.0, 0.5, 0.0);
                    }
                    else if (highlighted_node == gl_InstanceID) {
                        frag_color = vec3(1.0, 0.8, 0.2);
                    }
                    else {
                        frag_color = in_instance_color;
                    }
                }
            """,
            fragment_shader="""
                #version 330

                in vec3 frag_color;
                in vec3 frag_normal;
                in vec3 frag_view_dir;

                out vec4 outColor;

                void main() {
                    // Edge detection based on normal-view angle
                    float edge = 1.0 - abs(dot(frag_normal, frag_view_dir));

                    // Create sharp outline
                    float outline = smoothstep(0.8, 0.9, edge);

                    // Mix the sphere color with outline
                    vec3 final_color = mix(frag_color, vec3(0.0), outline);

                    outColor = vec4(final_color, 1.0);
                }
            """,
        )

        # Edge shader program with wide lines using geometry shader
        self.edge_prog = self.glctx.program(
            vertex_shader="""
                #version 330

                uniform mat4 mvp;

                in vec3 in_position;
                in vec3 in_color;

                out vec3 v_color;
                out vec4 v_position;

                void main() {
                    v_position = mvp * vec4(in_position, 1.0);
                    gl_Position = v_position;
                    v_color = in_color;
                }
            """,
            geometry_shader="""
                #version 330

                layout(lines) in;
                layout(triangle_strip, max_vertices = 4) out;

                uniform float edge_width;
                uniform vec2 viewport_size;

                in vec3 v_color[];
                in vec4 v_position[];
                out vec3 g_color;
                out float edge_coord;

                void main() {
                    // Get the two vertices of the line
                    vec4 p1 = v_position[0];
                    vec4 p2 = v_position[1];

                    // Perspective division
                    vec4 p1_ndc = p1 / p1.w;
                    vec4 p2_ndc = p2 / p2.w;

                    // Calculate line direction in screen space
                    vec2 dir = normalize((p2_ndc.xy - p1_ndc.xy) * viewport_size);
                    vec2 normal = vec2(-dir.y, dir.x);

                    // Calculate half width based on screen space
                    float half_width = edge_width * 0.5;
                    vec2 offset = normal * (half_width / viewport_size);

                    // Emit vertices with proper depth
                    gl_Position = vec4(p1_ndc.xy + offset, p1_ndc.z, 1.0);
                    gl_Position *= p1.w;  // Restore perspective
                    g_color = v_color[0];
                    edge_coord = 1.0;
                    EmitVertex();

                    gl_Position = vec4(p1_ndc.xy - offset, p1_ndc.z, 1.0);
                    gl_Position *= p1.w;
                    g_color = v_color[0];
                    edge_coord = -1.0;
                    EmitVertex();

                    gl_Position = vec4(p2_ndc.xy + offset, p2_ndc.z, 1.0);
                    gl_Position *= p2.w;
                    g_color = v_color[1];
                    edge_coord = 1.0;
                    EmitVertex();

                    gl_Position = vec4(p2_ndc.xy - offset, p2_ndc.z, 1.0);
                    gl_Position *= p2.w;
                    g_color = v_color[1];
                    edge_coord = -1.0;
                    EmitVertex();

                    EndPrimitive();
                }
            """,
            fragment_shader="""
                #version 330

                in vec3 g_color;
                in float edge_coord;

                out vec4 fragColor;

                void main() {
                    // Edge outline parameters
                    float outline_width = 0.2;  // Width of the outline relative to edge
                    float edge_softness = 0.1;  // Softness of the edge
                    float edge_dist = abs(edge_coord);

                    // Calculate outline
                    float outline_factor = smoothstep(1.0 - outline_width - edge_softness,
                                                    1.0 - outline_width,
                                                    edge_dist);

                    // Mix edge color with outline (black)
                    vec3 final_color = mix(g_color, vec3(0.0), outline_factor);

                    // Calculate alpha for anti-aliasing
                    float alpha = 1.0 - smoothstep(1.0 - edge_softness, 1.0, edge_dist);

                    fragColor = vec4(final_color, alpha);
                }
            """,
        )

        # Id framebuffer shader program
        self.node_id_prog = self.glctx.program(
            vertex_shader="""
                #version 330

                uniform mat4 mvp;
                uniform float scale;

                in vec3 in_position;
                in vec3 in_instance_position;
                in float in_instance_size;

                out vec3 frag_color;

                vec3 int_to_rgb(int value) {
                    float R = float((value >> 16) & 0xFF);
                    float G = float((value >> 8) & 0xFF);
                    float B = float(value & 0xFF);
                    // normalize to [0, 1]
                    return vec3(R / 255.0, G / 255.0, B / 255.0);
                }

                void main() {
                    vec3 pos = in_position * in_instance_size * scale + in_instance_position;
                    gl_Position = mvp * vec4(pos, 1.0);
                    frag_color = int_to_rgb(gl_InstanceID);
                }
                """,
            fragment_shader="""
                    #version 330
                    in vec3 frag_color;
                    out vec4 outColor;
                    void main() {
                        outColor = vec4(frag_color, 1.0);
                    }
                """,
        )

    def setup_buffers(self):
        """Setup vertex buffers for nodes and edges"""
        # We'll create these when loading the graph
        self.node_vbo = None
        self.node_color_vbo = None
        self.node_size_vbo = None
        self.edge_vbo = None
        self.edge_color_vbo = None
        self.inferred_edge_vbo = None
        self.inferred_edge_color_vbo = None
        self.node_vao = None
        self.edge_vao = None
        self.inferred_edge_vao = None
        self.node_id_vao = None
        self.sphere_pos_vbo = None
        self.sphere_index_buffer = None

    def load_file(self, filepath: str):
        """Load a GraphML file with error handling"""
        try:
            # Clear existing data
            self.id_node_map.clear()
            self.nodes.clear()
            self.selected_node = None
            self.highlighted_node = None
            self.setup_buffers()

            # Load new graph
            self.graph = nx.read_graphml(filepath)
            self.calculate_layout()
            self.update_buffers()
            self.show_load_error = False
            self.error_message = ""
        except Exception as _:
            self.show_load_error = True
            self.error_message = traceback.format_exc()
            print(self.error_message)

    def calculate_layout(self):
        """Calculate 3D layout for the graph"""
        if not self.graph:
            return

        # Detect communities for coloring
        self.communities = community.best_partition(self.graph)
        num_communities = len(set(self.communities.values()))
        self.community_colors = generate_colors(num_communities)

        # Calculate layout based on selected type
        if self.layout_type == "Spring":
            pos = nx.spring_layout(
                self.graph, dim=3, k=2.0, iterations=100, weight=None
            )
        elif self.layout_type == "Circular":
            pos_2d = nx.circular_layout(self.graph)
            pos = {node: np.array((x, 0.0, y)) for node, (x, y) in pos_2d.items()}
        elif self.layout_type == "Shell":
            # Group nodes by community for shell layout
            comm_lists = [[] for _ in range(num_communities)]
            for node, comm in self.communities.items():
                comm_lists[comm].append(node)
            pos_2d = nx.shell_layout(self.graph, comm_lists)
            pos = {node: np.array((x, 0.0, y)) for node, (x, y) in pos_2d.items()}
        else:  # Random
            pos = {node: np.random.rand(3) * 2 - 1 for node in self.graph.nodes()}

        # Scale positions
        positions = np.array(list(pos.values()))
        if len(positions) > 0:
            scale = 10.0 / max(1.0, np.max(np.abs(positions)))
            pos = {node: coords * scale for node, coords in pos.items()}

        # Calculate degree-based sizes
        degrees = dict(self.graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        min_degree = min(degrees.values()) if degrees else 1

        idx = 0
        # Create nodes with community colors
        for node_id in self.graph.nodes():
            position = glm.vec3(pos[node_id])
            color = self.get_node_color(node_id)

            # Normalize sizes between 0.5 and 2.0
            size = 1.0
            if max_degree != min_degree:
                # Normalize and scale size
                normalized = (degrees[node_id] - min_degree) / (max_degree - min_degree)
                size = 0.5 + normalized * 1.5

            if node_id in self.id_node_map:
                node = self.id_node_map[node_id]
                node.position = position
                node.base_color = color
                node.color = color
                node.size = size
            else:
                node = Node3D(position, color, str(node_id), size, idx)
                self.id_node_map[node_id] = node
                self.nodes.append(node)
                idx += 1

        self.update_buffers()

    def get_node_color(self, node_id: str) -> glm.vec3:
        """Get RGBA color based on community"""
        if self.communities and node_id in self.communities:
            comm_id = self.communities[node_id]
            color = self.community_colors[comm_id]
            return color
        return glm.vec3(0.5, 0.5, 0.5)

    def update_buffers(self):
        """Update vertex buffers with current node and edge data using batch rendering"""
        if not self.graph:
            return

        # Update node buffers
        node_positions = []
        node_colors = []
        node_sizes = []

        for node in self.nodes:
            node_positions.append(node.position)
            node_colors.append(node.color)  # Only use RGB components
            node_sizes.append(node.size)

        if node_positions:
            node_positions = np.array(node_positions, dtype=np.float32)
            node_colors = np.array(node_colors, dtype=np.float32)
            node_sizes = np.array(node_sizes, dtype=np.float32)

            self.node_vbo = self.glctx.buffer(node_positions.tobytes())
            self.node_color_vbo = self.glctx.buffer(node_colors.tobytes())
            self.node_size_vbo = self.glctx.buffer(node_sizes.tobytes())
            self.sphere_pos_vbo = self.glctx.buffer(self.sphere_data[0].tobytes())
            self.sphere_index_buffer = self.glctx.buffer(self.sphere_data[1].tobytes())

            self.node_vao = self.glctx.vertex_array(
                self.node_prog,
                [
                    (self.sphere_pos_vbo, "3f", "in_position"),
                    (self.node_vbo, "3f /i", "in_instance_position"),
                    (self.node_color_vbo, "3f /i", "in_instance_color"),
                    (self.node_size_vbo, "f /i", "in_instance_size"),
                ],
                index_buffer=self.sphere_index_buffer,
                index_element_size=4,
            )
            self.node_vao.instances = len(self.nodes)

            self.node_id_vao = self.glctx.vertex_array(
                self.node_id_prog,
                [
                    (self.sphere_pos_vbo, "3f", "in_position"),
                    (self.node_vbo, "3f /i", "in_instance_position"),
                    (self.node_size_vbo, "f /i", "in_instance_size"),
                ],
                index_buffer=self.sphere_index_buffer,
                index_element_size=4,
            )
            self.node_id_vao.instances = len(self.nodes)

        # Update edge buffers
        edge_positions = []
        edge_colors = []

        # Separate inferred edges for different rendering
        inferred_edge_positions = []
        inferred_edge_colors = []

        for edge in self.graph.edges():
            start_node = self.id_node_map[edge[0]]
            end_node = self.id_node_map[edge[1]]

            # Check if edge is inferred
            edge_data = self.graph.get_edge_data(edge[0], edge[1])
            is_inferred = edge_data and str(edge_data.get("inferred", "false")).lower() == "true"

            if is_inferred:
                # Inferred edges: use faded colors
                relationship_type = edge_data.get("relationship_type", "")

                # Color based on relationship type
                if relationship_type == "competitor":
                    edge_color = glm.vec3(1.0, 0.27, 0.23)  # Red
                elif relationship_type == "partnership":
                    edge_color = glm.vec3(0.2, 0.78, 0.35)  # Green
                elif relationship_type == "supply_chain":
                    edge_color = glm.vec3(0.35, 0.78, 0.98)  # Blue
                else:
                    edge_color = glm.vec3(0.6, 0.6, 0.6)  # Gray

                # Make it faded (lower intensity)
                edge_color *= 0.4

                inferred_edge_positions.append(start_node.position)
                inferred_edge_colors.append(edge_color)
                inferred_edge_positions.append(end_node.position)
                inferred_edge_colors.append(edge_color)
            else:
                # Explicit edges: normal rendering
                edge_positions.append(start_node.position)
                edge_colors.append(start_node.color)
                edge_positions.append(end_node.position)
                edge_colors.append(end_node.color)

        # Create VAO for explicit edges
        if edge_positions:
            edge_positions = np.array(edge_positions, dtype=np.float32)
            edge_colors = np.array(edge_colors, dtype=np.float32)

            self.edge_vbo = self.glctx.buffer(edge_positions.tobytes())
            self.edge_color_vbo = self.glctx.buffer(edge_colors.tobytes())

            self.edge_vao = self.glctx.vertex_array(
                self.edge_prog,
                [
                    (self.edge_vbo, "3f", "in_position"),
                    (self.edge_color_vbo, "3f", "in_color"),
                ],
            )
        else:
            self.edge_vao = None

        # Create VAO for inferred edges
        if inferred_edge_positions:
            inferred_edge_positions = np.array(inferred_edge_positions, dtype=np.float32)
            inferred_edge_colors = np.array(inferred_edge_colors, dtype=np.float32)

            self.inferred_edge_vbo = self.glctx.buffer(inferred_edge_positions.tobytes())
            self.inferred_edge_color_vbo = self.glctx.buffer(inferred_edge_colors.tobytes())

            self.inferred_edge_vao = self.glctx.vertex_array(
                self.edge_prog,
                [
                    (self.inferred_edge_vbo, "3f", "in_position"),
                    (self.inferred_edge_color_vbo, "3f", "in_color"),
                ],
            )
        else:
            self.inferred_edge_vao = None

    def update_view_proj_matrix(self):
        """Update view matrix based on camera parameters"""
        self.view_matrix = glm.lookAt(
            self.position, self.position + self.front, self.up
        )

        aspect_ratio = self.window_width / self.window_height
        self.proj_matrix = glm.perspective(
            glm.radians(60.0),  # FOV
            aspect_ratio,  # Aspect ratio
            0.001,  # Near plane
            1000.0,  # Far plane
        )

    def find_node_at(self, screen_pos: Tuple[int, int]) -> Optional[Node3D]:
        """Find the node at a specific screen position"""
        if (
            self.node_id_texture_np is None
            or self.node_id_texture_np.shape[1] != self.window_width
            or self.node_id_texture_np.shape[0] != self.window_height
            or screen_pos[0] < 0
            or screen_pos[1] < 0
            or screen_pos[0] >= self.window_width
            or screen_pos[1] >= self.window_height
        ):
            return None

        x = screen_pos[0]
        y = self.window_height - screen_pos[1] - 1
        pixel = self.node_id_texture_np[y, x]

        if pixel[3] == 0:
            return None

        R = int(round(pixel[0] * 255))
        G = int(round(pixel[1] * 255))
        B = int(round(pixel[2] * 255))
        index = (R << 16) | (G << 8) | B

        if index > len(self.nodes):
            return None
        return self.nodes[index]

    def is_node_visible_at(self, screen_pos: Tuple[int, int], node_idx: int) -> bool:
        """Check if a node exists at a specific screen position"""
        node = self.find_node_at(screen_pos)
        return node is not None and node.idx == node_idx

    def render_settings(self):
        """Render settings window"""
        if imgui.begin("Graph Settings"):
            # Layout type combo
            changed, value = imgui.combo(
                "Layout",
                self.available_layouts.index(self.layout_type),
                self.available_layouts,
            )
            if changed:
                self.layout_type = self.available_layouts[value]
                self.calculate_layout()  # Recalculate layout when changed

            # Node size slider
            changed, value = imgui.slider_float("Node Scale", self.node_scale, 0.01, 10)
            if changed:
                self.node_scale = value

            # Edge width slider
            changed, value = imgui.slider_float("Edge Width", self.edge_width, 0, 20)
            if changed:
                self.edge_width = value

            # Show labels checkbox
            changed, value = imgui.checkbox("Show Labels", self.show_labels)

            if changed:
                self.show_labels = value

            if self.show_labels:
                # Label size slider
                changed, value = imgui.slider_float(
                    "Label Size", self.label_size, 0.5, 10.0
                )
                if changed:
                    self.label_size = value

                # Label color picker
                changed, value = imgui.color_edit4(
                    "Label Color",
                    self.label_color,
                    imgui.ColorEditFlags_.picker_hue_wheel,
                )
                if changed:
                    self.label_color = (value[0], value[1], value[2], value[3])

                # Label culling distance slider
                changed, value = imgui.slider_float(
                    "Label Culling Distance", self.label_culling_distance, 0.1, 100.0
                )
                if changed:
                    self.label_culling_distance = value

            # Background color picker
            changed, value = imgui.color_edit4(
                "Background Color",
                self.background_color,
                imgui.ColorEditFlags_.picker_hue_wheel,
            )
            if changed:
                self.background_color = (value[0], value[1], value[2], value[3])

            imgui.end()

    def save_node_id_texture_to_png(self, filename):
        # Convert to a PIL Image and save as PNG
        from PIL import Image

        scaled_array = self.node_id_texture_np * 255
        img = Image.fromarray(
            scaled_array.astype(np.uint8),
            "RGBA",
        )
        img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
        img.save(filename)

    def render_id_map(self, mvp: glm.mat4):
        """Render an offscreen id map where each node is drawn with a unique id color."""
        # Lazy initialization of id framebuffer
        if self.node_id_texture is not None:
            if (
                self.node_id_texture.width != self.window_width
                or self.node_id_texture.height != self.window_height
            ):
                self.node_id_fbo = None
                self.node_id_texture = None
                self.node_id_texture_np = None
                self.node_id_depth = None

        if self.node_id_texture is None:
            self.node_id_texture = self.glctx.texture(
                (self.window_width, self.window_height), components=4, dtype="f4"
            )
            self.node_id_depth = self.glctx.depth_renderbuffer(
                size=(self.window_width, self.window_height)
            )
            self.node_id_fbo = self.glctx.framebuffer(
                color_attachments=[self.node_id_texture],
                depth_attachment=self.node_id_depth,
            )
            self.node_id_texture_np = np.zeros(
                (self.window_height, self.window_width, 4), dtype=np.float32
            )

        # Bind the offscreen framebuffer
        self.node_id_fbo.use()
        self.glctx.clear(0, 0, 0, 0)

        # Render nodes
        if self.node_id_vao:
            self.node_id_prog["mvp"].write(mvp.to_bytes())
            self.node_id_prog["scale"].write(np.float32(self.node_scale).tobytes())
            self.node_id_vao.render(moderngl.TRIANGLES)

        # Revert to default framebuffer
        self.glctx.screen.use()
        self.node_id_texture.read_into(self.node_id_texture_np.data)

    def render(self):
        """Render the graph"""
        # Clear screen
        self.glctx.clear(*self.background_color, depth=1)

        if not self.graph:
            return

        # Enable blending for transparency
        self.glctx.enable(moderngl.BLEND)
        self.glctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Update view and projection matrices
        self.update_view_proj_matrix()
        mvp = self.proj_matrix * self.view_matrix

        # Render edges first (under nodes)
        # First render explicit edges (solid, thicker)
        if self.edge_vao:
            self.edge_prog["mvp"].write(mvp.to_bytes())
            self.edge_prog["edge_width"].value = (
                float(self.edge_width) * 2.0
            )  # Thicker for explicit edges
            self.edge_prog["viewport_size"].value = (
                float(self.window_width),
                float(self.window_height),
            )
            self.edge_vao.render(moderngl.LINES)

        # Then render inferred edges (faded, thinner)
        if self.inferred_edge_vao:
            self.edge_prog["mvp"].write(mvp.to_bytes())
            self.edge_prog["edge_width"].value = (
                float(self.edge_width) * 0.8
            )  # Thinner for inferred edges
            self.edge_prog["viewport_size"].value = (
                float(self.window_width),
                float(self.window_height),
            )
            self.inferred_edge_vao.render(moderngl.LINES)

        # Render nodes
        if self.node_vao:
            self.node_prog["mvp"].write(mvp.to_bytes())
            self.node_prog["camera"].write(self.position.to_bytes())
            self.node_prog["selected_node"].write(
                np.int32(self.selected_node.idx).tobytes()
                if self.selected_node
                else np.int32(-1).tobytes()
            )
            self.node_prog["highlighted_node"].write(
                np.int32(self.highlighted_node.idx).tobytes()
                if self.highlighted_node
                else np.int32(-1).tobytes()
            )
            self.node_prog["scale"].write(np.float32(self.node_scale).tobytes())
            self.node_vao.render(moderngl.TRIANGLES)

        self.glctx.disable(moderngl.BLEND)

        # Render id map
        self.render_id_map(mvp)

    def render_labels(self):
        # Render labels if enabled
        if self.show_labels and self.nodes:
            # Save current font scale
            original_scale = imgui.get_font_size()

            self.update_view_proj_matrix()
            mvp = self.proj_matrix * self.view_matrix

            for node in self.nodes:
                # Project node position to screen space
                pos = mvp * glm.vec4(
                    node.position[0], node.position[1], node.position[2], 1.0
                )

                # Check if node is behind camera
                if pos.w > 0 and pos.w < self.label_culling_distance:
                    screen_x = (pos.x / pos.w + 1) * self.window_width / 2
                    screen_y = (-pos.y / pos.w + 1) * self.window_height / 2

                    if self.is_node_visible_at(
                        (int(screen_x), int(screen_y)), node.idx
                    ):
                        # Set font scale
                        imgui.set_window_font_scale(float(self.label_size) * node.size)

                        # Calculate label size
                        label_size = imgui.calc_text_size(node.label)

                        # Adjust position to center the label
                        screen_x -= label_size.x / 2
                        screen_y -= label_size.y / 2

                        # Set text color with calculated alpha
                        imgui.push_style_color(imgui.Col_.text, self.label_color)

                        # Draw label using ImGui
                        imgui.set_cursor_pos((screen_x, screen_y))
                        imgui.text(node.label)

                        # Restore text color
                        imgui.pop_style_color()

            # Restore original font scale
            imgui.set_window_font_scale(original_scale)

    def reset_view(self):
        """Reset camera view to default"""
        self.position = glm.vec3(0.0, -10.0, 0.0)
        self.front = glm.vec3(0.0, 1.0, 0.0)
        self.yaw = 90.0
        self.pitch = 0.0


def generate_colors(n: int) -> List[glm.vec3]:
    """Generate n distinct colors using HSV color space"""
    colors = []
    for i in range(n):
        # Use golden ratio to generate well-distributed hues
        hue = (i * 0.618033988749895) % 1.0
        # Fixed saturation and value for vibrant colors
        saturation = 0.8
        value = 0.95
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Add alpha channel
        colors.append(glm.vec3(rgb))
    return colors


def show_file_dialog() -> Optional[str]:
    """Show a file dialog for selecting GraphML files"""
    file_path = filedialog.askopenfilename(
        title="Select GraphML File",
        filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")],
    )
    return file_path if file_path else None


def create_sphere(sectors: int = 32, rings: int = 16) -> Tuple:
    """
    Creates a sphere.
    """
    R = 1.0 / (rings - 1)
    S = 1.0 / (sectors - 1)

    # Use those names as normals and uvs are part of the API
    vertices_l = [0.0] * (rings * sectors * 3)
    # normals_l = [0.0] * (rings * sectors * 3)
    uvs_l = [0.0] * (rings * sectors * 2)

    v, n, t = 0, 0, 0
    for r in range(rings):
        for s in range(sectors):
            y = np.sin(-np.pi / 2 + np.pi * r * R)
            x = np.cos(2 * np.pi * s * S) * np.sin(np.pi * r * R)
            z = np.sin(2 * np.pi * s * S) * np.sin(np.pi * r * R)

            uvs_l[t] = s * S
            uvs_l[t + 1] = r * R

            vertices_l[v] = x
            vertices_l[v + 1] = y
            vertices_l[v + 2] = z

            t += 2
            v += 3
            n += 3

    indices = [0] * rings * sectors * 6
    i = 0
    for r in range(rings - 1):
        for s in range(sectors - 1):
            indices[i] = r * sectors + s
            indices[i + 1] = (r + 1) * sectors + (s + 1)
            indices[i + 2] = r * sectors + (s + 1)

            indices[i + 3] = r * sectors + s
            indices[i + 4] = (r + 1) * sectors + s
            indices[i + 5] = (r + 1) * sectors + (s + 1)
            i += 6

    vbo_vertices = np.array(vertices_l, dtype=np.float32)
    vbo_elements = np.array(indices, dtype=np.uint32)

    return (vbo_vertices, vbo_elements)


def draw_text_with_bg(
    text: str,
    text_pos: imgui.ImVec2Like,
    text_size: imgui.ImVec2Like,
    bg_color: int,
):
    imgui.get_window_draw_list().add_rect_filled(
        (text_pos[0] - 5, text_pos[1] - 5),
        (text_pos[0] + text_size[0] + 5, text_pos[1] + text_size[1] + 5),
        bg_color,
        3.0,
    )
    imgui.set_cursor_pos(text_pos)
    imgui.text(text)


def main():
    """Main application entry point"""
    viewer = GraphViewer()

    show_fps = True
    text_bg_color = imgui.IM_COL32(0, 0, 0, 100)

    def gui():
        if not viewer.initialized:
            viewer.setup()
            # # Change the theme
            # tweaked_theme = hello_imgui.get_runner_params().imgui_window_params.tweaked_theme
            # tweaked_theme.theme = hello_imgui.ImGuiTheme_.darcula_darker
            # hello_imgui.apply_tweaked_theme(tweaked_theme)

        viewer.window_width = int(imgui.get_window_width())
        viewer.window_height = int(imgui.get_window_height())

        # Handle keyboard and mouse input
        viewer.handle_keyboard_input()
        viewer.handle_mouse_interaction()

        style = imgui.get_style()
        window_bg_color = style.color_(imgui.Col_.window_bg.value)

        window_bg_color.w = 0.8
        style.set_color_(imgui.Col_.window_bg.value, window_bg_color)

        # Main control window
        imgui.begin("Graph Controls")

        if imgui.button("Load GraphML"):
            filepath = show_file_dialog()
            if filepath:
                viewer.load_file(filepath)

        # Show error message if loading failed
        if viewer.show_load_error:
            imgui.push_style_color(imgui.Col_.text, (1.0, 0.0, 0.0, 1.0))
            imgui.text(f"Error loading file: {viewer.error_message}")
            imgui.pop_style_color()

        imgui.separator()

        # Camera controls help
        imgui.text("Camera Controls:")
        imgui.bullet_text("Hold Right Mouse - Look around")
        imgui.bullet_text("W/S - Move forward/backward")
        imgui.bullet_text("A/D - Move left/right")
        imgui.bullet_text("Q/E - Move up/down")
        imgui.bullet_text("Left Mouse - Select node")
        imgui.bullet_text("Wheel - Change the movement speed")

        imgui.separator()

        # Camera settings
        _, viewer.move_speed = imgui.slider_float(
            "Movement Speed", viewer.move_speed, 0.01, 2.0
        )
        _, viewer.mouse_sensitivity = imgui.slider_float(
            "Mouse Sensitivity", viewer.mouse_sensitivity, 0.01, 0.5
        )

        imgui.separator()

        imgui.begin_horizontal("buttons")

        if imgui.button("Reset Camera"):
            viewer.reset_view()

        if imgui.button("Update Layout") and viewer.graph:
            viewer.update_layout()

        # if imgui.button("Save Node ID Texture"):
        #     viewer.save_node_id_texture_to_png("node_id_texture.png")

        imgui.end_horizontal()

        imgui.end()

        # Render node details window if a node is selected
        viewer.render_node_details()

        # Render graph settings window
        viewer.render_settings()

        # Render FPS
        if show_fps:
            imgui.set_window_font_scale(1)
            fps_text = f"FPS: {hello_imgui.frame_rate():.1f}"
            text_size = imgui.calc_text_size(fps_text)
            cursor_pos = (10, viewer.window_height - text_size.y - 10)
            draw_text_with_bg(fps_text, cursor_pos, text_size, text_bg_color)

        # Render highlighted node ID
        if viewer.highlighted_node:
            imgui.set_window_font_scale(1)
            node_text = f"Node ID: {viewer.highlighted_node.label}"
            text_size = imgui.calc_text_size(node_text)
            cursor_pos = (
                viewer.window_width - text_size.x - 10,
                viewer.window_height - text_size.y - 10,
            )
            draw_text_with_bg(node_text, cursor_pos, text_size, text_bg_color)

        window_bg_color.w = 0
        style.set_color_(imgui.Col_.window_bg.value, window_bg_color)

        # Render labels
        viewer.render_labels()

    def custom_background():
        if viewer.initialized:
            viewer.render()

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_geometry.size = (
        viewer.window_width,
        viewer.window_height,
    )
    runner_params.app_window_params.window_title = "3D GraphML Viewer"
    runner_params.callbacks.show_gui = gui
    runner_params.callbacks.custom_background = custom_background

    def load_font():
        # You will need to provide it yourself, or use another font.
        font_filename = CUSTOM_FONT

        io = imgui.get_io()
        io.fonts.tex_desired_width = 4096  # Larger texture for better CJK font quality
        font_size_pixels = 14
        asset_dir = os.path.join(os.path.dirname(__file__), "assets")

        # Try to load custom font
        if not os.path.isfile(font_filename):
            font_filename = os.path.join(asset_dir, font_filename)
        if os.path.isfile(font_filename):
            custom_font = io.fonts.add_font_from_file_ttf(
                filename=font_filename,
                size_pixels=font_size_pixels,
                glyph_ranges_as_int_list=io.fonts.get_glyph_ranges_chinese_full(),
            )
            io.font_default = custom_font
            return

        # Load default fonts
        io.fonts.add_font_from_file_ttf(
            filename=os.path.join(asset_dir, DEFAULT_FONT_ENG),
            size_pixels=font_size_pixels,
        )

        font_config = imgui.ImFontConfig()
        font_config.merge_mode = True

        io.font_default = io.fonts.add_font_from_file_ttf(
            filename=os.path.join(asset_dir, DEFAULT_FONT_CHI),
            size_pixels=font_size_pixels,
            font_cfg=font_config,
            glyph_ranges_as_int_list=io.fonts.get_glyph_ranges_chinese_full(),
        )

    runner_params.callbacks.load_additional_fonts = load_font

    tk_root = tk.Tk()
    tk_root.withdraw()  # Hide the main window

    immapp.run(runner_params)

    tk_root.destroy()  # Destroy the main window


if __name__ == "__main__":
    main()
