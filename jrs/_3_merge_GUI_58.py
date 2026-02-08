import tkinter as tk
from tkinter import ttk, messagebox
import asyncio
import requests
import os
import json
import platform
import urllib.parse
import subprocess
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from llama_index.embeddings.openai import OpenAIEmbedding

# Configuration
WORKING_DIR = "/home/js/LightRAG/jrs/work/mod_linx_text/mod_linx_work_dir"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 3072))
API_KEY = os.getenv("EMBEDDING_BINDING_API_key")
MAX_TOKEN_SIZE = int(os.getenv("MAX_TOKEN_SIZE", 8192))
LIGHTRAG_SERVER_URL = "http://localhost:9621"


async def initialize_rag():
    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL, api_key=API_KEY, dimensions=EMBEDDING_DIM
    )

    async def async_embedding_func(texts):
        return embed_model.get_text_embedding_batch(texts)

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=MAX_TOKEN_SIZE,
        func=async_embedding_func,
    )
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=embedding_func,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def fetch_entities():
    try:
        response = requests.get(f"{LIGHTRAG_SERVER_URL}/graph/label/list")
        response.raise_for_status()
        entities = sorted(response.json(), key=lambda x: x.lower())
        print(f"Number of entities fetched: {len(entities)}")  # Logging
        return entities
    except requests.exceptions.ConnectionError:
        messagebox.showerror(
            "Connection Error", "Could not connect to LightRAG server. Is it running?"
        )
        return []
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch entities from server: {e}")
        return []


def fetch_entity_details(label):
    try:
        encoded_label = urllib.parse.quote_plus(label)
        response = requests.get(
            f"{LIGHTRAG_SERVER_URL}/graphs?label={encoded_label}&max_depth=1&max_nodes=20000"
        )
        response.raise_for_status()
        data = response.json()

        main_entity_desc = "No description found."
        main_entity_type = ""
        main_entity_source_id = ""
        main_entity_file_path = ""

        related_nodes_info = []

        for node in data.get("nodes", []):
            node_id = node.get("id")
            node_properties = node.get("properties", {})
            node_desc = node_properties.get("description", "No description found.")
            node_type = node_properties.get("entity_type", "")

            if node_id == label:
                main_entity_desc = node_desc
                main_entity_type = node_type
                main_entity_source_id = node_properties.get("source_id", "")
                main_entity_file_path = node_properties.get("file_path", "")
            else:
                related_nodes_info.append(
                    {"id": node_id, "description": node_desc, "type": node_type}
                )

        edges_info = []
        for edge in data.get("edges", []):
            edges_info.append(
                {
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "description": edge.get("properties", {}).get("description", ""),
                    "keywords": edge.get("properties", {}).get("keywords", ""),
                    "weight": edge.get("properties", {}).get("weight", 1.0),
                }
            )

        return {
            "desc": main_entity_desc,
            "type": main_entity_type,
            "srcid": main_entity_source_id,
            "fpath": main_entity_file_path,
            "related_nodes": related_nodes_info,
            "edges": edges_info,
        }
    except requests.exceptions.ConnectionError:
        print(
            "Connection Error: Could not connect to LightRAG server to fetch entity details."
        )
        return {
            "desc": "Error: Server not reachable.",
            "type": "",
            "srcid": "",
            "fpath": "",
            "related_nodes": [],
            "edges": [],
        }
    except Exception as e:
        print(f"Error fetching entity details for {label}: {e}")
        return {
            "desc": f"Error: {e}",
            "type": "",
            "srcid": "",
            "fpath": "",
            "related_nodes": [],
            "edges": [],
        }


def trigger_server_refresh():
    try:
        print("Attempting to trigger LightRAG server data refresh...")
        response = requests.post(f"{LIGHTRAG_SERVER_URL}/graph/refresh-data")
        response.raise_for_status()
        print("LightRAG server data refresh triggered successfully.")
        return True
    except requests.exceptions.ConnectionError:
        messagebox.showwarning(
            "Server Not Running",
            "Could not connect to LightRAG server to trigger refresh. Please ensure the server is running.",
        )
        return False
    except requests.exceptions.HTTPError as e:
        messagebox.showwarning(
            "API Error",
            f"Failed to trigger LightRAG server data refresh: {e.response.status_code} - {e.response.text}",
        )
        return False
    except Exception as e:
        messagebox.showwarning(
            "Error",
            f"An unexpected error occurred while triggering LightRAG server data refresh: {e}",
        )
        return False


def update_entity_description_api(entity_label, new_description):
    try:
        url = f"{LIGHTRAG_SERVER_URL}/graph/entity/edit"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        payload = {
            "entity_name": entity_label,
            "updated_data": {"description": new_description},
            "allow_rename": False,
        }

        print(f"Sending update request for {entity_label} with new description...")
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Successfully updated description for {entity_label}.")
        return True
    except requests.exceptions.ConnectionError:
        messagebox.showerror(
            "Connection Error",
            "Could not connect to LightRAG server to update description. Is it running?",
        )
        return False
    except requests.exceptions.HTTPError as e:
        messagebox.showerror(
            "API Error",
            f"Failed to update description for {entity_label}: {e.response.status_code} - {e.response.text}",
        )
        return False
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"An unexpected error occurred while updating description for {entity_label}: {e}",
        )
        return False


def update_relationship_api(
    source_id, target_id, new_description, new_keywords, weight=1.0
):
    try:
        url = f"{LIGHTRAG_SERVER_URL}/graph/relation/edit"
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        payload = {
            "source_id": source_id,
            "target_id": target_id,
            "updated_data": {
                "description": new_description,
                "keywords": new_keywords,
                "weight": weight,
            },
        }

        print(
            f"Sending update request for relationship from {source_id} to {target_id}..."
        )
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Successfully updated relationship from {source_id} to {target_id}.")
        return True
    except requests.exceptions.ConnectionError:
        messagebox.showerror(
            "Connection Error",
            "Could not connect to LightRAG server to update relationship. Is it running?",
        )
        return False
    except requests.exceptions.HTTPError as e:
        messagebox.showerror(
            "API Error",
            f"Failed to update relationship from {source_id} to {target_id}: {e.response.status_code} - {e.response.text}",
        )
        return False
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"An unexpected error occurred while updating relationship from {source_id} to {target_id}: {e}",
        )
        return False


def create_relationship_api(
    source_id, target_id, description, keywords, weight, source_file_id
):
    try:
        rag = asyncio.run(initialize_rag())
        relationship_data = {
            "relationships": [
                {
                    "src_id": source_id,
                    "tgt_id": target_id,
                    "description": description,
                    "keywords": keywords,
                    "weight": float(weight),
                    "source_id": source_file_id,
                }
            ]
        }
        asyncio.run(
            rag.ainsert_custom_kg(
                relationship_data, full_doc_id=source_file_id, file_path=source_file_id
            )
        )
        print(f"Successfully created relationship from {source_id} to {target_id}.")
        return True
    except requests.exceptions.ConnectionError:
        messagebox.showerror(
            "Connection Error",
            "Could not connect to LightRAG server to create relationship. Is it running?",
        )
        return False
    except requests.exceptions.HTTPError as e:
        messagebox.showerror(
            "API Error",
            f"Failed to create relationship from {source_id} to {target_id}: {e.response.status_code} - {e.response.text}",
        )
        return False
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"An unexpected error occurred while creating relationship from {source_id} to {target_id}: {e}",
        )
        return False
    finally:
        if "rag" in locals():
            asyncio.run(rag.finalize_storages())


class MergeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LightRAG Entity Merger")
        self.entity_list = fetch_entities()
        self.filtered_entity_list = self.entity_list.copy()
        self.current_page = 0  # Initialize page counter
        self.page_size = 35  # Number of entities per page
        self.all_check_vars = {entity: tk.BooleanVar() for entity in self.entity_list}
        self.description_windows = {}
        self.description_frames = {}
        self.entity_data = {}
        self.config_file = "merge_gui_config.json"

        self.load_window_config()
        self.setup_main_window()

        self.check_vars = {}
        self.first_entity_var = tk.StringVar()

        self.create_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def copy_selected_text(self, widget):
        try:
            if isinstance(widget, tk.Text):
                # For Text widgets
                selected_text = widget.get("sel.first", "sel.last")
            elif isinstance(widget, ttk.Combobox):
                # For Combobox widgets
                if widget.selection_present():  # Check if there's a selection
                    selected_text = widget.selection_get()
                else:
                    selected_text = (
                        widget.get()
                    )  # Fallback to entire text if no selection
            else:
                raise ValueError("Unsupported widget type")

            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.root.update()
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=selected_text.encode(),
                check=True,
            )
            print(f"Copied to clipboard: {selected_text}")
        except tk.TclError:
            print("No text selected to copy.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to copy to clipboard via xclip: {e}")
        except ValueError as e:
            print(f"Error: {e}")

    def load_window_config(self):
        self.window_config = {
            "geometry": "1200x800",
            "state": "normal",
            "paned_position": 300,
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    saved_config = json.load(f)
                    self.window_config.update(saved_config)
            except Exception:
                pass

    def set_initial_paned_position(self):
        try:
            self.root.update_idletasks()
            window_width = self.paned_window.winfo_width()

            if window_width > 100:
                position = int(window_width * 0.25)
            else:
                position = 300

            self.paned_window.sashpos(0, position)
        except Exception:
            self.paned_window.sashpos(0, 300)

    def save_window_config(self):
        try:
            geometry = self.root.geometry()
            state = self.root.state()
            paned_position = self.paned_window.sash_coord(0)[0]

            config = {
                "geometry": geometry,
                "state": state,
                "paned_position": paned_position,
            }

            with open(self.config_file, "w") as f:
                json.dump(config, f)
        except Exception:
            pass

    def setup_main_window(self):
        self.root.geometry(self.window_config["geometry"])

        if platform.system() == "Windows":
            try:
                self.root.state("zoomed")
            except tk.TclError:
                self.root.geometry(
                    f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0"
                )
        elif platform.system() == "Linux":
            self.root.geometry(
                f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0"
            )
        else:
            self.root.geometry(
                f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0"
            )

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def create_ui(self):
        self.paned_window = ttk.PanedWindow(self.root, orient="horizontal")
        self.paned_window.grid(row=0, column=0, sticky="nsew")

        self.left_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_panel, weight=1)

        self.right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_panel, weight=3)

        self.root.after(10, self.set_initial_paned_position)

        self.create_right_panel()
        self.create_left_panel()

    def create_left_panel(self):
        self.left_panel.grid_rowconfigure(2, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        top_controls_frame = ttk.Frame(self.left_panel)
        top_controls_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        top_controls_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(top_controls_frame, text="Filter:").grid(
            row=0, column=0, padx=(0, 5), sticky="w"
        )
        self.filter_var = tk.StringVar()
        self.filter_entry = ttk.Entry(top_controls_frame, textvariable=self.filter_var)
        self.filter_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.filter_var.trace("w", self.on_filter_change)

        # Pagination controls
        pagination_frame = ttk.Frame(self.left_panel)
        pagination_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        pagination_frame.grid_columnconfigure(2, weight=1)

        ttk.Button(
            pagination_frame, text="First", command=lambda: self.change_page_to(0)
        ).grid(row=0, column=0, padx=2)
        ttk.Button(
            pagination_frame, text="Previous", command=lambda: self.change_page(-1)
        ).grid(row=0, column=1, padx=2)
        self.page_label = ttk.Label(pagination_frame, text="Page 1")
        self.page_label.grid(row=0, column=2, sticky="w")
        ttk.Button(
            pagination_frame, text="Next", command=lambda: self.change_page(1)
        ).grid(row=0, column=3, padx=2)
        ttk.Button(pagination_frame, text="Last", command=self.go_to_last_page).grid(
            row=0, column=4, padx=2
        )

        ttk.Label(pagination_frame, text="Go to page:").grid(
            row=1, column=1, padx=(10, 2)
        )
        self.page_entry = ttk.Entry(pagination_frame, width=5)
        self.page_entry.grid(row=1, column=2, padx=2)
        ttk.Button(pagination_frame, text="Go", command=self.jump_to_page).grid(
            row=1, column=3, padx=2
        )

        header_frame = ttk.Frame(pagination_frame)
        header_frame.grid(row=2, column=0, columnspan=8, sticky="ew", padx=5, pady=5)
        header_frame.grid_columnconfigure(0, weight=0)
        header_frame.grid_columnconfigure(1, weight=0)
        header_frame.grid_columnconfigure(2, weight=1)

        self.desc_header_label = ttk.Label(
            header_frame,
            text="Show\nDesc",
            font=("TkDefaultFont", 9, "bold"),
            anchor="center",
            justify="center",
        )
        self.desc_header_label.grid(row=0, column=0, padx=5, sticky="ew")

        self.keep_first_label = ttk.Label(
            header_frame,
            text="Keep\nFirst",
            font=("TkDefaultFont", 9, "bold"),
            anchor="center",
            justify="center",
        )
        self.keep_first_label.grid(row=0, column=1, padx=5, sticky="ew")

        self.select_entities_label = ttk.Label(
            header_frame,
            text="Select\nEntities",
            font=("TkDefaultFont", 9, "bold"),
            anchor="w",
            justify="left",
        )
        self.select_entities_label.grid(row=0, column=2, padx=5, sticky="w")

        action_buttons_frame = ttk.Frame(top_controls_frame)
        action_buttons_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        action_buttons_frame.grid_columnconfigure(0, weight=1)
        action_buttons_frame.grid_columnconfigure(1, weight=1)
        action_buttons_frame.grid_columnconfigure(2, weight=1)
        action_buttons_frame.grid_rowconfigure(0, weight=1)
        action_buttons_frame.grid_rowconfigure(1, weight=1)

        self.select_all_of_entity_button = ttk.Button(
            action_buttons_frame,
            text="Select All Of Type",
            command=self.select_all_of_entity_type,
        )
        self.select_all_of_entity_button.grid(row=0, column=0, sticky="ew", padx=(0, 2))

        self.select_all_orphans_button = ttk.Button(
            action_buttons_frame, text="Select Orphans", command=self.select_all_orphans
        )
        self.select_all_orphans_button.grid(row=0, column=1, sticky="ew", padx=(2, 2))

        self.clear_selected_button = ttk.Button(
            action_buttons_frame,
            text="Clear Selected",
            command=self.clear_selected_entities,
        )
        self.clear_selected_button.grid(row=0, column=2, sticky="ew", padx=(2, 2))

        self.show_selected_button = ttk.Button(
            action_buttons_frame, text="Show Selected", command=self.show_selected_only
        )
        self.show_selected_button.grid(row=1, column=0, sticky="ew", padx=(0, 2))

        self.clear_filter_button = ttk.Button(
            action_buttons_frame, text="Show All", command=self.clear_filter
        )
        self.clear_filter_button.grid(row=1, column=1, sticky="ew", padx=(2, 2))

        self.clear_all_button = ttk.Button(
            action_buttons_frame, text="Reset All", command=self.clear_all_selections
        )
        self.clear_all_button.grid(row=1, column=2, sticky="ew", padx=(2, 0))

        content_frame = ttk.Frame(self.left_panel)
        content_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=1)

        self.toggle_desc_button = ttk.Button(
            content_frame, text="⇕", command=self.toggle_descriptions, width=3
        )
        self.toggle_desc_button.grid(row=0, column=0, sticky="ns", padx=(0, 5))

        list_frame = ttk.Frame(content_frame)
        list_frame.grid(row=0, column=1, sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(list_frame)
        self.scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.scrollable_frame.grid_columnconfigure(0, weight=0)
        self.scrollable_frame.grid_columnconfigure(1, weight=1)

        def _on_mousewheel(event):
            try:
                if platform.system() == "Windows":
                    if hasattr(event, "delta") and event.delta:
                        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                elif platform.system() == "Darwin":
                    if hasattr(event, "delta") and event.delta:
                        self.canvas.yview_scroll(int(-1 * event.delta), "units")
                else:
                    if event.num == 4:
                        self.canvas.yview_scroll(-1, "units")
                    elif event.num == 5:
                        self.canvas.yview_scroll(1, "units")
                    elif hasattr(event, "delta") and event.delta:
                        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass
            return "break"

        scroll_events = []
        if platform.system() == "Windows":
            scroll_events = ["<MouseWheel>", "<Shift-MouseWheel>"]
        elif platform.system() == "Darwin":
            scroll_events = ["<MouseWheel>", "<Button-4>", "<Button-5>"]
        else:
            scroll_events = ["<Button-4>", "<Button-5>", "<MouseWheel>"]

        for event in scroll_events:
            try:
                self.canvas.bind(event, _on_mousewheel)
            except Exception:
                pass

        def on_canvas_enter(event):
            self.canvas.focus_set()

        def on_canvas_leave(event):
            self.root.focus_set()

        self.canvas.bind("<Enter>", on_canvas_enter)
        self.canvas.bind("<Leave>", on_canvas_leave)
        self.canvas.config(takefocus=True)

        def _on_key(event):
            if event.keysym == "Up":
                self.canvas.yview_scroll(-1, "units")
                return "break"
            elif event.keysym == "Down":
                self.canvas.yview_scroll(1, "units")
                return "break"
            elif event.keysym == "Page_Up":
                self.canvas.yview_scroll(-5, "units")
                return "break"
            elif event.keysym == "Page_Down":
                self.canvas.yview_scroll(5, "units")
                return "break"

        self.canvas.bind("<Key>", _on_key)
        self.canvas.bind("<Button-1>", lambda e: e.widget.focus_set())

        self.create_entity_list()

    def create_entity_list(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.check_vars.clear()

        # Calculate pagination
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.filtered_entity_list))
        paginated_entities = self.filtered_entity_list[start_idx:end_idx]

        # Update page label
        total_pages = (
            len(self.filtered_entity_list) + self.page_size - 1
        ) // self.page_size
        self.page_label.config(text=f"Pg {self.current_page + 1} of {total_pages}")

        for i, ent in enumerate(paginated_entities):
            rb = ttk.Radiobutton(
                self.scrollable_frame,
                text="",
                variable=self.first_entity_var,
                value=ent,
            )
            rb.grid(row=i, column=0, padx=(5, 2), pady=1, sticky="w")

            var = self.all_check_vars.get(ent)
            if var is None:
                var = tk.BooleanVar()
                self.all_check_vars[ent] = var

            cb = ttk.Checkbutton(
                self.scrollable_frame,
                text=ent,
                variable=var,
                command=self.update_selection,
            )
            cb.grid(row=i, column=1, padx=(2, 5), pady=1, sticky="w")

        current_first_entity = self.first_entity_var.get()
        if current_first_entity and current_first_entity in self.filtered_entity_list:
            self.first_entity_var.set(current_first_entity)
        else:
            self.first_entity_var.set("")

        self.update_selection()

    def change_page(self, direction):
        total_pages = (
            len(self.filtered_entity_list) + self.page_size - 1
        ) // self.page_size
        self.current_page = self.current_page + direction
        if self.current_page < 0:
            self.current_page = 0
        elif self.current_page >= total_pages:
            self.current_page = total_pages - 1
        self.create_entity_list()

    def change_page_to(self, page):
        total_pages = (
            len(self.filtered_entity_list) + self.page_size - 1
        ) // self.page_size
        self.current_page = max(0, min(page, total_pages - 1))
        self.create_entity_list()

    def go_to_last_page(self):
        total_pages = (
            len(self.filtered_entity_list) + self.page_size - 1
        ) // self.page_size
        self.current_page = total_pages - 1
        self.create_entity_list()

    def jump_to_page(self):
        try:
            page_num = int(self.page_entry.get()) - 1
            self.change_page_to(page_num)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid page number.")

    def clear_selected_entities(self):
        for var in self.all_check_vars.values():
            var.set(False)
        self.first_entity_var.set("")
        self.update_selection()
        self.create_entity_list()

    def on_filter_change(self, *args):
        self.current_page = 0  # Reset to first page on filter change
        filter_text = self.filter_var.get().lower()

        if not filter_text:
            self.filtered_entity_list = self.entity_list.copy()
        else:
            self.filtered_entity_list = [
                entity for entity in self.entity_list if filter_text in entity.lower()
            ]

        self.create_entity_list()

    def clear_filter(self):
        self.filter_var.set("")
        self.current_page = 0
        self.create_entity_list()

    def select_all_of_entity_type(self):
        selected_type = self.entity_type.get()
        if not selected_type:
            messagebox.showinfo(
                "No Type Selected",
                "Please select an Entity Type from the dropdown first.",
            )
            return

        for var in self.all_check_vars.values():
            var.set(False)

        entities_of_selected_type = []
        for label in self.entity_list:
            if label not in self.entity_data:
                self.entity_data[label] = fetch_entity_details(label)

            if self.entity_data[label]["type"] == selected_type:
                self.all_check_vars[label].set(True)
                entities_of_selected_type.append(label)

        self.filter_var.set("")
        self.current_page = 0
        self.filtered_entity_list = sorted(
            entities_of_selected_type, key=lambda x: x.lower()
        )
        self.create_entity_list()

        if not entities_of_selected_type:
            messagebox.showinfo(
                "No Entities Found", f"No entities of type '{selected_type}' found."
            )

    def select_all_orphans(self):
        for var in self.all_check_vars.values():
            var.set(False)

        orphan_entities = []
        for label in self.entity_list:
            if label not in self.entity_data:
                self.entity_data[label] = fetch_entity_details(label)

            related_nodes = self.entity_data[label].get("related_nodes", [])
            edges = self.entity_data[label].get("edges", [])

            if not related_nodes and not edges:
                self.all_check_vars[label].set(True)
                orphan_entities.append(label)

        self.filter_var.set("")
        self.current_page = 0
        self.filtered_entity_list = sorted(orphan_entities, key=lambda x: x.lower())
        self.create_entity_list()

        if not orphan_entities:
            messagebox.showinfo(
                "No Orphans Found", "No entities without relationships were found."
            )

    def show_selected_only(self):
        selected_entities = [
            label for label, var in self.all_check_vars.items() if var.get()
        ]
        if not selected_entities:
            messagebox.showinfo("No Selection", "No entities are currently selected.")
            self.filter_var.set("")
            self.current_page = 0
            self.filtered_entity_list = self.entity_list.copy()
            self.create_entity_list()
            return

        self.filter_var.set("")
        self.current_page = 0
        self.filtered_entity_list = sorted(selected_entities, key=lambda x: x.lower())
        self.create_entity_list()

    def clear_all_selections(self):
        for var in self.all_check_vars.values():
            var.set(False)
        self.filter_var.set("")
        self.current_page = 0
        self.filtered_entity_list = self.entity_list.copy()
        self.first_entity_var.set("")
        self.strategy_desc.set("join_unique")
        self.strategy_srcid.set("join_unique")
        self.create_entity_list()

    def create_right_panel(self):
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)

        control_frame = ttk.Frame(self.right_panel)
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        control_frame.grid_columnconfigure(0, weight=0)
        control_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="Target Entity:").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        self.target_entry = ttk.Combobox(control_frame, values=[], width=40)
        self.target_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.target_entry.bind(
            "<Control-c>", lambda event: self.copy_selected_text(self.target_entry)
        )

        ttk.Label(control_frame, text="Merge Strategy - Description:").grid(
            row=1, column=0, sticky="w", padx=5, pady=2
        )
        self.strategy_desc = ttk.Combobox(
            control_frame, values=["concatenate", "keep_first", "join_unique"]
        )
        self.strategy_desc.set("join_unique")
        self.strategy_desc.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(control_frame, text="Merge Strategy - Source ID:").grid(
            row=2, column=0, sticky="w", padx=5, pady=2
        )
        self.strategy_srcid = ttk.Combobox(
            control_frame, values=["concatenate", "keep_first", "join_unique"]
        )
        self.strategy_srcid.set("join_unique")
        self.strategy_srcid.grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        self.entity_type_button = ttk.Button(
            control_frame,
            text="Select Entity Type",
            command=self.open_all_entity_types_modal,
        )
        self.entity_type_button.grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.entity_type = ttk.Combobox(control_frame, values=[], width=37)
        self.entity_type.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        self.entity_type.bind(
            "<Control-c>", lambda event: self.copy_selected_text(self.entity_type)
        )

        info_label = ttk.Label(
            control_frame,
            text="Note: 'Keep First' strategy uses the selected radio button item.",
            font=("TkDefaultFont", 8),
            foreground="gray",
            wraplength=300,
        )
        info_label.grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=10)

        button_row_frame = ttk.Frame(control_frame)
        button_row_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        button_row_frame.grid_columnconfigure(0, weight=1)
        button_row_frame.grid_columnconfigure(1, weight=1)
        button_row_frame.grid_columnconfigure(2, weight=1)

        self.merge_button = ttk.Button(
            button_row_frame, text="Merge Entities", command=self.submit_merge
        )
        self.merge_button.grid(row=0, column=0, sticky="w", padx=(5, 5))

        self.create_rel_button = ttk.Button(
            button_row_frame,
            text="Create Relationship",
            command=self.open_create_relationship_modal,
        )
        self.create_rel_button.grid(row=0, column=1, sticky="w", padx=(5, 5))

        self.delete_button = ttk.Button(
            button_row_frame,
            text="Delete Entities",
            command=lambda: asyncio.run(self.run_delete()),
        )
        self.delete_button.grid(row=0, column=2, sticky="w", padx=(5, 5))

        self.merge_button.grid_remove()
        self.create_rel_button.grid_remove()
        self.delete_button.grid_remove()

        self.description_area = ttk.Frame(self.right_panel)
        self.description_area.grid(
            row=1, column=0, sticky="nsew", padx=10, pady=(0, 10)
        )
        self.description_area.grid_columnconfigure(0, weight=1)
        self.description_area.grid_rowconfigure(0, weight=1)

        self.desc_canvas = tk.Canvas(self.description_area)
        self.desc_scrollbar = ttk.Scrollbar(
            self.description_area, orient="vertical", command=self.desc_canvas.yview
        )
        self.desc_scrollable_frame = ttk.Frame(self.desc_canvas)

        self.desc_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.desc_canvas.configure(
                scrollregion=self.desc_canvas.bbox("all")
            ),
        )
        self.desc_canvas.create_window(
            (0, 0), window=self.desc_scrollable_frame, anchor="nw"
        )
        self.desc_canvas.configure(yscrollcommand=self.desc_scrollbar.set)

        self.desc_canvas.grid(row=0, column=0, sticky="nsew")
        self.desc_scrollbar.grid(row=0, column=1, sticky="ns")

        def _on_desc_mousewheel(event):
            try:
                if platform.system() == "Windows":
                    if hasattr(event, "delta") and event.delta:
                        self.desc_canvas.yview_scroll(
                            int(-1 * (event.delta / 120)), "units"
                        )
                elif platform.system() == "Darwin":
                    if hasattr(event, "delta") and event.delta:
                        self.desc_canvas.yview_scroll(int(-1 * event.delta), "units")
                else:
                    if event.num == 4:
                        self.desc_canvas.yview_scroll(-1, "units")
                    elif event.num == 5:
                        self.desc_canvas.yview_scroll(1, "units")
                    elif hasattr(event, "delta") and event.delta:
                        self.desc_canvas.yview_scroll(
                            int(-1 * (event.delta / 120)), "units"
                        )
            except Exception:
                pass
            return "break"

        desc_scroll_events = []
        if platform.system() == "Windows":
            desc_scroll_events = ["<MouseWheel>", "<Shift-MouseWheel>"]
        elif platform.system() == "Darwin":
            desc_scroll_events = ["<MouseWheel>", "<Button-4>", "<Button-5>"]
        else:
            desc_scroll_events = ["<Button-4>", "<Button-5>", "<MouseWheel>"]

        for event in desc_scroll_events:
            try:
                self.desc_canvas.bind(event, _on_desc_mousewheel)
            except Exception:
                pass

    def open_all_entity_types_modal(self):
        all_types = set()
        for label in self.entity_list:
            if label not in self.entity_data:
                self.entity_data[label] = fetch_entity_details(label)
            entity_type = self.entity_data[label].get("type", "")
            if entity_type and not entity_type.startswith("Error:"):
                all_types.add(entity_type)
        all_types = sorted(list(all_types))

        if not all_types:
            messagebox.showinfo(
                "No Entity Types", "No entity types found for any entities."
            )
            return

        modal = tk.Toplevel(self.root)
        modal.title("Select Entity Type")
        modal.transient(self.root)
        modal.grab_set()
        modal.protocol("WM_DELETE_WINDOW", modal.destroy)

        modal.grid_columnconfigure(0, weight=1)
        modal.grid_rowconfigure(0, weight=1)

        list_frame = ttk.Frame(modal)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)

        listbox = tk.Listbox(list_frame, font=("TkDefaultFont", 10), height=10)
        listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")

        for entity_type in all_types:
            listbox.insert(tk.END, entity_type)

        def on_double_click(event):
            selection = listbox.curselection()
            if selection:
                selected_type = listbox.get(selection[0])
                self.entity_type.set(selected_type)
                modal.destroy()

        listbox.bind("<Double-1>", on_double_click)

        button_frame = ttk.Frame(modal)
        button_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        cancel_button = ttk.Button(button_frame, text="Cancel", command=modal.destroy)
        cancel_button.pack(side="right")

        modal.geometry("300x300")
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 150
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 150
        modal.geometry(f"+{x}+{y}")

    def update_selection(self):
        selected = [label for label, var in self.all_check_vars.items() if var.get()]

        if not hasattr(self, "target_entry"):
            return

        self.target_entry["values"] = selected

        types = set()
        for label in selected:
            if label not in self.entity_data:
                self.entity_data[label] = fetch_entity_details(label)

            if self.entity_data[label].get("type") and not str(
                self.entity_data[label]["type"]
            ).startswith("Error:"):
                types.add(self.entity_data[label]["type"])

        current_entity_type = self.entity_type.get()
        self.entity_type["values"] = sorted(list(types))

        if current_entity_type and current_entity_type in self.entity_type["values"]:
            self.entity_type.set(current_entity_type)
        else:
            self.entity_type.set("")

        if self.target_entry.get() not in selected:
            self.target_entry.set("")

        if self.first_entity_var.get() and self.first_entity_var.get() not in selected:
            self.first_entity_var.set("")

        self.update_button_visibility()

    def update_button_visibility(self):
        selected_from_all = {
            label for label, var in self.all_check_vars.items() if var.get()
        }

        if (
            selected_from_all
            and set(self.filtered_entity_list) == selected_from_all
            and len(self.filtered_entity_list) == len(selected_from_all)
        ):
            self.merge_button.grid()
            self.delete_button.grid()
        else:
            self.merge_button.grid_remove()
            self.delete_button.grid_remove()

        if len(selected_from_all) == 2:
            entity1, entity2 = list(selected_from_all)
            if entity1 not in self.entity_data:
                self.entity_data[entity1] = fetch_entity_details(entity1)
            if entity2 not in self.entity_data:
                self.entity_data[entity2] = fetch_entity_details(entity2)

            edges1 = self.entity_data[entity1].get("edges", [])
            has_relationship = any(
                (edge["source"] == entity1 and edge["target"] == entity2)
                or (edge["source"] == entity2 and edge["target"] == entity1)
                for edge in edges1
            )

            if not has_relationship:
                self.create_rel_button.grid()
            else:
                self.create_rel_button.grid_remove()
        else:
            self.create_rel_button.grid_remove()

    def calculate_tile_layout(
        self, available_width, available_height, num_items, min_height=600
    ):
        if num_items == 0:
            return 0, 0, 0, 0

        effective_min_width = max(300, (available_width // 2) - 10)

        cols = max(1, available_width // effective_min_width)

        if available_width >= 2 * effective_min_width and num_items > 1:
            cols = 2
        elif num_items == 1:
            cols = 1
        else:
            cols = 1

        cols = min(cols, num_items)

        rows = (num_items + cols - 1) // cols

        frame_width = available_width // cols
        frame_height = max(
            min_height, available_height // rows if rows > 0 else available_height
        )

        return cols, rows, frame_width, frame_height

    def toggle_descriptions(self):
        any_open = bool(self.description_frames)
        if any_open:
            for frame in list(self.description_frames.values()):
                frame.destroy()
            self.description_frames.clear()
            self.desc_header_label["text"] = "Show\nDesc"
            self.desc_scrollable_frame.update_idletasks()
            self.desc_canvas.configure(scrollregion=self.desc_canvas.bbox("all"))
        else:
            selected = [
                label for label, var in self.all_check_vars.items() if var.get()
            ]
            if not selected:
                messagebox.showinfo(
                    "No Selection",
                    "Please select some entities first to show descriptions.",
                )
                return

            self.root.update_idletasks()
            self.desc_scrollable_frame.update_idletasks()
            available_width = self.desc_scrollable_frame.winfo_width()
            available_height = self.desc_scrollable_frame.winfo_height()

            if available_width < 100:
                available_width = self.right_panel.winfo_width() - 20
                if available_width < 100:
                    available_width = 800
            if available_height < 100:
                available_height = 600

            cols, rows, frame_width, frame_height = self.calculate_tile_layout(
                available_width, available_height, len(selected), min_height=600
            )

            for c in range(cols):
                self.desc_scrollable_frame.grid_columnconfigure(c, weight=1)
            for r in range(rows):
                self.desc_scrollable_frame.grid_rowconfigure(r, weight=1)

            for idx, label in enumerate(selected):
                try:
                    if label not in self.entity_data:
                        self.entity_data[label] = fetch_entity_details(label)

                    entity_details = self.entity_data[label]
                    desc = entity_details.get("desc", "No description found.")
                    entity_type = entity_details.get("type", "No type found.")
                    srcid = entity_details.get("srcid", "")
                    fpath = entity_details.get("fpath", "")
                    related_nodes = entity_details.get("related_nodes", [])
                    edges = entity_details.get("edges", [])

                    row = idx // cols
                    col = idx % cols

                    desc_frame = ttk.LabelFrame(
                        self.desc_scrollable_frame, text="", padding=5
                    )
                    desc_frame.config(width=frame_width - 4, height=frame_height - 4)
                    desc_frame.grid_propagate(False)

                    desc_frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)

                    header_sub_frame = ttk.Frame(desc_frame)
                    header_sub_frame.pack(fill="x")
                    header_sub_frame.grid_columnconfigure(0, weight=1)
                    header_sub_frame.grid_columnconfigure(1, weight=0)
                    header_sub_frame.grid_columnconfigure(2, weight=0)

                    ttk.Label(
                        header_sub_frame, text=label, font=("TkDefaultFont", 10, "bold")
                    ).grid(row=0, column=0, sticky="w")

                    edit_button = ttk.Button(
                        header_sub_frame,
                        text="Edit Description",
                        command=lambda lbl=label: self.open_edit_description_modal(lbl),
                    )
                    edit_button.grid(row=0, column=1, sticky="e", padx=(5, 5))

                    edit_rel_button = ttk.Button(
                        header_sub_frame,
                        text="Edit/Delete Relationships",
                        command=lambda lbl=label: self.open_edit_relationships_modal(
                            lbl
                        ),
                    )
                    edit_rel_button.grid(row=0, column=2, sticky="e", padx=(0, 5))

                    text_frame = ttk.Frame(desc_frame)
                    text_frame.pack(fill="both", expand=True, pady=(5, 0))

                    text_widget = tk.Text(
                        text_frame, wrap="word", font=("TkDefaultFont", 9), height=15
                    )
                    text_scrollbar = ttk.Scrollbar(
                        text_frame, orient="vertical", command=text_widget.yview
                    )
                    text_widget.configure(yscrollcommand=text_scrollbar.set)

                    text_content_parts = []
                    if entity_type and entity_type != "No type found.":
                        text_content_parts.append(f"Type: {entity_type}")

                    if related_nodes:
                        text_content_parts.append(
                            f"Related Entities: {len(related_nodes)}"
                        )

                    if desc and desc != "No description found.":
                        text_content_parts.append(f"Description:\n{desc}")

                    if srcid:
                        text_content_parts.append(f"Source ID:\n{srcid}")

                    if fpath:
                        text_content_parts.append(f"File Path: {fpath}")

                    if related_nodes:
                        for idx, node in enumerate(related_nodes, start=1):
                            node_id = node.get("id", "N/A")
                            node_desc = node.get("description", "")
                            node_type = node.get("type", "")
                            text_content_parts.append(f"\nRelated Entity {idx}:")
                            text_content_parts.append(
                                f"- {node_id} (Type: {node_type})"
                            )
                            if node_desc and node_desc != "No description found.":
                                text_content_parts.append(f"  Description: {node_desc}")

                    if edges:
                        filtered_edges = [
                            edge
                            for edge in edges
                            if edge.get("source") == label
                            or edge.get("target") == label
                        ]
                        if filtered_edges:
                            text_content_parts.append("\nRelationships:")
                            for edge in filtered_edges:
                                source = edge.get("source", "N/A")
                                target = edge.get("target", "N/A")
                                edge_desc = edge.get(
                                    "description", "No description provided."
                                )
                                edge_keywords = edge.get("keywords", "")
                                edge_weight = edge.get("weight", 1.0)
                                text_content_parts.append(
                                    f"- From: {source}\n  To: {target}\n  Relation: {edge_desc}\n  Weight: {edge_weight}"
                                )
                                if edge_keywords:
                                    text_content_parts.append(
                                        f"  Keywords: {edge_keywords}"
                                    )

                    text_widget.insert("1.0", "\n\n".join(text_content_parts))
                    text_widget.config(state="disabled")

                    text_widget.pack(side="left", fill="both", expand=True)
                    text_scrollbar.pack(side="right", fill="y")

                    self.description_frames[label] = desc_frame

                except Exception as e:
                    print(f"Error showing description for {label}: {e}")

            self.desc_scrollable_frame.update_idletasks()
            self.desc_canvas.configure(scrollregion=self.desc_canvas.bbox("all"))

            if self.description_frames:
                self.desc_header_label["text"] = "Hide\nDesc"

    def open_edit_description_modal(self, entity_label):
        modal = tk.Toplevel(self.root)
        modal.title(f"Edit Description for {entity_label}")
        modal.transient(self.root)
        modal.grab_set()
        modal.protocol("WM_DELETE_WINDOW", modal.destroy)

        modal.grid_columnconfigure(0, weight=1)
        modal.grid_rowconfigure(1, weight=1)

        current_description = self.entity_data.get(entity_label, {}).get("desc", "")

        ttk.Label(
            modal,
            text=f"Editing Description for: {entity_label}",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        text_frame = ttk.Frame(modal)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        description_text = tk.Text(text_frame, wrap="word", font=("TkDefaultFont", 10))
        description_text.insert("1.0", current_description)

        text_scrollbar = ttk.Scrollbar(
            text_frame, orient="vertical", command=description_text.yview
        )
        description_text.configure(yscrollcommand=text_scrollbar.set)

        description_text.pack(side="left", fill="both", expand=True)
        text_scrollbar.pack(side="right", fill="y")

        button_frame = ttk.Frame(modal)
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        button_frame.grid_columnconfigure(0, weight=1)

        save_button = ttk.Button(
            button_frame,
            text="Save",
            command=lambda: self.save_entity_description(
                entity_label, description_text, modal
            ),
        )
        save_button.pack(side="right", padx=(5, 0))

        cancel_button = ttk.Button(button_frame, text="Cancel", command=modal.destroy)
        cancel_button.pack(side="right")

        self.root.update_idletasks()
        modal.update_idletasks()
        min_width = 400
        min_height = 300
        if modal.winfo_width() < min_width or modal.winfo_height() < min_height:
            modal.geometry(f"{min_width}x{min_height}")
            modal.update_idletasks()

        x = (
            self.root.winfo_x()
            + (self.root.winfo_width() // 2)
            - (modal.winfo_width() // 2)
        )
        y = (
            self.root.winfo_y()
            + (self.root.winfo_height() // 2)
            - (modal.winfo_height() // 2)
        )
        modal.geometry(f"+{x}+{y}")

        description_text.bind(
            "<Control-c>", lambda event: self.copy_selected_text(description_text)
        )

        self.root.wait_window(modal)

    def open_create_relationship_modal(self):
        selected = [label for label, var in self.all_check_vars.items() if var.get()]
        if len(selected) != 2:
            messagebox.showerror(
                "Error", "Please select exactly two entities to create a relationship."
            )
            return

        entity1, entity2 = selected
        modal = tk.Toplevel(self.root)
        modal.title("Create New Relationship")
        modal.transient(self.root)
        modal.grab_set()
        modal.protocol("WM_DELETE_WINDOW", modal.destroy)

        modal.grid_columnconfigure(0, weight=1)
        modal.grid_columnconfigure(1, weight=0)
        modal.grid_rowconfigure(5, weight=0)

        ttk.Label(
            modal,
            text="Create Relationship Between Entities",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        ttk.Label(modal, text="Source Entity:").grid(
            row=1, column=0, sticky="w", padx=10, pady=2
        )
        source_var = tk.StringVar(value=entity1)
        source_combobox = ttk.Combobox(
            modal, textvariable=source_var, values=selected, state="readonly"
        )
        source_combobox.grid(row=2, column=0, sticky="ew", padx=10, pady=2)

        ttk.Label(modal, text="Target Entity:").grid(
            row=3, column=0, sticky="w", padx=2
        )
        target_var = tk.StringVar(value=entity2)
        target_combobox = ttk.Combobox(
            modal, textvariable=target_var, values=selected, state="readonly"
        )
        target_combobox.grid(row=4, column=0, sticky="ew", padx=10, pady=2)

        ttk.Label(modal, text="Relationship Description:").grid(
            row=5, column=0, sticky="w", padx=10, pady=2
        )
        desc_text = tk.Text(modal, wrap="word", font=("TkDefaultFont", 10), height=5)
        desc_scrollbar = ttk.Scrollbar(
            modal, orient="vertical", command=desc_text.yview
        )
        desc_text.configure(yscrollcommand=desc_scrollbar.set)
        desc_text.grid(row=6, column=0, sticky="nsew", padx=10, pady=2)
        desc_scrollbar.grid(row=6, column=1, sticky="ns")

        ttk.Label(modal, text="Keywords (comma-separated):").grid(
            row=7, column=0, sticky="w", padx=10, pady=2
        )
        keywords_entry = ttk.Entry(modal)
        keywords_entry.grid(row=8, column=0, sticky="ew", padx=10, pady=2)

        ttk.Label(modal, text="Weight (1.0-10.0):").grid(
            row=9, column=0, sticky="w", padx=10, pady=2
        )
        weight_var = tk.StringVar(value="7.0")
        weight_entry = ttk.Entry(modal, textvariable=weight_var)
        weight_entry.grid(row=10, column=0, sticky="ew", padx=10, pady=2)

        ttk.Label(modal, text="Source File ID:").grid(
            row=11, column=0, sticky="w", padx=10, pady=2
        )
        source_file_id_entry = ttk.Entry(modal)
        source_file_id_entry.grid(row=12, column=0, sticky="ew", padx=10, pady=2)

        button_frame = ttk.Frame(modal)
        button_frame.grid(row=13, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        button_frame.grid_columnconfigure(0, weight=1)

        save_button = ttk.Button(
            button_frame,
            text="Save",
            command=lambda: self.save_new_relationship(
                source_var.get(),
                target_var.get(),
                desc_text,
                keywords_entry.get(),
                weight_var.get(),
                source_file_id_entry.get(),
                modal,
            ),
        )
        save_button.pack(side="right", padx=(5, 0))

        cancel_button = ttk.Button(button_frame, text="Cancel", command=modal.destroy)
        cancel_button.pack(side="right")

        modal.geometry("500x450")
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 250
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 300
        modal.geometry(f"+{x}+{y}")

        desc_text.bind("<Control-c>", lambda event: self.copy_selected_text(desc_text))

    def save_new_relationship(
        self, source_id, target_id, desc_text, keywords, weight, source_file_id, modal
    ):
        description = desc_text.get("1.0", tk.END).strip()
        if (
            not source_id
            or not target_id
            or not description
            or not keywords
            or not weight
            or not source_file_id
        ):
            messagebox.showerror("Error", "All fields are required.")
            return

        try:
            weight_float = float(weight)
            if not (1.0 <= weight_float <= 10.0):
                messagebox.showerror("Error", "Weight must be between 1.0 and 10.0.")
                return
        except ValueError:
            messagebox.showerror("Error", "Weight must be a valid number.")
            return

        if create_relationship_api(
            source_id, target_id, description, keywords, weight_float, source_file_id
        ):
            trigger_server_refresh()
            self.entity_data[source_id] = fetch_entity_details(source_id)
            self.entity_data[target_id] = fetch_entity_details(target_id)
            if self.description_frames:
                for frame in list(self.description_frames.values()):
                    frame.destroy()
                self.description_frames.clear()
                self.toggle_descriptions()
            modal.destroy()
            self.create_entity_list()

    def open_edit_relationships_modal(self, entity_label):
        modal = tk.Toplevel(self.root)
        modal.title(f"Edit Relationships for {entity_label}")
        modal.transient(self.root)
        modal.grab_set()
        modal.protocol("WM_DELETE_WINDOW", modal.destroy)

        modal.grid_columnconfigure(0, weight=1)
        modal.grid_rowconfigure(2, weight=1)

        ttk.Label(
            modal,
            text=f"Select Relationship for: {entity_label}",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, padx=10, pady=5, sticky="w")

        entity_details = self.entity_data.get(entity_label, {})
        edges = entity_details.get("edges", [])
        filtered_edges = [
            edge
            for edge in edges
            if edge.get("source") == entity_label or edge.get("target") == entity_label
        ]

        if not filtered_edges:
            ttk.Label(
                modal, text="No relationships found.", font=("TkDefaultFont", 10)
            ).grid(row=1, column=0, padx=10, pady=5, sticky="w")
            button_frame = ttk.Frame(modal)
            button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
            ttk.Button(button_frame, text="Close", command=modal.destroy).pack(
                side="right"
            )
            return

        relationship_options = [
            f"From: {edge['source']} To: {edge['target']}" for edge in filtered_edges
        ]
        selected_relationship = tk.StringVar()
        relationship_combobox = ttk.Combobox(
            modal,
            textvariable=selected_relationship,
            values=relationship_options,
            state="readonly",
        )
        relationship_combobox.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        if relationship_options:
            relationship_combobox.set(relationship_options[0])

        edit_frame = ttk.Frame(modal)
        edit_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        edit_frame.grid_columnconfigure(0, weight=1)
        edit_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(edit_frame, text="Relationship Description:").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        desc_text = tk.Text(
            edit_frame, wrap="word", font=("TkDefaultFont", 10), height=5
        )
        desc_scrollbar = ttk.Scrollbar(
            edit_frame, orient="vertical", command=desc_text.yview
        )
        desc_text.configure(yscrollcommand=desc_scrollbar.set)
        desc_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=2)
        desc_scrollbar.grid(row=1, column=1, sticky="ns")

        ttk.Label(edit_frame, text="Keywords (comma-separated):").grid(
            row=2, column=0, sticky="w", padx=5, pady=2
        )
        keywords_entry = ttk.Entry(edit_frame)
        keywords_entry.grid(row=3, column=0, sticky="ew", padx=5, pady=2)

        ttk.Label(edit_frame, text="Weight (1.0-10.0):").grid(
            row=4, column=0, sticky="w", padx=5, pady=2
        )
        weight_var = tk.StringVar()
        weight_entry = ttk.Entry(edit_frame, textvariable=weight_var)
        weight_entry.grid(row=5, column=0, sticky="ew", padx=5, pady=2)

        def update_fields(*args):
            selected_idx = relationship_combobox.current()
            if selected_idx >= 0:
                edge = filtered_edges[selected_idx]
                desc_text.delete("1.0", tk.END)
                desc_text.insert("1.0", edge.get("description", ""))
                keywords_entry.delete(0, tk.END)
                keywords_entry.insert(0, edge.get("keywords", ""))
                weight_var.set(str(edge.get("weight", 1.0)))

        selected_relationship.trace("w", update_fields)
        update_fields()

        button_frame = ttk.Frame(modal)
        button_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        button_frame.grid_columnconfigure(0, weight=1)

        save_button = ttk.Button(
            button_frame,
            text="Save",
            command=lambda: self.save_relationship(
                entity_label,
                filtered_edges,
                relationship_combobox.current(),
                desc_text,
                keywords_entry,
                weight_var.get(),
                modal,
            ),
        )
        save_button.pack(side="right", padx=(5, 0))

        cancel_button = ttk.Button(button_frame, text="Cancel", command=modal.destroy)
        cancel_button.pack(side="right")

        delete_button = ttk.Button(
            button_frame,
            text="Delete Relationship",
            command=lambda: asyncio.run(
                self.delete_relationship(
                    entity_label, filtered_edges, relationship_combobox.current(), modal
                )
            ),
        )
        delete_button.pack(side="left", padx=(0, 5))

        self.root.update_idletasks()
        modal.update_idletasks()
        min_width = 500
        min_height = 400
        modal.geometry(f"{min_width}x{min_height}")

        x = (
            self.root.winfo_x()
            + (self.root.winfo_width() // 2)
            - (modal.winfo_width() // 2)
        )
        y = (
            self.root.winfo_y()
            + (self.root.winfo_height() // 2)
            - (modal.winfo_height() // 2)
        )
        modal.geometry(f"+{x}+{y}")

        self.root.wait_window(modal)

    def save_relationship(
        self,
        entity_label,
        edges,
        selected_idx,
        desc_text,
        keywords_entry,
        weight,
        modal,
    ):
        if selected_idx < 0:
            messagebox.showerror("Error", "No relationship selected.")
            return

        edge = edges[selected_idx]
        source_id = edge["source"]
        target_id = edge["target"]
        new_description = desc_text.get("1.0", tk.END).strip()
        new_keywords = keywords_entry.get().strip()

        try:
            weight_float = float(weight)
            if not (1.0 <= weight_float <= 10.0):
                messagebox.showerror("Error", "Weight must be between 1.0 and 10.0.")
                return
        except ValueError:
            messagebox.showerror("Error", "Weight must be a valid number.")
            return

        if update_relationship_api(
            source_id, target_id, new_description, new_keywords, weight_float
        ):
            for e in self.entity_data[entity_label]["edges"]:
                if e["source"] == source_id and e["target"] == target_id:
                    e["description"] = new_description
                    e["keywords"] = new_keywords
                    e["weight"] = weight_float
                    break

            if self.description_frames:
                for frame in list(self.description_frames.values()):
                    frame.destroy()
                self.description_frames.clear()
                self.toggle_descriptions()

            modal.destroy()

    async def delete_relationship(self, entity_label, edges, selected_idx, modal):
        if selected_idx < 0 or selected_idx >= len(edges):
            messagebox.showerror("Error", "No relationship selected to delete.")
            return

        edge = edges[selected_idx]
        source_id = edge["source"]
        target_id = edge["target"]

        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete the relationship from:\n\n{source_id}\n\nto\n\n{target_id}?",
        )
        if not confirm:
            return

        try:
            rag = await initialize_rag()
            await rag.adelete_by_relation(source_id, target_id)

            messagebox.showinfo(
                "Success", f"Deleted relationship from '{source_id}' to '{target_id}'."
            )

            print("Triggering LightRAG server refresh...")
            if not trigger_server_refresh():
                print("Server data refresh failed or server not running.")
            else:
                print("Server refresh attempted.")

            if entity_label in self.entity_data:
                self.entity_data[entity_label] = fetch_entity_details(entity_label)

            modal.destroy()
            self.open_edit_relationships_modal(entity_label)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete relationship:\n{e}")
        finally:
            if "rag" in locals():
                await rag.finalize_storages()

    def save_entity_description(
        self, entity_label, description_text_widget, modal_window
    ):
        new_description = description_text_widget.get("1.0", tk.END).strip()

        if update_entity_description_api(entity_label, new_description):
            if entity_label in self.entity_data:
                self.entity_data[entity_label]["desc"] = new_description

            if self.description_frames:
                for frame in list(self.description_frames.values()):
                    frame.destroy()
                self.description_frames.clear()
                self.toggle_descriptions()

            modal_window.destroy()

    def submit_merge(self):
        if not self.entity_type.get() or not self.target_entry.get():
            messagebox.showerror(
                "Missing info", "Please select a target entity and entity type."
            )
            return

        selected = [label for label, var in self.all_check_vars.items() if var.get()]

        if not selected:
            messagebox.showerror("No entities", "Select at least one source entity.")
            return

        selected_from_filtered = {
            label
            for label in self.filtered_entity_list
            if self.all_check_vars.get(label, tk.BooleanVar()).get()
        }
        if not (
            set(selected) == selected_from_filtered
            and len(selected) == len(selected_from_filtered)
        ):
            messagebox.showerror(
                "Operation Not Allowed",
                "Merge operation can only be performed when 'Selected Only' mode is active and the displayed list exactly matches the selected entities.",
            )
            return

        for frame in list(self.description_frames.values()):
            frame.destroy()
        self.description_frames.clear()
        self.desc_header_label["text"] = "Show\nDesc"

        strategy = {
            "description": self.strategy_desc.get(),
            "source_id": self.strategy_srcid.get(),
        }

        if (
            strategy["description"] == "keep_first"
            or strategy["source_id"] == "keep_first"
        ):
            first_entity = self.first_entity_var.get()
            if not first_entity:
                messagebox.showerror(
                    "Missing Selection",
                    "Please select which entity should be 'first' using the radio buttons when using 'keep_first' strategy.",
                )
                return
            if first_entity not in selected:
                messagebox.showerror(
                    "Invalid Selection",
                    "The selected 'first' entity must be in the list of selected entities.",
                )
                return
            selected = [first_entity] + [e for e in selected if e != first_entity]

        asyncio.run(
            self.run_merge(
                selected,
                self.target_entry.get(),
                strategy,
                etype=self.entity_type.get(),
            )
        )

    async def run_merge(self, sources, target, strategy, etype):
        rag = await initialize_rag()
        try:
            await rag.amerge_entities(
                source_entities=sources,
                target_entity=target,
                merge_strategy=strategy,
                target_entity_data={"entity_type": etype},
            )
            messagebox.showinfo("Success", f"Entities merged into '{target}'")

            print("Refreshing LightRAG server data from disk...")
            if not trigger_server_refresh():
                print(
                    "Server data refresh failed or server not running. Manual restart might still be needed if changes don't appear."
                )
            else:
                print("Server refresh attempted.")

            self.entity_data.clear()

            newly_fetched_entities = fetch_entities()

            new_all_check_vars = {}
            for entity in newly_fetched_entities:
                new_all_check_vars[entity] = self.all_check_vars.get(
                    entity, tk.BooleanVar()
                )
            self.all_check_vars = new_all_check_vars
            self.entity_list = newly_fetched_entities

            self.filter_var.set("")
            self.current_page = 0
            self.filtered_entity_list = self.entity_list.copy()
            for var in self.all_check_vars.values():
                var.set(False)
            self.first_entity_var.set("")

            self.create_entity_list()

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            await rag.finalize_storages()

    async def run_delete(self):
        selected_for_deletion = [
            label for label, var in self.all_check_vars.items() if var.get()
        ]

        if not selected_for_deletion:
            messagebox.showinfo("No entities", "No entities are selected for deletion.")
            return

        selected_from_filtered = {
            label
            for label in self.filtered_entity_list
            if self.all_check_vars.get(label, tk.BooleanVar()).get()
        }
        if not (
            set(selected_for_deletion) == selected_from_filtered
            and len(selected_for_deletion) == len(selected_from_filtered)
        ):
            messagebox.showerror(
                "Operation Not Allowed",
                "Delete operation can only be performed when 'Selected Only' mode is active and the displayed list exactly matches the selected entities.",
            )
            return

        if len(selected_for_deletion) > 1:
            confirm_message = (
                f"Are you sure you want to delete the following {len(selected_for_deletion)} entities?\n\n"
                + "\n".join(selected_for_deletion[:10])
                + ("..." if len(selected_for_deletion) > 10 else "")
            )
        else:
            confirm_message = (
                f"Are you sure you want to delete '{selected_for_deletion[0]}'?"
            )

        if not messagebox.askyesno("Confirm Deletion", confirm_message):
            return

        for frame in list(self.description_frames.values()):
            frame.destroy()
        self.description_frames.clear()
        self.desc_header_label["text"] = "Show\nDesc"

        rag = await initialize_rag()
        try:
            success_count = 0
            fail_count = 0
            failed_entities = []

            for entity_to_delete in selected_for_deletion:
                print(f"Attempting to delete entity: {entity_to_delete}")
                try:
                    await rag.adelete_by_entity(entity_to_delete)
                    print(f"Successfully deleted: {entity_to_delete}")
                    success_count += 1
                except Exception as e:
                    print(f"Failed to delete {entity_to_delete}: {e}")
                    fail_count += 1
                    failed_entities.append(entity_to_delete)

            if success_count > 0:
                messagebox.showinfo(
                    "Deletion Complete",
                    f"Successfully deleted {success_count} entities. Failed to delete {fail_count} entities. Please check logs for details.",
                )
                if failed_entities:
                    messagebox.showerror(
                        "Deletion Errors",
                        "Failed to delete:\n" + "\n".join(failed_entities),
                    )

                print("Refreshing LightRAG server data from disk after deletion...")
                if not trigger_server_refresh():
                    print(
                        "Server data refresh failed or server not running. Manual restart might still be needed if changes don't appear."
                    )
                else:
                    print("Server refresh attempted.")

                self.entity_data.clear()
                newly_fetched_entities = fetch_entities()

                new_all_check_vars = {}
                for entity in newly_fetched_entities:
                    new_all_check_vars[entity] = self.all_check_vars.get(
                        entity, tk.BooleanVar()
                    )
                self.all_check_vars = new_all_check_vars
                self.entity_list = newly_fetched_entities

                self.filter_var.set("")
                self.current_page = 0
                self.filtered_entity_list = self.entity_list.copy()
                for var in self.all_check_vars.values():
                    var.set(False)
                self.first_entity_var.set("")

                self.create_entity_list()

            elif fail_count > 0:
                messagebox.showerror(
                    "Deletion Failed",
                    "No entities were successfully deleted. Please check logs for details.",
                )

        except Exception as e:
            messagebox.showerror(
                "Error", f"An unexpected error occurred during deletion: {e}"
            )
        finally:
            await rag.finalize_storages()

    def on_closing(self):
        self.save_window_config()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()

    # === ADD THIS BLOCK HERE ===
    import tkinter.font as tkfont

    # Force a thicker, larger, more readable default font
    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(family="DejaVu Sans", size=12, weight="bold")

    # Also fix the other standard Tk fonts for consistency
    tkfont.nametofont("TkTextFont").configure(family="DejaVu Sans", size=12)
    tkfont.nametofont("TkFixedFont").configure(family="DejaVu Sans Mono", size=12)

    # Optional: if you prefer a different thick font that's likely available
    # default_font.configure(family="Liberation Sans", size=12, weight="bold")
    # or
    # default_font.configure(family="Arial", size=12, weight="bold")
    # ===========================

    app = MergeGUI(root)
    root.mainloop()
