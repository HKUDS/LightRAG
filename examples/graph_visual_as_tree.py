import json
import os
import asyncio
import re
import sys
from typing import Dict, Any
import xml.etree.ElementTree as ET

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Current working directory:", os.getcwd())
WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    return await openai_complete_if_cache(
        "deepseek-chat",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="yourkey",
        base_url="https://api.deepseek.com",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="BAAI/bge-m3",
        api_key="yourkey",
        base_url="https://api.siliconflow.cn/v1",
    )


async def get_embedding_dim() -> int:
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    return embedding.shape[1]


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """从响应中提取有效的JSON内容"""
    # 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试去除Markdown代码块
    cleaned = re.sub(r"```(json)?|```", "", response).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 尝试提取第一个{...}之间的内容
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 最终尝试：去除所有可能的非JSON内容
    lines = []
    in_json = False
    for line in cleaned.split("\n"):
        if line.strip().startswith("{") or in_json:
            in_json = True
            lines.append(line)
        if line.strip().endswith("}"):
            break
    final_attempt = "\n".join(lines)
    try:
        return json.loads(final_attempt)
    except json.JSONDecodeError as e:
        raise ValueError(f"无法从响应中提取有效JSON: {e}\n原始响应:\n{response}")


def parse_graphml_nodes(graphml_file: str) -> Dict[str, Dict]:
    """解析GraphML文件中的节点信息"""
    tree = ET.parse(graphml_file)
    root = tree.getroot()

    # 定义XML命名空间
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

    nodes = {}
    for node in root.findall(".//g:node", ns):
        node_id = node.get("id")
        data_elements = node.findall(".//g:data", ns)

        node_data = {}
        for data in data_elements:
            key = data.get("key")
            text = data.text.strip() if data.text else ""
            node_data[key] = text

        nodes[node_id] = {
            "id": node_id,
            "label": node_data.get("d0", ""),  # 假设d0是标签键
            "description": node_data.get("d1", ""),  # 假设d1是描述键
            "type": node_data.get("d2", "concept"),  # 假设d2是类型键
        }

    return nodes


async def initialize_rag() -> LightRAG:
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # 首先解析GraphML文件中的节点
        graphml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graphml_file):
            raise FileNotFoundError(f"GraphML文件不存在: {graphml_file}")

        nodes = parse_graphml_nodes(graphml_file)
        print(f"从GraphML文件中解析出 {len(nodes)} 个节点")

        rag = await initialize_rag()
        custom_prompt = """
        你是一个严格的JSON生成器，并且将其中的英文翻译成中文，必须基于以下节点信息生成树状图结构：
        现有结点信息：
        {context_data}

        {{
          "id": "根节点名称",
          "entity_type": "类型",
          "description": "描述",
          "source_id": "来源标记",
          "style": {{"fill": "颜色代码"}},
          "children": []
        }}

        规则：
        1. 必须基于提供的节点信息
        2. 直接以{{开头，以}}结尾
        3. 不要有任何非JSON内容
        """

        response = await rag.aquery(
            "以python为根结点再用我给你的所有结点至少使用20个和python有关的结点无关的抛弃掉，生成一个json格式的树状图",
            param=QueryParam(mode="hybrid"),
            system_prompt=custom_prompt,
        )
        print("原始响应：\n", response)

        try:
            data = extract_json_from_response(response)
            with open("tree_structure4.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("树状图JSON文件保存成功！")
            # 替换原来的HTML模板生成部分为以下代码：
            html_template = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>G6 Graph Visualization</title>
              <script src="https://gw.alipayobjects.com/os/antv/pkg/_antv.g6-3.7.1/dist/g6.min.js"></script>
              <style>
                #graph-container {
                  width: 100%;
                  height: 100vh;
                  border: 1px solid #ddd;
                }
                body {
                  margin: 0;
                  padding: 0;
                  overflow: hidden;
                }
                .g6-tooltip {
                  padding: 10px;
                  background-color: rgba(255, 255, 255, 0.9);
                  border: 1px solid #e2e2e2;
                  border-radius: 4px;
                  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
                }
              </style>
            </head>
            <body>
              <div id="graph-container"></div>

              <script>
                // Initialize the graph
                const graph = new G6.TreeGraph({
                  container: 'graph-container',
                  width: window.innerWidth,
                  height: window.innerHeight,
                  pixelRatio: 2,
                  plugins: [],
                  modes: {
                    default: [
                      {
                        type: 'collapse-expand',
                        onChange: function onChange(item, collapsed) {
                          const data = item.get('model').data;
                          data.collapsed = collapsed;
                          return true;
                        }
                      },
                      'drag-canvas',
                      'zoom-canvas',
                      'drag-node'
                    ]
                  },
                  layout: {
                    type: 'compactBox',
                    direction: 'LR',
                    getId: function getId(d) {
                      return d.id;
                    },
                    getHeight: function getHeight() {
                      return 16;
                    },
                    getWidth: function getWidth() {
                      return 16;
                    },
                    getVGap: function getVGap() {
                      return 22;
                    },
                    getHGap: function getHGap() {
                      return 100;
                    }
                  },
                  defaultNode: {
                    size: [50, 26],
                    style: {
                      fill: '#40a9ff',
                      stroke: '#096dd9'
                    },
                    labelCfg: {
                      position: 'center',
                    }
                  },
                  defaultEdge: {
                    shape: 'cubic-horizontal',
                    style: {
                      stroke: '#A3B1BF'
                    }
                  }
                });

                // Custom node style
                graph.node(function(node) {
                  return {
                    size: [50, 26],
                    style: {
                      fill: '#40a9ff',
                      stroke: '#096dd9'
                    },
                    label: node.id,
                    labelCfg: {
                      position: 'center',
                    }
                  };
                });

                graph.edge(function (edge) {
                  return {
                    shape: 'cubic-horizontal',
                    style: {
                      stroke: '#A3B1BF',
                      lineWidth: 2
                    }
                  };
                });

                // Create tooltip
                const tooltip = new G6.Tooltip({
                  offsetX: 10,
                  offsetY: 10,
                  itemTypes: ['node'],
                  getContent: (e) => {
                    const model = e.item.getModel();
                    if (model.description) {
                      return `<div class="g6-tooltip">
                        <h4>${model.id}</h4>
                        <p>${model.description}</p>
                      </div>`;
                    }
                    return null;
                  },
                  shouldBegin: (e) => {
                    const model = e.item.getModel();
                    return model.description !== undefined;
                  }
                });
                graph.addPlugin(tooltip);

                // Handle window resize
                window.addEventListener('resize', function() {
                  graph.changeSize(window.innerWidth, window.innerHeight);
                  graph.fitView();
                });

                // Load JSON data
                fetch('tree_structure4.json')
                  .then(response => response.json())
                  .then(data => {
                    graph.data(data);
                    graph.render();
                    graph.fitView();
                  })
                  .catch(error => {
                    console.error('Error loading JSON file:', error);
                    // Fallback data if JSON fails to load
                    const fallbackData = {
                      id: "root",
                      description: "Root node description",
                      children: [
                        { id: "child1", description: "First child node" },
                        { id: "child2", description: "Second child node" }
                      ]
                    };
                    graph.data(fallbackData);
                    graph.render();
                    graph.fitView();
                  });
              </script>
            </body>
            </html>
            """

            with open("tree_visualization.html", "w", encoding="utf-8") as f:
                f.write(html_template)
            print("树状图可视化HTML文件已生成：tree_visualization.html")

        except ValueError as e:
            print(f"JSON处理失败: {e}")
            with open("failed_response.txt", "w", encoding="utf-8") as f:
                f.write(response)

    except Exception as e:
        print(f"程序发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
