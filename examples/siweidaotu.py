from pyecharts import options as opts
from pyecharts.charts import Tree
import markdown
import json


# 将 Markdown 转换为 JSON
def markdown_to_tree(markdown_text):
    lines = markdown_text.strip().split("\n")
    stack = []
    root = {"name": "root", "children": []}
    stack.append((0, root))

    for line in lines:
        depth = line.count("  ")  # 计算缩进
        content = line.strip("- ").strip()

        node = {"name": content, "children": []}

        while stack and stack[-1][0] >= depth:
            stack.pop()

        stack[-1][1]["children"].append(node)
        stack.append((depth, node))

    return root["children"][0]


# Markdown 输入
markdown_text = """
# 中心主题
- 一级主题 1
  - 二级主题 1.1
  - 二级主题 1.2
- 一级主题 2
  - 二级主题 2.1
  - 二级主题 2.2
"""

# 转换为树形结构
tree_data = markdown_to_tree(markdown_text)

# 用 ECharts 绘制思维导图
def draw_tree(data):
    chart = Tree(init_opts=opts.InitOpts(width="800px", height="600px"))
    chart.add(
        "",
        [data],
        orient="TB",
        label_opts=opts.LabelOpts(position="top", font_size=12),
    )
    chart.set_global_opts(title_opts=opts.TitleOpts(title="思维导图"))
    chart.render("mindmap.html")
    print("思维导图已生成，查看文件：mindmap.html")


# 生成思维导图
draw_tree(tree_data)
