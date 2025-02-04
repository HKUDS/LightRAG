# 3D GraphML Viewer

一个基于 Dear ImGui 和 ModernGL 的交互式 3D 图可视化工具。

## 功能特点

- **3D 交互式可视化**: 使用 ModernGL 实现高性能的 3D 图形渲染
- **多种布局算法**: 支持多种图布局方式
  - Spring 布局
  - Circular 布局
  - Shell 布局
  - Random 布局
- **社区检测**: 支持图社区结构的自动检测和可视化
- **交互控制**:
  - WASD + QE 键控制相机移动
  - 鼠标右键拖拽控制视角
  - 节点选择和高亮
  - 可调节节点大小和边宽度
  - 可控制标签显示
  - 可在节点的Connections间快速跳转
- **社区检测**: 支持图社区结构的自动检测和可视化
- **交互控制**:
  - WASD + QE 键控制相机移动
  - 鼠标右键拖拽控制视角
  - 节点选择和高亮
  - 可调节节点大小和边宽度
  - 可控制标签显示

## 技术栈

- **imgui_bundle**: 用户界面
- **ModernGL**: OpenGL 图形渲染
- **NetworkX**: 图数据结构和算法
- **NumPy**: 数值计算
- **community**: 社区检测

## 使用方法

1. **启动程序**:
   ```bash
   python -m pip install -r requirements.txt
   python graph_visualizer.py
   ```

2. **加载字体**:
   - 将中文字体文件 `font.ttf` 放置在 `assets` 目录下
   - 或者修改 `CUSTOM_FONT` 常量来使用其他字体文件

3. **加载图文件**:
   - 点击界面上的 "Load GraphML" 按钮
   - 选择 GraphML 格式的图文件

4. **交互控制**:
   - **相机移动**:
     - W: 前进
     - S: 后退
     - A: 左移
     - D: 右移
     - Q: 上升
     - E: 下降
   - **视角控制**:
     - 按住鼠标右键拖动来旋转视角
   - **节点交互**:
     - 鼠标悬停可高亮节点
     - 点击可选中节点

5. **可视化设置**:
   - 可通过 UI 控制面板调整:
     - 布局类型
     - 节点大小
     - 边的宽度
     - 标签显示
     - 标签大小
     - 背景颜色

## 自定义设置

- **节点缩放**: 通过 `node_scale` 参数调整节点大小
- **边宽度**: 通过 `edge_width` 参数调整边的宽度
- **标签显示**: 可通过 `show_labels` 开关标签显示
- **标签大小**: 使用 `label_size` 调整标签大小
- **标签颜色**: 通过 `label_color` 设置标签颜色
- **视距控制**: 使用 `label_culling_distance` 控制标签显示的最大距离

## 性能优化

- 使用 ModernGL 进行高效的图形渲染
- 视距裁剪优化标签显示
- 社区检测算法优化大规模图的可视化效果

## 系统要求

- Python 3.10+
- OpenGL 3.3+ 兼容的显卡
- 支持的操作系统：Windows/Linux/MacOS
