# PRD：面向查询用户的独立 WebUI 入口

- 状态：设计草案（2026-08-24）
- 适用范围：LightRAG Server（`lightrag/api/`）+ WebUI（`lightrag_webui/`）
- 产品入口：日常查询端点 `/workspace`；后台管理端点 `/webui`
- 关键约束：查询入口只提供知识库问答，不暴露查询参数、文档管理或知识图谱管理能力；复用 `ChatMessage.tsx` 及从现有查询页抽出的共享会话能力，不复用后台页面壳；完整支持移动端；欢迎内容和 Logo 通过只读、多语言 UI Bundle 定制

## TL;DR

新增与 `/webui` 并存的 `/workspace` WebUI 入口。同一份前端构建产物根据服务端注入的入口模式渲染两种应用壳：`admin` 模式保留现有文档、知识图谱和查询页面，`workspace` 模式渲染独立的 `WorkspaceQueryView`。两个查询页面共同复用 `ChatMessage.tsx` 和抽出的查询会话、消息滚动、输入操作能力，但分别拥有自己的页面布局。访问 `/workspace` 的未登录用户先看到可定制欢迎页，登录后返回 `/workspace`；访问 `/webui` 的用户登录后仍返回 `/webui`。工作区查询空白页中央显示可定制 Logo 和欢迎词，返回查询内容后 Logo 和欢迎词消失。定制内容以多语言 UI Bundle 提供：构建产物携带内置默认 Bundle，生产环境可通过 `UI_TEMPLATES_DIR` 指向外部只读 Bundle；Server 启动时完整校验并生成不可变快照，前端通过公开 API 按语言读取内容和带内容哈希的资源 URL。

> **术语边界**：本文件中的 `/workspace` 只是“面向日常查询用户的 WebUI URL”，不代表 [服务端多工作空间方案](./LR2-multi-workspace-phase1.md) 中的知识库 `workspace` ID，也不新增知识库选择、数据隔离或多租户语义。后续若同时启用多工作空间能力，二者应分别命名为“查询入口”和“知识库工作区”，避免在代码和文案中混用。

---

## 1. 背景与问题

当前 WebUI 统一挂载在 `/webui`，登录后进入包含以下能力的后台界面：

- 文档上传、扫描、状态查看和删除；
- 知识图谱浏览与编辑；
- 知识库查询及完整查询参数配置。

普通知识库用户的主要任务只是提问和阅读答案。现有界面会向这类用户暴露大量无关操作，增加学习成本，也容易让“日常使用入口”和“知识库维护入口”混在一起。与此同时，当前登录成功、认证失效后的跳转固定指向应用根路由，不能可靠保留用户原本进入的是查询入口还是后台入口。

因此需要在不复制查询实现、不破坏现有 `/webui` 的前提下，增加一个职责单一、可品牌定制、适合手机使用的查询入口。

## 2. 产品目标与成功标准

### 2.1 目标

1. 提供固定的日常查询入口 `/workspace`，用户登录后只看到知识库查询页面。
2. 将现有 `/webui` 明确定位为后台管理入口，并保持其现有功能和直接链接可用。
3. 新增独立的 `WorkspaceQueryView` 页面，直接复用 `ChatMessage.tsx`，并从 `RetrievalView.tsx` 抽取共享的查询会话、流式响应、停止、复制、清空、历史记录、输入和滚动跟随能力，不维护第二套查询状态机。
4. 查询入口不显示、也不接受面向用户的查询参数覆盖；查询请求沿用同一浏览器中 `/webui` 保存的查询配置，未配置时使用现有前端默认值。
5. 登录、退出和认证失效流程保留入口来源，登录后回到原入口。
6. 未登录的查询用户先看到可定制欢迎页；查询空白态显示可定制 Logo 和欢迎词；返回查询内容后 Logo 和欢迎词消失，仅显示查询历史。
7. 在主流手机宽度和软键盘场景下完成提问、阅读流式答案、复制、停止和清空操作。
8. 根路径 `/` 的现有默认跳转继续使用 `/webui`，可以通过环境变量改为 `/workspace`。
9. 欢迎页和查询空白态支持当前 WebUI 的全部语言，并允许部署方在不重建前端的情况下替换整套多语言文案与品牌资源。

### 2.2 成功标准

- 新用户从 `/workspace` 进入后，无需理解文档、图谱或 RAG 参数即可完成首次提问。
- `/workspace` 页面不存在文档管理、图谱管理、API 文档入口和查询参数侧栏。
- 从 `/workspace` 发起登录的用户登录后返回 `/workspace`；从 `/webui` 发起登录的用户登录后返回 `/webui`。
- 在 320 px 宽度下无页面级横向滚动，输入框和发送/停止按钮始终可操作，软键盘弹出后当前输入区仍可见。
- 未设置 `UI_TEMPLATES_DIR` 时使用内置默认 Bundle；显式设置的外部 Bundle 通过完整校验后才能启动，避免定制内容部分生效而造成品牌或语言混用。

## 3. 非目标

- 本期不新增用户、角色、注册、找回密码、SSO 或权限管理能力。
- 本期不把 `/webui` 的“后台管理”定位当作新的安全边界。是否允许调用文档、图谱或查询 API，仍由服务端现有认证/授权机制决定；仅隐藏前端入口不能代替 API 授权。
- 本期不实现知识库工作区选择、多工作空间路由、多租户或按用户隔离知识库。
- 本期不直接复用整个 `RetrievalView.tsx` 作为工作区页面，也不复制或重写其中的查询状态机；只把可复用逻辑抽成两个页面共同依赖的模块。
- 本期不改变 `/query`、`/query/stream` 的公开 API 契约。
- 本期不允许最终用户在 `/workspace` 中设置 query mode、Top K、Token Budget、rerank、stream、user prompt 等参数；这些参数沿用同一浏览器配置中 `/webui` 保存的值，未配置时使用现有前端默认值。
- 本期不要求模板热重载；修改模板文件或 Logo 后重启 Server 生效。
- 本期不允许通过模板注入任意 HTML、Jinja、JavaScript、CSS 或替换系统控制项。

## 4. 用户角色与核心场景

| 角色 | 主要入口 | 核心任务 | 页面能力 |
| --- | --- | --- | --- |
| 查询用户 | `/workspace` | 提问、阅读答案、复制内容、停止生成、清空会话 | 仅查询 |
| 知识库管理员 | `/webui` | 文档管理、图谱管理、查询调试 | 保持现有后台能力 |
| 运维人员 | Server 配置 | 定义欢迎页、空白态 Logo 和欢迎词、根路径默认跳转页 | 文件和环境变量配置，无管理 UI |

核心用户故事：

1. 作为查询用户，我访问 `/workspace` 时先看到产品介绍和登录入口，登录后直接进入查询页面。
2. 作为回访用户，我持有有效登录态时访问 `/workspace`，无需再次经过欢迎页或登录页。
3. 作为管理员，我从 `/webui` 登录后仍进入后台，而不会被送到查询入口。
4. 作为手机用户，我可以在单手可操作的页面中输入多行问题、发送、停止生成并阅读长答案。
5. 作为运维人员，我可以通过文件替换欢迎页文案、空白态文案和 Logo，并通过环境变量选择根路径的默认入口，而不修改或重新构建前端源码。

## 5. 信息架构与路由契约

### 5.1 服务端入口

| URL | 定位 | 有有效登录态 | 无有效登录态 |
| --- | --- | --- | --- |
| `/workspace` | 查询入口规范化地址 | 308/307 到 `/workspace/` | 308/307 到 `/workspace/` |
| `/workspace/` | 查询应用 | 查询主界面 | 查询欢迎页 |
| `/webui` | 后台入口规范化地址 | 保持现有尾斜杠处理 | 保持现有尾斜杠处理 |
| `/webui/` | 后台应用 | 现有后台主界面 | 现有登录页 |
| `/` | 可配置的默认入口 | 按 `WEBUI_DEFAULT_ENTRY` 跳转 | 按 `WEBUI_DEFAULT_ENTRY` 跳转 |

本 PRD 将“未登录用户跳转欢迎页”限定为新增的查询入口 `/workspace`。后台 `/webui` 是面向明确知道后台地址的管理员入口，继续直接显示登录页，避免为既有管理流程增加一次无意义点击。

根路径的唯一配置项为 `WEBUI_DEFAULT_ENTRY`：

- 允许值仅为 `webui` 或 `workspace`，默认值为 `webui`，因此升级后 `/` 仍跳转 `/webui/`。
- 配置为 `workspace` 时，`/` 跳转 `/workspace/`，再由查询入口按登录态显示查询主界面或欢迎页。
- 配置值是枚举而不是 URL；非法值应在启动配置校验时明确报错，不得原样拼接为重定向目标。
- 重定向必须带上实际 `root_path`，且 `WEBUI_DEFAULT_ENTRY` 不控制两个入口是否挂载。

两个入口必须同时支持 `LIGHTRAG_API_PREFIX` 和反向代理 `root_path`。任何重定向、运行时配置、静态资源地址和登录回跳都不得绕过部署前缀。例如前缀为 `/site01` 时，浏览器可见入口应为 `/site01/workspace/` 和 `/site01/webui/`。

### 5.2 单构建产物、双挂载

前端只生成一份静态构建产物，Server 将同一目录分别挂载到 `/webui` 和 `/workspace`。每个 HTML 响应注入该挂载点自己的运行时配置：

```ts
type UIMode = 'admin' | 'workspace'

interface RuntimeConfig {
  apiPrefix: string
  uiPrefix: string       // 当前入口的完整前缀，尾部带 /
  uiMode: UIMode
  webuiPrefix: string    // 兼容现有 /webui 使用者，迁移期保留
}
```

要求：

- `uiMode='admin'` 渲染现有 `App`；`uiMode='workspace'` 渲染精简的查询应用壳。
- 前端不得通过 `window.location.pathname.includes(...)` 猜入口模式；`uiMode` 是唯一权威来源。
- 静态资源继续使用内容哈希和长期缓存；两个入口的 `index.html` 均不得缓存，并分别注入正确的 `uiPrefix`。
- 不复制 `webui` 构建目录，不增加第二套 Vite 构建流程。
- WebUI 资源未打包时，`/workspace` 与 `/webui` 遵循同一降级原则：API 文档可用则跳转 API 文档，否则返回现有服务信息 JSON；不得出现一个入口可用、另一个入口静默 404 的状态。

### 5.3 前端路由

`HashRouter` 可继续使用，但路由守卫必须感知当前 `uiMode`：

| 模式 | SPA 路由 | 说明 |
| --- | --- | --- |
| `workspace` | `#/welcome` | 公开欢迎页 |
| `workspace` | `#/login` | 登录页，携带可信回跳目标 |
| `workspace` | `#/` | 受保护的查询主界面 |
| `admin` | `#/login` | 现有后台登录页 |
| `admin` | `#/` | 现有后台主界面 |

未知 hash 路由应回到当前入口的默认页，不得跨入口跳转。

## 6. 登录与回跳

### 6.1 “从哪里来，回哪里去”

回跳目标至少包含入口类型，不再把所有成功路径写死为 `navigate('/')`：

- 从 `/workspace` 欢迎页进入登录页，成功后返回 `/workspace/#/`。
- 从 `/webui` 进入登录页，成功后返回 `/webui/#/`。
- 已登录用户直接访问任一入口时留在该入口。
- 日常查询端点发生 token 失效时，保存 `/workspace` 为回跳入口并进入欢迎页；用户重新登录后回到查询页面。
- 后台管理端点发生 token 失效时，保存 `/webui` 为回跳入口并进入后台登录页。
- 退出登录后回到当前入口的未登录默认页：查询入口回欢迎页，后台入口回登录页。

### 6.2 回跳目标安全规则

回跳参数不能形成开放重定向：

1. 只接受当前部署 `root_path` 下的 `/workspace/`、`/webui/` 及其已注册内部 hash 路由。
2. 拒绝带 scheme、host、`//`、反斜杠、编码后路径穿越或未知入口的值。
3. 无有效回跳目标时，按当前 `uiMode` 回到其默认页。
4. 回跳状态使用结构化值保存，例如 `{ mode, route }`；不得把未经解析的完整 URL 直接传给 `window.location`。
5. 登录请求失败不消费回跳目标；用户重试成功后仍回到原入口。

### 6.3 认证未启用时

当 Server 未配置账号认证时：

- 首次访问 `/workspace` 仍显示欢迎页，主按钮文案为“进入工作区”而不是“登录”。
- 点击后沿用现有 `/auth-status` guest token 流程，成功后进入查询主界面。
- `/webui` 保持现有自动 guest 登录行为。
- guest token 续期和 401 重试沿用现有 API 客户端逻辑，但最终导航必须保留当前入口。

## 7. 查询主界面

### 7.1 页面组成

`/workspace` 登录后的可见界面只包含：

- 精简页头：品牌标识，标题，状态与设置（用户名、主题、语言、退出）；
- 查询消息区域；
- 清空、问题输入、发送/停止操作；
- 流式进度、答案、引用、复制和回到底部等现有查询反馈。

不得显示：

- 文档、知识图谱或 API 文档导航；
- `QuerySettings` 侧栏或其移动端抽屉；
- Query mode、Top K、Token Budget、rerank、stream、history turns、user prompt 等控件；
- 后台健康状态、数据维护按钮或图谱编辑入口。

### 7.2 页面与复用边界

工作区不直接渲染或给 `RetrievalView` 增加大范围 `variant` 分支，而是新增独立的 `WorkspaceQueryView`。原因是两者的页面职责已经不同：后台查询页包含参数侧栏并服从 Tab 生命周期，工作区查询页需要独立的空白态、页头、移动端布局和始终活跃的消息区域。把这些差异全部塞入 `RetrievalView` 会让一个已超过千行的组件继续承担两套页面布局。

但 `ChatMessage.tsx` 只负责一条消息的展示，不能单独构成完整复用边界。现有查询页中的以下能力仍必须抽出并由两个页面共享：

| 共享层 | 职责 | 不应包含 |
| --- | --- | --- |
| 查询会话控制层（hook/controller） | 消息状态、历史加载与持久化、请求参数序列化、流式增量、COT/LaTeX 完整性、进度、计时、停止与清理 | 页面布局、后台 Tab、参数控件 |
| 消息列表层 | 遍历消息、调用 `ChatMessage`、复制按钮、滚动跟随、回到底部、文本选择行为 | 查询参数、欢迎页、页头 |
| 输入操作层 | 单/多行输入切换、草稿、清空、发送/停止及冷却 | `QuerySettings`、后台导航 |
| `ChatMessage.tsx` | 单条用户/助手消息及 Markdown、COT、LaTeX、Mermaid、计时和进度展示 | 请求发起、历史、输入框、列表滚动 |

组合关系：

```text
RetrievalView (后台布局) ─┐
                          ├─ shared query session
WorkspaceQueryView ──────┘   ├─ message list ── ChatMessage
                              └─ query composer

RetrievalView ─────────────── QuerySettings
WorkspaceQueryView ────────── WorkspaceEmptyState
```

具体文件名可以在实现时调整，但依赖方向必须保持：共享层不得 import `RetrievalView`、`WorkspaceQueryView`、`QuerySettings` 或后台 Tab 状态。

- 当前由 `ChatMessage.tsx` 导出的 `MessageWithError` 属于两个页面和会话控制层共同使用的领域模型，应迁移到独立的 retrieval types 模块，再由 `ChatMessage` 和共享层共同 import；不能让无 UI 的会话控制层反向依赖消息渲染组件。
- 查询会话控制层通过显式策略接收“是否允许 mode 前缀”等页面差异，不得自行读取 `uiMode` 或后台 `currentTab`；这样共享状态机保持单一，入口差异仍由页面组合层决定。
- `RetrievalView` 保持后台页面定位，继续渲染当前 `QuerySettings`，继续支持现有查询参数和 `/mode` 输入前缀。
- `WorkspaceQueryView` 不渲染 `QuerySettings`，也不解析 `/naive`、`/local`、`/global`、`/hybrid`、`/mix`、`/bypass` 等参数覆盖前缀；以 `/` 开头的普通问题按普通文本提交。
- `WorkspaceQueryView` 始终把消息区域视为活跃；不得读取后台 `currentTab` 决定 Markdown、Mermaid 或动画是否更新。
- 两个页面必须通过同一个请求序列化和流式状态机发起查询，不能各自维护一份近似实现。

### 7.3 工作区查询请求

工作区不提供参数编辑界面，但使用与 `/webui` 相同的持久化 `querySettings`。查询会话控制层从当前浏览器的 settings store 读取一次一致快照，经同一个 serializer 转换为 API 请求：

```json
{
  "query": "用户的问题",
  "mode": "mix",
  "top_k": 40,
  "chunk_top_k": 20,
  "stream": true,
  "include_progress": true
}
```

上例只用于说明参数来源，不固定具体数值。规则如下：

- 同一浏览器 profile 中，管理员在 `/webui` 保存的新参数从下一次 `/workspace` 查询开始生效；工作区本身只读这些参数。
- 从未访问或配置过 `/webui` 时，使用现有 settings store 的前端默认值。
- 这是浏览器本地配置共享，不是 Server 全局配置，也不跨设备同步。若未来需要运维统一控制所有查询用户的参数，应另行设计服务端查询 profile，不能把 localStorage 描述成全局策略。
- serializer 只发送 `QueryRequest` 接受的字段；`history_turns` 等纯前端字段应转换为相应的 `conversation_history`，不得作为未知字段原样发送。
- 工作区禁止通过输入前缀临时覆盖 mode，但不篡改或重置 `/webui` 已保存的 mode。
- stream 继续决定调用 `/query` 或 `/query/stream`；无论选择哪一条，两个页面必须走相同的请求和错误处理路径。

本期保持现有查询会话语义，不借本功能改变 conversation history、答案缓存或引用格式。

### 7.4 查询历史

- 沿用现有按浏览器保存的查询历史和“切换登录用户时清空历史”规则。
- `/workspace` 与 `/webui` 的查询页面可共享同一用户的消息历史，以便管理员和查询入口之间延续同一会话展示。
- 共享消息历史和浏览器级查询参数；工作区请求始终遵循 §7.3，且不提供参数编辑入口。
- token 失效、欢迎页和重新登录过程不得提前清空当前用户历史；如果登录成不同用户，则沿用现有规则清空。

## 8. 欢迎页与空白态定制

### 8.1 总体方案与唯一配置入口

定制内容采用“内置默认 UI Bundle + 外部完整覆盖 Bundle”的模型，而不是在 Server 启动时把客户文件复制到或覆盖前端构建目录。

| 项目 | 位置/配置 | 说明 |
| --- | --- | --- |
| 内置默认源文件 | `lightrag_webui/public/ui_defaults/` | 随前端源码维护，覆盖当前 WebUI 全部语言 |
| 内置默认构建产物 | `lightrag/api/webui/ui_defaults/` | 由 Vite 构建自动复制并随 PyPI/容器发布；Server 不修改该目录 |
| 生产定制目录 | `UI_TEMPLATES_DIR` | 可选，指向打包目录之外的完整 UI Bundle；建议以只读卷挂载 |

生产部署示例：

```yaml
services:
  lightrag:
    environment:
      UI_TEMPLATES_DIR: /app/ui_templates
    volumes:
      - ./ui_templates:/app/ui_templates:ro
```

只保留一个 `UI_TEMPLATES_DIR` 配置，不再分别提供欢迎页、空白态和 Logo 的三个文件变量。这样可以一次校验并原子切换一整套品牌内容，也能让多语言文件和其资源保持一致。

“模板”在本期指受限 Markdown 内容，不是可执行的 Jinja/JS/HTML 模板。系统拥有页面框架、登录/进入按钮、查询输入框、语言选择和导航行为；Bundle 只能提供品牌资源及内容区域，不能覆盖这些产品控制项。

### 8.2 Bundle 目录与 manifest

内置和外部 Bundle 使用相同目录契约：

```text
ui_templates/
├── manifest.json
├── assets/
│   ├── logo.svg
│   └── logo-zh-TW.svg
└── locales/
    ├── en/
    │   ├── welcome.md
    │   └── query_empty.md
    ├── zh/
    │   ├── welcome.md
    │   └── query_empty.md
    └── zh-TW/
        ├── welcome.md
        └── query_empty.md
```

`manifest.json` 是目录内文件的唯一索引。建议的一期 Schema 如下：

```json
{
  "schema_version": 1,
  "default_locale": "en",
  "fallbacks": {
    "zh-HK": ["zh-TW", "en"]
  },
  "brand": {
    "logo": "assets/logo.svg"
  },
  "locales": {
    "en": {
      "welcome": "locales/en/welcome.md",
      "query_empty": "locales/en/query_empty.md",
      "logo_alt": "LightRAG"
    },
    "zh": {
      "welcome": "locales/zh/welcome.md",
      "query_empty": "locales/zh/query_empty.md",
      "logo_alt": "LightRAG"
    },
    "zh-TW": {
      "welcome": "locales/zh-TW/welcome.md",
      "query_empty": "locales/zh-TW/query_empty.md",
      "logo": "assets/logo-zh-TW.svg",
      "logo_alt": "LightRAG"
    }
  }
}
```

规则如下：

- Locale key 使用 BCP 47 风格的连字符形式，例如 `zh-TW`，不在 Bundle 中使用前端内部的 `zh_TW`；前端/Server 边界负责规范化。
- 每个声明的 locale 必须同时提供 `welcome`、`query_empty` 和非空 `logo_alt`。
- `brand.logo` 是所有语言共享的默认 Logo；locale 条目可选的 `logo` 覆盖该语言的 Logo，以满足不同语言使用不同品牌图的需求。
- `fallbacks` 是可选的显式有序映射。所有目标必须存在，且不能出现环。
- 未在 manifest 中引用的文件不对外提供，也不参与 revision 计算。
- 内置默认 Bundle 必须至少覆盖当前 WebUI 的 `en`、`zh`、`zh-TW`、`fr`、`ar`、`ru`、`ja`、`de`、`uk`、`ko`、`vi`。

`WEBUI_TITLE` 和 `WEBUI_DESCRIPTION` 继续是部署级、非本地化的站点标题和描述，由现有 Server 配置拥有；它们不在 manifest 中重复定义。欢迎正文、查询空白态正文和 `logo_alt` 由多语言 Bundle 提供。若未来需要本地化站点标题，应另行扩展 Schema，不能同时维护两个权威来源。

### 8.3 语言选择与回退

前端请求定制内容时按以下顺序确定目标语言：

1. 用户在 WebUI 中显式选择并持久化的语言；
2. 浏览器语言经现有支持语言表规范化后的结果；
3. manifest 的 `default_locale`。

Server 对目标语言按以下顺序解析模板：

1. 精确匹配 locale；
2. manifest 中该 locale 的显式 `fallbacks` 顺序；
3. 当前 Bundle 的 `default_locale`。

不自动进行区域或书写系统推断，例如不能自行断言 `zh-HK → zh`，因为部署方可能希望它回退到 `zh-TW`。外部 Bundle 缺少用户所选语言时，必须回退到外部 Bundle 自己的默认语言，不能逐字段混入 LightRAG 内置 Bundle，否则同一页面可能出现客户 Logo、LightRAG 文案或语言错配。

阿拉伯语等 RTL 布局方向从系统维护的可信 locale 注册表推导，不允许模板提供 CSS 或任意 `dir` 值。

### 8.4 Server 启动快照与失败语义

Server 启动时构造不可变的 `UICustomizationSnapshot`：

1. 始终加载并校验随构建发布的内置默认 Bundle，内置 Bundle 错误属于发布缺陷，应阻止启动。
2. 未设置 `UI_TEMPLATES_DIR` 时，直接激活内置 Bundle 快照。
3. 显式设置 `UI_TEMPLATES_DIR` 时，从该目录读取并完整校验一个外部 Bundle；全部通过后再原子激活它。
4. 外部 Bundle 的 manifest、任一已声明 locale 或任一被引用资源缺失、不可读、超限或格式非法时，Server 启动失败并给出可操作但不泄露敏感内容的错误；不得悄悄改用默认 Bundle。
5. 不把外部文件复制到 `lightrag/api/webui/ui_defaults/`，也不修改 Python 包、容器镜像或前端构建产物。

启动快照包含已解析的文案、资源字节、MIME、内容哈希、locale revision 和 bundle revision。请求阶段不重新读取磁盘，多个 worker 各自从同一只读目录构造相同快照。修改 `UI_TEMPLATES_DIR` 中的内容后需要重启所有 Server worker；本期不做文件监听或热重载。

模板文件单个限制为 64 KiB，Logo 单个限制为 2 MiB。具体限制应集中定义并在运维文档中说明。

### 8.5 公开读取 API

欢迎页出现在登录前，因此定制内容必须通过无需登录但严格限域的公开 API 提供，而不是放入认证后的通用配置接口：

```http
GET /ui/customization?locale=zh-TW
GET /ui/customization/assets/{asset_hash}/{asset_id}
```

两个端点必须感知 `LIGHTRAG_API_PREFIX` 和代理 `root_path`。配置响应示例：

```json
{
  "requested_locale": "zh-TW",
  "locale": "zh-TW",
  "fallback_used": false,
  "direction": "ltr",
  "revision": "sha256:...",
  "brand": {
    "title": "LightRAG",
    "description": "Simple and Fast RAG",
    "logo_url": "/ui/customization/assets/8c1f.../brand-logo",
    "logo_alt": "LightRAG"
  },
  "welcome": {
    "format": "markdown",
    "content": "..."
  },
  "query_empty": {
    "format": "markdown",
    "content": "..."
  }
}
```

其中 `brand.title` 和 `brand.description` 只映射现有 `WEBUI_TITLE` / `WEBUI_DESCRIPTION`，manifest 不能覆盖它们。API 不返回 Bundle 根目录、绝对文件路径或其它 Server 配置。

### 8.6 Revision、资源标识与缓存失效

缓存模型区分以下四个概念：

| 名称 | 作用 | 变化条件 |
| --- | --- | --- |
| `asset_id` | 稳定的语义标识，例如 `brand-logo` | 资源角色变化时才变，不作为缓存失效值 |
| `asset_hash` | 资源原始字节的 SHA-256 | Logo 等资源内容变化时改变，并进入 URL |
| `bundle_revision` | 整个已激活 Bundle 的确定性摘要 | 任一 manifest、文案或被引用资源变化时改变；用于健康状态和日志 |
| locale `revision` | 当前 locale API 表示的确定性摘要 | 该 locale 最终返回的文案、Logo、替代文本、方向或部署级标题/描述变化时改变 |

缓存规则：

- `GET /ui/customization` 返回 `Cache-Control: no-cache, must-revalidate` 和基于 locale `revision` 的 `ETag`，支持 `If-None-Match`/`304`。
- 资源响应使用包含 `asset_hash` 的 URL，并返回 `Cache-Control: public, max-age=31536000, immutable`；同一字节内容可永久复用。
- 只修改日语文案时，日语 locale revision 和 bundle revision 改变；中文 locale revision 及未变化的 Logo URL 保持不变。
- 修改 `WEBUI_TITLE` 或 `WEBUI_DESCRIPTION` 时，所有 locale revision 改变，但 bundle revision、asset hash 和 Logo URL 不变。
- 修改外部 `UI_TEMPLATES_DIR` 内容并重启后，Server 重新计算上述摘要；文案靠 ETag 失效，资源靠 URL 中的 `asset_hash` 失效。
- 修改 `lightrag_webui/public/ui_defaults/` 后，必须重新构建并部署 WebUI；新 Server 以相同算法加载新默认 Bundle，因此缓存失效机制完全相同。
- 前端和页面模板不得直接引用 `/ui_defaults/logo.svg` 等 Vite public 固定路径；否则文件名不带内容哈希，浏览器或 CDN 可能继续使用旧资源。所有可定制资源均经上述资源 API 访问。

`bundle_revision` 和激活来源（`builtin`/`custom`）可写入启动日志与健康信息，但不得暴露服务器目录。若配置中含 locale 列表，可公开返回语言标识，但不返回文件结构。

摘要计算必须跨进程、平台和重启保持确定性：manifest 先按约定进行规范化序列化，引用路径按规范化后的相对路径排序，再将路径、内容字节和必要的响应元数据纳入 SHA-256；locale `revision` 对最终响应表示计算，并排除 `revision` 字段自身。多个 worker 对相同输入必须得到相同结果。

### 8.7 内容与资源安全

- manifest 使用严格 Schema，拒绝未知字段、错误类型、重复或非法 locale、fallback 环和不存在的目标。
- 所有相对路径必须解析后仍位于 Bundle 根目录内；拒绝绝对路径、`..` 穿越以及通过符号链接逃逸根目录。
- 只读取并公开 manifest 明确引用的文件。
- 由于 Vite 会把 `public/ui_defaults/` 复制到静态构建目录，Server 的静态文件处理必须显式拒绝直接访问 `ui_defaults/` 原始路径；内置内容也只能通过 customization API 暴露，避免绕过 manifest 与响应头策略。
- Markdown 禁用原始 HTML、脚本、iframe、表单和事件属性；链接只允许明确的安全 scheme，并为新窗口链接添加 `noopener noreferrer`。
- Logo 支持 PNG、JPEG、WebP 和 SVG，按实际内容校验 MIME，不只信任扩展名。
- 自定义 SVG 仅作为 `<img>` 资源加载，不以内联 DOM 注入；资源响应设置正确的 `Content-Type`、`X-Content-Type-Options: nosniff` 和限制性 CSP。
- 前端渲染仍需使用统一的受限 Markdown 组件。单个视图发生意外渲染错误时显示最小内置纯文本提示，不能让登录或查询页面白屏。
- Bundle 是公开展示内容，运维文档必须提示不得在其中放入密钥、内部路径或其它敏感信息。
- 公开 customization API 必须纳入路由和响应字段审计，不能演变为任意文件读取接口。

### 8.8 前端加载与切换行为

- 应用启动时并行请求认证状态和当前 locale 的 customization 配置，避免串行增加首屏时间。
- 用户切换语言时重新请求该 locale；新响应成功前保留当前完整内容，成功后一次性切换 Logo、替代文本和两处文案，不能逐字段闪烁。
- 定制快照仅缓存在当前页面内存中，不写入 localStorage，避免部署更新后长期残留旧内容。
- 配置请求暂时失败时可以保留本页面内最近一次成功快照并提供重试；首次加载失败时显示前端最小安全默认内容，登录和查询操作仍可使用。
- 主题或语言切换不能绕过 Server 返回的资源 URL，也不能直接拼接 Bundle 路径。

### 8.9 欢迎页行为

欢迎页至少包含：

- 当前 locale 解析后的 Logo、替代文本和欢迎 Markdown；
- 系统控制的主操作按钮；
- 主题和语言切换；
- 可访问的页面标题和键盘焦点顺序。

账号认证开启时，主按钮进入当前 `/workspace` 挂载点的登录页；认证关闭时，主按钮获取 guest token 并进入查询页。

### 8.10 查询空白态

当消息历史为空时，查询区域中央按“Logo + 欢迎词”垂直排列：

- Logo 保持宽高比，桌面端最大边 120 px，手机端最大边 88 px；不得挤压输入区。
- 欢迎词使用当前 locale 解析后的 `query_empty` 内容，内容居中但长段落保持可读行宽。
- 发送第一条问题后空白态消失；清空消息后重新出现。
- Logo 加载失败时隐藏破损图片并继续显示欢迎词。
- 后台 `/webui` 的查询空白态保持现有行为，除非后续另行决定统一品牌展示。

## 9. 移动端要求

### 9.1 支持范围

最低验收视口：

- 320 × 568；
- 375 × 667；
- 390 × 844；
- 768 × 1024。

同时覆盖 iOS Safari、Android Chrome 的触摸滚动和软键盘场景；桌面端继续支持 Chrome、Firefox、Safari 和 Edge 的当前版本。

### 9.2 布局与交互

1. 页面使用动态视口高度（如 `100dvh`）和 safe-area inset，不能依赖固定 `100vh` 导致输入区被浏览器工具栏或刘海遮挡。
2. 页面级无横向滚动。代码块、宽表格和长 URL 在消息内容内部横向滚动或换行，不撑宽整个页面。
3. 查询参数侧栏在工作区模式根本不挂载，而不是通过负坐标藏在屏幕外。
4. 输入区在软键盘弹出后仍可见；输入多行时高度有上限，消息区让出空间并保持可滚动。
5. 主要触控目标最小 44 × 44 CSS px；图标按钮必须有可访问名称。
6. 发送和停止按钮在同一稳定位置切换，避免重复点击误触；保留现有停止冷却行为。
7. 用户向上滚动或选择文本时暂停自动跟随；回到底部按钮不能遮挡输入区。
8. 消息气泡在窄屏下使用可用宽度，复制按钮不得把正文压缩到不可读。
9. 横竖屏切换不清空输入草稿、历史或正在显示的回答。

## 10. 可用性、错误与无障碍

- 欢迎页、登录页和查询页均提供明确的 loading 状态，初始化期间不闪现后台页面。
- 查询失败沿用现有消息内错误展示；401 与普通查询错误分流，不把认证失效渲染成模型回答失败。
- customization API 的临时请求或渲染失败只影响定制内容，不影响登录和查询；显式配置的外部 Bundle 若在启动校验阶段失败，则 Server 按 §8.4 fail-fast。
- 所有页面支持键盘导航和可见焦点；颜色对比满足 WCAG 2.1 AA 的常见文本要求。
- Logo 使用有意义的替代文本；纯装饰图标标记为隐藏。
- `prefers-reduced-motion` 下减少非必要动画。
- 当前 WebUI 的全部语言继续可选；Bundle 为每种语言提供已经翻译完成的内容，前端不对自定义 Markdown 做运行时机器翻译。缺少精确语言时严格使用 §8.3 的显式回退规则。

## 11. 兼容性与系统同步

### 11.1 兼容性

- `/webui` URL、后台功能、现有书签和静态资源构建流程保持可用。
- `RetrievalView` 继续只表示后台查询页；新增 `WorkspaceQueryView` 不改变现有调用方语义。
- 现有 token、API key、guest token 和登录接口契约不变。
- 当前 `WEBUI_TITLE` / `WEBUI_DESCRIPTION` 继续作为部署级、非本地化站点标题和描述，同时可供查询入口页头使用；欢迎页和空白态正文由 UI Bundle 拥有。
- 未设置 `UI_TEMPLATES_DIR` 的部署自动使用打包的 `ui_defaults`；升级无需新增配置。显式设置该变量的部署必须提供符合当前 Schema 的完整 Bundle。
- 根路径 `/` 默认仍跳转 `/webui`；只有显式配置 `WEBUI_DEFAULT_ENTRY=workspace` 时才跳转 `/workspace`。

### 11.2 必须同步的面

实现不能只新增一个前端路由，还必须同步：

| 面 | 要求 |
| --- | --- |
| Server mount | 同一静态目录挂载 `/webui` 与 `/workspace`，两者均支持 `root_path` |
| Runtime config | 每个 mount 注入唯一 `uiMode` 和正确 `uiPrefix` |
| 根路径/降级 | `WEBUI_DEFAULT_ENTRY` 默认 `webui`；选择 `workspace` 时保留 `root_path`；无资源时两个入口一致降级 |
| 健康状态 | 保留 `webui_available` 语义，并新增或明确 `workspace_available`；两者由同一资源检查结果派生，不能各自漂移 |
| UI 定制加载 | 从内置或 `UI_TEMPLATES_DIR` 构造一个只读快照；绝不修改 WebUI 构建目录 |
| 定制读取 API | 公开端点只返回当前 locale 内容和 manifest 引用资源，并正确处理前缀、ETag、内容哈希与安全响应头 |
| 启动日志 | 同时打印后台和查询入口的实际带前缀 URL，以及 customization 来源、bundle revision 和校验结果；不打印服务端目录 |
| 登录导航 | 所有登录成功、退出、401、guest token 更新路径保留入口来源 |
| 打包 | PyPI/容器仍只打包一份 `lightrag/api/webui` 产物，其中包含完整的内置 `ui_defaults` |
| 文档 | 更新 Server/WebUI 启动文档、`env.example`、Bundle Schema、Docker 只读挂载示例和多站点部署说明 |
| 路由审计 | 将 `/workspace` mount 和公开 customization/config 面加入完整路由/mount 清单 |

`workspace_available` 不应拥有独立构建开关。两个入口共用一份构建产物，所以可用性必须由一个资源检查结果派生；未来若增加关闭查询入口的产品开关，再单独设计其默认值和健康语义。

## 12. 验收标准

### 12.1 路由与认证矩阵

| 场景 | 操作 | 期望结果 |
| --- | --- | --- |
| 查询入口、未登录 | 打开 `/workspace/` | 显示欢迎页，不闪现后台或查询参数 |
| 查询入口、已登录 | 打开 `/workspace/` | 直接显示查询页 |
| 查询入口登录 | 欢迎页点击登录并成功 | 返回 `/workspace/#/` |
| 后台登录 | 打开 `/webui/` 并成功登录 | 返回 `/webui/#/` |
| 查询 token 失效 | 查询接口返回不可续期 401 | 保存查询入口，进入欢迎/登录流程；重登后回查询页 |
| 后台 token 失效 | 后台接口返回不可续期 401 | 回后台登录；重登后回后台 |
| 认证关闭 | 首次访问 `/workspace/` 并点击进入 | 获取 guest token 后进入查询页 |
| 恶意回跳 | 提交外部 URL 或路径穿越 return target | 拒绝该值并回当前模式默认页 |
| API 前缀 | 在 `/site01` 下完成以上流程 | 所有资源和跳转保留 `/site01` |

### 12.2 查询功能

- `/workspace` 只渲染独立的 `WorkspaceQueryView`，不挂载 `RetrievalView` 或 `QuerySettings`。
- `RetrievalView` 与 `WorkspaceQueryView` 复用同一个查询会话控制层、消息列表、输入操作层和 `ChatMessage`，不存在复制的流式/停止/历史状态机。
- 不存在 `QuerySettings` DOM、后台 Tab、文档/图谱入口或隐藏移动端参数抽屉。
- 输入 `/mix what is RAG` 时整段作为普通问题提交，不覆盖 mode。
- 请求携带当前浏览器中 `/webui` 保存的合法 `top_k`、`mode`、`user_prompt` 等参数；工作区不能编辑或临时覆盖它们。
- 流式答案、停止、复制、清空、引用、思考区、公式、Mermaid、滚动跟随与现有后台查询页功能一致。
- 后台查询页仍显示查询参数，并保持现有 `/mode` 前缀行为。
- `WEBUI_DEFAULT_ENTRY` 未设置时 `/` 仍进入 `/webui/`；设置为 `workspace` 时进入 `/workspace/`；非法值启动失败且不会成为重定向 URL。

### 12.3 定制化

- 未设置 `UI_TEMPLATES_DIR` 时，所有支持语言均显示内置 Logo、欢迎页和查询欢迎词。
- 指向合法完整 Bundle 并重启后，欢迎页与空白态按当前语言显示客户内容，无需重建前端，也不修改 `lightrag/api/webui/ui_defaults/`。
- 显式配置的 Bundle 出现 manifest Schema 错误、缺少默认语言、fallback 环、文件缺失/超限、路径逃逸或 Logo MIME 不匹配时，Server 启动失败并报告错误，不发生内置与客户内容的逐字段混用。
- 精确语言不存在时按 manifest 显式 fallback、再按外部 Bundle 自己的 `default_locale` 回退；不会回退到内置 Bundle。
- 用户切换语言后，欢迎页和空白态使用相同解析结果原子更新，RTL 方向正确，切换过程中不出现跨语言字段混合。
- 模板中的 `<script>`、事件属性、iframe 和危险 URL 不执行。
- customization API 支持 locale ETag/304；文案变化后对应 locale revision 改变，未变化的 locale 可继续 304。
- Logo 字节变化后 `asset_hash` 和资源 URL 改变；仅文案变化时 `asset_id` 和未修改 Logo URL 保持稳定。
- 修改内置 `ui_defaults` 并重新构建/部署后，也通过相同的 revision 与 asset hash 机制使缓存失效。
- 浏览器响应、健康信息和日志不暴露模板/Logo 的服务端绝对路径或文件内容之外的配置。

### 12.4 移动端

- 在 §9.1 四种视口中无页面级横向滚动。
- iOS/Android 软键盘展开后输入框和发送/停止按钮可见、可点击。
- 长代码块和表格不撑宽页面；长回答可以平滑滚动。
- 所有主要按钮满足 44 px 触控尺寸并具有可访问名称。
- 旋转屏幕、停止流式查询、向上滚动阅读和清空会话均不产生布局跳变或历史回弹。

## 13. 测试与验证计划

### 13.1 前端单元测试

- `AppRouter` 在 `admin/workspace × authenticated/anonymous × auth enabled/disabled` 下的路由矩阵。
- 安全回跳解析器对合法前缀、外部 URL、双斜杠、编码路径穿越和未知 hash 的测试。
- `RetrievalView` 与 `WorkspaceQueryView` 的 DOM 差异，以及共享会话控制层的请求 payload 测试。
- 工作区不解析 query mode 前缀，但读取后台持久化的合法 `querySettings`；未配置时使用前端默认值。
- 两个页面的流式完成、失败、停止、清空、历史持久化和卸载清理使用同一组共享层测试。
- 空白态首次显示、发送后隐藏、清空后恢复。
- customization 加载失败、语言切换原子更新、RTL、fallback 和 Markdown 安全渲染测试。

### 13.2 后端测试

- `/webui`、`/workspace` 双 mount 的资源检查、尾斜杠、缓存头和无资源降级。
- `WEBUI_DEFAULT_ENTRY` 的默认值、两个合法枚举、非法值拒绝及 `root_path` 重定向。
- 不设置、设置和组合 `LIGHTRAG_API_PREFIX` 时的运行时配置注入与重定向。
- 内置/外部 Bundle 加载、严格 manifest Schema、全部支持语言、locale 规范化、显式 fallback、fallback 环和完整 Bundle 原子校验。
- 路径穿越、绝对路径、符号链接逃逸、未引用文件、文件超限、MIME 不匹配和 SVG 响应头。
- customization 配置端点的 locale ETag/304、部署前缀，以及资源端点的 `asset_hash`、长期 immutable 缓存和未知资源拒绝。
- 分别修改单一语言文案、共享 Logo、`WEBUI_TITLE`，验证 locale revision、bundle revision 和 asset hash 的变化边界。
- `/health` 中两个入口可用性字段与实际 mount 一致。
- 实际 `app.routes` 审计包含两个 mount 和公开 customization/config 面，并验证原始 `ui_defaults/` URL 被拒绝。

### 13.3 浏览器验证

- 使用真实浏览器分别验证桌面 Chrome/Firefox 和移动端 Safari/Chrome。
- 覆盖软键盘、触摸滚动、文本选择、流式增长、停止、复制、横竖屏切换和长 Markdown 内容。
- Vite 开发环境使用 DOM selector 等待，不以长期连接下的 `networkidle` 作为完成条件。

### 13.4 建议验证命令

```bash
cd lightrag_webui
bun test
bun run lint
bun run build

cd ..
./scripts/test.sh tests/api
```

开发迭代期间应优先运行实际改动模块对应的前端测试文件及 `tests/api/` 下新增的 WebUI mount/config 子集；完整 `tests/api` 和前端全套在里程碑执行。

## 14. 交付拆分

建议按以下顺序拆分，确保每一步都可独立验证：

1. **双入口基础设施**：统一 runtime config，Server 双 mount，`root_path`、健康状态、降级和打包测试。
2. **入口感知认证**：安全回跳模型，欢迎页路由，登录/退出/401/guest 全链路保留入口。
3. **查询共享层与工作区 UI**：从 `RetrievalView` 抽出查询会话、消息列表和输入操作层；新增复用 `ChatMessage` 的 `WorkspaceQueryView`、空白态和精简应用壳；保持后台页面行为不变。
4. **多语言品牌定制**：默认 Bundle、严格 manifest、外部只读 Bundle 启动快照、locale/fallback、公开读取 API、revision/asset hash 缓存、安全渲染和运维文档。
5. **移动端收口**：响应式布局、safe-area/软键盘、真实浏览器回归和无障碍检查。

每个 PR 都必须保持 `/webui` 可用，不能等最后一个 PR 才恢复后台入口。

## 15. 风险与对策

| 风险 | 后果 | 对策 |
| --- | --- | --- |
| 只复用 `ChatMessage` | 工作区仍会复制请求、流式、停止、历史和滚动状态机 | 同时抽取共享查询会话、消息列表和输入操作层 |
| 给 `RetrievalView` 堆叠 variant 分支 | 千行组件同时承担后台与移动工作区两套布局 | 两个页面壳组合同一组共享能力 |
| 只隐藏参数侧栏 | `/mode` 前缀仍可临时覆盖参数，或两个页面组装出不同请求 | 使用同一 serializer 读取持久化参数，并在工作区禁用前缀解析 |
| 用 URL 字符串猜模式 | API 前缀、代理改写或未来路径变更后误判 | Server 注入唯一 `uiMode` |
| 登录固定回 `/` | 查询用户进入后台或管理员进入查询页 | 结构化、白名单校验的入口回跳状态 |
| 两套静态构建 | 包体、版本和发布流程漂移 | 一次构建、同目录双挂载 |
| 原始 HTML 模板 | XSS、钓鱼表单或布局劫持 | 受限 Markdown，系统拥有操作控件 |
| 启动时复制客户文件覆盖构建目录 | 只读容器/PyPI 安装不可写，多 worker 竞态，升级后残留旧文件 | 外部目录只读加载为内存快照，构建产物永不修改 |
| 显式错误配置被静默回退 | 运维误以为客户品牌已生效，实际展示 LightRAG 默认内容 | 未配置时用默认 Bundle；显式配置校验失败时启动失败 |
| 对每个字段独立回退 | 同页混合客户/内置品牌或不同语言 | 以完整 Bundle 和 locale 表示为原子单位加载与切换 |
| 用稳定文件名直接缓存 Logo | 客户替换文件后浏览器/CDN 继续展示旧图 | `asset_id` 保持语义稳定，内容 SHA-256 进入资源 URL |
| 自动猜测区域语言回退 | `zh-HK` 等语言回退到错误书写体系 | 只做精确匹配、manifest 显式 fallback 和默认语言回退 |
| customization API 变成任意文件读取 | 未登录攻击者读取 Server 文件 | 严格 manifest、根目录约束、拒绝 symlink 逃逸且只提供已引用资源 |
| Logo/模板响应或渲染失败导致白屏 | 日常入口不可用 | 前端最小安全默认内容、错误隔离和重试，不阻断认证/查询交互 |
| 把“后台入口”当成授权 | 用户可直接调用管理 API | 明确 UI 分流不是安全边界，API 继续强制授权 |
| `workspace` 术语与多工作空间冲突 | 代码、文档和用户理解混乱 | UI 层使用 `uiMode`/query workspace，数据层保留 workspace ID |

## 16. 后续可选项

以下能力不进入本期验收：

- 无重启热加载或集中配置推送；
- 在 Bundle 中本地化 `WEBUI_TITLE` / `WEBUI_DESCRIPTION`；
- 按用户/角色显示不同欢迎内容；
- 运维品牌配置管理 UI；
- 面向查询入口的独立域名、PWA、分享链接或会话云同步；
- 查询入口与服务端多工作空间选择器集成。
