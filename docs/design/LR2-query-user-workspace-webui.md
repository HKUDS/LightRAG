# PRD：面向查询用户的独立 WebUI 入口

- 状态：待复核（2026-08-24；2026-08-25 改为双 HTML 入口方案；2026-08-26 评审通过；2026-08-26 修订：钳制 `only_need_*`、取消服务端内置 UI Bundle、把 `<ns>` 分区与 token 本地校验拆出到 [LR2-client-state-partitioning.md](./LR2-client-state-partitioning.md)、取消配置端点的 locale revision 与 ETag；2026-08-26 二轮评审修订：可信 Bundle 信任模型、legacy 历史迁移映射、依赖闭包双证据、多键迁移原子性表述、`bundle_revision` 只进日志；2026-08-26 三轮评审修订：明确旧标签页回写丢失边界）
- 适用范围：LightRAG Server（`lightrag/api/`）+ WebUI（`lightrag_webui/`）
- 产品入口：日常查询端点 `/workspace`；后台管理端点 `/webui`
- 关键约束：查询入口只提供知识库问答，不暴露查询参数、文档管理、知识图谱管理或 API 文档入口；即使 WebUI 资源缺失也不得跳转或引导到 API 文档；复用 `ChatMessage.tsx` 及从现有查询页抽出的共享会话能力，不复用后台页面壳；完整支持移动端；欢迎内容和 Logo 通过只读、多语言 UI Bundle 定制

## TL;DR

新增与 `/webui` 并存的 `/workspace` WebUI 入口。同一次前端构建产出两个 HTML 入口：`index.html` 保留现有文档、知识图谱和查询页面，`workspace.html` 渲染独立的 `WorkspaceQueryView`。入口身份就是被加载的那个入口产物，运行时不存在“入口模式”标志。两个查询页面共同复用 `ChatMessage.tsx` 和抽出的查询会话、消息滚动、输入操作能力，但分别拥有自己的页面布局。访问 `/workspace` 的未登录用户先看到可定制欢迎页，登录后返回 `/workspace`；访问 `/webui` 的用户登录后仍返回 `/webui`。`/workspace` 在正常页面和资源缺失降级态都不显示、跳转或引导到 API 文档。工作区查询空白页中央显示可定制 Logo 和欢迎词，返回查询内容后 Logo 和欢迎词消失。定制内容默认由前端 i18n 资源与打包 Logo 提供，服务端不携带内置 Bundle；生产环境可通过 `UI_TEMPLATES_DIR` 指向外部只读的多语言 UI Bundle，Server 启动时完整校验并生成不可变快照，前端通过公开 API 按语言读取内容和带内容哈希的资源 URL。未配置该变量即为「无定制」，是常态而非降级。

> **术语边界**：本文件中的 `/workspace` 只是“面向日常查询用户的 WebUI URL”，不代表 [服务端多工作空间方案](./LR2-multi-workspace-phase1.md) 中的知识库 `workspace` ID，也不新增知识库选择、数据隔离或多租户语义。后续若同时启用多工作空间能力，二者应分别命名为“查询入口”和“知识库工作区”，避免在代码和文案中混用。
>
> 两者不存在 URL 命名空间冲突：多工作空间按 `LIGHTRAG-WORKSPACE` 请求头（Ollama 面按 model tag）选择知识库，数据面 URL 不按工作区分区，只有管理端点在路径里带 ID（`/workspaces*`）。因此本入口占用 `/workspace/` 不会挡住多工作空间的任何路由形态。真正需要处理的是**请求头的静默继承**，见 §7.3 末段。

---

## 1. 背景与问题

当前 WebUI 统一挂载在 `/webui`，登录后进入包含以下能力的后台界面：

- 文档上传、扫描、状态查看和删除；
- 知识图谱浏览与编辑；
- 知识库查询及完整查询参数配置。

普通知识库用户的主要任务只是提问和阅读答案。现有界面会向这类用户暴露大量无关操作，增加学习成本，也容易让“日常使用入口”和“知识库维护入口”混在一起。与此同时，现有前端只有一个应用壳：`AppRouter` 在未认证时一律导航到 `#/login`，没有“入口”这个概念，因此无法区分“未登录的查询用户应看到欢迎页”和“未登录的管理员应看到登录页”。需要澄清的是，这不是一个“跳转丢失入口”的问题——WebUI 使用 `HashRouter`，全部导航走 `navigate()` 只改写 hash 而不改写路径，所以在双挂载下入口天然保留（见 §6.1）。真正缺的是第二个入口产物及其独立的路由表。

因此需要在不复制查询实现、不破坏现有 `/webui` 的前提下，增加一个职责单一、可品牌定制、适合手机使用的查询入口。

## 2. 产品目标与成功标准

### 2.1 目标

1. 提供固定的日常查询入口 `/workspace`，用户登录后只看到知识库查询页面。
2. 将现有 `/webui` 明确定位为后台管理入口，并保持其现有功能和直接链接可用。
3. 新增独立的 `WorkspaceQueryView` 页面，直接复用 `ChatMessage.tsx`，并从 `RetrievalView.tsx` 抽取共享的查询会话、流式响应、停止、复制、清空、历史记录、输入和滚动跟随能力，不维护第二套查询状态机。
4. 查询入口不显示、也不接受面向用户的查询参数覆盖；查询请求沿用同一浏览器中 `/webui` 保存的查询配置，未配置时使用现有前端默认值。**查询参数共享，查询历史与运行中的会话状态各自独立**（§7.2）。
5. 登录、退出和认证失效流程始终停留在当前入口；未登录默认页按入口区分（查询入口为欢迎页，后台为登录页）。
6. 未登录的查询用户先看到可定制欢迎页；查询空白态显示可定制 Logo 和欢迎词；返回查询内容后 Logo 和欢迎词消失，仅显示查询历史。
7. 在主流手机宽度和软键盘场景下完成提问、阅读流式答案、复制、停止和清空操作；查询入口的首屏只加载查询所需代码，不下载图谱与文档管理模块。
8. 根路径 `/` 的现有默认跳转继续使用 `/webui`，可以通过环境变量改为 `/workspace`。
9. 欢迎页和查询空白态支持当前 WebUI 的全部语言，并允许部署方在不重建前端的情况下替换整套多语言文案与品牌资源。

### 2.2 成功标准

- 新用户从 `/workspace` 进入后，无需理解文档、图谱或 RAG 参数即可完成首次提问。
- `/workspace` 页面不存在文档管理、图谱管理、API 文档入口和查询参数侧栏。
- 从 `/workspace` 发起登录的用户登录后返回 `/workspace`；从 `/webui` 发起登录的用户登录后返回 `/webui`。
- 在 320 px 宽度下无页面级横向滚动，输入框和发送/停止按钮始终可操作，软键盘弹出后当前输入区仍可见。
- 未设置 `UI_TEMPLATES_DIR` 时显示前端默认的品牌内容，无需任何配置；显式设置时，外部 Bundle 必须通过完整校验才能启动，避免定制内容部分生效而造成品牌或语言混用（§8.4）。

## 3. 非目标

- 本期不新增用户、角色、注册、找回密码、SSO 或权限管理能力。
- 本期不把 `/webui` 的“后台管理”定位当作新的安全边界。是否允许调用文档、图谱或查询 API，仍由服务端现有认证/授权机制决定；仅隐藏前端入口不能代替 API 授权。
- 本期不实现知识库工作区选择、多工作空间路由、多租户或按用户隔离知识库。
- 本期不直接复用整个 `RetrievalView.tsx` 作为工作区页面，也不复制或重写其中的查询状态机；只把可复用逻辑抽成两个页面共同依赖的模块。
- 本期不改变 `/query`、`/query/stream` 的公开 API 契约。
- 本期不允许最终用户在 `/workspace` 中设置 query mode、Top K、Token Budget、rerank、stream、user prompt 等参数；这些参数沿用同一浏览器配置中 `/webui` 保存的值，未配置时使用现有前端默认值。
- 本期不把 `only_need_context` / `only_need_prompt` 提供给查询入口。二者是**调试开关**（返回检索上下文原文或最终提示词而不是答案），只在 `/webui` 的参数侧栏提供；`/workspace` 强制以 `false` 提交，见 §7.3。
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
| `/` | 可配置的默认入口 | 按 `LIGHTRAG_DEFAULT_UI` 跳转 | 按 `LIGHTRAG_DEFAULT_UI` 跳转 |

本 PRD 将“未登录用户跳转欢迎页”限定为新增的查询入口 `/workspace`。后台 `/webui` 是面向明确知道后台地址的管理员入口，继续直接显示登录页，避免为既有管理流程增加一次无意义点击。

根路径的唯一配置项为 `LIGHTRAG_DEFAULT_UI`：

- 允许值仅为 `webui` 或 `workspace`，默认值为 `webui`，因此升级后 `/` 仍跳转 `/webui/`。
- 配置为 `workspace` 时，`/` 跳转 `/workspace/`，再由查询入口按登录态显示查询主界面或欢迎页。
- 归入 `LIGHTRAG_*` 家族（与 `LIGHTRAG_API_PREFIX` 同属路由/部署类），而不是 `WEBUI_TITLE` / `WEBUI_DESCRIPTION` 所在的展示串家族；按 `--api-prefix` 的形状同时提供 env 与 CLI 两条通道。文档中必须把它的作用域钉死为**唯一一条行为：`/` 的跳转目标**，避免日后被理解成“默认界面”而累积其它含义。
- **配置值是枚举而不是 URL。** 自由 URL 会同时引入四个问题：开放重定向面、值在 `root_path` 之前还是之后的语义歧义、无法参与下面的可用性降级判断，以及没人要求的通用性。两值枚举一次性消除全部四项。
- **非法值必须启动期 fail-fast。** 注意 argparse 的 `choices` 只校验命令行显式传入的值，**不校验来自环境变量的默认值**（仓库现有的 `--rerank-binding` 就是这个形状），因此除 `choices` 外还需对 env 取到的值做显式校验。否则把 `workspace` 拼错成 `workspaces` 的结果是静默落回 `webui`，而用户报“打开首页进了后台”时没人会怀疑到环境变量。
- 重定向必须带上实际 `root_path`，且 `LIGHTRAG_DEFAULT_UI` 不控制两个入口是否挂载。
- **`/` 的行为等于所配置默认入口的行为，包括该入口的降级分支**；默认入口不可用时不改投另一个入口（见 §5.2 末段）。

之所以做成服务端配置而不是让运维在反向代理写一条根路径规则，是因为代理层做不到三件事：无反代的直连部署（compose 直接暴露 9621）没有落点；重定向必须叠加 ASGI `root_path` 才能在剥前缀的代理后仍指向 `/site01/webui/`，手写规则在多站点场景下正好会写错；以及只有服务端知道两个入口产物在不在，代理无法降级。在有反代、无部署前缀、且不需要降级感知的部署中，反代层的根路径规则仍是可用的替代方案，文档应并列说明。

两个入口必须同时支持 `LIGHTRAG_API_PREFIX` 和反向代理 `root_path`。任何重定向、运行时配置、静态资源地址和登录回跳都不得绕过部署前缀。例如前缀为 `/site01` 时，浏览器可见入口应为 `/site01/workspace/` 和 `/site01/webui/`。

### 5.2 一次构建、两个 HTML 入口、双挂载

前端只进行一次 Vite 构建，但通过 `rollupOptions.input` 产出**两个 HTML 入口**，二者都落在输出目录根：

| 产物 | 加载的应用壳 | 挂载点 |
| --- | --- | --- |
| `index.html` | 现有后台 `App` | `/webui` |
| `workspace.html` | 精简的 `WorkspaceQueryView` 应用壳 | `/workspace` |

Server 把**同一个静态目录**挂载两次，每个 mount 只把属于自己的那个文件当作目录索引。入口身份因此完全由 URL 决定，并在浏览器侧体现为“加载了哪个入口产物”——**运行时不存在 `uiMode` 之类的入口模式标志**，前端也无需从任何地方推断当前入口。

**用户可见 URL 不变。** 入口文件名是服务端实现细节，只用于决定每个 mount 的目录索引，从不出现在用户输入或对外宣传的地址中：查询入口仍是 `/workspace/`（`/workspace` 由 Router 的 `redirect_slashes` 307 到带尾斜杠形式），后台仍是 `/webui/`，与今天 `/webui/` 内部读取 `index.html` 的方式完全一致。

要求：

- 两个 HTML 都在输出目录根，因此 `base: './'` 下都发出 `./assets/...`；各自作为 mount 根被访问时，浏览器分别解析到 `/webui/assets/...` 与 `/workspace/assets/...`，在服务端命中同一份文件。共用模块由 Vite 自动抽成公共 chunk，**但浏览器 HTTP 缓存以完整 URL 为键，两个入口各缓存一份**——同一份 JS 会被下载两次。这是可接受的代价：绝大多数用户只使用其中一个入口，而 `base: './'` 的相对资源路径正是双挂载与 `root_path` 部署得以成立的前提，不值得为跨入口缓存复用去增设统一的 assets mount。
- **运行时配置回到已发布的形状 `{ apiPrefix, webuiPrefix }`，两个 mount 注入的内容逐字节相同。** 因此现有的模块级常量 `runtime_config_script` 无需改造为按 mount 实例参数化。运行时配置中不得出现入口模式字段。
- `webuiPrefix` 在本期不再有消费点（见下一条），但它已作为公开的运行时配置字段写入 [多站点部署文档](../MultiSiteDeployment.md)，故保留字段本身；若决定删除，须同步该文档。
- **品牌链接改用文档相对的 `<a href="./">`。** 当前 `App.tsx` 与 `SiteHeader.tsx` 使用 `<a href={webuiPrefix}>`；`HashRouter` 下 pathname 恒等于当前入口的挂载根，`./` 因此永远解析回本入口。这既避免查询用户点一下品牌标识就被送到 `/webui/`（一次完整的跨入口泄漏，而 §12.1 的路由矩阵抓不到它），也比任何注入前缀更能适应改写路径的反向代理。
- **每个 mount 只提供属于自己的入口 HTML**：`/workspace/index.html` 与 `/webui/workspace.html` 必须返回 404；而 `/webui/index.html`、`/workspace/workspace.html` 这类**同入口显式文件名**继续可用——前者今天就已存在，不得破坏。默认行为不是报错，而是更糟：两个 HTML 位于同一目录，Starlette 对命中的常规文件直接返回，且页内 `./assets/...` 相对当前 mount 根解析后同样命中，于是凭空多出两个**完整可用的别名**——`/webui/workspace.html` 给出可用的查询界面，`/workspace/index.html` 给出可用的后台界面，而 URL 不再指示入口。这直接推翻本节“入口身份完全由 URL 决定”的立论，也让 §12.1 中“页面内不存在指向另一入口的链接”这类验收失去意义。它不构成安全边界（接口授权仍由服务端按 router 强制，两个入口本来也都可直接访问），但必须处理。
- **索引文件名必须在 `SmartStaticFiles` 中覆写。** Starlette 的 `StaticFiles` 在 `html=True` 下把目录索引硬编码为 `index.html`，没有可配置参数，因此查询入口 mount 需覆写才能以 `workspace.html` 作为索引；上一条的跨入口 404 规则落在同一处覆写里。覆写时注意尾斜杠语义：Router 层的 `redirect_slashes` 已保证 mount 根一定带尾斜杠，因此把目录请求改写为具体文件不会绕过任何重定向。
- HTML 响应的 no-cache 头与运行时配置占位符替换对两个入口同等适用。`<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->` 必须同时存在于两个 HTML 源文件中，`SmartStaticFiles` 中“与 `lightrag_webui/index.html` 保持同步”的注释需改为点名两个文件。
- 静态资源继续使用内容哈希和长期 immutable 缓存。
- **不需要 `React.lazy`：代码分割在构建层完成。** 当前 `App.tsx` 静态 import `GraphViewer`、`DocumentManager` 与 `RetrievalView`，构建出的入口 chunk 约 3.2 MB（未压缩，另含 cytoscape、mermaid 等依赖），一个只想提问的移动端用户要下载整个后台应用才能看到输入框，与 §9 的移动端定位直接冲突。双入口在页面壳这一层天然消除了它，但**仅换掉入口不足以达成目标**：现有代码里存在两条会把重依赖拖进公共 chunk 的静态链，必须一并拆开。
- **导航核心不得静态依赖图谱 store。** 当前链路是 `api/lightrag.ts` → `services/navigation.ts` → `stores/graph.ts` → `graphology`，也就是说**任何发起查询的代码都会静态拉入图谱库**。修法与 §5.3 的 bootstrap 配置同源：导航核心只持有一个可选的“重置适配器”接口，后台入口在 bootstrap 时注册图谱清理逻辑，工作区入口不注册。导航核心自身不得 import `stores/graph`。
- **Mermaid 必须动态加载。** `ChatMessage.tsx` 目前静态 import `mermaid`，而两个入口共用 `ChatMessage`，因此“工作区不含 mermaid”与“工作区保留 Mermaid 渲染”只能靠动态 import 同时成立：在消息中发现完整的 mermaid 代码块时才 `import()`。这不是可选优化，而是本条约束能否成立的前提。
- 因此首屏约束的准确表述是**首屏静态依赖闭包**而非“chunk 闭包”，而它需要**两种互补的证据**，因为任何一种单独都不充分：

  | 证据 | 来源 | 能证明什么 | 不能证明什么 |
  | --- | --- | --- | --- |
  | Vite `build.manifest` | `ManifestChunk.imports` / `dynamicImports` | 入口存在、chunk 间的静态/动态可达关系、mermaid 只经动态边到达、首屏传输字节 | **chunk 内部包含哪些源模块** |
  | 构建审计插件 | Rollup/Rolldown `OutputChunk.modules` 的键（模块 id） | 首屏可达 chunk 的**源模块清单**中不含图谱、文档管理及其依赖 | chunk 之间的加载时机 |

  **只用 manifest 断言是不够的**：`ManifestChunk` 只有 `src`/`file`/`css`/`assets`/`isEntry`/`name`/`isDynamicEntry`/`imports`/`dynamicImports` 九个字段，没有任何字段列出 chunk 内部的源模块（`vite/dist/node/index.d.ts` 的 `ManifestChunk` 定义）。一旦 graphology 或 cytoscape 被合并进某个工作区入口也会静态加载的公共 chunk，“`imports` 里没有图谱 chunk”依然成立，而字节已经在首屏里了——这正是本条要防的退化，用 manifest 单独去防它是自证。`OutputChunk.modules` 是 `Record<string, RenderedModule>`，键即模块 id，可在 `generateBundle` 钩子里导出为清单。

- **首屏传输字节必须有阈值，不能只“记录”。** 取实现完成时的实测值为基线，写入仓库并设定相对容差（建议 +10%）；超出即 CI 失败。只记录不设限的指标不会阻止任何退化，只会在事后被用来解释退化。
- **开发环境无需任何模式开关。** `bun run dev` 下 `/` 提供后台入口，`/workspace.html` 提供查询入口（Vite 的 HTML fallback 对磁盘上存在的 `.html` 直接放行）。若希望 dev 的 URL 形态与生产一致，可加一段把 `/workspace/` 改写到 `/workspace.html` 的 dev 中间件；这是便利项而非必需项。不引入 `VITE_DEV_UI_MODE` 一类的构建期变量。
- 不复制 `webui` 构建目录，不增加第二套 Vite 构建流程或第二个输出目录。
- **两个入口的可用性各自独立判断。** `check_frontend_build()` 目前只检查 `webui/index.html` 并返回单个 `assets_exist`，同时驱动 mount 条件、根路径跳转和 `/health`。双产物后必须分别检查 `index.html` 与 `workspace.html`：正常构建两者同时存在（`emptyOutDir: true`），但“新服务端 + 旧构建目录”（镜像里烘焙的旧 WebUI、版本错配的安装包）会出现只有 `index.html` 的情况。此时**不得把整个 WebUI 判为未构建**——那会让后台一并失效，是过激的回归；正确行为是后台照常挂载，查询入口进入下述降级分支。
- WebUI 资源未打包时，两个入口仍必须显式注册，但采用与各自定位一致的降级行为：`/webui` 保持现有逻辑，API 文档可用时可跳转到 API 文档，否则返回服务信息 JSON；`/workspace` 和 `/workspace/` 无论 `ENABLE_API_DOCS` 是否开启，都只返回固定的查询入口不可用服务信息 JSON，不得重定向到 `/docs`、`/redoc` 或其它 API 文档页面，响应也不得包含 API 文档链接或引导文案。
- 当 `LIGHTRAG_DEFAULT_UI=workspace` 且查询入口产物缺失时，根路径 `/` 必须进入上述 `/workspace` 降级分支：既不因 API 文档可用而改跳 API 文档，**也不改投仍然可用的 `/webui/`**。把查询用户送进后台登录页与送进 API 文档属于同一类跨入口泄漏。两个入口都不得静默 404；“产物来自同一次构建”不等于“共享同一降级目标”。

### 5.3 前端路由

`HashRouter` 继续使用，但**两个入口各自拥有独立的 router 与路由表**，不存在一张按入口模式分支的共享路由表。

后台入口（`index.html`）：

| SPA 路由 | 说明 |
| --- | --- |
| `#/login` | 现有后台登录页；未登录默认页 |
| `#/` | 现有后台主界面 |

查询入口（`workspace.html`）：

| SPA 路由 | 说明 |
| --- | --- |
| `#/welcome` | 公开欢迎页；未登录默认页 |
| `#/login` | 登录页，成功后经 `navigate('/')` 落回本入口 `#/` |
| `#/` | 受保护的查询主界面 |

未知 hash 路由应回到当前入口自己的默认页，不得跨入口跳转。

被两个入口共用的单例——尤其是 `services/navigation.ts` 中处理 401 与退出的导航策略——必须由各入口的 bootstrap 代码在启动时**显式配置**（查询入口把“未认证目标”设为 `#/welcome`，后台设为 `#/login`），不得在运行时读取某个全局入口标志来分支。这是入口差异唯一会静默出错的接线点：共享层拿不到策略时会沉默地沿用后台默认值，症状只在查询入口的 401 路径上出现。

## 6. 登录与入口保留

### 6.1 入口由挂载路径天然保留

WebUI 使用 `HashRouter`，全部导航都经 react-router 的 `navigate()` 完成，只改写 hash 而不改写路径；现有代码中不存在任何 `location.href=` / `location.replace` / `location.assign` 形式的硬跳转（唯一一处 `window.location` 是 `AppRouter` 对 `hash` 的读取）。因此在双挂载下，`navigate('/')` 在 `/workspace/#/login` 上落回的是 `/workspace/#/`，在 `/webui/#/login` 上落回的是 `/webui/#/`——**保留入口不需要额外机制，现有写死的相对路径恰恰就是保留入口的实现**。

本节要求的是每条路径在正确的入口内落到正确的页面：

- 从 `/workspace` 欢迎页进入登录页，成功后返回 `/workspace/#/`。
- 从 `/webui` 进入登录页，成功后返回 `/webui/#/`。
- 已登录用户直接访问任一入口时留在该入口。
- 日常查询端点发生 token 失效时进入 `/workspace` 的欢迎页；用户重新登录后回到查询页面。
- 后台管理端点发生 token 失效时进入 `/webui` 的后台登录页。
- 退出登录后回到当前入口的未登录默认页：查询入口回欢迎页，后台入口回登录页。

### 6.2 不引入回跳参数

既然入口由挂载路径天然保留，本方案**不引入任何承载回跳目标的 URL 参数或持久化状态**，因此也不存在开放重定向面：

1. 未登录时的默认页由**各入口自己的路由表**定义：查询入口为 `#/welcome`，后台为 `#/login`（见 §5.3）；共享导航单例由入口 bootstrap 显式配置，不读取全局入口标志。
2. 未知或非法的 hash 路由回落到当前入口的默认页，不跨入口跳转。
3. 任何导航都必须经 `navigate()` 完成。改写路径的硬跳转（`window.location.href` / `location.replace` / `location.assign`）与带路径的 `<a href>` 是仅有的两类能让页面离开当前挂载点的途径：前者一律禁止，后者按 §5.2 一律使用文档相对的 `href="./"`，并受 §12.1 的品牌链接验收管辖。
4. 登录请求失败不改变当前路由；用户重试成功后仍留在原入口。

若未来确有跨入口回跳需求（本期没有任何用户故事需要它），再单独设计结构化的 `{ mode, route }` 状态并对已注册 hash 路由做白名单校验；无论如何都不得把未经解析的 URL 字符串交给 `window.location`。

### 6.3 未登录默认页与 guest 登录态

判定口径统一为一句话：**“首次访问显示欢迎页”等价于“当前没有有效 token”**。guest token 是有效登录态，与普通用户 token 同等对待。

| 当前状态 | 访问 `/workspace/` |
| --- | --- |
| 持有有效普通用户 token | 直接进入查询主界面 |
| 持有有效 guest token（例如先访问过 `/webui`） | 直接进入查询主界面，不再展示欢迎页 |
| 无有效 token，认证已关闭 | 展示欢迎页；点击后才激活 guest 登录态 |
| 无有效 token，认证已启用 | 展示欢迎页；点击后进入登录页 |

采用这一口径是为了不在 auth store 里长期区分 token 来源：任何“持 guest token 时仍展示欢迎页”的变体，都要为一次性的展示差异引入一个必须长期维护的状态位。

**guest token 的获取与激活必须分开。** `/auth-status` 在认证关闭时**必然**同时返回 guest token，所以“启动时不请求 `/auth-status`”是做不到的（§8.8 还要求它与 customization 并行发起以压首屏时间）。要禁止的不是取，而是**激活与持久化**：

1. 启动时调用 `/auth-status`，只读取 `auth_configured`。
2. 响应若附带 guest token，在启动阶段**丢弃**——不调用 `login()`，不写 localStorage，不改 auth store。
3. 用户点击“进入工作区”后再次调用 `/auth-status`，此时才写入 auth store 并进入查询主界面。

否则欢迎页会被自己刚写入的 token 判定为已登录而跳过自己。

其余保持现状：

- 主按钮文案在认证关闭时为“进入工作区”而不是“登录”。
- `/webui` 保持现有自动 guest 登录行为。
- guest token 续期和 401 重试沿用现有 API 客户端逻辑，但最终导航必须保留当前入口。

**token 有效性判断必须收紧，这是本节成立的前提。** 现有 `initAuthState`（`stores/state.ts`）只检查 `LIGHTRAG-API-TOKEN` 是否**存在**，随后直接置 `isAuthenticated: true`；它虽然顺带算出了 `tokenExpiresAt`，却不用它来把关。因此格式损坏或早已过期的 token 同样会被判为已登录，上表“无有效 token”这一行在今天的代码里无法成立。

该收紧属于共享认证层改造，两个入口同时受益，且与本功能无因果关系（`/webui` 今天同样会先渲染整个后台再被 401 打回），因此**规格与测试计划见[客户端状态分区与共享认证层](./LR2-client-state-partitioning.md) §5**。本节只声明依赖关系：**它必须先于查询入口上线**，否则 §6.3 的状态表退化为「有无 token」两行，欢迎页会对持过期 token 的用户失效。

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
- Query mode、Top K、Token Budget、rerank、stream、history turns、user prompt、`only_need_context` / `only_need_prompt` 等控件；
- 后台健康状态、数据维护按钮或图谱编辑入口。

### 7.2 页面与复用边界

工作区不直接渲染或给 `RetrievalView` 增加大范围 `variant` 分支，而是新增独立的 `WorkspaceQueryView`。原因是两者的页面职责已经不同：后台查询页包含参数侧栏并服从 Tab 生命周期，工作区查询页需要独立的空白态、页头、移动端布局和始终活跃的消息区域。把这些差异全部塞入 `RetrievalView` 会让一个已超过千行的组件继续承担两套页面布局。

但 `ChatMessage.tsx` 只负责一条消息的展示，不能单独构成完整复用边界。现有查询页中的以下能力仍必须抽出并由两个页面共享：

| 共享层 | 职责 | 不应包含 |
| --- | --- | --- |
| 查询会话控制层（hook/controller） | 消息状态、按**注入的**历史存储读写、请求参数序列化、流式增量、COT/LaTeX 完整性、进度、计时、停止与清理 | 页面布局、后台 Tab、参数控件、历史存储键的选择 |
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

**共享的是实现，不是状态。** 两个入口共享查询参数与查询实现，但**不共享会话历史，也不共享运行中的会话实例**：

| 状态 | `/webui` | `/workspace` | 是否共享 |
| --- | --- | --- | --- |
| `querySettings` | 可读写 | 只读 | 共享 |
| 查询 serializer | 使用 | 使用 | 共享代码 |
| 流式 / 停止 / 计时状态机 | 独立实例 | 独立实例 | 共享实现，不共享状态 |
| 查询历史 | 后台调试历史 | 工作区用户历史 | **不共享** |
| `conversation_history` | 取自后台历史 | 取自工作区历史 | **不共享** |
| 草稿、清空、停止 | 仅影响后台 | 仅影响工作区 | **不共享** |

历史隔离不是保守取舍，而是产品语义要求：后台历史是管理员的调参调试记录，工作区历史是终端用户的正常会话。把前者混入后者的展示，甚至混入后者 `bypass` 模式的 LLM 上下文，是缺陷而不是特性。共享层因此**不得自行决定读写哪个历史存储**——存储由页面组合层在构造控制层时注入。

- 当前由 `ChatMessage.tsx` 导出的 `MessageWithError` 属于两个页面和会话控制层共同使用的领域模型，应迁移到独立的 retrieval types 模块，再由 `ChatMessage` 和共享层共同 import；不能让无 UI 的会话控制层反向依赖消息渲染组件。
- 查询会话控制层通过显式策略接收“是否允许 mode 前缀”等页面差异，不得自行探测当前入口，也不得读取后台 `currentTab`；这样共享状态机保持单一，入口差异仍由页面组合层决定。`querySettings` 快照同样是**由页面组合层构造并传入**的入参，`only_need_*` 的钳制因此落在 `WorkspaceQueryView` 里，共享层不含该字段的任何特判（§7.3）。
- `RetrievalView` 保持后台页面定位，继续渲染当前 `QuerySettings`，继续支持现有查询参数和 `/mode` 输入前缀。
- `WorkspaceQueryView` 不渲染 `QuerySettings`，也不解析 `/naive`、`/local`、`/global`、`/hybrid`、`/mix`、`/bypass` 等参数覆盖前缀；以 `/` 开头的普通问题按普通文本提交。**输入前缀解析是两个页面在请求序列化上的唯一差异**；`conversation_history` 来源不同不属于 serializer 差异——历史是显式入参，由各页面从自己的存储提供（§7.3）。
- `WorkspaceQueryView` 始终把消息区域视为活跃；不得读取后台 `currentTab` 决定 Markdown、Mermaid 或动画是否更新。
- 两个页面必须通过同一个请求序列化和流式状态机发起查询，不能各自维护一份近似实现。

### 7.3 工作区查询请求

工作区不提供参数编辑界面，但使用与 `/webui` **共享**的持久化 `querySettings`。查询会话控制层读取一次一致快照，连同**本入口自己的**消息历史，经同一个 serializer 转换为 API 请求：

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

上例只用于说明参数来源，不固定具体数值。

**核心规则是 serializer 等价性**：

> 给定相同的 `querySettings`、**显式传入的相同历史**和相同问题，共享 serializer 必须产生**完全相同**的请求体，包括 `mode`、`conversation_history` 以及 bypass 模式下的默认历史轮数。**共享 serializer 内部**不得含有任何按入口分支的模式钳制、字段过滤或 history 特判。
>
> 运行时两个页面各自从**自己的**历史存储提供那段历史，因此实际请求体中的 `conversation_history` 本就不同——这是 §7.2 状态边界的直接结果，不是等价性的例外。
>
> 同理，页面组合层对**传入的 `querySettings` 快照**做的任何处理（如下文对 `only_need_*` 的钳制）都不是等价性的例外：等价性约束的是 serializer 这个**函数**，快照是它的入参。serializer 自身不得含有任何按入口分支的钳制、过滤或特判。

这条约束取代逐字段列举规则：它更强、更容易测（一条参数化跑遍全部 mode 的 serializer 测试即可覆盖），也精确划出了“共享实现、隔离状态”的边界——等价性约束的是**函数**，不是两个页面某一时刻的实际请求。派生规则如下：

- `mode`、Top K、Token Budget、rerank、stream、user prompt 等参数两侧必须一致；`conversation_history` 两侧必须各取自本入口历史。
- `/webui` 的调试提问**不得**进入 `/workspace` 的 LLM 上下文，反之亦然；在任一入口清空会话不影响另一入口。
- 两个页面共享的是 serializer 与状态机的**实现**，不是消息数组、`AbortController` 或持久化历史；同时打开两个入口时，在一侧点停止不得中断另一侧正在进行的流式响应。
- 同一浏览器 profile 中，管理员在 `/webui` 保存的新参数从下一次 `/workspace` 查询开始生效；工作区本身只读这些参数。
- 从未访问或配置过 `/webui` 时，使用现有 settings store 的前端默认值。
- 这是浏览器本地配置共享，不是 Server 全局配置，也不跨设备同步。若未来需要运维统一控制所有查询用户的参数，应另行设计服务端查询 profile，不能把 localStorage 描述成全局策略。
- serializer 发送的字段以**服务端** `QueryRequest` 模型的声明为准。注意前端 TS 的 `QueryRequest` 类型里还留着 `history_turns`，而服务端没有这个字段（pydantic 默认 `extra='ignore'`，今天它被静默丢弃）：`history_turns` 的职责是决定往 `conversation_history` 里放几轮，展开后应从请求体中剥离。这是对共享 serializer 的清理，两个入口同时受益，需有回归测试钉住。
- 工作区禁止通过输入前缀临时覆盖 mode，但**不钳制也不篡改**继承到的 mode 值。
- `only_need_context` 与 `only_need_prompt` 是唯二在页面组合层被钳制的字段：`WorkspaceQueryView` 传给会话控制层的快照中，这两项恒为 `false`。钳制发生在构造入参时，serializer 不感知。
- stream 继续决定调用 `/query` 或 `/query/stream`；无论选择哪一条，两个页面必须走相同的请求和错误处理路径。

**mode 继承是全量的，包括 `bypass`。** `bypass` 是后台参数侧栏 mode 下拉框中的一个可选项，会被持久化并由工作区继承。管理员选中它之后，工作区的行为与 `/webui` 在**规则上**完全一致：跳过知识检索直接问 LLM（无引用），共享 serializer 中“`bypass` 且 `history_turns=0` 时取 3 轮”的既有逻辑照常生效，因而该模式下会带上最近 3 轮对话并绕过服务端答案缓存。区别只在数据来源——**各入口取自己最近 3 轮**，工作区用户不会看到管理员的调试上下文。这是有意为之的一致性，不是缺陷；但由于工作区用户看不到也改不了 mode，运维文档必须写明**后台的 mode 选择同时决定查询入口的行为**。

**`only_need_context` / `only_need_prompt` 不继承，工作区强制为 `false`。** 这两项与 mode 不同：它们不是「换一种回答方式」，而是**让服务端不回答**——分别返回检索到的上下文原文和最终提示词。它们在后台参数侧栏中是可勾选开关（`QuerySettings.tsx`），会被持久化，并经 `...querySettings` 整体展开进请求体（`RetrievalView.tsx`）。若按全量继承处理，管理员调试后忘记关闭，`/workspace` 的**每一次**查询都会返回一段原始上下文或提示词而不是答案，而查询用户既看不到也改不了这个开关——这与 §2.1「无需理解 RAG 参数即可完成首次提问」直接冲突。

它与 `bypass` 的区别值得写明，以免日后被当成同一类问题重开：`bypass` 仍然产出一个面向用户的答案（只是无检索、无引用），是一个**有意接受**的一致性取舍；`only_need_*` 产出的根本不是答案，属于调试出口。因此前者继承并由运维文档兜底，后者钳制。

除 `bypass` 外的全部模式，前端 `history_turns` 被固定为 0（store 中无 UI 控件且迁移强制归零），因此工作区与 `/webui` 一样发送空 `conversation_history`，答案缓存正常生效。

本期保持现有查询会话语义，不借本功能改变 conversation history、答案缓存或引用格式。

**与服务端多工作空间的前置约定**：[多工作空间方案](./LR2-multi-workspace-phase1.md) 按 `LIGHTRAG-WORKSPACE` 请求头选择知识库，并在前端统一请求层注入该头——这一层被两个入口共用。按本节的继承口径，工作区会继承 `/webui` 选中的知识库，而 §3 又规定工作区不提供选择器，用户将无从知道自己在问哪个库。因此多工作空间落地时，§7.1 的精简页头**必须只读显示当前知识库名称**（显示不等于选择，不违反 §3）。该头缺失时服务端会落到默认工作区，所以在多工作空间启用前，本期无需任何改动。

### 7.4 查询历史与存储拆分

**两个入口的查询历史相互独立**，各自沿用现有“按浏览器保存”和“切换登录用户时清空历史”的规则。

现有 `settings-storage` 无法直接承载这一边界。它是单个 Zustand persist key，同时保存 `querySettings`、`retrievalHistory`、语言、主题和图谱设置。Zustand 的每次 `set` 都会把内存中的整份持久化切片重新写回，于是两个入口同时打开时会出现：`/webui` 修改查询参数并写入 → 已打开的 `/workspace` 仍持有旧内存状态 → `/workspace` 更新自己的历史或切换语言 → 整份旧状态被写回，覆盖 `/webui` 刚保存的参数。这个竞态在今天两个 `/webui` 标签页之间就已存在，但本方案让**跨入口共享 `querySettings`** 成为契约，因此必须先解决它才能声称参数共享可靠。

存储拆分为：

| localStorage key | 内容 | 写入方 |
| --- | --- | --- |
| `lightrag::query-settings-storage` | `querySettings` | 仅 `/webui`（运行期） |
| `lightrag::webui-retrieval-history` | 后台查询历史 | 仅 `/webui` |
| `lightrag::workspace-retrieval-history` | 工作区查询历史 | 仅 `/workspace` |
| `settings-storage`（保留） | 主题、语言、图谱显示偏好，以及 `apiKey` / `userPromptHistory` / `queryLabel` / `backendMaxGraphNodes` 等尚未搬迁的站点相关状态 | 两个入口 |

“写入方”一列描述的是**运行期**权限。一次性迁移是唯一例外，它可以由任一入口执行并写入其涉及的全部新键。

**迁移的字段映射是唯一且穷尽的**，不得由实现者推断：

```text
legacy settings-storage.querySettings      → lightrag::query-settings-storage
legacy settings-storage.retrievalHistory   → lightrag::webui-retrieval-history
（无来源）                                  → lightrag::workspace-retrieval-history = []
```

第二条必须写死。今天只有一份 `retrievalHistory`（`stores/settings.ts`），它记录的是**后台查询页**的提问——把它同时复制到两份，工作区就会展示管理员的调参调试记录，`bypass` 模式下还会把它当作 `conversation_history` 发给 LLM，正是 §7.2 要消除的缺陷；两份都置空则违背“不丢失现有历史”的兼容承诺。三种实现都能通过一个只检查“目标键都存在”的测试，所以映射必须是规格而不是约定。

**只拆本功能必需的键。** 上表只搬走了 `querySettings` 与两份历史——它们是「参数共享、历史隔离」这一契约无法绕开的部分。`apiKey`、`userPromptHistory`、`queryLabel`、`backendMaxGraphNodes` 以及 `lightrag_search_history`、`LIGHTRAG-PREVIOUS-USER` 等独立键留在原处，由[客户端状态分区与共享认证层](./LR2-client-state-partitioning.md)统一搬迁。理由是这些键的搬迁只有连同 `<ns>` 分区才有意义：单独搬一次、分区时再动一次，等于让同一批数据承受两次迁移风险。

**键名形状预留 `<ns>` 位，本期恒为空串。** 键写作 `lightrag:<ns>:<name>`，`<ns>` 在本期固定为空（故形如 `lightrag::query-settings-storage`，双冒号是有意保留的，使键形状统一、解析无歧义）。分区文档随后只把 `<ns>` 从「恒为空」改为「由 `apiPrefix` 计算」，**根部署与直连端口部署因此一个字节都不用动**，只有带前缀的部署会经历一次命名空间变化。把不可避免的那次迁移（拆 key）与可选的那次（分区）错开，两者的失败面就不会叠加。

**残留竞态明确记账。** `settings-storage` 中剩下的主题/语言仍有整份写回竞态；与之同键的 `apiKey` 等站点状态因此也可能被另一入口的主题切换回写覆盖。这个机制今天在两个 `/webui` 标签页之间就已存在，本期不处理，随上述键一并在分区文档中解决。

要求：

- 拆分后 `/workspace` 的任何写入都不再触及 `query-settings-storage`，参数覆盖竞态从源头消失。
- 已打开的 `/workspace` 必须监听 `storage` 事件（或以等效方式重新 hydrate `query-settings-storage`），使 `/webui` 的新参数**从下一次查询开始生效**，无需刷新页面。不得在流式响应进行中途切换参数快照。
- 迁移器由两个入口共同调用，在任何依赖上述键的 store 被求值之前执行，语义如下：

  1. **版本规范化先行。** 现有 `settings-storage` 带有 v1 → v21 的逐级迁移链，直接“只接受 v21 envelope”会把从 v20 及更早版本升级的用户判为无旧数据，绕开既有迁移链并把参数与历史清回默认值。迁移器必须先**复用**（而非复制）现有 `migrate` 链把任意 ≤21 的 envelope 规范化到 v21，再执行拆分。
  2. **该链必须抽成无副作用的纯模块。** 它目前内联在 `stores/settings.ts` 里，而该模块一被 import 就会创建并 hydrate persist store——迁移器为复用它而 import，就等于在迁移之前先把待拆分的 store 建了起来。因此把 v1→v21 链抽到一个纯模块，由迁移器与 Zustand 的 `persist.migrate` 共同 import；该纯模块不得 import 任何 store、`App`、API 客户端或导航模块。
  3. `version > 21`（回滚后再升级、或未知的更高版本）时**不读、不清理、不覆盖任何旧字段**，新键一律用默认值初始化——否则一次降级会永久破坏数据。
  4. **只为不存在的新键写入迁移值**；新键一旦存在，其值永远优先，重复迁移绝不用旧数据覆盖它。
  5. 先成功写入全部新键，**再**清理旧 envelope 中已迁出的字段并升级其版本号。顺序不可颠倒：清理先于复制会在中途崩溃时直接丢数据。
  6. 部分失败时保留未清理的旧字段，下次启动按同样规则重试；已写入的新键因第 4 条不会被二次覆盖，所以重跑安全且收敛到同一结果。
  7. **物理上允许存在“只写了一部分新键”的中间状态；要保证的是应用永远观察不到它。** `localStorage.setItem()` 对多个键没有事务，页面在两次写入之间崩溃必然留下部分新键，任何声称崩溃点原子的验收都无法实现。因此边界划在应用侧：**迁移未成功完成时，依赖这些键的 store 一律不得被 hydrate**，入口显示可重试的错误而不是带着默认值继续运行。这条不是保守，而是第 4 条的必然推论——若允许应用在半迁移状态下跑起来，store 会把缺失键初始化成默认值并在用户第一次改动时落盘，下次启动该键便“已存在”，legacy 值被永久遮蔽。下次启动只补齐**缺失的**键，已存在的新键保持优先。
  8. `/workspace` **允许**执行这一次性迁移写入，且覆盖迁移涉及的**全部**目标键（包括 `lightrag::webui-retrieval-history`——workspace 先被打开时必然由它写入）。运行期对 `query-settings-storage` 仍严格只读。

- **由此不需要任何锁，也不需要 IndexedDB 事务。** 目标键固定（本期只有一个命名空间）；所有执行者只写不存在的新键；目标键一旦出现便成为权威值；复制全部完成后才开始清理。在 legacy 源不被旧代码改写的前提下，运行新版本代码的并发标签页读到的是同一份升级前 legacy 状态，任意交错都收敛到同一结果；若旧标签页在两次读取之间改写了 legacy 源，结果是若干 legacy 快照之一——仍落在上一条明确接受的丢失范围内，不构成新的失效模式；重复执行天然自门控——字段已迁出，后续运行自动成为空操作。上一条的「应用不得在半迁移状态下 hydrate」同样不需要互斥原语：它是单页面内的顺序约束，不是跨页面的协调。

- **启动顺序：迁移模块必须是入口文件的第一个静态 import。** 迁移是纯同步的 localStorage 操作，不需要 `await`，也不需要动态 `import()`；但当前 `main.tsx` 静态 import `AppRouter`，后者又静态 import auth store 与 `App`，模块求值阶段 `initAuthState()` 就已读过 localStorage 并建好 store。ESM 按源码顺序深度优先求值同级 import，因此只要把这个带副作用的迁移模块放在入口文件 import 列表的**第一位**，它就会在任何 store 模块求值之前完整跑完。约束是两条：**位置在最前**，且它**递归地不 import 任何 store**（由第 2 条的纯模块保证）。

- **旧标签页兼容边界明确接受数据丢失。** 迁移开始时已经持久化在 legacy envelope 中的参数和后台历史按上述映射保留；但升级部署完成后，仍运行旧 JavaScript 的标签页若继续改写 `settings-storage`，其新增修改不再受兼容承诺保护。新键永远优先，后续新页面只清理旧标签页重新写回的 legacy 字段而不把它们覆盖进新键，因此这部分迁移后修改允许丢失。理由是旧代码不了解新键，试图双向合并既无法判断新旧，又会重新引入整份状态覆盖竞态。

  **这条边界是自足的，不需要任何强制旧页面下线的机制来兜底。** 需要保证的只有一件事——旧标签页写回的内容**不影响新版本**，而「新键永远优先、legacy 字段只清理不合并」已经完整给出了这个保证：无论旧页面写回多少次、写回什么，新键都不会被它改变。用版本握手强制重载旧页面并不能改善这个不变式，却要以中断用户正在进行的操作（流式回答、未发送的草稿）为代价，是拿确定的可用性损失换一个已经成立的保证。

- 作用域始终是**同源、同一浏览器 profile**。它不跨设备、不跨浏览器、不跨域名同步，也不是 Server 全局配置。若未来需要运维统一控制所有查询用户的参数，应另行设计服务端 query profile，不能把 localStorage 描述成全局策略。
- token 失效、欢迎页和重新登录过程不得提前清空当前用户历史；如果登录成不同用户，则沿用现有规则清空——该规则对两份历史分别独立生效。

## 8. 欢迎页与空白态定制

### 8.1 总体方案与唯一配置入口

定制内容采用“前端内置默认内容 + 外部完整覆盖 Bundle”的模型，而不是在 Server 启动时把客户文件复制到或覆盖前端构建目录。

| 项目 | 位置/配置 | 说明 |
| --- | --- | --- |
| 默认文案 | `lightrag_webui/src/locales/*.json` | 走现有 i18n 流程，天然覆盖当前 WebUI 全部语言；不经服务端 |
| 默认 Logo | `lightrag_webui/src/assets/` 下以模块方式 import | 由 Vite 加内容哈希后随前端产物发布；不经服务端 |
| 生产定制目录 | `UI_TEMPLATES_DIR` | 可选，指向打包目录之外的完整 UI Bundle；建议以只读卷挂载 |

**服务端不携带任何内置 Bundle。** §8.8 本来就要求前端具备一条「拿不到定制内容时渲染自身默认内容」的路径，因此一份服务端内置 Bundle 只是同一件事的第二套实现——它同时把 `ui_defaults/` 的构建复制、打包、原始路径拒绝规则、11 语言的服务端维护，以及「内置 Bundle 缺失是否阻止启动」这一整串交叉判断带了进来。把默认内容留在前端，这些全部消失，而用户可见结果不变；额外收益是默认文案进入现有 i18n 翻译流程，而不是一份没有翻译工作流的服务端资源。

由此得到两条贯穿本节的边界：

- **`UI_TEMPLATES_DIR` 是 customization 子系统存在与否的唯一开关。** 未设置即「无定制」，是常态而非故障；设置了就必须完整可用。
- **customization 与 `/workspace` 产物可用性完全正交。** Bundle 校验是纯文件读取，不依赖 `workspace.html`；两者不存在需要联合判断的交叉状态，因此不存在 active/inactive 概念。

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

“模板”在本期指受限 Markdown 内容，不是可执行的 Jinja/JS/HTML 模板。系统拥有页面框架、登录/进入按钮、查询输入框、语言选择和导航行为；Bundle 只能提供品牌资源及内容区域，不能覆盖这些产品控制项。**这是产品格式边界，不是对 Bundle 的不信任**——见下。

### 8.1.1 信任模型

**Bundle 内容是可信的部署内容。** `UI_TEMPLATES_DIR` 只能由系统管理员设置，目录由管理员挂载，其内容与 `.env`、证书、compose 文件属于同一信任层级：能写入 Bundle 的主体本来就能改配置、换镜像、直接替换前端产物。因此本节各项校验的定位是：

| 校验 | 定位 |
| --- | --- |
| manifest 严格 Schema、字段完整性 | **结构正确性**——让配置错误在启动期可诊断，而不是在用户面前表现为缺文案、缺 Logo |
| 路径 containment、拒绝绝对路径 / `..` / 符号链接逃逸 | **避免误读任意服务器文件**——一条写错的相对路径不应把 `/etc/` 下的东西当作品牌资源公开出去 |
| 文件存在性、单文件大小上限、MIME 与实际内容一致 | **部署可诊断性**——尽早报错，而不是在浏览器里表现为破图或超长响应 |
| Markdown 不含 HTML/JS/CSS | **产品格式边界**——系统拥有页面壳与控制项；不是因为不信任作者 |

由此明确**不做**的事：

- 不对 Markdown 中的链接、图片等内容施加额外安全过滤（`javascript:` 一类仍被 react-markdown 的默认 `urlTransform` 挡住，这是库的默认行为而非本方案新增的过滤层；主动覆写它去放行反而需要额外代码）。
- 不要求 Bundle 总字节、locale 数量或快照总内存有界；保留单文件上限即可。
- 不把 Bundle 当作攻击面来设计威胁模型。**唯一仍然成立的告诫是运维性的**：Bundle 是公开展示内容，不要在其中放入密钥或内部路径——那是信息暴露，与作者是否可信无关。

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
    "ko": ["en"]
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

- Locale key 使用 BCP 47 风格的连字符形式，例如 `zh-TW`，不在 Bundle 中使用前端内部的 `zh_TW`（前端 i18n 资源键与 `src/locales/*.json` 文件名用的是下划线形式）。规范化必须有**单一咽喉**：前端在发起 customization 请求时把内部标识转成连字符形式，Server 只接受连字符形式并拒绝下划线值。不得两侧各写一份转换。
- 每个声明的 locale 必须同时提供 `welcome`、`query_empty` 和非空 `logo_alt`。
- **`brand.logo` 必填**，是所有语言共享的默认 Logo；locale 条目可选的 `logo` 覆盖该语言的 Logo，以满足不同语言使用不同品牌图的需求。不展示任何 Logo 的 Bundle 必须显式写 `"logo": null`，**不得省略该字段**——省略在过去会静默落到 LightRAG 内置 Logo，产出「客户文案 + LightRAG 图标」这种比语言错配更糟的品牌事故。取消内置 Bundle 后服务端已无可回落的对象，把它定成必填是为了让这个空档在 Schema 层就不可表达，而不是留给实现去猜。
- `fallbacks` 是可选的显式有序映射，解析**非递归**：source 可以不是 `locales` 的成员（正是为了处理 Bundle 未覆盖的受支持语言），但每个 target 必须是已声明的 locale。因为只查一层且 target 必然存在，环在结构上不可能出现，故**不做环检测**——上一版要求的“不能出现环”是在“target 必须存在”前提下永远为真的空条款。
- 未在 manifest 中引用的文件不对外提供，也不参与 `bundle_revision` 计算。
- 外部 Bundle **允许只覆盖部分语言**，未覆盖的语言按 §8.3 整体回退到该 Bundle 自己的 `default_locale`。全部受支持语言（`en`、`zh`、`zh-TW`、`fr`、`ar`、`ru`、`ja`、`de`、`uk`、`ko`、`vi`）的默认内容由前端 i18n 资源覆盖，不是 Bundle 的义务。

`WEBUI_TITLE` 和 `WEBUI_DESCRIPTION` 继续是部署级、非本地化的站点标题和描述，由现有 Server 配置拥有；它们不在 manifest 中重复定义。欢迎正文、查询空白态正文和 `logo_alt` 由多语言 Bundle 提供。若未来需要本地化站点标题，应另行扩展 Schema，不能同时维护两个权威来源。

### 8.3 语言选择与回退

前端请求定制内容时按以下顺序确定目标语言：

1. 用户在 WebUI 中显式选择并持久化的语言；
2. 浏览器语言经现有支持语言表规范化后的结果；
3. manifest 的 `default_locale`。

Server 对目标语言按以下顺序解析模板，**单层、不递归**：

1. 精确匹配 locale；
2. 该 locale 在 manifest `fallbacks` 中声明的数组，按顺序取第一个可用 target（target 必然是已声明 locale，因此至多查一层）；
3. 当前 Bundle 的 `default_locale`。

Server 在匹配前对 `requested_locale` 做 BCP 47 规范化：校验格式与长度、统一大小写（语言子标签小写、区域子标签大写），但**继续拒绝下划线形式**——规范化只吸收大小写与格式差异，不承担形态转换，那是前端单一咽喉的职责（§8.2）。

按 §8.3 的前两步，`requested_locale` 通常已被规范化为当前 WebUI 支持的语言之一；`fallbacks` 主要服务于两类情形：Bundle 未覆盖某个受支持语言（例如只翻译了部分语言的客户 Bundle），以及直接调用配置端点时传入的任意 BCP 47 值。

不自动进行区域或书写系统推断，例如不能自行断言 `zh-HK → zh`，因为部署方可能希望它回退到 `zh-TW`。

**原子性规则（取消内置 Bundle 后必须重新表述）**：过去这条规则的对手方是「LightRAG 内置 Bundle」，服务端侧的逐字段混用现在已不可能；新的混用面移到了前端。规则改为：

> Bundle 一旦激活，某个 locale 的**完整表示只能来自该 Bundle 自身**——缺语言时整体回退到它的 `default_locale`，缺字段在启动校验阶段就已被拒绝。**前端默认内容只在两种情形下整体使用：未配置 `UI_TEMPLATES_DIR`，以及 customization 请求硬失败**（§8.8）。任何情况下都不得把 Bundle 内容与前端默认内容逐字段拼接。

具体到最容易发生的一例：Bundle 提供了 `welcome.md` 却没提供 Logo 时，**不得**回落到 LightRAG 默认 Logo；这由 §8.2 的 `brand.logo` 必填在 Schema 层拦住。

阿拉伯语等 RTL 布局方向从系统维护的可信 locale 注册表推导，不允许模板提供 CSS 或任意 `dir` 值。

### 8.4 Server 启动快照与失败语义

Server 启动时按以下规则构造 customization 状态：

1. **未设置 `UI_TEMPLATES_DIR`（默认形态）**：不存在任何 Bundle，customization 快照为空。这不是故障，不产生告警，也不影响任何端点的可用性——`/ui/customization` 照常注册并返回「无定制」表示（§8.5），前端渲染自身默认内容。
2. **显式设置 `UI_TEMPLATES_DIR`**：从该目录读取并**完整校验**一个外部 Bundle，全部通过后原子激活为不可变的 `UICustomizationSnapshot`。
3. manifest、任一已声明 locale、任一被引用资源缺失、不可读、超限或格式非法时，**Server 启动失败**并给出可操作但不泄露敏感内容的错误；不得悄悄改用前端默认内容——那正是运维以为客户品牌已生效、实际展示 LightRAG 内容的失效模式。
4. 不修改 Python 包、容器镜像或前端构建产物；外部目录只以只读方式加载为内存快照。

**校验与 `workspace.html` 是否存在完全正交。** Bundle 校验是纯文件读取，不依赖前端产物；即便查询入口按 §5.2 进入降级，“你显式配置的 Bundle 是坏的”依然是真话，静默忽略只会让运维在补齐构建产物之后才发现配错了。因此这里没有 active/inactive 状态，没有需要联合判断的交叉项，`/health` 也不新增任何 customization 字段。

这条规则相对上一版收紧了一个组合：**已配置 `UI_TEMPLATES_DIR`、Bundle 非法、且恰好使用了陈旧构建目录的部署会整体启动失败。** 明确接受——失败信息是准确的，且陈旧构建目录本身不会凭空注入这个环境变量，触发面比上一版要防的场景窄得多。而上一版为覆盖该场景引入的整套 active/inactive、503 契约与 `UI_TEMPLATES_DIR ignored` 告警，其真正成因是「内置 Bundle 是每个正常部署都存在的强制依赖」；这一前提已经不存在。

启动快照包含已解析的文案、资源字节、MIME、`asset_hash` 和 `bundle_revision`。请求阶段不重新读取磁盘，多个 worker 各自从同一只读目录构造相同快照。修改 `UI_TEMPLATES_DIR` 中的内容后需要重启所有 Server worker；本期不做文件监听或热重载。

模板文件单个限制为 64 KiB，Logo 单个限制为 2 MiB。具体限制应集中定义并在运维文档中说明。

**示例 Bundle 作为文档资产提供。** 取消内置 Bundle 后，客户失去了一份可照着改的参照物。补偿方式是在仓库文档目录中放一份 example bundle：它是纯文档，服务端不加载、不打包进运行时，也不参与任何校验或缓存路径。

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

**未配置 Bundle 时的返回形态**。默认部署没有 Bundle，前端在启动时无从预知这一点，因此必须发起这次请求；它的响应是常态而非故障：

```json
{
  "customized": false,
  "brand": {
    "title": "LightRAG",
    "description": "Simple and Fast RAG"
  }
}
```

规则：

- 两种情形**统一返回 200**，Bundle 激活时 `customized` 为 `true` 并附带上面的完整表示。前端只判这一个布尔，`false` 即整体渲染自身默认内容（§8.3 原子性规则）。
- 不使用 404 或 503 表达「没有定制」。404 会与 §8.7 的「未知资源被拒绝」混淆，503 则暗示一个暂时不可用、稍后会恢复的子系统——而未配置 `UI_TEMPLATES_DIR` 是一个稳定且正确的终态。
- 资源端点 `/ui/customization/assets/...` 在无 Bundle 时不存在任何合法 `asset_hash`，一律按 §8.7 的未知资源处理，返回 404。
- `brand.title` / `brand.description` 在两种情形下都返回，因为它们来自 `WEBUI_TITLE` / `WEBUI_DESCRIPTION` 而不是 Bundle。

### 8.6 Revision、资源标识与缓存失效

缓存模型区分以下三个概念：

| 名称 | 作用 | 变化条件 |
| --- | --- | --- |
| `asset_id` | 稳定的语义标识，例如 `brand-logo` | 资源角色变化时才变，不作为缓存失效值 |
| `asset_hash` | 资源原始字节的 SHA-256 | Logo 等资源内容变化时改变，并进入 URL |
| `bundle_revision` | 整个已激活 Bundle 的确定性摘要 | 任一 manifest、文案或被引用资源变化时改变；**只用于启动日志**。无 Bundle 时不存在 |

缓存规则：

- `GET /ui/customization` 返回 `Cache-Control: no-store`，**不带 `ETag`，不支持条件请求**。响应是一份 64 KiB 上限的文本、每次页面加载取一次、只在 Server 重启后变化，条件请求省下的字节买不回它的代价：一个 locale `revision` 必须对**最终响应表示**计算，也就是要为响应定义一套跨进程、跨平台、跨重启稳定的规范化序列化，并让多 worker 逐字节一致。删掉它，摘要的确定性要求收缩到 `asset_hash`（文件字节的 SHA-256）与 `bundle_revision`（排序后的相对路径 + 原始字节），两者都天然确定性。
- 资源响应使用包含 `asset_hash` 的 URL，并返回 `Cache-Control: public, max-age=31536000, immutable`；同一字节内容可永久复用。
- 只修改日语文案时，`bundle_revision` 改变；未变化的 Logo URL 保持不变，浏览器已缓存的 Logo 继续复用。
- 修改 `WEBUI_TITLE` 或 `WEBUI_DESCRIPTION` 时，`bundle_revision`、`asset_hash` 和 Logo URL 都不变——它们不属于 Bundle。
- 修改外部 `UI_TEMPLATES_DIR` 内容并重启后，文案随 `no-store` 天然失效，资源靠 URL 中的 `asset_hash` 失效。
- **默认内容不经本节机制**：前端默认文案随 JS chunk 走 Vite 的内容哈希，默认 Logo 以模块方式 import 后同样带哈希，两者都由既有的 `assets/` 长期 immutable 策略覆盖，无需服务端参与。
- 因此 `lightrag_webui/public/logo.svg` 若要作为默认 Logo 使用，**必须先移到 `src/assets/` 并以模块方式 import**。留在 `public/` 下的文件名不带内容哈希，浏览器或 CDN 会在版本更新后继续使用旧图——这正是本节要避免的失效模式。
- Bundle 提供的可定制资源一律经上述资源 API 访问，不得由前端拼接任何 Bundle 路径。

`bundle_revision` **只写入启动日志，既不进入匿名层也不进入认证层的 `/health`**，且不得暴露服务器目录。理由是没有它的用武之地：Bundle 非法时 Server 根本起不来，合法快照又不会在运行期变化，健康接口重复暴露一个恒定值只会制造第二个可能漂移的真相源，还要为它回答“属于哪一层”这个问题。取消内置 Bundle 后不再存在 `builtin`/`custom` 两种来源：要么有一个外部 Bundle，要么没有 Bundle，日志相应地只需表达这一件事。若配置中含 locale 列表，可公开返回语言标识，但不返回文件结构。

两个摘要的计算必须跨进程、平台和重启保持确定性，但都不需要任何规范化序列化：`asset_hash` 是文件原始字节的 SHA-256；`bundle_revision` 把被引用文件按规范化后的相对路径排序，再将路径与**原始字节**（manifest 本身也只作为一个文件参与）纳入 SHA-256。多个 worker 对相同输入必须得到相同结果。

### 8.7 内容与资源安全

- manifest 使用严格 Schema，拒绝未知字段、错误类型、重复或非法 locale，以及非法的 fallback source/target、空数组、数组内重复项和不存在的 target。按 §8.2 的单层解析不做环检测。
- 所有相对路径必须解析后仍位于 Bundle 根目录内；拒绝绝对路径、`..` 穿越以及通过符号链接逃逸根目录。定位是避免一条写错的路径把服务器上的任意文件当作品牌资源公开出去（§8.1.1），不是防御恶意管理员。
- 只读取并公开 manifest 明确引用的文件。
- Markdown 不支持原始 HTML、脚本、iframe、表单和事件属性——这是 §8.1.1 的产品格式边界，不是内容过滤；新窗口链接按常规做法带上 `noopener noreferrer`（tab-nabbing 卫生，零成本）。链接与图片的 URL 不做额外过滤。
- Logo 支持 PNG、JPEG、WebP 和 SVG，按实际内容校验 MIME，不只信任扩展名。
- 自定义 SVG 仅作为 `<img>` 资源加载，不以内联 DOM 注入；资源响应设置正确的 `Content-Type`、`X-Content-Type-Options: nosniff` 和限制性 CSP。
- **定制内容使用独立的 Markdown 档位，不复用聊天的 sanitize schema。** 现有 `chatMarkdownSanitizeSchema`（`lightrag_webui/src/utils/markdownSanitizeSchema.ts`，为修复 GHSA-xpjq-3w4w-w5wr 引入）**刻意保留了 `rehypeRaw`**——脚注插件与内联格式化标签依赖原始 HTML，安全性由 allow-list 兜底。定制内容没有这些需求，因此使用不挂 `rehypeRaw`、直接 `skipHtml` 的独立档位。理由是**格式边界与实现简单**（不引入一条只为品牌文案服务的 HTML 通路），不是把 Bundle 当作不可信输入。两档需在代码中显式命名（如 `chat` / `customization`），避免实现时因“统一”二字把两者的行为混在一起。
- 单个视图发生意外渲染错误时显示最小内置纯文本提示，不能让登录或查询页面白屏。
- Bundle 是公开展示内容，运维文档必须提示不得在其中放入密钥、内部路径或其它敏感信息（信息暴露，与作者可信与否无关）。
- 公开 customization API 必须纳入路由和响应字段审计，不能演变为任意文件读取接口——这一条与 Bundle 是否可信无关：端点本身面向未认证的公网，只能提供 manifest 明确引用的那几个文件。

### 8.8 前端加载与切换行为

- 应用启动时并行请求认证状态和当前 locale 的 customization 配置，避免串行增加首屏时间。
- 用户切换语言时重新请求该 locale；新响应成功前保留当前完整内容，成功后一次性切换 Logo、替代文本和两处文案，不能逐字段闪烁。
- 定制快照仅缓存在当前页面内存中，不写入 localStorage，避免部署更新后长期残留旧内容。
- 配置请求暂时失败时可以保留本页面内最近一次成功快照并提供重试；首次加载失败时显示前端默认内容，登录和查询操作仍可使用。
- 主题或语言切换不能绕过 Server 返回的资源 URL，也不能直接拼接 Bundle 路径。
- **首屏不得先渲染前端默认内容再被 Bundle 内容替换。** 请求未完成期间显示 loading 占位（与 §10 第一条一致）；只有在响应明确为 `customized: false` 或请求硬失败时才落到前端默认内容。否则配置了 Bundle 的部署会在每次首屏看到一次 LightRAG 默认内容闪过——这与 §8.3 的原子性规则是同一条要求在时间维度上的表达。
- 前端默认内容必须走**与 Bundle 内容相同的渲染器与 `customization` sanitize 档位**（§8.7），否则「有无 Bundle」会带来可观察的渲染行为差异，且默认路径会失去 allow-list 保护。

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
- Logo 加载失败时隐藏破损图片并继续显示欢迎词；Bundle 显式声明 `"logo": null` 时同样只显示欢迎词，布局不留空位。
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
- customization API 的临时请求或渲染失败只影响定制内容，不影响登录和查询；显式配置的外部 Bundle 若在启动校验阶段失败，则 Server 按 §8.4 fail-fast。收到 `customized: false` 或请求硬失败时，前端整体回落到自身默认内容，不向用户暴露内部状态。
- 所有页面支持键盘导航和可见焦点；颜色对比满足 WCAG 2.1 AA 的常见文本要求。
- Logo 使用有意义的替代文本；纯装饰图标标记为隐藏。
- `prefers-reduced-motion` 下减少非必要动画。
- 当前 WebUI 的全部语言继续可选，前端不对自定义 Markdown 做运行时机器翻译。默认内容由前端 i18n 资源覆盖全部受支持语言；**外部 Bundle 允许只覆盖部分语言**，但每个**已声明**的 locale 都必须字段完整（§8.2），未覆盖的语言按 §8.3 的显式 fallback 与 `default_locale` **整体**回退到该 Bundle 内部，不做逐字段拼接，也不与前端默认内容混用。

## 11. 兼容性与系统同步

### 11.1 兼容性

- `/webui` URL、后台功能、现有书签和静态资源构建流程保持可用。
- `RetrievalView` 继续只表示后台查询页；新增 `WorkspaceQueryView` 不改变现有调用方语义。
- 一处有意的例外：共享 serializer 收紧后，`/webui` 的请求体不再携带 `history_turns`。该字段服务端从未声明、一直被静默丢弃，因此对服务端行为无影响，但属于可观测的请求体变化，需在变更记录中写明。
- 现有 token、API key、guest token 和登录接口契约不变。服务端契约确实不变，但前端会新增**本地** token 有效性校验（§6.3）：过期或结构损坏的 token 不再先渲染应用再被 401 打回，而是直接进入欢迎页/登录页。这是两个入口共同受益的可观测行为变化。
- `settings-storage` 会被部分拆分（§7.4）：`querySettings` 与两份查询历史迁出到独立的 `lightrag::…` 键，主题/语言、图谱偏好以及 `apiKey`、`userPromptHistory`、`queryLabel`、`backendMaxGraphNodes` 等尚未搬迁的状态留在原键。迁移在前端 persist 层完成，复用既有 v1→v21 版本链，须幂等且对缺失字段回落到默认值。
- 本期迁移完整保留**迁移开始前已经持久化**的参数与后台历史；升级后仍运行旧 JavaScript 的标签页继续写回 legacy 状态属于明确例外，其迁移后新增修改允许丢失，新键不被它覆盖（§7.4）。
- 键名已预留 `<ns>` 位并在本期恒为空串。[客户端状态分区与共享认证层](./LR2-client-state-partitioning.md)随后把 `<ns>` 改为由 `apiPrefix` 计算，届时只有带前缀的部署会经历一次状态迁移；本期不涉及。
- 该文档同时收敛 token 本地有效性校验（§6.3 的前置依赖）。
- 当前 `WEBUI_TITLE` / `WEBUI_DESCRIPTION` 继续作为部署级、非本地化站点标题和描述，同时可供查询入口页头使用；欢迎页和空白态正文由 UI Bundle 拥有。
- 未设置 `UI_TEMPLATES_DIR` 的部署升级后自动显示前端默认品牌内容，无需新增配置；显式设置该变量的部署必须提供符合当前 Schema 的完整 Bundle，否则启动失败（与 `workspace.html` 是否存在无关，§8.4）。
- 根路径 `/` 默认仍跳转 `/webui`；只有显式配置 `LIGHTRAG_DEFAULT_UI=workspace` 时才跳转 `/workspace`。
- 构建产物新增 `workspace.html`。用新服务端配旧构建目录时后台仍完整可用，只有查询入口进入不可用降级分支（§5.2）。

### 11.2 必须同步的面

实现不能只新增一个前端路由，还必须同步：

| 面 | 要求 |
| --- | --- |
| Server mount | 同一静态目录挂载 `/webui` 与 `/workspace`，两者均支持 `root_path` |
| 构建配置 | `rollupOptions.input` 产出 `index.html` 与 `workspace.html` 两个入口，均落在输出目录根；两个 HTML 源文件都含运行时配置占位符 |
| Runtime config | 两个 mount 注入逐字节相同的 `{ apiPrefix, webuiPrefix }`；模块级常量无需改造；运行时配置不含入口模式字段；dev 不引入模式开关 |
| 跨入口 HTML | 每个 mount 只提供自己的索引文件（需覆写 Starlette 硬编码的 `index.html`）；另一入口的 HTML 返回 404，同入口显式文件名仍可用 |
| 前端入口分流 | 两个入口各自组合自己的 router 与应用壳；工作区入口的**首屏静态依赖闭包**不含图谱/文档管理（由构建审计插件按 `OutputChunk.modules` 断言，manifest 只能证明 chunk 图与动态边），mermaid 改为动态 import，首屏字节设基线阈值；品牌链接改为 `href="./"`；共享导航单例（含图谱重置适配器）由入口 bootstrap 显式配置 |
| 客户端状态存储 | 按 §7.4 拆出三个 `lightrag::…` 键：`querySettings` 独立 persist key（仅后台运行期可写），两份查询历史各自独立；键名预留 `<ns>` 位、本期恒为空串；迁移复用既有版本链（抽为无副作用纯模块）、单一目标因而无需任何锁、新键优先、先写全部再清理、部分失败可重试；迁移模块是入口文件的第一个静态 import；工作区监听 `storage` 事件重新 hydrate 参数。迁移后旧标签页回写允许丢失且不得覆盖新键。其余站点相关键的搬迁与 `<ns>` 分区见[独立文档](./LR2-client-state-partitioning.md) |
| 根路径/降级 | `LIGHTRAG_DEFAULT_UI` 默认 `webui`，env + CLI 双通道，非法值启动期 fail-fast（含 env 取值）；选择 `workspace` 时保留 `root_path`；无资源时 `/webui` 可沿用 API 文档降级，`/workspace` 只返回无 API 文档链接或引导的固定服务信息 JSON；根路径遵循所选入口自己的降级分支，不改投另一入口 |
| 健康状态 | 保留 `webui_available` 语义，并新增 `workspace_available`；两者由**各自产物**的检查结果派生，且**都留在 `/health` 的公开 liveness 层**——`webui_available` 今天就是匿名可见的 liveness 信号，把它挪进认证层会破坏既有契约，而两个入口是否挂载本就可由请求该路径直接探得。文件系统路径与 Bundle 目录不出现在任何一层；`bundle_revision` **只在启动日志中出现**（§8.6）。customization 不向 `/health` 的任何一层新增字段，它与两个入口的可用性正交 |
| UI 定制加载 | 仅当设置 `UI_TEMPLATES_DIR` 时从该目录构造一个只读快照；未设置即无定制，由前端渲染自身默认内容；绝不修改 WebUI 构建目录 |
| 定制读取 API | 公开端点只返回当前 locale 内容和 manifest 引用资源，并正确处理前缀、内容哈希与安全响应头；配置端点 `no-store` 且不带 `ETag`，资源端点长期 immutable；无 Bundle 时统一返回 200 `customized: false`，不使用 404/503 表达该状态（§8.5） |
| 启动日志 | 同时打印后台和查询入口的实际带前缀 URL；配置了 Bundle 时打印 `bundle_revision` 与校验结果，未配置时打印一行「无定制」；不打印服务端目录 |
| 登录导航 | 所有登录成功、退出、401、guest token 更新路径继续只经 `navigate()` 完成（入口由挂载路径天然保留），未登录默认页由各入口自己的路由表定义 |
| 打包 | PyPI/容器仍只打包一份 `lightrag/api/webui` 产物；不新增任何服务端默认内容目录。默认 Logo 需从 `public/` 移入 `src/assets/` 以获得 Vite 内容哈希（§8.6）。示例 Bundle 只作为文档资产提供，不进运行时 |
| 文档 | 更新 Server/WebUI 启动文档、`env.example`、Bundle Schema、示例 Bundle、Docker 只读挂载示例和多站点部署说明；[多站点部署文档](../MultiSiteDeployment.md) 中描述 `webuiPrefix` 注入与 `/` 降级链路的段落需同步 |
| 路由审计 | 将 `/workspace` mount 和公开 customization/config 面加入完整路由/mount 清单。注意多工作空间方案会引入 `/workspaces*` 管理端点，与本入口的 `/workspace/` 仅差一个字母——不构成前缀冲突，但清单与运维文档中并排出现时需分别标注用途 |

两个入口现在各有自己的构建产物，因此 `webui_available` 与 `workspace_available` 是两个独立的检查结果，而不是同一个布尔的两个视图；`/health` 必须能表达“后台可用、查询入口缺失”这一真实状态。这不等于引入产品级开关：本期没有“关闭查询入口”的配置，未来若增加，再单独设计其默认值和健康语义。

## 12. 验收标准

### 12.1 路由与认证矩阵

| 场景 | 操作 | 期望结果 |
| --- | --- | --- |
| 查询入口、未登录 | 打开 `/workspace/` | 显示欢迎页，不闪现后台或查询参数 |
| 查询入口、已登录 | 打开 `/workspace/` | 直接显示查询页 |
| 查询入口登录 | 欢迎页点击登录并成功 | 返回 `/workspace/#/` |
| 后台登录 | 打开 `/webui/` 并成功登录 | 返回 `/webui/#/` |
| 查询 token 失效 | 查询接口返回不可续期 401 | 停留在 `/workspace/`，进入欢迎/登录流程；重登后回查询页 |
| 后台 token 失效 | 后台接口返回不可续期 401 | 停留在 `/webui/`，回后台登录；重登后回后台 |
| 认证关闭 | 首次访问 `/workspace/` 并点击进入 | 获取 guest token 后进入查询页 |
| 未知 hash 路由 | 在任一入口访问未注册的 `#/xxx` | 回落到当前入口自己的默认页，不跨入口跳转 |
| 跨入口 HTML | 请求 `/workspace/index.html` 与 `/webui/workspace.html` | 均返回 404，不返回另一入口的应用壳 |
| 同入口显式文件名 | 请求 `/webui/index.html` 与 `/workspace/workspace.html` | 正常返回本入口应用壳，资源解析与运行时配置注入均正确 |
| 无硬跳转 | 静态检查前端源码 | 不存在 `location.href=` / `location.replace` / `location.assign` 等改写路径的导航 |
| 品牌链接 | 检查 `/workspace/` 页面壳、导航与控制项中的全部链接（**不含定制 Markdown 正文渲染出的链接**） | 不存在指向 `/webui` 的链接。可信管理员在欢迎文案里自行写的链接不受此约束，也不得使该用例失败 |
| API 前缀 | 在 `/site01` 下完成以上流程 | 所有资源和跳转保留 `/site01` |
| 查询入口资源缺失、API 文档开启 | 打开 `/workspace` 或 `/workspace/` | 返回固定的查询入口不可用服务信息 JSON；不重定向、不链接或引导到 API 文档 |
| 查询入口资源缺失、根路径选择查询入口 | 设置 `LIGHTRAG_DEFAULT_UI=workspace` 后打开 `/` | 进入 `/workspace` 降级分支；即使 API 文档开启也不进入 API 文档 |
| 仅后台产物存在、根路径选择查询入口 | 目录中只有 `index.html` 时设置 `LIGHTRAG_DEFAULT_UI=workspace` 并打开 `/` | 进入查询入口降级 JSON；不改跳 `/webui/`，也不改跳 API 文档；`/webui/` 本身仍正常可用 |

### 12.2 查询功能

- `/workspace` 只渲染独立的 `WorkspaceQueryView`，不挂载 `RetrievalView` 或 `QuerySettings`。
- `RetrievalView` 与 `WorkspaceQueryView` 复用同一个查询会话控制层、消息列表、输入操作层和 `ChatMessage`，不存在复制的流式/停止/历史状态机。
- 不存在 `QuerySettings` DOM、后台 Tab、文档/图谱/API 文档入口或隐藏移动端参数抽屉。
- 输入 `/mix what is RAG` 时整段作为普通问题提交，不覆盖 mode。
- 请求携带当前浏览器中 `/webui` 保存的合法 `top_k`、`mode`、`user_prompt` 等参数；工作区不能编辑或临时覆盖它们。
- **`only_need_*` 钳制**：在 `/webui` 中把 `only_need_context` 或 `only_need_prompt` 置为 `true` 后，`/workspace` 的请求体中两项均为 `false` 且用户仍得到正常答案；同一份 `querySettings` 下 `/webui` 自身的请求体保持 `true`，调试能力不受影响。
- **serializer 等价性**：对同一份 `querySettings`、同一段**显式传入**的历史和同一个问题，共享 serializer 产生逐字段相等的请求体。用例覆盖全部 mode，含 `mode='bypass'`（带最近 3 轮 `conversation_history`）与非 bypass（空 `conversation_history`）。
- **状态隔离**：两个入口的历史相互不可见；在 `bypass` 模式下各自只带本入口最近 3 轮；在一侧清空会话或点停止，不影响另一侧的历史与正在进行的流式响应。
- **参数共享**：`/webui` 保存新参数后，已打开的 `/workspace` 无需刷新即可在下一次查询中使用；`/workspace` 的任何写入都不改变 `query-settings-storage`。
- 两个页面的请求体均不含 `history_turns`。
- **chunk 图证据**：按构建 manifest 递归展开 `workspace.html` 入口的**静态** `imports`，断言 mermaid 只经 `dynamicImports` 到达。
- **源模块证据**：由构建审计插件从首屏可达 chunk 的 `OutputChunk.modules` 生成模块清单，断言其中不含 `GraphViewer`、`DocumentManager`、`RetrievalView`、`stores/graph`、graphology 与 cytoscape。这条不可由 manifest 代替（§5.2）。
- **字节阈值**：工作区首屏传输字节不超过仓库中记录的基线 + 10%，超出即失败。
- 工作区渲染含 mermaid 代码块的答案时能正常出图（动态加载不得降级功能）。
- 流式答案、停止、复制、清空、引用、思考区、公式、Mermaid、滚动跟随与现有后台查询页功能一致。
- 后台查询页仍显示查询参数，并保持现有 `/mode` 前缀行为。
- `LIGHTRAG_DEFAULT_UI` 未设置时 `/` 仍进入 `/webui/`；设置为 `workspace` 时进入 `/workspace/`；非法值启动失败且不会成为重定向 URL。
- **旧标签页回写边界**：升级后仍运行旧 JavaScript 的标签页可以继续访问；它在迁移完成后写回的 legacy 状态只被清理、绝不合并进新键，这部分修改允许丢失。新页面的参数与历史不因旧页面的任何写入而改变。

### 12.3 定制化

- 未设置 `UI_TEMPLATES_DIR` 时，所有支持语言均显示前端默认 Logo、欢迎页和查询欢迎词；`/ui/customization` 返回 200 `customized: false`，不产生任何告警。
- 指向合法完整 Bundle 并重启后，欢迎页与空白态按当前语言显示客户内容，无需重建前端，也不修改任何构建产物。
- 显式配置的 Bundle 出现 manifest Schema 错误、缺少默认语言、缺少 `brand.logo`、非法或不存在的 fallback target、文件缺失/超限、路径逃逸或 Logo MIME 不匹配时，Server 启动失败并报告错误；**该行为与 `workspace.html` 是否存在无关**。
- 精确语言不存在时按 manifest 显式 fallback、再按外部 Bundle 自己的 `default_locale` 回退；**不会回落到前端默认内容**，也不与之逐字段拼接。
- Bundle 激活时首屏不出现「前端默认内容闪过再被客户内容替换」；请求未完成期间显示 loading 占位。
- 用户切换语言后，欢迎页和空白态使用相同解析结果原子更新，RTL 方向正确，切换过程中不出现跨语言字段混合。
- 模板中的 `<script>`、事件属性和 iframe 按格式边界被丢弃；普通链接与图片正常渲染，不被当作可疑内容剥离。
- customization 配置端点返回 `Cache-Control: no-store` 且响应中不含 `ETag`；重启后内容变化立即对客户端可见，无需任何条件请求。
- Logo 字节变化后 `asset_hash` 和资源 URL 改变；仅文案变化时 `asset_id` 和未修改 Logo URL 保持稳定。
- 默认 Logo 与默认文案随前端产物的内容哈希失效，不经 customization API；`public/` 下不存在被直接引用的可定制资源。
- 浏览器响应、健康信息和日志不暴露模板/Logo 的服务端绝对路径或文件内容之外的配置；`bundle_revision` 只出现在启动日志，`/health` 两层均无。

### 12.4 移动端

- 在 §9.1 四种视口中无页面级横向滚动。
- iOS/Android 软键盘展开后输入框和发送/停止按钮可见、可点击。
- 长代码块和表格不撑宽页面；长回答可以平滑滚动。
- 所有主要按钮满足 44 px 触控尺寸并具有可访问名称。
- 旋转屏幕、停止流式查询、向上滚动阅读和清空会话均不产生布局跳变或历史回弹。

## 13. 测试与验证计划

### 13.1 前端单元测试

- 两个入口各自的 router 在 `authenticated/anonymous × auth enabled/disabled` 下的路由矩阵；测试直接渲染对应入口的 router，不注入任何入口标志。
- 未知 hash 路由回落到本入口自己的默认页，不跨入口跳转。
- 共享导航单例在两种 bootstrap 配置下的 401 与退出目标分别为 `#/welcome` 和 `#/login`；未配置时不得静默沿用后台默认值。
- 导航核心在未注册重置适配器时可独立工作，且其模块图不含 `stores/graph`；后台注册适配器后图谱清理照常触发。
- §6.3 的 guest token 矩阵四种状态：有效普通 token、有效 guest token、无有效 token 且认证关闭、无有效 token 且认证开启；并断言欢迎页展示期间 `/auth-status` 附带的 guest token 未被写入 auth store 或 localStorage。
- token 本地校验的用例矩阵见[客户端状态分区与共享认证层](./LR2-client-state-partitioning.md) §7；本 PRD 只需断言「无有效 token」时查询入口落到 `#/welcome`、后台落到 `#/login`。
- `RetrievalView` 与 `WorkspaceQueryView` 的 DOM 差异，以及共享会话控制层的请求 payload 测试。
- **serializer 等价性测试**：参数化跑遍全部 mode，断言共享 serializer 对同一份 `querySettings`、同一段显式传入的历史和同一个问题产生逐字段相等的请求体，并断言 `history_turns` 已被剥离。历史 fixture 是必需的——`bypass` 用例的请求体正是从它切片而来。
- **`only_need_*` 钳制测试**：以 `only_need_context=true` / `only_need_prompt=true` 的 `querySettings` 分别构造两个页面的入参，断言 `WorkspaceQueryView` 交给共享层的快照中两项为 `false`、`RetrievalView` 的保持原值；并断言**共享 serializer 源码中不存在这两个字段的任何分支**（钳制必须在组合层，否则等价性契约被绕过而测试仍会通过）。
- **状态隔离测试**：两个控制层实例注入不同历史存储后，各自的 `bypass` 请求体只含本存储的最近 3 轮；一侧 `clear()` 或 `abort()` 不改变另一侧的消息数组与流式状态。
- **存储拆分测试**：`/workspace` 的历史写入不修改 `query-settings-storage`；`storage` 事件触发后重新 hydrate，下一次查询使用新参数、进行中的流式响应不换快照。
- **迁移测试**：workspace 先打开、`/webui` 先打开、新键已存在（不得被旧值覆盖）、只成功写入部分新键后重试、重复执行结果一致、写入抛错后旧字段仍保留；切换登录用户时两份历史都被清空。
- **历史映射测试**：给定一份非空 legacy `retrievalHistory`，迁移后 `lightrag::webui-retrieval-history` 逐条相等、`lightrag::workspace-retrieval-history` 为空数组。**回归用例**：升级后首次打开 `/workspace`，消息区为空白态（不展示旧后台历史），且此时以 `bypass` 模式发起查询，请求体的 `conversation_history` 为空——旧调试记录既不显示也不发送。
- **版本链测试**：从 v1、v6、v20、v21 各自升级后，查询参数与历史都不丢失（证明复用了既有迁移链而不是把旧版本判为无数据）；损坏 envelope 按无旧数据处理；`version > 21` 时旧字段不被读取也不被清理。
- **重复执行自门控测试**：连续执行迁移三次，断言第二、三次为空操作，不覆盖已被用户改动过的新键值。
- **旧标签页回写测试**：迁移完成后模拟旧 bundle 重新写回一份包含不同参数与历史的完整 `settings-storage` envelope，断言下次启动只清理该 envelope，不覆盖任何新键，且新键的值与旧页面写回的内容无关；明确断言这部分迁移后修改被丢弃，测试名称和说明不得再把它描述成“无损吸收”。
- **崩溃重跑测试**：复制到一半崩溃、复制完成后清理旧 envelope 字段前崩溃——两种情形重跑后都收敛到同一结果，新键不被旧值二次覆盖。
- **版本边界测试**：`version > 21` 时不读、不清理、不覆盖任何旧字段，新键为默认值；`version ≤ 21` 时经**复用**（而非复制）的 v1→v21 链规范化后再拆分，v20 envelope 的参数与历史完整保留。
- **半迁移不可观察测试**：在写入第二个新键前注入失败，断言（1）物理上确实只存在部分新键——不要求崩溃点原子；（2）依赖这些键的 store **未被 hydrate**，入口呈现可重试错误；（3）下次启动补齐缺失键后结果与一次成功执行完全一致；（4）中断那次运行没有把任何默认值写入尚未迁移的键（否则第 4 条会让 legacy 值被永久遮蔽）。
- **启动顺序测试**：断言迁移模块求值完成之前没有任何待拆分 store 被求值（未读取也未写回 localStorage）；并断言迁移纯模块的传递 import 图中不含任何 store、`App`、API 客户端或导航模块。
- **纯模块边界测试**：承载 v1→v21 链的模块其 import 图不含任何 store、`App`、API 客户端或导航模块；import 它不产生 localStorage 读写。
- 工作区不解析 query mode 前缀，但读取后台持久化的合法 `querySettings`；未配置时使用前端默认值。
- 两个页面的流式完成、失败、停止、清空、历史持久化和卸载清理使用同一组共享层测试（历史存储由测试注入，不由共享层选择）。
- 空白态首次显示、发送后隐藏、清空后恢复。
- customization 加载失败、语言切换原子更新、RTL、fallback 测试。
- `customized: false` 响应下整体渲染前端默认内容；Bundle 响应下不出现「默认内容先渲染再被替换」（请求未完成期间为 loading 占位）；断言两条路径经过同一渲染器与 `customization` sanitize 档位。
- Markdown 分档渲染断言：`customization` 档对原始 HTML 直接丢弃（而非净化后保留）——这是格式边界的行为断言；`chat` 档维持现有 `chatMarkdownSanitizeSchema` 行为不回归。定制文案中的普通链接与图片必须正常渲染，不得被当作可疑内容剥离。

### 13.2 后端测试

- `/webui`、`/workspace` 双 mount 的资源检查、尾斜杠、缓存头和无资源降级；覆盖 `ENABLE_API_DOCS=true/false`，断言资源缺失时 `/workspace` 两种尾斜杠形式都不重定向到 API 文档，响应不包含 API 文档 URL 或引导文案。
- 构建产物断言：输出目录根同时存在 `index.html` 与 `workspace.html`，两者的资源引用均为 `./assets/` 相对形式，且都包含运行时配置占位符。
- 每个 mount 只提供自己的索引文件：`/workspace/index.html` 与 `/webui/workspace.html` 返回 404，而 `/webui/index.html` 与 `/workspace/workspace.html` 仍正常返回本入口应用壳——防止实现成“拒绝一切显式 `.html`”而破坏既有别名。
- 只存在 `index.html` 时：`/webui` 正常挂载，`/workspace` 进入固定 JSON 降级，`/health` 的两个可用性字段分别为 true 与 false。
- `LIGHTRAG_DEFAULT_UI` 的默认值、两个合法枚举、非法值拒绝（命令行与环境变量两条通道都必须拒绝）及 `root_path` 重定向。
- 查询入口产物缺失且 `LIGHTRAG_DEFAULT_UI=workspace` 时，根路径在 `ENABLE_API_DOCS=true/false` 下都进入查询入口的固定 JSON 降级响应，既不被 API 文档可用性改写，也不改投 `/webui/`。
- 不设置、设置和组合 `LIGHTRAG_API_PREFIX` 时的运行时配置注入与重定向。
- 外部 Bundle 加载、严格 manifest Schema（含 `brand.logo` 必填与显式 `null` 的接受）、locale 规范化（含大小写归一与下划线拒绝）、单层显式 fallback 与 `default_locale` 兜底、完整 Bundle 原子校验。
- 未设置 `UI_TEMPLATES_DIR` 时：配置端点返回 200 且 `customized=false` 并携带 `brand.title` / `brand.description`，资源端点对任意 `asset_hash` 返回 404，启动无告警。
- **正交性测试**：`workspace.html` 存在与否 × `UI_TEMPLATES_DIR` 未设置/合法/非法，六种组合下 customization 的行为只由后一维决定——非法配置在两种产物状态下都启动失败，合法配置在两种产物状态下都成功加载，未设置时都无告警。
- `/health` 的**匿名层与认证层都不含**任何 customization 字段（含 `bundle_revision`）；该值只出现在启动日志中。
- 路径穿越、绝对路径、符号链接逃逸、未引用文件、文件超限、MIME 不匹配和 SVG 响应头。
- customization 配置端点的部署前缀与 `no-store` 响应头（断言不含 `ETag`），以及资源端点的 `asset_hash`、长期 immutable 缓存和未知资源拒绝。
- 分别修改单一语言文案、共享 Logo、`WEBUI_TITLE`，验证 `bundle_revision` 与 `asset_hash` 的变化边界：改文案只动 `bundle_revision`，改 Logo 两者都动，改 `WEBUI_TITLE` 两者都不动。
- 多 worker 下对同一 Bundle 计算出的 `bundle_revision` 与 `asset_hash` 逐字节一致。
- `/health` 中两个入口可用性字段与实际 mount 一致。
- 实际 `app.routes` 审计包含两个 mount 和公开 customization/config 面。

### 13.3 浏览器验证

- 使用真实浏览器分别验证桌面 Chrome/Firefox 和移动端 Safari/Chrome。移动端验证跑在 dev server 上，查询入口通过 `/workspace.html` 访问，无需任何模式开关。
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

1. **双入口基础设施**：Vite 双 HTML 入口构建，Server 双 mount（各自的索引文件、跨入口 HTML 拒绝与独立产物检查），`LIGHTRAG_DEFAULT_UI` 与根路径跳转，`root_path`、健康状态、降级和打包测试。
2. **入口感知认证**：两个入口各自的 router 与未登录默认页，欢迎页路由，共享导航单例的 bootstrap 配置，登录/退出/401/guest 全链路只经 `navigate()`。
3. **客户端状态边界**：按 §7.4 拆出 `query-settings-storage` 与两份查询历史，把迁移模块置为入口文件的第一个静态 import、把 v1→v21 链抽为纯模块并复用、实现单目标无锁迁移、`storage` 事件重新 hydrate；明确迁移后旧标签页回写允许丢失且不得影响新键。解耦导航核心与图谱 store（重置适配器），`ChatMessage` 的 mermaid 改为动态 import。这一步不引入新页面，可独立在 `/webui` 上验证无回归。

   **前置与顺序**：[客户端状态分区与共享认证层](./LR2-client-state-partitioning.md) 的 token 本地校验（§5）须**先于**第 2 步的欢迎页上线。其 `<ns>` 分区与 legacy 键搬迁则必须**后于**本步：本步会清理 `settings-storage` 中已迁出的字段并升级 envelope 版本号，分区文档的迁移器要按那个升级后的版本作为规范化上界，否则会把新 envelope 判为未知的更高版本而拒绝读取，`apiKey` 永远搬不出来。两份文档的版本上界必须对齐，不得各写各的。
4. **查询共享层与工作区 UI**：从 `RetrievalView` 抽出查询会话、消息列表和输入操作层（历史存储由页面注入）；新增复用 `ChatMessage` 的 `WorkspaceQueryView`、空白态和精简应用壳；工作区入口只 import 查询所需模块，品牌链接改为 `href="./"`；保持后台页面行为不变。
5. **多语言品牌定制**：前端默认内容（i18n 文案 + 移入 `src/assets/` 的默认 Logo）、严格 manifest、外部只读 Bundle 启动快照、locale/fallback、公开读取 API（含无 Bundle 时的 `customized: false`）、`asset_hash` 资源缓存与 `bundle_revision`、安全渲染、示例 Bundle 与运维文档。
6. **移动端收口**：响应式布局、safe-area/软键盘、真实浏览器回归和无障碍检查。

每个 PR 都必须保持 `/webui` 可用，不能等最后一个 PR 才恢复后台入口。

## 15. 风险与对策

| 风险 | 后果 | 对策 |
| --- | --- | --- |
| 只复用 `ChatMessage` | 工作区仍会复制请求、流式、停止、历史和滚动状态机 | 同时抽取共享查询会话、消息列表和输入操作层 |
| 给 `RetrievalView` 堆叠 variant 分支 | 千行组件同时承担后台与移动工作区两套布局 | 两个页面壳组合同一组共享能力 |
| 只隐藏参数侧栏 | `/mode` 前缀仍可临时覆盖参数，或两个页面组装出不同请求 | 使用同一 serializer 读取持久化参数，工作区禁用前缀解析，并以 serializer 等价性测试兜底 |
| 共享历史 | 管理员的调试提问进入查询用户的展示与 `bypass` LLM 上下文 | 两份历史独立存储，共享层只接受注入的存储 |
| 全量继承 `only_need_context` / `only_need_prompt` | 管理员调试后忘记关闭，查询用户每次提问都拿到检索上下文原文或提示词而不是答案，且无从察觉或修改 | 页面组合层在构造快照时强制置 `false`；serializer 不含该字段分支，等价性契约不受影响 |
| 迁移语义不完整 | 谁迁移、写入顺序、部分失败与重复执行未定义，升级后可能丢历史或用旧值覆盖新值 | 新键优先、先写新后清旧、部分失败保留旧字段重试 |
| 把迁移后旧标签页回写也承诺为无损 | 旧代码只认识整份 legacy envelope，合并它会覆盖已经独立演进的新键；不合并则与“绝对无损”措辞冲突 | 兼容承诺截止到迁移开始时已持久化的数据；迁移后旧标签页修改明确允许丢失，新键始终优先。该保证由存储规则自足给出，不依赖任何强制旧页面重载的机制 |
| 一次性把站点分区一起做 | 迁移的影响半径超过本功能：带前缀部署要为一个可能永不打开的查询入口重新录入 `apiKey` | 本期只拆本功能必需的键并预留 `<ns>` 位；分区与 token 收紧移入[独立文档](./LR2-client-state-partitioning.md)，可独立排期与回滚 |
| 清理早于复制 | 中途崩溃直接丢数据 | 顺序固定为先写全部新键、再清理；重跑幂等（只写不存在的键，删已删的键为空操作） |
| 引入 `claimed`/`completed` 状态机 | 状态推进顺序一旦写错（`completed` 早于清理），崩溃后所有运行永久跳过，凭据与历史残留 | 不引入状态机：重复执行天然自门控——字段已迁出、legacy 键已删，后续运行自动成为空操作 |
| 为复用迁移链而 import store 模块 | 迁移器在迁移之前就创建并 hydrate 了待拆分的 persist store | v1→v21 链抽到无副作用纯模块，迁移器与 `persist.migrate` 共同 import；纯模块递归不含 store |
| 迁移模块不在 import 首位 | ESM 按源码顺序求值同级 import，晚于 store 模块就等于没迁 | 约束写死为“入口文件的第一个静态 import”，并以 import 图测试守护 |
| 只接受 v21 envelope | 从 v20 及更早升级的用户被判为无旧数据，参数与历史清回默认值 | 复用既有 v1→v21 迁移链先规范化再拆分；`version > 21` 一律不读不清 |
| `querySettings` 与历史同 persist key | 工作区写入历史时整份写回，覆盖后台刚保存的参数 | 拆出 `query-settings-storage`，仅后台可写 |
| 前端解析 pathname 判断入口 | API 前缀、代理改写或未来路径变更后误判 | 入口由加载的 HTML 产物决定，前端无需也不得解析路径 |
| 两个 HTML 源文件不同步 | 某一入口缺运行时配置占位符，`apiPrefix` 静默为空，部署前缀下该入口全面 404 | 占位符与注入对两个 HTML 同等适用，并加构建产物断言 |
| 品牌链接使用绝对前缀 | 查询用户点 logo 被送到后台，路由矩阵抓不到 | 一律改用文档相对的 `href="./"`，并加“页面内无 `/webui` 链接”验收 |
| 另一入口的 HTML 可直接取到 | `/workspace/index.html` 吐出后台壳，跨入口验收失真 | 每个 mount 只提供自己的索引文件，其余 HTML 返回 404 |
| 旧构建目录缺 `workspace.html` | 若按“缺任一即未构建”处理，后台会被一并判为不可用 | 两个产物独立检查；后台照常挂载，只有查询入口降级 |
| 两套静态构建 | 包体、版本和发布流程漂移 | 一次构建、同目录双挂载 |
| 共享模块间接 import 重依赖 | 查询 API → 导航服务 → 图谱 store → graphology，以及 `ChatMessage` 静态 import mermaid，使工作区入口重新膨胀 | 导航核心改用重置适配器、mermaid 动态 import；chunk 图用 manifest 断言，**源模块闭包用构建审计插件按 `OutputChunk.modules` 断言**，首屏字节设基线 +10% 阈值 |
| 仅用 Vite manifest 证明依赖闭包 | `ManifestChunk` 不列出 chunk 内部的源模块；重依赖被合入公共 chunk 后断言照样通过，退化无人察觉 | 两种证据并用，源模块清单来自 Rollup/Rolldown `OutputChunk.modules` |
| 两档 Markdown 行为被“统一” | 一条只为品牌文案服务的原始 HTML 通路被引入，两档行为漂移 | `chat` / `customization` 两档显式命名分离 |
| 把 Bundle 当作不可信输入设计 | 为可信部署内容堆叠过滤层与容量上限，实现变重、正常的链接与图片被误剥离 | §8.1.1 明确信任模型：校验只服务于结构正确性、避免误读任意服务器文件和部署可诊断性 |
| 启动时复制客户文件覆盖构建目录 | 只读容器/PyPI 安装不可写，多 worker 竞态，升级后残留旧文件 | 外部目录只读加载为内存快照，构建产物永不修改 |
| 显式错误配置被静默回退 | 运维误以为客户品牌已生效，实际展示 LightRAG 默认内容 | 未配置 `UI_TEMPLATES_DIR` 即无定制；显式配置校验失败即启动失败，且与前端产物状态无关 |
| 服务端内置一份默认 Bundle | 与 §8.8 必需的前端默认路径构成同一件事的两套实现，并连带 `ui_defaults/` 打包、原始路径拒绝规则、11 语言服务端维护，以及「内置 Bundle 缺失是否阻止启动」这一整串交叉判断 | 默认内容留在前端 i18n 与 Vite 资源；服务端只认外部 Bundle，customization 与 `/workspace` 可用性正交 |
| Bundle 缺 Logo 时回落到默认 Logo | 客户文案配 LightRAG 图标，比语言错配更糟 | `brand.logo` 必填，不展示须显式写 `null`；Schema 层即不可表达 |
| 首屏先渲染前端默认再被 Bundle 替换 | 配置了 Bundle 的部署每次首屏闪一次 LightRAG 内容 | 请求未完成期间显示 loading 占位，只在 `customized: false` 或硬失败时落到前端默认 |
| 对每个字段独立回退 | 同页混合客户/前端默认品牌或不同语言 | 以完整 Bundle 和 locale 表示为原子单位加载与切换 |
| 用稳定文件名直接缓存 Logo | 客户替换文件后浏览器/CDN 继续展示旧图 | `asset_id` 保持语义稳定，内容 SHA-256 进入资源 URL |
| 为配置端点做 ETag/304 | 需要为响应定义跨进程、跨平台、跨重启稳定的规范化序列化并保证多 worker 逐字节一致——§8 中实现难度最高、最易出微妙 bug 的一条，换来的只是一份 64 KiB 上限、每页取一次的响应偶尔省几 KB | 配置端点改为 `no-store` 不带 `ETag`；确定性要求收缩到 `asset_hash` 与 `bundle_revision`，两者都只对原始字节计算 |
| 自动猜测区域语言回退 | `zh-HK` 等语言回退到错误书写体系 | 只做精确匹配、manifest 显式 fallback 和默认语言回退 |
| customization API 变成任意文件读取 | 未登录攻击者读取 Server 文件 | 严格 manifest、根目录约束、拒绝 symlink 逃逸且只提供已引用资源 |
| Logo/模板响应或渲染失败导致白屏 | 日常入口不可用 | 整体回落到前端默认内容、错误隔离和重试，不阻断认证/查询交互 |
| 把“后台入口”当成授权 | 用户可直接调用管理 API | 明确 UI 分流不是安全边界，API 继续强制授权 |
| `workspace` 术语与多工作空间冲突 | 代码、文档和用户理解混乱 | UI 层称“查询入口”，数据层保留 workspace ID；`/workspace/`（WebUI 入口）与 `/workspaces*`（管理端点）在路由清单中分别标注 |
| 静默继承 `LIGHTRAG-WORKSPACE` 头 | 多工作空间启用后，查询用户不知道自己在问哪个知识库 | 多工作空间落地时页头只读显示当前知识库名（§7.3） |

## 16. 后续可选项

以下能力不进入本期验收：

- 无重启热加载或集中配置推送；
- 在 Bundle 中本地化 `WEBUI_TITLE` / `WEBUI_DESCRIPTION`；
- 按用户/角色显示不同欢迎内容；
- 运维品牌配置管理 UI；
- 面向查询入口的独立域名、PWA、分享链接或会话云同步；
- 查询入口与服务端多工作空间**选择器**集成（只读显示当前知识库名不属于可选项，是多工作空间落地时的前置约束，见 §7.3）。
- 为查询入口单独设计的多轮对话开关及配套的答案缓存策略（本期与 `/webui` 保持一致：仅 `bypass` 模式发送历史）。
