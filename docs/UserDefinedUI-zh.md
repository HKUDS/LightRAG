# 用户自定义 UI 内容（`UI_TEMPLATES_DIR`）

LightRAG 支持用**你自己的**欢迎页、登录页文案、用户协议、查询空白页和品牌
Logo 替换内置内容，并且可以按语言分别提供——无需重新构建前端。你只需要写一个
小目录：若干 Markdown 文件加一个 `manifest.json`，把环境变量 `UI_TEMPLATES_DIR`
指向它，然后重启服务器。

本文是完整指南：可定制的内容、Bundle（模板包）的精确格式、源码 / Docker /
Kubernetes 三种部署方式、如何验证，以及每一条启动错误的含义。

可直接复制的示例包位于 [`docs/ui_templates_example/`](./ui_templates_example/)。

---

## 1. 可以定制什么

| 界面 | 访问者在哪里看到 | Bundle 字段 |
|---|---|---|
| **欢迎页** | `/workspace` 入口，登录之前（`/workspace/#/welcome`） | `welcome`（必填） |
| **查询空白页** | `/workspace` 查询页，对话为空时（点击 *Clear* 后会再次出现） | `query_empty`（必填） |
| **登录页文案** | 用户名/密码表单上方，**两个入口都生效**（`/webui/#/login` 与 `/workspace/#/login`） | `login`（可选） |
| **用户协议 + 同意勾选框** | 登录页上的勾选框，其链接以弹窗打开协议文档；未勾选时前端登录按钮保持禁用——这是前端提示，不是服务端强制（见 [§6](#6-登录同意门禁)） | `agreements`（可选，与 `login` 成对生效） |
| **品牌 Logo** | 欢迎页与查询空白页 | `brand.logo`、locale 级 `logo`、`logo_alt` |

Bundle **不能**设置的内容：

- 浏览器标签标题，以及登录卡片自身的 `LightRAG` 标题与副标题。这些都**硬编码在前端**——
  `index.html` 与 `workspace.html` 里的 `<title>Lightrag</title>`，以及 `LoginPage.tsx`
  里的标题和走 i18n 的 `login.description`。Bundle 和任何环境变量都改不了它们，只能修改
  前端源码并重新构建。
- **登录之后**显示的应用头部（`/webui` 的 `SiteHeader`、`/workspace` 的工作区头部）。这一处
  由 `WEBUI_TITLE` / `WEBUI_DESCRIPTION` 环境变量设置，manifest 有意不允许覆盖。
- 内容周围的按钮、菜单和设置项——参见
  [§5.3 界面语言与 Bundle 语言](#53-界面语言与-bundle-语言)。
- 文字方向。方向由 locale 推导（见 [§5.2](#52-文字方向rtl)），绝不取自
  Bundle 中的标记。

**按 locale 全有或全无。** 访问者要么*完整*看到你 Bundle 中该 locale 的内容，
要么*完整*看到 LightRAG 的内置品牌内容。字段之间绝不逐项混合，因此你无法只提供
一个 Logo 而沿用 LightRAG 的欢迎文案。

---

## 2. 快速开始

### 2.1 源码部署

```bash
# 源码检出时的约定位置，已加入 .gitignore
cp -r docs/ui_templates_example lightrag_webui/ui_templates

# 修改文案，换上自己的 Logo
$EDITOR lightrag_webui/ui_templates/locales/zh/welcome.md
cp /path/to/your-logo.svg lightrag_webui/ui_templates/assets/logo.svg
```

然后在 `.env` 中加入：

```ini
UI_TEMPLATES_DIR=./lightrag_webui/ui_templates
```

启动服务器：

```bash
lightrag-server
```

启动日志会明确告诉你当前处于哪种状态：

```
INFO: UI customization bundle active: bundle_revision=1f0c… locales=['en', 'zh', 'zh-TW']
```

### 2.2 Docker Compose 部署

仓库自带的 compose 文件已经以只读方式挂载了模板目录：

```yaml
    volumes:
      - ./data/ui_templates:/app/data/ui_templates:ro
```

因此只需：

```bash
mkdir -p ./data/ui_templates
cp -r docs/ui_templates_example/* ./data/ui_templates/
```

然后在 `lightrag` 服务的 `environment:` 块中取消注释这一行：

```yaml
      UI_TEMPLATES_DIR: "/app/data/ui_templates"
```

重启：

```bash
docker compose up -d --force-recreate lightrag
```

完整说明（含向导生成的 `docker-compose.final.yml` 与 Kubernetes）见
[§7 部署](#7-部署)。

---

## 3. 目录结构

```
ui_templates/
├── manifest.json            # 唯一索引——其他文件不会被自动发现
├── assets/
│   └── logo.svg             # PNG / JPEG / WebP / SVG
└── locales/
    ├── en/
    │   ├── welcome.md       # 必填
    │   ├── query_empty.md   # 必填
    │   ├── login.md         # 可选
    │   └── agreements.md    # 可选
    ├── zh/
    │   └── …
    └── zh-TW/
        └── …
```

上面的目录名只是约定，不是硬性规则：所有文件都通过 `manifest.json` 定位，manifest
没有引用的文件永远不会被读取，也不会被对外提供。manifest 中的路径相对于 Bundle
根目录；绝对路径、`..` 片段，以及指向 Bundle 之外的符号链接都会在启动时被拒绝。

---

## 4. `manifest.json` 参考

```jsonc
{
  "schema_version": 1,
  "default_locale": "zh",
  "fallbacks": {
    "ko": ["en"],
    "ja": ["en"]
  },
  "brand": {
    "logo": "assets/logo.svg"
  },
  "locales": {
    "zh": {
      "welcome": "locales/zh/welcome.md",
      "query_empty": "locales/zh/query_empty.md",
      "login": "locales/zh/login.md",
      "agreements": "locales/zh/agreements.md",
      "logo_alt": "示例公司"
    },
    "en": {
      "welcome": "locales/en/welcome.md",
      "query_empty": "locales/en/query_empty.md",
      "login": "locales/en/login.md",
      "agreements": "locales/en/agreements.md",
      "logo_alt": "Example Corp"
    }
  }
}
```

> JSON 不支持注释——上面标注 `jsonc` 只是便于阅读。真实的 manifest 必须是纯 JSON。

### 4.1 顶层字段

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `schema_version` | 是 | number | 必须为 `1`。 |
| `default_locale` | 是 | string | 必须是 `locales` 中已声明的某个 key。访问者的 locale 无法匹配时使用。 |
| `brand` | 是 | object | 有且仅有一个 key：`logo`。 |
| `brand.logo` | 是 | string \| `null` | 默认 Logo 的路径；显式写 `null` 表示「本部署不显示 Logo」。**缺省该 key 会导致启动失败**——绝不允许在你的文案下方悄悄回退到 LightRAG 的 Logo。 |
| `locales` | 是 | object | 非空的 locale → 条目映射。 |
| `fallbacks` | 否 | object \| `null` | 把*未覆盖*的 locale 映射到已声明的 locale。见 [§5.1](#51-locale-解析)。 |

顶层出现未知字段会报错，因此拼写错误（如 `defaultLocale`）会在启动时被指出，而不是
被悄悄忽略。

### 4.2 locale 条目

| 字段 | 必填 | 类型 | 说明 |
|---|---|---|---|
| `welcome` | 是 | string | 欢迎页 Markdown 的路径。 |
| `query_empty` | 是 | string | 查询空白页 Markdown 的路径。 |
| `logo_alt` | 是 | string | 非空的 Logo 替代文本，使用该 locale 的语言书写。 |
| `logo` | 否 | string \| `null` | 覆盖本 locale 的 `brand.logo`（`null` = 本 locale 不显示 Logo）。key **不存在**时继承 `brand.logo`。 |
| `login` | 否 | string \| `null` | 登录页文案的路径。 |
| `agreements` | 否 | string \| `null` | 单份用户协议文档的路径。 |

`login` 与 `agreements` 同时声明才会开启登录同意门禁——见
[§6](#6-登录同意门禁)。

### 4.3 locale key 的写法

key 是 BCP 47 标签的**连字符**形式，且必须写成已归一化的形态：

- 语言小写（`zh`、`en`、`ar`）；
- 4 字母 script 首字母大写（`Hant`、`Arab`）；
- 2 字母地区全大写（`TW`、`CN`）；
- `zh_TW`（下划线）会被拒绝——请写 `zh-TW`；
- `zh-tw` 同样会被拒绝：key 必须已经是归一化形式（`zh-TW`）。

接受的范围是**以语言子标签开头的 BCP 47 标签**——远比前端自带的语言宽，但并非字面意义上
的「任何合法标签」：

| 规则 | 接受 | 拒绝 |
|---|---|---|
| 主语言子标签为 2–8 个字母 | `en`、`zh`、`art-lojban` | `x-acme`、`i-klingon`——私有用途与 grandfathered 标签以单个字母开头 |
| 其后每个子标签为 1–8 个字母数字 | `zh-Hant-TW`、`de-CH-1901`、`sl-rozaj-biske` | `abcdefghi` |
| 整个标签不超过 35 个字符 | | 更长的标签 |

因此 script、地区、variant 和 extension 子标签都可用——`zh-Hant-TW`、`ar-aao-Latn`、
`en-US-u-VA-posix`——实际排除的是私有用途（`x-…`）和 grandfathered（`i-…`）两种形式。

另请注意：归一化是按位置作用于*每一个*子标签的，包括 extension 自身的取值——
`en-US-u-va-posix` 会被归一化为 `en-US-u-VA-posix`，而 key 必须已经是归一化形式，
所以要按后者书写。

若要声明前端自带语言之外的 locale，请先阅读 [§5.3](#53-界面语言与-bundle-语言)。

### 4.4 限制与格式

| 规则 | 数值 |
|---|---|
| 单个 Markdown 模板大小 | ≤ 64 KiB |
| `manifest.json` 大小 | ≤ 64 KiB |
| 单个 Logo 大小 | ≤ 2 MiB |
| 模板编码 | UTF-8（非法 UTF-8 会导致启动失败） |
| Logo 格式 | PNG、JPEG、WebP、SVG——**依据文件字节判断，而不是扩展名** |

SVG 必须真正存在 `<svg>` 根元素（其前可以有 XML 声明、注释、DOCTYPE 或处理指令）。
带命名空间前缀的根元素（`<s:svg>`）以及 UTF-16/32 编码的文件不被识别——没有任何
SVG 工具会产出这类文件。

---

## 5. 多语言

### 5.1 locale 解析

前端只会请求一个 locale：它为当前访问者解析出的界面语言（界面中显式设置 > 浏览器
语言 > `en`）。服务端随后进行**单跳**解析：

1. 与已声明 locale **精确匹配** → 采用；
2. 否则取 `fallbacks.<请求的 locale>` 中第一个*已声明*的目标；
3. 否则使用 `default_locale`。

`fallbacks` 的每个目标都必须是已声明的 locale，这正是让解析保持单跳、结构上不可能
出现环的原因。而来源侧可以是任意 locale——把未覆盖的语言导向合适的位置正是这张映射
表的全部意义：

```json
"fallbacks": {
  "ko": ["en"],
  "ja": ["en"],
  "de": ["en"]
}
```

不写条目时，未覆盖的 locale 会直接落到 `default_locale`，所以只有当不同的未覆盖语言
需要落到*不同*位置时（例如 `zh-HK` → `zh-TW`，其余 → `en`）才需要 `fallbacks`。

### 5.2 文字方向（RTL）

方向由解析后的 locale 对照 CLDR 派生的注册表推导后下发给浏览器，Bundle 无法设置。
显式的 script 子标签优先于语言本身，这也是默认判断不符合预期时的逃生舱：

- `ar`、`he`、`fa`、`ur`、`ps`、`ckb`、`dv`… → 从右向左；
- `ku` → 从左向右，但 `ku-Arab` → 从右向左；
- `az-Arab`、`pa-Arab`、`ha-Arab` 同理。

### 5.3 界面语言与 Bundle 语言

Bundle 的语言集合与前端界面的语言集合是**相互独立**的。前端自带界面翻译（按钮、
设置、登录标签、同意勾选框文案）的语言为：

`en`、`zh`、`zh-TW`、`fr`、`ar`、`ru`、`ja`、`de`、`uk`、`ko`、`vi`

Bundle 可以声明该列表之外的 locale，例如 `nl`。其内容（包括方向）会被正确渲染，
但周围的控件仍停留在访问者解析出的界面语言上，因为并不存在可切换的荷兰语界面翻译。
启动时会记录一条警告，列出这些 locale：

```
WARNING: UI customization: the WebUI ships no interface translation for ['nl'] …
```

若希望整页语言一致，请声明上述列表中的 locale。

---

## 6. 登录同意门禁

当某个 locale **同时**声明了 `login` 和 `agreements` 时，该 locale 的登录页会显示：

- 表单上方你的 `login` Markdown 文案；
- 一个勾选框，中文界面下文案为「同意《用户隐私协议》和《模型服务协议》」，其中
  唯一的链接会以弹窗打开你的 `agreements` 文档。

未勾选时**登录**按钮保持禁用，在表单中按回车同样会被拒绝。

> **这个门禁的确切边界——依赖它之前请先读这一段。** 它是**前端界面的控制，而不是服务端强制**。
> 服务端只负责计算 `consent_required` 并由前端遵守，但 `POST /login` 只接受标准的凭据字段：
> 它既不要求也不记录「已接受」，更不会保存用户同意的是文档的哪个版本。任何直接向 `/login`
> 提交凭据的客户端——curl、脚本、Ollama 兼容接口、另一个前端——都能拿到 token，根本不会看到
> 勾选框。
>
> 因此请把它当作**面向 WebUI 使用者的知情同意提示**，而不是访问控制；也不要把它当作「某用户
> 接受过某版本协议」的证据。如果你的部署需要可强制、可审计的接受记录，必须在服务端另行实现，
> 本功能不提供。

### 6.1 一份文档，而不是两份

勾选框只承载**一个**链接。请把隐私协议和模型服务协议写进同一个 `agreements.md`，
用标题分隔：

```markdown
## 用户隐私协议

…

## 模型服务协议

…
```

勾选框的文案本身不可定制——它来自前端自身的翻译，因此在每种界面语言下都已经是自然的
表达。请让你文档中的标题与该文案保持一致。

### 6.2 需要知道的规则

- **要么都写，要么都不写。** 只写 `login` 会得到一个有品牌文案但没有门禁的登录页；
  只写 `agreements` 则是一份没有任何入口链接的文档。两者都不会开启门禁——半份配置
  就按半份配置处理，不会被当作已获得同意。
- **按 locale 生效。** 解析到的 locale 若两个字段都没声明，访问者就看不到勾选框。
  请为每个需要门禁的 locale 都声明这一对字段，或者用 `fallbacks` 把未覆盖的
  locale 导向已声明的 locale。
- **声明了但内容为空的文件会导致启动失败。** 门禁绝不能指向一份空白文档。
- **勾选状态不会被记住。** 它只存活于登录页本身，并且绑定到屏幕上那份文档的确切
  文本，因此每次访问都需要重新勾选。中途切换界面语言会替换文档并清除勾选，除非新
  locale 的文本逐字节相同。
- **仅作用于 WebUI，不覆盖 API。** 同上：`POST /login` 没有同意相关字段，因此该门禁只
  约束前端登录表单，此外别无约束。
- **仅覆盖账号密码登录。** 未配置认证（`AUTH_ACCOUNTS` 未设置）的部署会以 guest
  身份直接放行，不受门禁约束。这是刻意设计而非缺口：没有认证就没有可被约束的用户
  身份，而免认证本就是开发/演示形态。**若必须要求接受协议，请配置
  `AUTH_ACCOUNTS`**（并配置 `TOKEN_SECRET`）。
- **服务不可达时 fail-open。** 若定制内容接口在重试后仍无法加载，门禁会保持关闭，
  而不是把所有人锁在部署之外。

---

## 7. 部署

### 7.1 模板目录放在哪里

| 部署方式 | 模板目录 | `UI_TEMPLATES_DIR` |
|---|---|---|
| 源码部署 | `lightrag_webui/ui_templates/`（已 gitignore） | `./lightrag_webui/ui_templates` |
| Docker / Compose | 宿主机 `./data/ui_templates/` → 容器 `/app/data/ui_templates` | `/app/data/ui_templates` |
| Kubernetes | ConfigMap 或 PVC 挂载到 `/app/data/ui_templates` | `/app/data/ui_templates` |

相对路径以服务器进程的工作目录为基准解析，因此在不确定进程从哪里启动时，使用绝对
路径更稳妥。

### 7.2 Docker Compose

`docker-compose.yml` 和 `docker-compose-full.yml` 都已内置该挂载，并附带一行注释掉的
启用配置：

```yaml
services:
  lightrag:
    volumes:
      - ./data/ui_templates:/app/data/ui_templates:ro
    environment:
      # UI_TEMPLATES_DIR: "/app/data/ui_templates"
```

挂载本身是惰性的：只要 `UI_TEMPLATES_DIR` 未设置，`./data/ui_templates` 不存在或为空
都不会有任何影响。取消注释那一行环境变量，是启用该功能的唯一步骤。

两点实务提示：

- **首次 `up` 之前先创建该目录。** 如果交给 Docker 自动创建，目录属主会是 `root`，
  之后往里复制文件需要 `sudo`。
- `:ro` 是刻意的——服务器只读取模板包，从不写入。
- **Podman**：`docker-compose.podman.yml` 里连挂载本身也是注释掉的。Podman 对「宿主机源
  不存在」的绑定挂载比 Docker 更严格，无条件挂载会把一个默认关闭的功能变成所有人的启动
  前置条件。在那里请先 `mkdir -p ./data/ui_templates`，再同时取消注释挂载和
  `UI_TEMPLATES_DIR`。

在 `.env` 里设置 `UI_TEMPLATES_DIR` 同样有效（该文件会被挂载进容器），但更适合放在
compose 的 `environment:` 块中：这个值是*容器内*路径，把容器路径挡在 `.env` 之外，
才能让同一份 `.env` 在源码运行时依然可用。另外请注意：compose 的 `environment:`
条目优先级高于 `.env` 中的同名 key。

### 7.3 向导生成的 compose 文件

`make env-base` / `make env-storage` / `make env-server` 会生成
`docker-compose.final.yml`。生成器现在会在该挂载不存在时补上同样的只读挂载，因此
重新生成已有文件即可获得：

```bash
make env-server        # 或任意其他 make env-* 目标
grep ui_templates docker-compose.final.yml
```

向导**不会**替你写入 `UI_TEMPLATES_DIR`——请自行把它加到
`docker-compose.final.yml` 中 `lightrag` 服务的 `environment:` 块里。向导在重新生成
时会保留该服务中用户新增的环境变量和绑定挂载。

### 7.4 Kubernetes

自带的 Helm Chart 目前还没有专门的配置项，因此需要自行修改 Deployment 来挂载模板包。

**ConfigMap 没有目录结构。** 它的 key 是扁平的，每个 key 直接成为挂载点下的一个文件；
`--from-file=<目录>` 只会把该目录下的文件按其基本名打包，并跳过所有子目录。因此以
ConfigMap 方式交付的模板包必须是扁平的，且 manifest 中的路径要与这些 key 完全一致。
直接照搬嵌套的示例布局会导致启动失败，报
`'locales/zh/welcome.md' does not exist or is not a file`。

为这种部署单独写一份扁平的 manifest：

```json
{
  "schema_version": 1,
  "default_locale": "zh",
  "brand": { "logo": "logo.svg" },
  "locales": {
    "zh": {
      "welcome": "welcome.zh.md",
      "query_empty": "query_empty.zh.md",
      "login": "login.zh.md",
      "agreements": "agreements.zh.md",
      "logo_alt": "示例公司"
    }
  }
}
```

并逐个显式指定被引用的文件（包括 Logo）——`key=路径` 的形式已经完成了扁平化，因此你的
源码目录可以保持嵌套。manifest 引用了但 ConfigMap 中缺失的文件会导致启动失败：

```bash
kubectl create configmap lightrag-ui-templates \
  --from-file=manifest.json=./k8s/ui-manifest.json \
  --from-file=welcome.zh.md=./ui_templates/locales/zh/welcome.md \
  --from-file=query_empty.zh.md=./ui_templates/locales/zh/query_empty.md \
  --from-file=login.zh.md=./ui_templates/locales/zh/login.md \
  --from-file=agreements.zh.md=./ui_templates/locales/zh/agreements.md \
  --from-file=logo.svg=./ui_templates/assets/logo.svg \
  --dry-run=client -o yaml | kubectl apply -f -
```

非 UTF-8 的 Logo（PNG、JPEG、WebP）会被 `kubectl` 自动放入 ConfigMap 的
`binaryData`。但请注意 ConfigMap 总大小上限约为 1 MiB，低于本功能单个 Logo 2 MiB 的
限制：Logo 较大的模板包需要改用 PVC，PVC 同时也允许保留嵌套目录结构。

然后在容器上添加：

```yaml
          volumeMounts:
            - name: ui-templates
              mountPath: /app/data/ui_templates
              readOnly: true
          env:
            - name: UI_TEMPLATES_DIR
              value: /app/data/ui_templates
      volumes:
        - name: ui-templates
          configMap:
            name: lightrag-ui-templates
```

`mountPath` 要与 `UI_TEMPLATES_DIR` 保持一致。投射卷内部的 `..data` 符号链接始终指向
挂载点内部，因此模板包的路径包含性校验可以通过——扁平的 ConfigMap 挂载与磁盘上的普通
目录加载表现完全一致。更新 ConfigMap **不会**重新加载模板包，请重启 Pod（见 §7.5）。

### 7.5 让修改生效

**没有热加载。** 整个模板包在启动时被一次性校验，并作为不可变的内存快照激活；此后
处理请求不再访问磁盘。修改模板包后需要重启服务器——使用 `lightrag-gunicorn` 或任何
多 worker 部署时，必须重启**所有** worker，否则部分 worker 仍在提供旧版本内容。

不需要手动清缓存：文案接口以 `Cache-Control: no-store` 返回，Logo 的 URL 内嵌文件
内容哈希，字节变化就会产生新的 URL。

---

## 8. 验证部署结果

**启动日志。** 以下两行必有其一：

```
INFO: UI customization: no bundle configured (UI_TEMPLATES_DIR unset)
INFO: UI customization bundle active: bundle_revision=<sha256> locales=['en', 'zh']
```

`bundle_revision` 是对所有被引用文件计算出的哈希。如果你改了文件后它没有变化，说明
服务器读取的目录并非你以为的那个——或者根本没有真正重启。

**接口。** 该接口是公开的（欢迎页在登录之前显示），因此直接 `curl` 即可：

```bash
curl -s 'http://localhost:9621/ui/customization?locale=zh' | jq
```

```json
{
  "customized": true,
  "requested_locale": "zh",
  "locale": "zh",
  "fallback_used": false,
  "direction": "ltr",
  "brand": {
    "title": "My Graph KB",
    "description": "Simple and Fast Graph Based RAG System",
    "logo_url": "/ui/customization/assets/9f2a…/brand-logo",
    "logo_alt": "示例公司"
  },
  "welcome": { "format": "markdown", "content": "## 欢迎…" },
  "query_empty": { "format": "markdown", "content": "…" },
  "login": { "format": "markdown", "content": "…" },
  "agreements": { "format": "markdown", "content": "…" },
  "consent_required": true
}
```

几个有用的判断点：

- `"customized": false` → 当前没有激活任何模板包（`UI_TEMPLATES_DIR` 未设置或为空），
  前端显示的是 LightRAG 内置品牌内容。
- `"fallback_used": true` → 请求的 locale 未被声明，`locale` 字段告诉你最终落到了哪里。
- `"consent_required"` → 该 locale 下是否会出现同意勾选框。
- `logo_url: null` → 该 locale 解析结果是*不显示* Logo（某处显式写了 `null`），而不是
  回退到 LightRAG 的 Logo。

**浏览器。** 访问 `/workspace` 查看欢迎页，访问 `/workspace/#/login` 或
`/webui/#/login` 查看登录页，并在设置菜单中切换界面语言逐一检查各个 locale。

---

## 9. 故障排查

只要设置了 `UI_TEMPLATES_DIR`，模板包中任何一处非法都会让服务器**拒绝启动**，错误
信息以 `UI_TEMPLATES_DIR bundle invalid:` 开头。这是刻意设计：悄悄回退到 LightRAG
内容会让你误以为客户品牌已经生效，而实际上并没有。

| 错误信息（节选） | 原因 / 处理 |
|---|---|
| `directory '…' does not exist or is not a directory` | 路径错误，或容器挂载缺失。进容器确认：`docker compose exec lightrag ls /app/data/ui_templates`。 |
| `manifest.json is missing` | `manifest.json` 必须直接位于模板包根目录，而不是下一层。常见原因是复制了父目录。 |
| `manifest.json is not valid JSON` | 多余的逗号或注释。JSON 两者都不允许。 |
| `unknown field(s) [...]` | 字段名拼写错误；schema 是封闭的，属于有意设计。 |
| `missing required field(s) [...]` | 补上该字段。注意 `brand.logo` 是必填的——不显示 Logo 请显式写 `null`。 |
| `unsupported schema_version` | 必须恰好为 `1`。 |
| `locales: key 'zh_TW' uses the underscore form` | 请写 `zh-TW`。 |
| `locales: key 'x-acme' has an invalid language subtag` | 主语言子标签必须是 2–8 个字母，因此私有用途（`x-…`）和 grandfathered（`i-…`）标签不受支持——见 [§4.3](#43-locale-key-的写法)。 |
| `locales: key 'zh-tw' must be written in its normalized form 'zh-TW'` | 修正大小写。 |
| `default_locale '…' is not a declared locale` | `default_locale` 必须出现在 `locales` 中。 |
| `fallbacks.xx: target 'yy' is not a declared locale` | fallback 的目标必须已声明；只有来源可以是未覆盖的 locale。 |
| `…: '…' does not exist or is not a file` | manifest 中某个路径指向了不存在的文件。路径相对于模板包根目录，且区分大小写。 |
| `…: absolute paths are not allowed` / `path traversal is not allowed` / `escapes the bundle directory` | 所有被引用的文件（包括符号链接的目标）都必须位于模板包内部。 |
| `…: file exceeds the … byte limit` | 模板 64 KiB、Logo 2 MiB。 |
| `…: file is not valid UTF-8` | 用 UTF-8 重新保存该 Markdown。 |
| `logo '…': content is not PNG, JPEG, WebP or SVG` | 文件字节不匹配任何受支持格式——例如后缀是 `.svg` 实际却是 HTML，或 SVG 缺少根元素/根元素带命名空间前缀。 |
| `locales.xx.login: template file is empty` | `login` / `agreements` 是「声明与否」的开关，声明了却为空会被拒绝。`welcome` / `query_empty` 允许为空。 |
| `locales.xx.logo_alt must be a non-empty string` | 为每个 locale 提供真实的替代文本。 |

以下现象**不会**导致启动失败：

| 现象 | 原因 |
|---|---|
| 服务器正常启动，但页面仍是 LightRAG 品牌内容 | `UI_TEMPLATES_DIR` 未设置或为空——检查启动日志和 `/ui/customization`。Docker 下请注意 compose 的 `environment:` 会覆盖 `.env`。 |
| 修改后不生效 | 没有热加载。请重启服务器（所有 worker）。 |
| 看不到同意勾选框 | 解析到的 locale 只声明了 `login` / `agreements` 之一；或未配置认证（`AUTH_ACCOUNTS` 未设置）；或访问者解析到的 locale 与你预期的不同——检查接口返回中的 `locale` 与 `consent_required`。 |
| 内容是你的语言，按钮却不是 | 该 locale 不在前端界面语言集合内——见 [§5.3](#53-界面语言与-bundle-语言) 及启动警告。 |
| Logo 不显示 | 该 locale（或 `brand`）解析结果为 `null`；或浏览器加载资源 URL 失败——直接请求 `logo_url` 查看状态码。 |

---

## 10. 内容撰写规则

- **Markdown 并支持 GFM**（表格、删除线、任务列表）。链接和图片正常可用，链接会在
  新标签页打开。
- **原始 HTML 会被丢弃。** 这是格式边界而非不信任：定制内容这一层根本不开放 HTML
  通道。请用 Markdown 表达排版。
- **方向不由你设置。** 不要写 `dir` 属性或 CSS——见 [§5.2](#52-文字方向rtl)。
- **保持简短。** 欢迎页文案在手机上位于首屏；查询空白页只是 Logo 下方居中的一段话。

---

## 11. 安全说明

- 模板包属于**受信任的部署内容**——与 `.env`、compose 文件同一信任层级。校验的目的是
  捕获配置错误，而不是防范模板包作者本人。
- 其内容以**免认证**方式对外提供，因为欢迎页和登录页都在登录之前。切勿在其中放入
  密钥、内部主机名或私有路径。
- 只有 manifest 引用到的文件才会被读取和提供。该接口无法用于读取服务器上的任意文件，
  逃逸出模板包根目录的路径会在加载时被拒绝。
- Logo 响应携带 `X-Content-Type-Options: nosniff` 和严格的
  `Content-Security-Policy`，因此即便被当作顶层文档打开，这里提供的 SVG 也是惰性的。

---

## 12. 参见

- [`docs/ui_templates_example/`](./ui_templates_example/) —— 完整可复制的示例包
  （`en`、`zh`、`zh-TW`）。
- [LightRAG-API-Server-zh.md](./LightRAG-API-Server-zh.md) —— 完整的服务器指南，
  包含 `AUTH_ACCOUNTS` 与 `TOKEN_SECRET`。
- [DockerDeployment.md](./DockerDeployment.md) —— Docker 与 Compose 部署。
- [InteractiveSetup.md](./InteractiveSetup.md) —— `make env-*` 向导与
  `docker-compose.final.yml`。
