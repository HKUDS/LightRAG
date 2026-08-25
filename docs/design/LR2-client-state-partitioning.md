# 设计：客户端状态按站点分区与共享认证层收紧

- 状态：草案（2026-08-26 从 [查询用户 WebUI 入口 PRD](./LR2-query-user-workspace-webui.md) §7.4 / §6.3 拆出）
- 适用范围：WebUI（`lightrag_webui/`）的 localStorage / sessionStorage 边界与启动期认证状态
- 关键约束：修复同源多站点部署下的**凭据串用**，不改变服务端契约；可独立于任何 UI 功能上线并在 `/webui` 上单独验证

## 1. 为什么单独成篇

本篇解决的两个问题**今天就存在**，与是否新增查询入口无关：

1. **`apiKey` 跨站点串用。** localStorage 按 origin 隔离而**不按路径**隔离，而项目明确支持同一 host 下的多站点部署（`https://host/site01/webui/` 与 `https://host/site02/webui/`，见[多站点部署文档](../MultiSiteDeployment.md)）。固定键名意味着 site01 的 `apiKey` 会被作为 `X-API-Key` 发给 site02——这是凭据泄露，不是数据串用。查询历史、`queryLabel`、后端上报的上限值同理。
2. **token 只判存在性。** `initAuthState`（`stores/state.ts`）只检查 `LIGHTRAG-API-TOKEN` 是否**存在**就置 `isAuthenticated: true`；它虽然顺带算出了 `tokenExpiresAt`，却不用它来把关。过期或结构损坏的 token 因此会被判为已登录，用户先看到完整应用再被 401 打回。

两者都属于**共享层**改造：不引入新页面，可独立在 `/webui` 上验证无回归。之所以从查询入口 PRD 拆出，是因为它们的影响半径超过那个功能——分区迁移会让**带前缀的部署**重新录入一次 `apiKey`，而其中相当一部分永远不会打开查询入口。用一个 UI 功能的排期去绑定一次面向全体部署的凭据修复，两头都不划算。

**前置关系**：查询入口 PRD 的 §6.3「无有效 token → 欢迎页」这一行依赖本篇第 4 节的 token 本地校验，本篇必须先于该功能上线。第 3 节的分区则不构成阻塞——查询入口 PRD 已经把它需要的键拆到了 `lightrag::` 形状（空命名空间），本篇只是把 `<ns>` 从「恒为空」改为「由 `apiPrefix` 计算」。

## 2. 起点：查询入口 PRD 已经完成的部分

查询入口 PRD 为了让「两个入口共享查询参数、隔离查询历史」成立，已经把 `settings-storage` 拆成：

| localStorage key | 内容 |
| --- | --- |
| `lightrag::query-settings-storage` | `querySettings` |
| `lightrag::webui-retrieval-history` | 后台查询历史 |
| `lightrag::workspace-retrieval-history` | 工作区查询历史 |
| `settings-storage`（保留） | 主题、语言、图谱显示偏好，以及尚未搬迁的站点相关状态 |

键的**形状**（`lightrag:<ns>:<name>`）在那一步就已固定，`<ns>` 恒为空串——因此根部署与直连端口部署在本篇落地时**一个字节都不用动**，只有带前缀的部署会看到命名空间变化。这不是巧合，是刻意的排序：把不可避免的那次迁移（拆 key）与可选的那次（分区）错开，各自的失败面就不会叠加。

## 3. `<ns>` 分区

### 3.1 判别式的选择

`<ns>` 取 `normalizeApiPrefix()` 的返回值（根部署为空串，故键形如 `lightrag::query-settings-storage`——双冒号是有意保留的，使键形状统一、解析无歧义）。同一站点的 `/webui` 与 `/workspace` 共用同一命名空间；`storage` 事件处理器只响应本命名空间的键。

**`<ns>` 是服务实例的属性，不是 URL 的属性——因此它是精确判别式，而非保守近似。** 注入给前端的 `apiPrefix` 由 `lightrag/api/lightrag_server.py` 在创建 app 时从该实例自己的 `LIGHTRAG_API_PREFIX` 烘焙而成（`normalize_api_prefix(args.api_prefix)`），既不逐请求从浏览器 URL 推导，也不读取 `X-Forwarded-Prefix`（该头在 `lightrag/` 中无任何消费者，多站点文档里那行只是给 nginx 侧的惯例声明）。由此：

| 部署形态 | 注入的 `apiPrefix` | `<ns>` | 结果 |
| --- | --- | --- | --- |
| 两实例、两前缀（多站点文档形态，`:9621` / `:9622`） | `/site01` / `/site02` | 不同 | 按预期隔离 |
| 同一实例被挂在两个反向代理 location 下 | 都是该实例配置的同一个值 | **相同** | 状态共享，不会被无谓劈成两份 |
| 单实例根部署 | `""` | `lightrag::…` | 与今天等价 |

第二行是这个取值方式的关键收益：隔离只在**后端确实不同**时发生。若把 `<ns>` 取自浏览器路径，同一个知识库经两条 location 访问就会被拆成两份历史与两份 API key，用户须重复配置——那才是过度隔离。

唯一的 `<ns>` 碰撞边界是**两个不同实例都用空前缀挂在同一 origin 下**。该部署今天就已不可用：SPA 会把 API 调用发往 origin 根的 `/documents/...`，任一 location 都路由不到。它不是本方案引入的新失败模式，而正是多站点文档要求每个实例配置自己前缀的原因，故不额外设防。

### 3.2 完整键盘点

**划界规则**：凡取值语义依赖于「哪个站点、哪个后端」的状态一律进入站点命名空间；只有与后端完全无关的纯展示偏好才留在 origin 级共享。

不能把 `settings-storage` 的其余部分当作无害：它的 persist 配置**没有 `partialize`**，整个 store 切片都会落盘。更重要的是，客户端存储不止这一个键。下表是**完整盘点**，实现必须逐条落到某一类，不得留白：

| 键 | 载体 | 分类 | 依据 |
| --- | --- | --- | --- |
| `settings-storage`（拆分后剩余部分） | localStorage | origin 全局偏好 | 主题、语言及与后端无关的纯 UI 开关 |
| `lightrag:<ns>:query-settings-storage` | localStorage | 分区 | 查询入口 PRD 已拆出，本篇只改 `<ns>` 取值 |
| `lightrag:<ns>:webui-retrieval-history` / `…:workspace-retrieval-history` | localStorage | 分区 | 同上 |
| `lightrag:<ns>:site-settings` → `apiKey` | localStorage | 分区 | 被读出后作为 `X-API-Key` 附加到普通请求与流式请求。跨站点共享等于**把 site01 的凭据发给 site02**——凭据泄露，不是数据串用 |
| `lightrag:<ns>:site-settings` → `userPromptHistory` | localStorage | 分区 | 用户针对某个知识库写下的提示词 |
| `lightrag:<ns>:site-settings` → `queryLabel` | localStorage | 分区 | 取自某站点知识图谱的标签，在另一站点无意义甚至误导 |
| `lightrag:<ns>:site-settings` → `backendMaxGraphNodes` | localStorage | 分区 | 由后端上报，串用会得到错误上限 |
| `lightrag_search_history` | localStorage | 分区 | `SearchHistoryManager` 的固定键，保存具体知识图谱标签 |
| `LIGHTRAG-CORE-VERSION` / `LIGHTRAG-API-VERSION` | localStorage | 分区 | 来自具体后端 |
| `LIGHTRAG-WEBUI-TITLE` / `LIGHTRAG-WEBUI-DESCRIPTION` | localStorage | 分区 | 来自具体后端的 `WEBUI_TITLE` / `WEBUI_DESCRIPTION` |
| `LIGHTRAG-PREVIOUS-USER` | localStorage | 分区 | 直接决定登录时是否清空查询历史；不分区会让一个站点的登录改变另一个站点的历史保留判断 |
| `VERSION_CHECKED_FROM_LOGIN` | sessionStorage | 分区 | sessionStorage 同样不按路径隔离，同一标签页切换站点会跳过新站点的版本信息获取 |
| `LIGHTRAG-API-TOKEN` / `LIGHTRAG-LAST-TOKEN-RENEWAL` | localStorage | **明确接受的既有风险** | 认证凭据及其伴随项整体留在 origin 级，改造属于认证层议题，不在本期范围 |

划线原则由此收敛为一句话：**只有认证凭据及其伴随项作为既有风险留在 origin 级，其余依赖站点/后端的状态一律分区。** `LIGHTRAG-API-TOKEN` 不再顺带豁免任何别的键。

实现上可拆成两个 persist store（全局偏好 + 站点作用域）或采用等价机制；本文不规定形式，但规定划界。两处连带修改：`i18n.ts` 对 `settings-storage` 的直接读取不受影响（语言仍留在该键）；而 `services/navigation.ts` 退出时的 `sessionStorage.clear()` 会连带清掉其它站点的 session 键，分区后必须改为只清除本命名空间的键。

## 4. 迁移

### 4.1 目标静态确定为空命名空间

**遗留数据一律迁入 `lightrag::`。** legacy 键是在还没有命名空间概念时写下的，**无法归因**到任何站点。「谁先打开谁继承」式的动态归属既要引入一整套跨标签页协调，又只是在两个同样可能错的答案里随机挑一个——多站点场景下旧数据本就是若干站点混写的结果，把它整体判给任何一个活动站点都可能把别的站点的 `apiKey` 交出去。因此把「没有命名空间」直接**定义**为「空命名空间」：所有 legacy 站点作用域状态一律迁往 `lightrag::…`，与执行迁移的入口是谁、它自己的 `<ns>` 是什么都无关。

**由此不需要任何锁。** 目标键固定；源数据在升级后不再被任何代码写入；目标值是源的确定性函数；只写不存在的键（下节第 4 条）；复制全部完成后才开始清理（第 5 条）。这五条合起来意味着任意交错的并发执行都是**把相同的字节写向相同的键**，竞态因构造而良性。不需要 IndexedDB，不需要 Web Locks，不需要任何互斥原语，也不需要 `claimed` / `completed` 状态机——重复执行天然自门控：字段已迁出、legacy 键已删除，后续运行自动成为空操作。这同时吸收了「升级前打开的旧标签页把整份 `settings-storage` 重新写回」的情形：下次启动再清理一次即可，且因第 4 条绝不会覆盖新键。

**每个入口都无条件执行迁移**，不按自己的 `<ns>` 分支。单一代码路径既是上述收敛论证成立的前提，也保证 legacy 键在任何部署形态下都会被清理，而不是只在恰好存在根部署时才被清理。

### 4.2 迁移器语义

迁移器由两个入口共同调用，在任何依赖上述键的 store 被求值之前执行，语义如下：

1. **版本规范化先行。** `settings-storage` 带有 v1 → v21 的逐级迁移链，直接「只接受 v21 envelope」会把从 v20 及更早版本升级的用户判为无旧数据，绕开既有迁移链并把状态清回默认值。迁移器必须先**复用**（而非复制）现有 `migrate` 链把任意 ≤21 的 envelope 规范化到 v21，再执行搬迁。
2. **该链必须抽成无副作用的纯模块。** 它目前内联在 `stores/settings.ts` 里，而该模块一被 import 就会创建并 hydrate persist store——迁移器为复用它而 import，就等于在迁移之前先把待拆分的 store 建了起来。因此把 v1→v21 链抽到一个纯模块，由迁移器与 Zustand 的 `persist.migrate` 共同 import；该纯模块不得 import 任何 store、`App`、API 客户端或导航模块。（查询入口 PRD 已经因同样的理由要求了这一步，本篇复用其成果。）
3. `version > 21`（回滚后再升级、或未知的更高版本）时**不读、不清理、不覆盖任何旧字段**，新键一律用默认值初始化——否则一次降级会永久破坏数据。
4. **只为不存在的新键写入迁移值**；新键一旦存在，其值永远优先，重复迁移绝不用旧数据覆盖它。
5. 先成功写入全部新键，**再**清理旧 envelope 中已迁出的字段并升级其版本号、删除已迁出的独立 legacy 键。顺序不可颠倒：清理先于复制会在中途崩溃时直接丢数据。
6. 部分失败时保留未清理的旧字段，下次启动按同样规则重试；已写入的新键因第 4 条不会被二次覆盖，清理已删除的键是空操作，所以重跑安全且收敛到同一结果。
7. `/workspace` **允许**执行这一次性迁移写入，且覆盖迁移涉及的**全部**目标键。运行期对 `query-settings-storage` 仍严格只读。

**启动顺序：迁移模块必须是入口文件的第一个静态 import。** 迁移是纯同步的 localStorage 操作，不需要 `await`，也不需要动态 `import()`；但 `main.tsx` 静态 import `AppRouter`，后者又静态 import auth store 与 `App`，模块求值阶段 `initAuthState()` 就已读过 localStorage 并建好 store。ESM 按源码顺序深度优先求值同级 import，因此只要把这个带副作用的迁移模块放在入口文件 import 列表的**第一位**，它就会在任何 store 模块求值之前完整跑完。约束是两条：**位置在最前**，且它**递归地不 import 任何 store**（由第 2 条的纯模块保证）。

### 4.3 legacy 键的升级策略

每个入口都执行，目标固定为空命名空间：

| 旧状态 | 策略 | 理由 |
| --- | --- | --- |
| `settings-storage` 中的 `apiKey`、`userPromptHistory`、`queryLabel`、`backendMaxGraphNodes` | 迁往 `lightrag::site-settings` | 见盘点表 |
| `lightrag_search_history`、`LIGHTRAG-CORE-VERSION` / `-API-VERSION` / `-WEBUI-TITLE` / `-WEBUI-DESCRIPTION`、`LIGHTRAG-PREVIOUS-USER`、`VERSION_CHECKED_FROM_LOGIN` | 一律迁往 `lightrag::` 对应键（sessionStorage 项迁往 sessionStorage） | 单一规则，无需逐键判断。后端应答缓存那几项迁与不迁无可观察差异（各站点本就会重新请求），统一迁移以保持规则唯一 |
| `LIGHTRAG-API-TOKEN` / `-LAST-TOKEN-RENEWAL` | 原键保留不动 | 盘点表中明确接受的既有风险 |

`LIGHTRAG-PREVIOUS-USER` 不是特例。目标静态化之后它必须**跟着一起迁移**：否则根部署站点首次登录会因「无上一用户记录」执行一次保守清理，把刚迁过来的历史立即清空。带前缀的站点没有这条记录，而它们的历史本就为空，保守清理是空操作。

### 4.4 两条明确接受的代价

1. **单实例、非空前缀的部署在升级时丢失站点作用域状态**（`apiKey` 需重新录入一次，历史与 `querySettings` 回到默认；登录态不受影响，因为 `LIGHTRAG-API-TOKEN` 仍留在 origin 级）。这类部署实际上没有归属歧义，但代码无从判断，不猜就是代价。旧数据被搬到 `lightrag::…` 而非销毁，不做导入引导 UI。
2. **根部署与带前缀部署混合在同一 origin 时，根站点确定性地继承这份混合数据**（nginx `location /` → 无前缀实例、`location /site01/` → 带前缀实例是合法且可路由的配置）。不变式仍然成立——legacy 数据**至多进入一个命名空间，且该命名空间是静态确定的**，不存在两个站点各拿一份。相比「随机挑一个活动站点」，确定性归属既可预期也可在部署文档中说明。

作用域始终是**同源、同一浏览器 profile**。它不跨设备、不跨浏览器、不跨域名同步，也不是 Server 全局配置。

## 5. token 本地有效性校验

现有 `initAuthState` 只检查 token 是否存在，随后直接置 `isAuthenticated: true`。要求：

- 启动时对 token 做**本地**校验：JWT 结构可解析，且 `exp` 未过期。
- 校验不通过即清除该 token 及其伴随的 localStorage 项，按「无有效 token」处理。
- 仅凭本地信息无法确认的情形（签名无效、服务端已吊销）仍由后续 401 纠正，本地校验不承担鉴权职责。

这是共享认证层的改动，所有入口同时受益：`/webui` 今天会先渲染整个后台再被 401 打回，收紧后直接落到登录页。属于可观测的行为变化，需在变更记录中写明。

查询入口 PRD 的 §6.3 依赖这一条：它的「无有效 token → 欢迎页」一行在今天的代码里无法成立，因为过期 token 会被判为已登录。

## 6. 兼容性

- 服务端契约不变，token、API key、guest token 和登录接口一律不动。
- 根部署（`LIGHTRAG_API_PREFIX` 为空）与直连端口部署的用户**完整保留**已保存的参数、历史与 `apiKey`，因为它们的 `<ns>` 本就是空命名空间。
- 单实例、非空前缀的部署会丢失站点作用域状态并需重新录入一次 `apiKey`；登录态不受影响。须在升级说明中写明。
- 带前缀站点的后端版本、标题与描述缓存在升级后重新请求一次，用户无感。
- 同源多站点部署的用户升级后，`apiKey` 需在每个带前缀的站点各自重新填写一次。这是修正凭据跨站点共享的必要代价。

## 7. 测试

- **存储拆分测试**：`storage` 事件中其它命名空间的键被忽略；两个不同 `apiPrefix` 命名空间之间互不可见。
- **迁移测试**：新键已存在（不得被旧值覆盖）、只成功写入部分新键后重试、重复执行结果一致、写入抛错后旧字段仍保留。
- **版本链测试**：从 v1、v6、v20、v21 各自升级后，站点作用域状态都不丢失（证明复用了既有迁移链而不是把旧版本判为无数据）；损坏 envelope 按无旧数据处理；`version > 21` 时旧字段不被读取也不被清理。
- **静态目标测试**：分别以 `<ns>` 为空、`/site01` 的入口执行迁移，断言两者都把 legacy 站点作用域状态写入**同一组** `lightrag::…` 键，且带前缀的入口自己的 `lightrag:/site01:…` 键保持默认值。
- **并发收敛测试**：以任意交错顺序执行两个命名空间入口的迁移（含一方读完 legacy 后另一方已完成清理的交错），断言结果与任一方单独执行完全一致，`lightrag::…` 每个键恰好一份且值相同，legacy 键被清理。
- **重复执行自门控测试**：连续执行迁移三次，断言第二、三次为空操作，不覆盖已被用户改动过的新键值。
- **旧标签页回写测试**：迁移完成后模拟旧 bundle 重新写回一份完整 `settings-storage` envelope，断言下次启动只清理该 envelope 而不覆盖任何新键。
- **崩溃重跑测试**：复制到一半崩溃、复制完成后清理前崩溃、清理多个 legacy 键到一半崩溃——三种情形重跑后都收敛到同一结果，legacy 键不残留，新键不被旧值二次覆盖。
- **统一策略测试**：断言 `apiKey`、`userPromptHistory`、`queryLabel`、`backendMaxGraphNodes`、`lightrag_search_history`、后端版本/标题/描述缓存、`LIGHTRAG-PREVIOUS-USER` 与 `VERSION_CHECKED_FROM_LOGIN` 全部迁往对应的 `lightrag::` 键（sessionStorage 项仍在 sessionStorage），而 `LIGHTRAG-API-TOKEN` / `-LAST-TOKEN-RENEWAL` 原地不动。
- **历史误清回归测试**：根部署下迁移后首次登录，断言刚迁移过来的查询历史**不被**清空（`LIGHTRAG-PREVIOUS-USER` 已随迁移进入空命名空间）。
- **站点作用域字段测试**：盘点表中标为「分区」的全部键在两个命名空间之间互不可见；site01 的请求头不携带 site02 的 `X-API-Key`；在 site01 退出登录不清除 site02 的 session 键，也不改变 site02 的历史保留判断。
- **启动顺序测试**：断言迁移模块求值完成之前没有任何待拆分 store 被求值（未读取也未写回 localStorage）；并断言迁移纯模块的传递 import 图中不含任何 store、`App`、API 客户端或导航模块。
- **token 本地校验测试**必须直接钉住「非空即登录」这个旧行为，否则旧实现会连同新测试一起通过。用例：结构损坏（非三段式）、payload 非合法 Base64URL、payload 可解码但非 JSON、缺少 `exp`、`exp` 已过期、合法且未过期（唯一应判为已登录的一例）；并断言前五种情形下 token 及其伴随的 localStorage 项（`LIGHTRAG-LAST-TOKEN-RENEWAL` 等）都被清除。
- 本地看似有效但服务端拒绝（签名无效或已吊销）时仍由 401 纠正，本地校验不得吞掉该路径。

## 8. 风险与对策

| 风险 | 后果 | 对策 |
| --- | --- | --- |
| 固定 localStorage 键名 | 同 host 多站点部署下查询参数、历史与 **`apiKey`** 互相串用——后者是把一个站点的凭据发给另一个站点 | 站点相关状态一律按 `apiPrefix` 分区，`storage` 事件只响应本命名空间 |
| 并发首次打开两个站点 | 若按「谁先打开谁继承」动态归属，双方各自复制一份旧历史与凭据，隔离不变式被破坏，且幂等测试发现不了 | 目标命名空间**静态固定为空命名空间**：任意交错都是把相同字节写向相同键，竞态因构造而良性，无需任何互斥原语 |
| 带前缀的单实例部署升级 | 站点作用域状态（含 `apiKey`）不再可见，用户需重新录入一次 | **明确接受**：无归属证据时不猜。登录态不受影响；旧数据搬入 `lightrag::…` 而非销毁，不做导入引导 UI |
| 根部署与带前缀部署共存于同一 origin | 根站点会继承那份混合的 legacy 数据 | **明确接受**：不变式是「至多一个命名空间继承，且是静态确定的」，不存在两站各拿一份 |
| 清理早于复制 | 中途崩溃直接丢数据 | 顺序固定为先写全部新键、再清理；重跑幂等 |
| 引入 `claimed`/`completed` 状态机 | 状态推进顺序一旦写错（`completed` 早于清理），崩溃后所有运行永久跳过，凭据与历史残留 | 不引入状态机：重复执行天然自门控 |
| 为复用迁移链而 import store 模块 | 迁移器在迁移之前就创建并 hydrate 了待拆分的 persist store | v1→v21 链抽到无副作用纯模块，迁移器与 `persist.migrate` 共同 import |
| 迁移模块不在 import 首位 | ESM 按源码顺序求值同级 import，晚于 store 模块就等于没迁 | 约束写死为「入口文件的第一个静态 import」，并以 import 图测试守护 |
| 存储盘点不完整 | `lightrag_search_history`、后端版本/标题、`LIGHTRAG-PREVIOUS-USER`、`VERSION_CHECKED_FROM_LOGIN` 继续跨站点串用 | 维护完整盘点表，每个键必须落到全局/分区/既有风险三类之一 |
| 只接受 v21 envelope | 从 v20 及更早升级的用户被判为无旧数据，状态清回默认值 | 复用既有 v1→v21 迁移链先规范化再拆分；`version > 21` 一律不读不清 |
| token 只判存在性 | 过期或损坏的 token 被判为已登录，依赖它的未登录分支无法成立 | 启动时本地校验 JWT 结构与 `exp`，不通过即清除；签名与吊销仍由 401 纠正 |
