<div align="center">

<div style="margin: 20px 0;">
  <img src="./assets/logo.png" width="120" height="120" alt="LightRAG Logo" style="border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);">
</div>

# 🚀 LightRAG: シンプルかつ高速な検索拡張生成（RAG）

<div align="center">
    <a href="https://trendshift.io/repositories/13043" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13043" alt="HKUDS%2FLightRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>
<p>
</p>
<div align="center">
  <div style="width: 100%; height: 2px; margin: 20px 0; background: linear-gradient(90deg, transparent, #00d9ff, transparent);"></div>
</div>

<div align="center">
  <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 25px; text-align: center;">
    <p>
      <a href='https://github.com/HKUDS/LightRAG'><img src='https://img.shields.io/badge/🔥Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e'></a>
      <a href='https://arxiv.org/abs/2410.05779'><img src='https://img.shields.io/badge/📄arXiv-2410.05779-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e'></a>
      <a href="https://github.com/HKUDS/LightRAG/stargazers"><img src='https://img.shields.io/github/stars/HKUDS/LightRAG?color=00d9ff&style=for-the-badge&logo=star&logoColor=white&labelColor=1a1a2e' /></a>
    </p>
    <p>
      <img src="https://img.shields.io/badge/🐍Python-3.10-4ecdc4?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
      <a href="https://pypi.org/project/lightrag-hku/"><img src="https://img.shields.io/pypi/v/lightrag-hku.svg?style=for-the-badge&logo=pypi&logoColor=white&labelColor=1a1a2e&color=ff6b6b"></a>
    </p>
    <p>
      <a href="https://discord.gg/yF2MmDJyGJ"><img src="https://img.shields.io/badge/💬Discord-Community-7289da?style=for-the-badge&logo=discord&logoColor=white&labelColor=1a1a2e"></a>
      <a href="https://github.com/HKUDS/LightRAG/issues/285"><img src="https://img.shields.io/badge/💬WeChat-Group-07c160?style=for-the-badge&logo=wechat&logoColor=white&labelColor=1a1a2e"></a>
    </p>
    <p>
      <a href="README-zh.md"><img src="https://img.shields.io/badge/🇨🇳中文版-1a1a2e?style=for-the-badge"></a>
      <a href="README.md"><img src="https://img.shields.io/badge/🇺🇸English-1a1a2e?style=for-the-badge"></a>
      <a href="README-ja.md"><img src="https://img.shields.io/badge/🇯🇵日本語版-1a1a2e?style=for-the-badge"></a>
    </p>
    <p>
      <a href="https://pepy.tech/projects/lightrag-hku"><img src="https://static.pepy.tech/personalized-badge/lightrag-hku?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads"></a>
      <a href="https://hvtracker.net/agents/lightrag/"><img src="https://hvtracker.net/badge/lightrag.svg"></a>
    </p>
  </div>
</div>

</div>

<div align="center" style="margin: 30px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="800">
</div>

<div align="center" style="margin: 30px 0;">
    <img src="./README.assets/b2aaf634151b4706892693ffb43d9093.png" width="800" alt="LightRAG Diagram">
</div>

---

<div align="center">
  <table>
    <tr>
      <td style="vertical-align: middle;">
        <img src="./assets/LiteWrite.png"
             width="56"
             height="56"
             alt="LiteWrite"
             style="border-radius: 12px;" />
      </td>
      <td style="vertical-align: middle; padding-left: 12px;">
        <a href="https://litewrite.ai">
          <img src="https://img.shields.io/badge/🚀%20LiteWrite-AI%20Native%20LaTeX%20Editor-ff6b6b?style=for-the-badge&logoColor=white&labelColor=1a1a2e">
        </a>
      </td>
    </tr>
  </table>
</div>

---

## 🎉 ニュース
- [2026.05]🎯[新機能]: **RagAnything を LightRAG に統合**🎉。**MinerU / Docling** サービスによるマルチモーダルコンテンツの解析・抽出に対応。
- [2026.05]🎯[新機能]: 選択可能な4種類のテキストチャンキング戦略を導入: `Fix`、`Recursive`、`Vector`、`Paragraph`。
- [2026.05]🎯[新機能]: **ロール別 LLM 設定**に対応。EXTRACT、QUERY、KEYWORDS、VLM の4つの異なるロールに対し、それぞれ独立した LLM 設定が可能。
- [2026.03]🎯[新機能]: **OpenSearch** を統合ストレージバックエンドとして統合し、LightRAG の4つのストレージすべてを包括的にサポート。
- [2026.03]🎯[新機能]: セットアップウィザードを導入。Docker による埋め込み・リランキング・ストレージバックエンドのローカルデプロイに対応。
- [2025.11]🎯[新機能]: **評価のための RAGAS** と **トレーシングのための Langfuse** を統合。コンテキスト精度メトリクスをサポートするため、クエリ結果とともに取得したコンテキストを返すよう API を更新。
- [2025.10]🎯[スケーラビリティ強化]: 処理上のボトルネックを排除し、**大規模データセットを効率的に**サポート。
- [2025.09]🎯[新機能] Qwen3-30B-A3B などの**オープンソース LLM** に対する知識グラフ抽出精度を向上。
- [2025.08]🎯[新機能] **リランカー**に対応。混合クエリのパフォーマンスを大幅に向上（デフォルトのクエリモードとして設定）。
- [2025.08]🎯[新機能] **ドキュメント削除**機能を追加し、最適なクエリ性能を保つために KG の自動再生成を実施。
- [2025.06]🎯[新リリース] 当チームは [RAG-Anything](https://github.com/HKUDS/RAG-Anything) をリリースしました。テキスト・画像・表・数式をシームレスに処理する**オールインワンのマルチモーダル RAG** システムです。
- [2025.06]🎯[新機能] LightRAG は [RAG-Anything](https://github.com/HKUDS/RAG-Anything) 統合により包括的なマルチモーダルデータ処理に対応しました。PDF、画像、Office ドキュメント、表、数式を含む多様な形式にわたって、シームレスなドキュメント解析と RAG 機能を実現します。詳細は新しい[マルチモーダルセクション](#マルチモーダル機能のアップグレード)を参照してください。
- [2025.03]🎯[新機能] LightRAG は引用機能に対応し、適切な出典の明示とドキュメントのトレーサビリティ向上を実現しました。
- [2025.02]🎯[新機能] MongoDB を統合データ管理のためのオールインワンストレージソリューションとして利用できるようになりました。
- [2025.02]🎯[新リリース] 当チームは [VideoRAG](https://github.com/HKUDS/VideoRAG) をリリースしました。極めて長いコンテキストの動画を理解するための RAG システムです。
- [2025.01]🎯[新リリース] 当チームは [MiniRAG](https://github.com/HKUDS/MiniRAG) をリリースしました。小規模モデルで RAG をよりシンプルにします。
- [2025.01]🎯PostgreSQL をデータ管理のためのオールインワンストレージソリューションとして利用できるようになりました。
- [2024.11]🎯[新リソース] LightRAG の包括的なガイドが [LearnOpenCV](https://learnopencv.com/lightrag) で公開されました。詳細なチュートリアルとベストプラクティスをご覧ください。素晴らしい貢献をいただいたブログ著者に感謝します！
- [2024.11]🎯[新機能] LightRAG WebUI を導入。直感的な Web ベースのダッシュボードを通じて、LightRAG の知識を挿入・クエリ・可視化できるインターフェースです。
- [2024.11]🎯[新機能] [Neo4J をストレージとして利用](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#using-neo4j-for-storage)できるようになり、グラフデータベースのサポートが可能になりました。
- [2024.10]🎯[新機能] [LightRAG 紹介動画](https://youtu.be/oageL-1I0GE)へのリンクを追加しました。LightRAG の機能のウォークスルーです。素晴らしい貢献をいただいた著者に感謝します！
- [2024.10]🎯[新チャンネル] [Discord チャンネル](https://discord.gg/yF2MmDJyGJ)を作成しました！💬 共有・議論・コラボレーションのため、ぜひコミュニティにご参加ください！🎉🎉

<details>
  <summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; display: list-item;">
    アルゴリズムフローチャート
  </summary>

![LightRAG Indexing Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-VectorDB-Json-KV-Store-Indexing-Flowchart-scaled.jpg)
*図1: LightRAG インデックス作成フローチャート - 画像出典: [Source](https://learnopencv.com/lightrag/)*
![LightRAG Retrieval and Querying Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-Querying-Flowchart-Dual-Level-Retrieval-Generation-Knowledge-Graphs-scaled.jpg)
*図2: LightRAG 検索・クエリフローチャート - 画像出典: [Source](https://learnopencv.com/lightrag/)*

</details>

## インストール

**💡 パッケージ管理に uv を使用**: 本プロジェクトでは、高速かつ信頼性の高い Python パッケージ管理のために [uv](https://docs.astral.sh/uv/) を使用しています。まず uv をインストールしてください: `curl -LsSf https://astral.sh/uv/install.sh | sh`（Unix/macOS）または `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`（Windows）

> **注記**: お好みであれば pip も使用できますが、より良いパフォーマンスと信頼性の高い依存関係管理のため、uv を推奨します。
>
> **📦 オフラインデプロイ**: オフライン環境やエアギャップ環境については、すべての依存関係とキャッシュファイルを事前インストールする手順を記した[オフラインデプロイガイド](./docs/OfflineDeployment.md)を参照してください。

### LightRAG サーバーのインストール

* PyPI からのインストール

```bash
### uv を使って LightRAG サーバーをツールとしてインストール（推奨）
uv tool install "lightrag-hku[api]"

### または pip を使用
# python -m venv .venv
# source .venv/bin/activate  # Windows: .venv\Scripts\activate
# pip install "lightrag-hku[api]"

### フロントエンド成果物のビルド
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..

# env ファイルのセットアップ
# env.example ファイルは GitHub リポジトリのルートからダウンロードするか、
# ローカルのソースチェックアウトからコピーして入手してください。
cp env.example .env  # .env を自分の LLM・埋め込み設定で更新
# サーバーの起動
lightrag-server
```

* ソースからのインストール

```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG

# 開発環境のブートストラップ（推奨）
make dev
source .venv/bin/activate  # 仮想環境の有効化（Linux/macOS）
# Windows の場合: .venv\Scripts\activate

# make dev はテストツールチェーンに加えて、完全なオフラインスタック
#（API、ストレージバックエンド、プロバイダー統合）をインストールし、フロントエンドをビルドします。
# サーバーを起動する前に、make env-base を実行するか env.example を .env にコピーしてください。

# uv による同等の手動手順
# 注記: uv sync は .venv/ ディレクトリに自動的に仮想環境を作成します
uv sync --extra test --extra offline
source .venv/bin/activate  # 仮想環境の有効化（Linux/macOS）
# Windows の場合: .venv\Scripts\activate

### または仮想環境付きで pip を使用
# python -m venv .venv
# source .venv/bin/activate  # Windows: .venv\Scripts\activate
# pip install -e ".[test,offline]"

# フロントエンド成果物のビルド
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..

# env ファイルのセットアップ
make env-base  # または: cp env.example .env して手動で更新
# API-WebUI サーバーの起動
lightrag-server
```

* Docker Compose による LightRAG サーバーの起動

```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
cp env.example .env  # .env を自分の LLM・埋め込み設定で更新
# .env で LLM と埋め込みの設定を変更
docker compose up
```

> LightRAG docker イメージの過去バージョンはこちらで確認できます: [LightRAG Docker Images]( https://github.com/HKUDS/LightRAG/pkgs/container/lightrag)
>
> GitHub Actions により公開された公式 GHCR イメージは、GitHub OIDC を用いた Sigstore Cosign で署名されています。検証コマンドについては [docs/DockerDeployment.md](./docs/DockerDeployment.md#verify-official-ghcr-images-with-cosign) を参照してください。

### セットアップツールによる .env ファイルの作成

`env.example` を手作業で編集する代わりに、対話型のセットアップウィザードを使って設定済みの `.env`、必要に応じて `docker-compose.final.yml` を生成できます:

```bash
make env-base           # 必須の最初のステップ: LLM、埋め込み、リランカー
make env-storage        # 任意: ストレージバックエンドとデータベースサービス
make env-server         # 任意: サーバーポート、認証、SSL
make env-base-rewrite   # 任意: ウィザード管理の compose サービスを強制再生成
make env-storage-rewrite # 任意: ウィザード管理の compose サービスを強制再生成
make env-security-check # 任意: 現在の .env のセキュリティリスクを監査
```

各ターゲットの詳細な説明については [docs/InteractiveSetup.md](./docs/InteractiveSetup.md) を参照してください。

## LightRAG について

### 軽量なグラフベース RAG フレームワーク

**LightRAG** は、軽量なナレッジグラフRAGフレームワークであり、Microsoft GraphRAGの効率的な代替手段です。KG（ナレッジグラフ）とベクトル埋め込みを同時に管理する二層アーキテクチャを採用しており、従来のベクトルベースRAGとグラフベースRAGの間にある技術的なギャップを効果的に埋めます。高い拡張性を前提に設計されたLightRAGは、大規模なグラフのインデックス作成および検索における、計算コストの大きさ、応答の遅さ、増分更新コストの高さといった主要課題を解決します。大規模データセットをサポートしながら、30B規模のオープンソース大規模言語モデル（LLM）を用いた場合でも、非常に高いRAG品質を維持できます。

### 機能と利点

- **深いコンテキスト理解:** グラフ構造化インデックスを通じて、LightRAG はエンティティ間の複雑な意味的依存関係を捉え、従来のチャンクベース検索手法に典型的な断片化したコンテキストの限界を克服します。その生成品質とコンテキスト認識は、グローバルな理解や論理的推論を必要とする垂直ドメイン（例: 法律、金融）において特に優れています。
- **卓越した網羅性と多様性:** LightRAG のデュアルレベル検索メカニズムにより、詳細な事実と抽象的な概念を同時に統合できます。これにより、クエリ結果の網羅性と多様性において顕著なパフォーマンスを達成し、複雑なクロスドキュメントクエリの処理に極めて効果的です。
- **極めて高い検索効率と低コスト:** LightRAG は、複雑なクエリに対して非効率なコミュニティレポートやマルチホップ推論に依存しません。これにより、インデックス作成段階とクエリ段階の双方で必要となる LLM 呼び出し回数を大幅に削減し、応答レイテンシと LLM の計算コストを著しく低減します。
- **動的データへの迅速な適応:** LightRAG は、シームレスな増分知識ベース更新をサポートします。新しいデータは標準的なグラフインデックス作成パイプラインを通すだけでローカルグラフを生成し、集合のマージによって既存のグラフに直接統合されます。このプロセスにより、元の構造を破壊したりグローバルインデックスを再構築したりする必要がなくなり、動的なデータ環境におけるリアルタイムな関連性を保証します。ドキュメント削除時には、構築段階での LLM キャッシュを活用して、影響を受けたエンティティ関係を迅速に再構築し、知識ベースの更新効率を大幅に向上させます。

### マルチモーダル機能のアップグレード

バージョン v1.5 から、LightRAG はマルチモーダルドキュメントの分析・検索機能を正式に導入しました:

- **マルチエンジンによるドキュメント解析:** ドキュメント処理パイプラインは MinerU、Docling、Native などの解析エンジンをサポートし、ドキュメントからテキスト・表・数式・画像を高効率に抽出できます。
- **クロスモーダルなエンティティ・関係マッピング:** 統一されたフレームワーク内でクロスモーダルなエンティティ抽出と関係マッピングを実現し、シームレスなインデックス作成とクエリをもたらします。
- **応用シナリオの強化:** まったく新しいマルチモーダル処理パイプラインにより、操作マニュアルや学術論文といったマルチモーダルコンテンツに富むドキュメントに対する RAG 品質が大幅に向上します。

### LightRAG API サーバー

LightRAG サーバーは、LightRAG の機能を探索するための Web ベース UI だけでなく、包括的な REST API も提供します。LightRAG サーバーの詳細については [LightRAG Server](./docs/LightRAG-API-Server.md) を参照してください。

![iShot_2025-03-23_12.40.08](./README.assets/iShot_2025-03-23_12.40.08.png)

## 主要な設定ガイド

### LLM モデルの選択

LightRAG はワークフロー中に4つの異なるロールの LLM/VLM を必要とします。パフォーマンスと処理速度のバランスを取るため、ロールごとに異なる能力と速度のモデルを設定すべきです。LightRAG は、ドキュメントから複雑なエンティティ関係抽出タスクを実行するために LLM を必要とするため、従来の RAG よりも大規模言語モデル（LLM）に対する能力要件が高くなります。クエリ段階では、LLM はエンティティ、関係、テキストチャンクを含む大量の取得情報を処理する必要があります。これには、長くノイズの多いコンテキストの中で高品質な応答を生成する能力がモデルに求められます。詳細なモデル設定については [RoleSpecificLLMConfiguration.md](./docs/RoleSpecificLLMConfiguration.md) を参照してください。

### クエリモードの選択

LightRAG は5つのクエリモードをサポートします:

- **local**: ローカルなコンテキストと特定のエンティティの精密なマッチングに焦点を当てます。知識グラフから候補エンティティとその直接関連する属性を取得します。このモードは、特定の対象、具体的な概念、詳細な事実を狙った Q&A に適しており、関連性が高く詳細なローカルコンテキストのサポートを提供します。
- **global**: マクロなテーマ、クロスドキュメント推論、エンティティ間の深い関係に焦点を当てます。広範なテーマと概念をカバーする関係チェーンを取得します。このモードは、複数のコンテキストにまたがる要約、トレンド分析、複雑な意味的依存関係の理解を必要とするクエリに適しています。
- **hybrid**: local モードと global モードの両方の検索結果をマージします。特定のエンティティとグローバルな関係コンテキストを同時に再現することで、包括的な推論と生成を実行します。
- **naive**: テキストチャンクに基づく従来の RAG 検索です。知識グラフを使用せず、ベクトル類似度に直接依存して元のテキストチャンクから取得します。
- **mix**: local、global、naive モードの検索結果をマージし、最も包括的で豊富な検索結果を提供するフル機能のモードです。

LightRAG のデフォルトのクエリモードは `mix` です。`mix` モードを使用すると、一般に最も理想的なクエリ結果が得られます。`mix` モードは `naive` よりわずかに時間がかかりますが、その他のクエリモードはレイテンシがおおむね同等です。

### 埋め込みモデル

埋め込みモデルを選ぶ際は、その多言語サポート能力に注意してください。LightRAG の検索品質は埋め込みモデルへの依存度が限定的であるため、低次元で高速なモデルを選ぶことを推奨します。通常、`BAAI/bge-m3` で十分です。最良のパフォーマンスを得るため、埋め込みモデルをローカルにデプロイすることを強く推奨します。

**重要な注記**: 埋め込みモデルはドキュメントのインデックス作成前に確定する必要があり、クエリ段階でも同じモデルを使用しなければなりません。一度選択すると、埋め込みモデルは一般に変更できません。変更した場合は、すべてのテキストチャンク、エンティティ、関係を再埋め込みする必要があります。LightRAG は現在、再埋め込みツールを提供していません。一部のストレージバックエンド（例: PostgreSQL）では、テーブルの初回作成時にベクトル次元を定義する必要があるため、埋め込みモデルを変更するにはベクトル関連テーブルを削除し、LightRAG が再作成できるようにする必要があります。

### リランキングの有効化

クエリ段階で Rerank オプションを有効にすると、クエリ品質が大幅に向上します。ただし、Rerank を有効にすると通常 1～2 秒の遅延が生じます。レイテンシを最小化するため、Rerank モデルをローカルにデプロイすることを強く推奨します。設定の詳細については `.env.example` ファイルを参照してください。埋め込みモデルとは異なり、Rerank モデルはクエリ段階でいつでも変更できます。

### ドキュメント処理パイプラインの設定

LightRAG のデフォルトのパイプライン設定では、システムが最高の性能を発揮できません。ドキュメント解析の品質はドキュメントのインデックス作成とクエリに大きく影響します。そのため、MinerU 解析エンジンを有効にし、パイプラインの画像分析機能を有効化するようパイプラインを設定することを推奨します。推奨設定:

```
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R

VLM_PROCESS_ENABLE=true
VLM_LLM_MODEL=<your_vlm_model_name>
```

クラウドベースの MinerU サービスには利用量・ファイルサイズ・ページ数の制限があるため、ローカルにデプロイした MinerU を使用することを推奨します。ファイル処理パイプラインの設定の詳細については [FileProcessingPipeline.md](./docs/FileProcessingPipeline.md) を参照してください。

### ファイル処理の並行性最適化

大規模なドキュメント処理では、並行性を高める必要があります。ファイルの並行処理に関連する主要な環境変数は以下のとおりです:

- **MAX_ASYNC_LLM/EXTRACT_ASYNC_LLM**: LLM モデルの最大並行数を制御します。
- **MAX_PARALLEL_INSERT**: 並行処理されるファイルの最大数を制御します。1つのファイル内のテキスト・表・数式・画像の処理も並行して行われます。`MAX_PARALLEL_INSERT` は理想的には `MAX_ASYNC_LLM` の約1/3に設定すべきです。
- **MAX_PARALLEL_PARSE_MINERU**: MinerU 解析で並行処理されるファイル数を制御します。
- **MAX_PARALLEL_PARSE_DOCLING**: Docling 解析で並行処理されるファイル数を制御します。
- **EMBEDDING_FUNC_MAX_ASYNC**: 埋め込みモデルの最大並行数を制御します。
- **EMBEDDING_BATCH_NUM**: 埋め込みモデルの各リクエストに含めるテキスト数（1バッチあたりの埋め込み数）を制御します。この数を増やすと、埋め込みモデルへの API 呼び出し回数を大幅に削減でき、埋め込みストレージへのデータ永続化を高速化できます。

```
# 設定例
MAX_ASYNC_LLM=8
MAX_PARALLEL_INSERT=3
EMBEDDING_FUNC_MAX_ASYNC=16
EMBEDDING_BATCH_NUM=32
```

### バックエンドストレージの選択

LightRAG は4種類のバックエンドストレージを必要とします:

- **KV_STORAGE**: LLM 応答キャッシュ、テキストチャンキング結果、エンティティ関係抽出結果などの保存に使用します。
- **VECTOR_STORAGE**: テキストチャンク、エンティティ、関係のベクトル情報の保存に使用します。
- **GRAPH_STORAGE**: 知識グラフの保存に使用します。
- **DOC_STATUS_STORAGE**: ドキュメントリストの保存に使用します。

デフォルトでは、LightRAG のストレージバックエンドはファイル永続化されたインメモリデータベースです。これらのデフォルトストレージは開発・デバッグ用途のみを想定しており、本番環境には適していません。本番環境で、4種類すべてのストレージを単一のバックエンドで扱いたい場合は、PostgreSQL、MongoDB、または OpenSearch を選択できます。あるいは、ベクトルストレージに Milvus や Qdrant、グラフストレージに Neo4j や Memgraph を使用するなど、ベクトルやグラフのストレージに専用データベースを選択することもできます。

### ドキュメント処理に関するその他の重要な設定

ドキュメント挿入段階では、ニーズに応じて以下の環境変数の調整を検討するとよいでしょう:

- **SUMMARY_LANGUAGE**: LLM がエンティティ関係名や要約を出力する際に使用する言語を制御します（例: `Chinese`、`English`）。
- **ENTITY_EXTRACTION_USE_JSON**: LLM がエンティティ関係抽出を JSON 形式で出力するかどうかを制御します。JSON 形式を使用すると通常より安定した結果が得られますが、より多くのトークンを消費し、やや遅くなることがあります。
- **ENABLE_CONTENT_HEADINGS**: クエリ段階でテキストチャンクのセクション見出し情報を LLM に送信するかどうかを制御します（デフォルトで有効。LLM により多くのコンテキストを提供します）。
- **FORCE_LLM_SUMMARY_ON_MERGE / MAX_SOURCE_IDS_PER_RELATION**: 1つの `entity/relation` が関連付けられるテキストチャンクの最大数を制御します。
- **SOURCE_IDS_LIMIT_METHOD**: ある `entity/relation` が関連テキストチャンク数の上限を超えた後も、エンティティ/関係の説明を更新し続けるかどうかを制御します（デフォルトでは更新を停止します。その時点でエンティティ関係の説明はすでに十分豊富であり、さらなる更新はほとんど価値を加えないためです。更新をスキップすることで知識ベースの構築を大幅に高速化できます）。
- **DEFAULT_MAX_FILE_PATHS**: 1つの `entity/relation` が関連付けられるソースファイルの最大数を制御します。この上限を超えると、新しいファイル名はベクトルストレージに書き込まれなくなります。

### エンティティ・関係抽出時の LLM タイムアウトの解消

エンティティ・関係抽出中の LLM タイムアウトは、通常3つの原因のいずれかに起因します。原因を特定し、対応する対策を適用してください（パラメータは併用できます）:

- **モデルが遅い。** 約50トークン/秒を下回るモデルでは、多数のエンティティと関係を含むチャンクを、リクエストがタイムアウトする前に処理しきれない場合があります。`*_LLM_TIMEOUT`（グローバルの `LLM_TIMEOUT`、または抽出フェーズ用のロール別 `EXTRACT_LLM_TIMEOUT`）でタイムアウトを延長してください。実際の実行タイムアウトは設定値の**2倍**になるため、`EXTRACT_LLM_TIMEOUT=300` は最大**600秒**を許容します。
- **チャンクから生成されるエンティティ・関係が多すぎる。** 例えば参考文献のチャンクでは、モデルが膨大な数のレコードを出力し、時間内に完了できないことがあります。`OPENAI_LLM_MAX_TOKENS` または `OPENAI_LLM_MAX_COMPLETION_TOKENS` で出力長を制限してください（正しいパラメータ名は LLM プロバイダーによって異なります。`env.example` を参照）。目安として `max_output_tokens < LLM_TIMEOUT × tokens_per_second`（例: `9000 < 240s × 50 tps`）が有用です。
- **モデルが出力ループに陥る。** 一部のモデル（特にローカル展開された Qwen モデル）は、特定のテキストで際限のない出力ループに陥ることがあります。これが断続的に発生する場合は、ドキュメントを一度再処理するだけで通常は解消します。
- **特に参考文献の場合（P チャンク戦略）。** 段落セマンティック（`P`）チャンク戦略（例: `LIGHTRAG_PARSER=...-iteP`）を使用している場合、`CHUNK_P_DROP_REFERENCES=true` を設定すると、チャンク化の前に末尾の参考文献セクションを自動的に削除します。これにより、参考文献が大量の低価値なエンティティ・関係を生成すること（タイムアウトの一般的な原因）を防ぎます。ファイル名のヒント `paper.[-P(drop_rf=true)].pdf` でファイルごとに有効化することもできます。関連する検出パラメータ（`CHUNK_P_REFERENCES_TAIL_N`、`CHUNK_P_REFERENCES_HEADINGS`）は `env.example` に記載されています。

### ドキュメントクエリに関するその他の重要な設定

ドキュメントクエリ段階では、ニーズに応じて以下の環境変数の調整を検討するとよいでしょう:
- **MAX_ENTITY_TOKENS / MAX_RELATION_TOKENS / MAX_TOTAL_TOKENS**: LLM コンテキストに送信される取得コンテンツのトークン長を制御します。取得コンテンツは `entities`、`relations`、`text chunks` の3つの部分から構成されます。エンティティと関係の長さは独立して制御でき、テキストチャンクの長さは総長からエンティティと関係の長さを差し引いて決まります。
- **ENABLE_CONTENT_HEADINGS**: テキストチャンクが存在するセクション見出しを LLM に送信するかどうかを制御します。デフォルトで有効で、LLM により豊富なコンテキストを提供し、回答品質を向上させます。
- **ENABLE_LLM_CACHE**: クエリ結果をキャッシュするかどうか。デフォルトで有効です。同一のクエリ質問、クエリモード、LLM モデルパラメータであれば同じ結果を返します。

## SDK としての LightRAG の利用

> ⚠️ **プロジェクトへの統合には、LightRAG サーバーが提供する REST API の使用を強く推奨します。** LightRAG SDK は主に組み込みアプリケーションや学術研究・評価目的を想定しています。

### LightRAG SDK のインストール

* ソースコードからのインストール

```bash
cd LightRAG
# 注記: uv sync は .venv/ ディレクトリに自動的に仮想環境を作成します
uv sync
source .venv/bin/activate  # 仮想環境の有効化（Linux/macOS）
# Windows の場合: .venv\Scripts\activate

# または: pip install -e .
```

* PyPI からのインストール

```bash
uv pip install lightrag-hku
# または: pip install lightrag-hku
```

### LightRAG SDK サンプルコード

LightRAG コアを使い始めるには、`examples` フォルダにあるサンプルコードを参照してください。さらに、ローカルセットアップ手順を案内する[デモ動画](https://www.youtube.com/watch?v=g21royNJ4fw)も用意されています。すでに OpenAI API キーをお持ちであれば、すぐにデモを実行できます:

```bash
### デモコードはプロジェクトフォルダ内で実行してください
cd LightRAG
### OpenAI 用の API-KEY を指定
export OPENAI_API_KEY="sk-...your_opeai_key..."
### Charles Dickens 著「A Christmas Carol」のデモドキュメントをダウンロード
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
### デモコードを実行
python examples/lightrag_openai_demo.py
```

ストリーミング応答の実装例については、`examples/lightrag_openai_compatible_demo.py` を参照してください。実行前に、サンプルコードの LLM と埋め込みの設定を適宜変更してください。

**注記1**: デモプログラムを実行する際は、テストスクリプトによって異なる埋め込みモデルが使用される場合があることに注意してください。別の埋め込みモデルに切り替える場合は、データディレクトリ（`./dickens`）をクリアする必要があります。そうしないとプログラムでエラーが発生する可能性があります。LLM キャッシュを保持したい場合は、データディレクトリをクリアする際に `kv_store_llm_response_cache.json` ファイルを残すことができます。

**注記2**: 公式にサポートされているサンプルコードは `lightrag_openai_demo.py` と `lightrag_openai_compatible_demo.py` のみです。その他のサンプルファイルはコミュニティによる貢献であり、完全なテストと最適化を経ていません。

### **SDK 利用に関する注記**

SDK の利用に関する詳細な手順については、**[docs/ProgramingWithCore.md](./docs/ProgramingWithCore.md)** を参照してください。一部の LightRAG 機能は REST API では公開されておらず、SDK 経由でのみアクセス可能です。これらの機能は通常、実験的なものであり、将来のバージョンと互換性がない場合があります。

## 論文の結果の再現

LightRAG は、農業、コンピュータサイエンス、法律、混合ドメインにわたって、NaiveRAG、RQ-RAG、HyDE、GraphRAG を一貫して上回ります。完全な評価方法論、プロンプト、再現手順については、**[docs/Reproduce.md](./docs/Reproduce.md)** を参照してください。

**全体性能テーブル**

||**Agriculture**||**CS**||**Legal**||**Mix**||
|----------------------|---------------|------------|------|------------|---------|------------|-------|------------|
||NaiveRAG|**LightRAG**|NaiveRAG|**LightRAG**|NaiveRAG|**LightRAG**|NaiveRAG|**LightRAG**|
|**Comprehensiveness**|32.4%|**67.6%**|38.4%|**61.6%**|16.4%|**83.6%**|38.8%|**61.2%**|
|**Diversity**|23.6%|**76.4%**|38.0%|**62.0%**|13.6%|**86.4%**|32.4%|**67.6%**|
|**Empowerment**|32.4%|**67.6%**|38.8%|**61.2%**|16.4%|**83.6%**|42.8%|**57.2%**|
|**Overall**|32.4%|**67.6%**|38.8%|**61.2%**|15.2%|**84.8%**|40.0%|**60.0%**|
||RQ-RAG|**LightRAG**|RQ-RAG|**LightRAG**|RQ-RAG|**LightRAG**|RQ-RAG|**LightRAG**|
|**Comprehensiveness**|31.6%|**68.4%**|38.8%|**61.2%**|15.2%|**84.8%**|39.2%|**60.8%**|
|**Diversity**|29.2%|**70.8%**|39.2%|**60.8%**|11.6%|**88.4%**|30.8%|**69.2%**|
|**Empowerment**|31.6%|**68.4%**|36.4%|**63.6%**|15.2%|**84.8%**|42.4%|**57.6%**|
|**Overall**|32.4%|**67.6%**|38.0%|**62.0%**|14.4%|**85.6%**|40.0%|**60.0%**|
||HyDE|**LightRAG**|HyDE|**LightRAG**|HyDE|**LightRAG**|HyDE|**LightRAG**|
|**Comprehensiveness**|26.0%|**74.0%**|41.6%|**58.4%**|26.8%|**73.2%**|40.4%|**59.6%**|
|**Diversity**|24.0%|**76.0%**|38.8%|**61.2%**|20.0%|**80.0%**|32.4%|**67.6%**|
|**Empowerment**|25.2%|**74.8%**|40.8%|**59.2%**|26.0%|**74.0%**|46.0%|**54.0%**|
|**Overall**|24.8%|**75.2%**|41.6%|**58.4%**|26.4%|**73.6%**|42.4%|**57.6%**|
||GraphRAG|**LightRAG**|GraphRAG|**LightRAG**|GraphRAG|**LightRAG**|GraphRAG|**LightRAG**|
|**Comprehensiveness**|45.6%|**54.4%**|48.4%|**51.6%**|48.4%|**51.6%**|**50.4%**|49.6%|
|**Diversity**|22.8%|**77.2%**|40.8%|**59.2%**|26.4%|**73.6%**|36.0%|**64.0%**|
|**Empowerment**|41.2%|**58.8%**|45.2%|**54.8%**|43.6%|**56.4%**|**50.8%**|49.2%|
|**Overall**|45.2%|**54.8%**|48.0%|**52.0%**|47.2%|**52.8%**|**50.4%**|49.6%|


## 🔗 関連プロジェクト

*エコシステムと拡張機能*

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/HKUDS/RAG-Anything">
          <div style="width: 100px; height: 100px; background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 217, 255, 0.05) 100%); border-radius: 15px; border: 1px solid rgba(0, 217, 255, 0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
            <span style="font-size: 32px;">📸</span>
          </div>
          <b>RAG-Anything</b><br>
          <sub>マルチモーダル RAG</sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/HKUDS/VideoRAG">
          <div style="width: 100px; height: 100px; background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 217, 255, 0.05) 100%); border-radius: 15px; border: 1px solid rgba(0, 217, 255, 0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
            <span style="font-size: 32px;">🎥</span>
          </div>
          <b>VideoRAG</b><br>
          <sub>極めて長いコンテキストの動画 RAG</sub>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/HKUDS/MiniRAG">
          <div style="width: 100px; height: 100px; background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 217, 255, 0.05) 100%); border-radius: 15px; border: 1px solid rgba(0, 217, 255, 0.2); display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
            <span style="font-size: 32px;">✨</span>
          </div>
          <b>MiniRAG</b><br>
          <sub>極めてシンプルな RAG</sub>
        </a>
      </td>
    </tr>
  </table>
</div>

---

## ⭐ スター履歴

[![Star History Chart](https://api.star-history.com/svg?repos=HKUDS/LightRAG&type=Date)](https://star-history.com/#HKUDS/LightRAG&Date)

## 🤝 貢献

<div align="center">
  バグ修正、新機能、ドキュメントの改善など、あらゆる種類の貢献を歓迎します。<br>
  プルリクエストを送信する前に、<a href=".github/CONTRIBUTING.md"><strong>貢献ガイド</strong></a>をお読みください。
</div>

<br>

<div align="center">
  貴重な貢献をいただいたすべてのコントリビューターに感謝します。
</div>

<div align="center">
  <a href="https://github.com/HKUDS/LightRAG/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=HKUDS/LightRAG" style="border-radius: 15px; box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);" />
  </a>
</div>


## 📖 引用

```python
@article{guo2024lightrag,
title={LightRAG: Simple and Fast Retrieval-Augmented Generation},
author={Zirui Guo and Lianghao Xia and Yanhua Yu and Tu Ao and Chao Huang},
year={2024},
eprint={2410.05779},
archivePrefix={arXiv},
primaryClass={cs.IR}
}
```

---

<div align="center" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 30px; margin: 30px 0;">
  <div>
    <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="500">
  </div>
  <div style="margin-top: 20px;">
    <a href="https://github.com/HKUDS/LightRAG" style="text-decoration: none;">
      <img src="https://img.shields.io/badge/⭐%20Star%20us%20on%20GitHub-1a1a2e?style=for-the-badge&logo=github&logoColor=white">
    </a>
    <a href="https://github.com/HKUDS/LightRAG/issues" style="text-decoration: none;">
      <img src="https://img.shields.io/badge/🐛%20Report%20Issues-ff6b6b?style=for-the-badge&logo=github&logoColor=white">
    </a>
    <a href="https://github.com/HKUDS/LightRAG/discussions" style="text-decoration: none;">
      <img src="https://img.shields.io/badge/💬%20Discussions-4ecdc4?style=for-the-badge&logo=github&logoColor=white">
    </a>
  </div>
</div>

<div align="center">
  <div style="width: 100%; max-width: 600px; margin: 20px auto; padding: 20px; background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 217, 255, 0.05) 100%); border-radius: 15px; border: 1px solid rgba(0, 217, 255, 0.2);">
    <div style="display: flex; justify-content: center; align-items: center; gap: 15px;">
      <span style="font-size: 24px;">⭐</span>
      <span style="color: #00d9ff; font-size: 18px;">LightRAG をご覧いただきありがとうございます！</span>
      <span style="font-size: 24px;">⭐</span>
    </div>
  </div>
</div>
