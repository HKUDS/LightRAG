#!/usr/bin/env bash
# Auto-generated script to apply Wave 4 commits
set -e

echo "Cherry-picking dde728a3: Bump core version to 1.5.0 and API to 0236"
git cherry-pick -x dde728a3

echo "Cherry-picking 289337b2: Bump API version to 0237"
git cherry-pick -x 289337b2

echo "Cherry-picking 79f623a2: Bump core version to 1.4.9.3"
git cherry-pick -x 79f623a2

echo "Cherry-picking f402ad27: Bump API version to 0238"
git cherry-pick -x f402ad27

echo "Cherry-picking a81c122f: Bump API version to 0240"
git cherry-pick -x a81c122f

echo "Cherry-picking f3740d82: Bump API version to 0239"
git cherry-pick -x f3740d82

echo "Cherry-picking 91b8722b: Bump core version to 1.4.9.4"
git cherry-pick -x 91b8722b

echo "Cherry-picking 7bf9d1e8: Bump API version to 0241"
git cherry-pick -x 7bf9d1e8

echo "Cherry-picking ef4acf53: Update pandas requirement from <2.3.0,>=2.0.0 to >=2.0.0,<2.4.0"
git cherry-pick -x ef4acf53

echo "Cherry-picking 1101562e: Bump API version to 0243"
git cherry-pick -x 1101562e

echo "Cherry-picking fdf0fe04: Bump API version to 0244"
git cherry-pick -x fdf0fe04

echo "Cherry-picking 3eb3a075: Bump core version to 1.4.9.5 and API version to 0245"
git cherry-pick -x 3eb3a075

echo "Cherry-picking 94f24a66: Bump API version to 0246"
git cherry-pick -x 94f24a66

echo "Cherry-picking f81dd4e7: Update redis requirement from <7.0.0,>=5.0.0 to >=5.0.0,<8.0.0"
git cherry-pick -x f81dd4e7

echo "Cherry-picking 4bf41abe: Merge pull request #2272 from HKUDS/dependabot/pip/redis-gte-5.0.0-and-lt-8.0.0"
git cherry-pick -x 4bf41abe

echo "Cherry-picking 3a7f7535: Bump core version to 1.4.9.6 and API version to 0248"
git cherry-pick -x 3a7f7535

echo "Cherry-picking da2e9efd: Bump API version to 0247"
git cherry-pick -x da2e9efd

echo "Cherry-picking c9e73bb4: Bump core version to 1.4.9.7 and API version to 0249"
git cherry-pick -x c9e73bb4

echo "Cherry-picking e5414c61: Bump core version to 1.4.9.8 and API version to 0250"
git cherry-pick -x e5414c61

echo "Cherry-picking 5bcd2926: Bump API version to 0251"
git cherry-pick -x 5bcd2926

echo "Cherry-picking cf732dbf: Bump core version to 1.4.9.9 and API to 0252"
git cherry-pick -x cf732dbf

echo "Cherry-picking 1f9d0735: Bump API version to 0253"
git cherry-pick -x 1f9d0735

echo "Cherry-picking 9262f66d: Bump API version to 0255"
git cherry-pick -x 9262f66d

echo "Cherry-picking d16c7840: Bump API version to 0256"
git cherry-pick -x d16c7840

echo "Cherry-picking f83b475a: Remove Dependabot configuration file"
git cherry-pick -x f83b475a

echo "Cherry-picking 881b8d3a: Bump API version to 0257"
git cherry-pick -x 881b8d3a

echo "Cherry-picking 112ed234: Bump API version to 0258"
git cherry-pick -x 112ed234

echo "Cherry-picking 13fc9f33: Reduce dependabot open pull request limits"
git cherry-pick -x 13fc9f33

echo "Cherry-picking 381ddfff: Bump API version to 0259"
git cherry-pick -x 381ddfff

echo "Cherry-picking 883c5dc0: Update dependabot config with new groupings and patterns"
git cherry-pick -x 883c5dc0

echo "Cherry-picking 9425277f: Improve dependabot config with better docs and numpy ignore rule"
git cherry-pick -x 9425277f

echo "Cherry-picking b2f1de4a: Merge pull request #2447 from danielaskdd/dependabot"
git cherry-pick -x b2f1de4a

echo "Cherry-picking f93bda58: Enable numpy updates in dependabot configuration"
git cherry-pick -x f93bda58

echo "Cherry-picking a250d881: Update webui assets"
git cherry-pick -x a250d881

echo "Cherry-picking cec784f6: Update webui assets"
git cherry-pick -x cec784f6

echo "Cherry-picking 1574fec7: Update webui assets"
git cherry-pick -x 1574fec7

echo "Cherry-picking b5f83767: Update webui assets"
git cherry-pick -x b5f83767

echo "Cherry-picking d473f635: Update webui assets"
git cherry-pick -x d473f635

echo "Cherry-picking 5734f51e: Merge pull request #2198 from danielaskdd/webui"
git cherry-pick -x 5734f51e

echo "Cherry-picking 8a009899: Update webui assets"
git cherry-pick -x 8a009899

echo "Cherry-picking e1af1c6d: Update webui assets"
git cherry-pick -x e1af1c6d

echo "Cherry-picking c0b1552e: Remove .gitkeep file by ensuring webui dir exists on bun build"
git cherry-pick -x c0b1552e

echo "Cherry-picking df52ce98: Revert vite.config.ts to origin version"
git cherry-pick -x df52ce98

echo "Cherry-picking 8070d0cf: Merge pull request #2234 from danielaskdd/fix-webui"
git cherry-pick -x 8070d0cf

echo "Cherry-picking 80f2e691: Remove redundant i18n import triggered the Vite “dynamic + static import” warning"
git cherry-pick -x 80f2e691

echo "Cherry-picking 2539b4e2: Merge pull request #2418 from danielaskdd/start-without-webui"
git cherry-pick -x 2539b4e2

echo "Cherry-picking 48b67d30: Handle missing WebUI assets gracefully without blocking server startup"
git cherry-pick -x 48b67d30

echo "Cherry-picking 15bfd9fa: Bump the ui-components group in /lightrag_webui with 7 updates"
git cherry-pick -x 15bfd9fa

echo "Cherry-picking 1f3d7006: Bump @stylistic/eslint-plugin-js from 3.1.0 to 4.4.1 in /lightrag_webui"
git cherry-pick -x 1f3d7006

echo "Cherry-picking 245c0c32: Bump the build-tools group in /lightrag_webui with 4 updates"
git cherry-pick -x 245c0c32

echo "Cherry-picking 587a930b: Bump the react group in /lightrag_webui with 3 updates"
git cherry-pick -x 587a930b

echo "Cherry-picking 9ae1c7fc: Bump react-error-boundary from 5.0.0 to 6.0.0 in /lightrag_webui"
git cherry-pick -x 9ae1c7fc

echo "Cherry-picking e2431b67: Bump @vitejs/plugin-react-swc from 3.11.0 to 4.2.0 in /lightrag_webui"
git cherry-pick -x e2431b67

echo "Cherry-picking f4acb25c: Bump the frontend-minor-patch group in /lightrag_webui with 6 updates"
git cherry-pick -x f4acb25c

echo "Cherry-picking 09aa8483: Merge branch 'dependabot/bun/lightrag_webui/stylistic/eslint-plugin-js-4.4.1'"
git cherry-pick -x 09aa8483

echo "Cherry-picking 0c2a653c: Merge branch 'main' into dependabot/bun/lightrag_webui/vitejs/plugin-react-swc-4.2.0"
git cherry-pick -x 0c2a653c

echo "Cherry-picking 0ca71a57: Bump sonner from 1.7.4 to 2.0.7 in /lightrag_webui"
git cherry-pick -x 0ca71a57

echo "Cherry-picking 0d89cd26: Merge pull request #2462 from HKUDS/dependabot/bun/lightrag_webui/faker-js/faker-10.1.0"
git cherry-pick -x 0d89cd26

echo "Cherry-picking 0f045a52: Merge branch 'dependabot/bun/lightrag_webui/vitejs/plugin-react-swc-4.2.0' of github.com:HKUDS/LightRAG into dependabot/bun/lightrag_webui/vitejs/plugin-react-swc-4.2.0"
git cherry-pick -x 0f045a52

echo "Cherry-picking 13a285d4: Merge pull request #2451 from HKUDS/dependabot/bun/lightrag_webui/build-tools-0944ec6cea"
git cherry-pick -x 13a285d4

echo "Cherry-picking 19cae272: Merge pull request #2463 from HKUDS/dependabot/bun/lightrag_webui/types/node-24.9.2"
git cherry-pick -x 19cae272

echo "Cherry-picking 1d12f497: Bump i18next from 24.2.3 to 25.6.0 in /lightrag_webui"
git cherry-pick -x 1d12f497

echo "Cherry-picking 29bd027a: Bump @vitejs/plugin-react-swc"
git cherry-pick -x 29bd027a

echo "Cherry-picking 2bb9ec13: Bump eslint-plugin-react-hooks from 5.2.0 to 7.0.1 in /lightrag_webui"
git cherry-pick -x 2bb9ec13

echo "Cherry-picking 35c79341: Merge pull request #2450 from HKUDS/dependabot/bun/lightrag_webui/ui-components-018be29f1c"
git cherry-pick -x 35c79341

echo "Cherry-picking 42b09b10: Bump globals from 15.15.0 to 16.5.0 in /lightrag_webui"
git cherry-pick -x 42b09b10

echo "Cherry-picking 57d9cc8f: Merge pull request #2464 from HKUDS/dependabot/bun/lightrag_webui/react-syntax-highlighter-16.1.0"
git cherry-pick -x 57d9cc8f

echo "Cherry-picking 59b1b58f: Merge pull request #2456 from HKUDS/dependabot/bun/lightrag_webui/globals-16.5.0"
git cherry-pick -x 59b1b58f

echo "Cherry-picking 5ca4792c: Bump @faker-js/faker from 9.9.0 to 10.1.0 in /lightrag_webui"
git cherry-pick -x 5ca4792c

echo "Cherry-picking 67d9455c: Merge pull request #2466 from HKUDS/dependabot/bun/lightrag_webui/react-i18next-16.2.3"
git cherry-pick -x 67d9455c

echo "Cherry-picking 68bee74d: Bump the frontend-minor-patch group in /lightrag_webui with 2 updates"
git cherry-pick -x 68bee74d

echo "Cherry-picking 6e2f125a: Merge pull request #2471 from HKUDS/dependabot/bun/lightrag_webui/frontend-minor-patch-172e1e6fcf"
git cherry-pick -x 6e2f125a

echo "Cherry-picking 7545fa72: Bump vite in /lightrag_webui in the build-tools group"
git cherry-pick -x 7545fa72

echo "Cherry-picking 7f7ce9d3: Bump i18next in /lightrag_webui in the frontend-minor-patch group"
git cherry-pick -x 7f7ce9d3

echo "Cherry-picking 8cdf8a12: Merge pull request #2455 from HKUDS/dependabot/bun/lightrag_webui/i18next-25.6.0"
git cherry-pick -x 8cdf8a12

echo "Cherry-picking 964b53e7: Merge pull request #2458 from HKUDS/dependabot/bun/lightrag_webui/eslint-plugin-react-hooks-7.0.1"
git cherry-pick -x 964b53e7

echo "Cherry-picking a47414f7: Merge pull request #2461 from HKUDS/dependabot/bun/lightrag_webui/sonner-2.0.7"
git cherry-pick -x a47414f7

echo "Cherry-picking a8e79a8a: Merge remote-tracking branch 'upstream/dependabot/bun/lightrag_webui/react-error-boundary-6.0.0'"
git cherry-pick -x a8e79a8a

echo "Cherry-picking ab718218: Merge pull request #2449 from HKUDS/dependabot/bun/lightrag_webui/react-b0cb288b9e"
git cherry-pick -x ab718218

echo "Cherry-picking b2b5f80b: Merge pull request #2467 from HKUDS/dependabot/bun/lightrag_webui/build-tools-ecae90f21c"
git cherry-pick -x b2b5f80b

echo "Cherry-picking b38f4dd7: Bump react-syntax-highlighter from 15.6.6 to 16.1.0 in /lightrag_webui"
git cherry-pick -x b38f4dd7

echo "Cherry-picking bd487a45: Bump @vitejs/plugin-react-swc from 3.11.0 to 4.2.0 in /lightrag_webui"
git cherry-pick -x bd487a45

echo "Cherry-picking c6c201d7: Merge branch 'main' into dependabot/bun/lightrag_webui/frontend-minor-patch-a28ecac770"
git cherry-pick -x c6c201d7

echo "Cherry-picking d3b5cb63: Bump vite from 6.3.6 to 7.1.12 in /lightrag_webui"
git cherry-pick -x d3b5cb63

echo "Cherry-picking d5e7b230: Bump @types/node from 22.18.9 to 24.9.2 in /lightrag_webui"
git cherry-pick -x d5e7b230

echo "Cherry-picking dd4c988b: Bump react-markdown from 9.1.0 to 10.1.0 in /lightrag_webui"
git cherry-pick -x dd4c988b

echo "Cherry-picking dd95813f: Merge pull request #2465 from HKUDS/dependabot/bun/lightrag_webui/react-markdown-10.1.0"
git cherry-pick -x dd95813f

echo "Cherry-picking ddd32f58: Merge pull request #2470 from HKUDS/dependabot/bun/lightrag_webui/build-tools-939f50a5f3"
git cherry-pick -x ddd32f58

echo "Cherry-picking e20f86a0: Bump react-i18next from 15.7.4 to 16.2.3 in /lightrag_webui"
git cherry-pick -x e20f86a0

echo "Cherry-picking e429d553: Merge pull request #2459 from HKUDS/dependabot/bun/lightrag_webui/frontend-minor-patch-9aaf02af10"
git cherry-pick -x e429d553

echo "Cherry-picking e547c003: Merge pull request #2460 from HKUDS/dependabot/bun/lightrag_webui/vite-7.1.12"
git cherry-pick -x e547c003

echo "Cherry-picking e7966712: Merge pull request #2452 from HKUDS/dependabot/bun/lightrag_webui/frontend-minor-patch-a28ecac770"
git cherry-pick -x e7966712

echo "Cherry-picking ea826a38: Merge branch 'dependabot/bun/lightrag_webui/vitejs/plugin-react-swc-4.2.0'"
git cherry-pick -x ea826a38

echo "Cherry-picking ffcd75a4: decalre targetNode after check sourceNode"
git cherry-pick -x ffcd75a4

echo "Cherry-picking 1bd84f00: Marchuk Merge branch 'main' into fix/dark-mode-graph-text-colors"
git cherry-pick -x 1bd84f00

echo "Cherry-picking 7297ca1d: Marchuk Fix dark mode graph labels for system theme and improve colors"
git cherry-pick -x 7297ca1d

echo "Cherry-picking bb6138e7: fix(prompt): Clarify reference section restrictions in prompt templates"
git cherry-pick -x bb6138e7

echo "Cherry-picking f83cde14: fix(prompt): Improve markdown formatting requirements and reference style"
git cherry-pick -x f83cde14

echo "Cherry-picking 112349ed: Modernize type hints and remove Python 3.8 compatibility code"
git cherry-pick -x 112349ed

echo "Cherry-picking 181525ff: Merge branch 'main' into zl7261/main"
git cherry-pick -x 181525ff

echo "Cherry-picking 19a41584: Fix linting"
git cherry-pick -x 19a41584

echo "Cherry-picking 1f07d4b1: Remove .env_example from .gitignore"
git cherry-pick -x 1f07d4b1

echo "Cherry-picking 6bf6f43d: Remove bold formatting from instruction headers in prompts"
git cherry-pick -x 6bf6f43d

echo "Cherry-picking b1a4e7d7: Fix linting"
git cherry-pick -x b1a4e7d7

echo "Cherry-picking d2196a4e: Merge remote-tracking branch 'origin/main'"
git cherry-pick -x d2196a4e

echo "Cherry-picking d4abe704: Hide dev options in production builds"
git cherry-pick -x d4abe704

echo "Cherry-picking f6b71536: Merge branch 'main' into fix/dark-mode-graph-text-colors"
git cherry-pick -x f6b71536

echo "Cherry-picking b9c37bd9: Fix linting"
git cherry-pick -x b9c37bd9

echo "Cherry-picking cf2a024e: feat: Add endpoint and UI to retry failed documents"
git cherry-pick -x cf2a024e

echo "Cherry-picking 0aef6a16: Add theme-aware edge highlighting colors for graph control"
git cherry-pick -x 0aef6a16

echo "Cherry-picking 0c1cb7b7: Improve document tooltip display with track ID and better formatting"
git cherry-pick -x 0c1cb7b7

echo "Cherry-picking 0d694962: Merge branch 'feat/retry-failed-documents-upstream'"
git cherry-pick -x 0d694962

echo "Cherry-picking 1b274706: Merge pull request #2171 from danielaskdd/doc-name-in-full-docs"
git cherry-pick -x 1b274706

echo "Cherry-picking 4fe41f76: Merge branch 'doc-name-in-full-docs'"
git cherry-pick -x 4fe41f76

echo "Cherry-picking 7b1f8e0f: Update scan tooltip to clarify it also reprocesses failed documents"
git cherry-pick -x 7b1f8e0f

echo "Cherry-picking bf6ca9dd: Add retry failed button translations and standardize button text"
git cherry-pick -x bf6ca9dd

echo "Cherry-picking d550f1c5: Fix linting"
git cherry-pick -x d550f1c5

echo "Cherry-picking dad90d25: Merge pull request #2170 from danielaskdd/tooltips-optimize"
git cherry-pick -x dad90d25

echo "Cherry-picking 6190fa89: Fix linting"
git cherry-pick -x 6190fa89

echo "Cherry-picking d8a9617c: fix: fix: asyncpg bouncer connection pool error"
git cherry-pick -x d8a9617c

echo "Cherry-picking fdcb034d: chore: distinguish settings"
git cherry-pick -x fdcb034d

echo "Cherry-picking ea5e390b: Merge pull request #2178 from aleksvujic/patch-1"
git cherry-pick -x ea5e390b

echo "Cherry-picking f1e01107: Merge branch 'kevinnkansah/main'"
git cherry-pick -x f1e01107

echo "Cherry-picking 9f44e89d: Add knowledge graph manipulation endpoints"
git cherry-pick -x 9f44e89d

echo "Cherry-picking ae9f4ae7: Rangana fix: Remove trailing whitespace for pre-commit linting"
git cherry-pick -x ae9f4ae7

echo "Cherry-picking b7c77396: Fix entity/relation creation endpoints to properly update vector stores"
git cherry-pick -x b7c77396

echo "Cherry-picking f6d1fb98: Fix Linting errors"
git cherry-pick -x f6d1fb98

echo "Cherry-picking 12facac5: Enhance graph API endpoints with detailed docs and field validation"
git cherry-pick -x 12facac5

echo "Cherry-picking 648d7bb1: Refactor Helm template to handle optional envFrom values safely"
git cherry-pick -x 648d7bb1

echo "Cherry-picking 85d1a563: Merge branch 'adminunblinded/main'"
git cherry-pick -x 85d1a563

echo "Cherry-picking b81b8620: chore: update deps"
git cherry-pick -x b81b8620

echo "Cherry-picking d0ae7a67: Fix typos and grammar in env.example configuration comments"
git cherry-pick -x d0ae7a67

echo "Cherry-picking fea10cd0: Merge branch 'chart-enchancment'"
git cherry-pick -x fea10cd0

echo "Cherry-picking 1a4d6775: i18n: fix mustache brackets"
git cherry-pick -x 1a4d6775

echo "Cherry-picking 49326f2b: Merge pull request #2194 from danielaskdd/offline"
git cherry-pick -x 49326f2b

echo "Cherry-picking 766f27da: Merge pull request #2193 from kevinnkansah/main"
git cherry-pick -x 766f27da

echo "Cherry-picking 7cddd564: Revert core version to 1.4.9..2"
git cherry-pick -x 7cddd564

echo "Cherry-picking b7216ede: Fix linting"
git cherry-pick -x b7216ede

echo "Cherry-picking e1e4f1b0: Fix get_by_ids to return None for missing records consistently"
git cherry-pick -x e1e4f1b0

echo "Cherry-picking 074f0c8b: Update docstring for adelete_by_doc_id method clarity"
git cherry-pick -x 074f0c8b

echo "Cherry-picking 2d9334d3: Simplify Root component by removing async i18n initialization"
git cherry-pick -x 2d9334d3

echo "Cherry-picking 44f51f88: Add fallback value for accordion content height CSS variable"
git cherry-pick -x 44f51f88

echo "Cherry-picking 5290b60e: Merge pull request #2196 from zl7261/main"
git cherry-pick -x 5290b60e

echo "Cherry-picking f2fb1202: Move accordion keyframes from CSS to Tailwind config and add fallback 'auto' value"
git cherry-pick -x f2fb1202

echo "Cherry-picking 6c05f0f8: Fix linting"
git cherry-pick -x 6c05f0f8

echo "Cherry-picking be9e6d16: Exclude Frontend Build Artifacts from Git Repository"
git cherry-pick -x be9e6d16

echo "Cherry-picking cc436910: Merge branch 'kevinnkansah/main'"
git cherry-pick -x cc436910

echo "Cherry-picking 130b4959: Add PREPROCESSED (multimodal_processed) status for multimodal document processing"
git cherry-pick -x 130b4959

echo "Cherry-picking 4e740af7: Import from env and use default if none and removed useless import"
git cherry-pick -x 4e740af7

echo "Cherry-picking 50210e25: Add @tailwindcss/typography plugin and fix Tailwind config"
git cherry-pick -x 50210e25

echo "Cherry-picking 5ace200b: Merge pull request #2208 from danielaskdd/remove-fontend-artifact"
git cherry-pick -x 5ace200b

echo "Cherry-picking 64900b54: Add frontend source code update warning"
git cherry-pick -x 64900b54

echo "Cherry-picking 8bf41131: Standardize build commands and remove --emptyOutDir flag"
git cherry-pick -x 8bf41131

echo "Cherry-picking 92a66565: Merge pull request #2211 from HKUDS/add-preprocessed-status"
git cherry-pick -x 92a66565

echo "Cherry-picking 965d8b16: Merge branch 'add-preprocessed-status'"
git cherry-pick -x 965d8b16

echo "Cherry-picking a8bbce3a: Use frozen lockfile for consistent frontend builds"
git cherry-pick -x a8bbce3a

echo "Cherry-picking d52c3377: Import from env and use default if none and removed useless import"
git cherry-pick -x d52c3377

echo "Cherry-picking ee45ab51: Move frontend build check from setup.py to runtime server startup"
git cherry-pick -x ee45ab51

echo "Cherry-picking 17c2a929: Get max source Id config from .env and lightRAG init"
git cherry-pick -x 17c2a929

echo "Cherry-picking 29bac49f: Handle empty query results by returning None instead of fail responses"
git cherry-pick -x 29bac49f

echo "Cherry-picking 433ec813: Improve offline installation with constraints and version bounds"
git cherry-pick -x 433ec813

echo "Cherry-picking 7f223d5a: Fix linting"
git cherry-pick -x 7f223d5a

echo "Cherry-picking 83b10a52: Merge pull request #2218 from danielaskdd/issue-2215"
git cherry-pick -x 83b10a52

echo "Cherry-picking c06522b9: Get max source Id config from .env and lightRAG init"
git cherry-pick -x c06522b9

echo "Cherry-picking 8cc8bbf4: Change Docker build cache mode from max to min"
git cherry-pick -x 8cc8bbf4

echo "Cherry-picking 8e3497dc: Update comments"
git cherry-pick -x 8e3497dc

echo "Cherry-picking c61b7bd4: Remove torch and transformers from offline dependency groups"
git cherry-pick -x c61b7bd4

echo "Cherry-picking ef6ed429: Optimize Docker builds with layer caching and add pip for runtime installs"
git cherry-pick -x ef6ed429

echo "Cherry-picking 03333d63: Merge branch 'main' into limit-vdb-metadata-size"
git cherry-pick -x 03333d63

echo "Cherry-picking 04d23671: Fix redoc access problem in front-end dev mode"
git cherry-pick -x 04d23671

echo "Cherry-picking 06ed2d06: Merge branch 'main' into remove-dotenv"
git cherry-pick -x 06ed2d06

echo "Cherry-picking 1642710f: Remove dotenv dependency from project"
git cherry-pick -x 1642710f

echo "Cherry-picking 46ac5dac: Improve API description formatting and add ReDoc link"
git cherry-pick -x 46ac5dac

echo "Cherry-picking 4c3ab584: Improve AsyncSelect layout and text overflow handling"
git cherry-pick -x 4c3ab584

echo "Cherry-picking 6b37d3ca: Merge branch 'feat-entity-size-caps' into limit-vdb-metadata-size"
git cherry-pick -x 6b37d3ca

echo "Cherry-picking 9f49e56a: Merge branch 'main' into feat-entity-size-caps"
git cherry-pick -x 9f49e56a

echo "Cherry-picking c0a87ca7: Merge branch 'remove-dotenv'"
git cherry-pick -x c0a87ca7

echo "Cherry-picking f45dce34: Fix cache control error of index.html"
git cherry-pick -x f45dce34

echo "Cherry-picking f5558240: Fix tuple delimiter corruption handling in regex patterns"
git cherry-pick -x f5558240

echo "Cherry-picking 012aaada: Update Swagger API key status description text"
git cherry-pick -x 012aaada

echo "Cherry-picking 813f4af9: Fix linting"
git cherry-pick -x 813f4af9

echo "Cherry-picking bdadaa67: Merge branch 'main' into limit-vdb-metadata-size"
git cherry-pick -x bdadaa67

echo "Cherry-picking 637b850e: Add truncation indicator and update property labels in graph view"
git cherry-pick -x 637b850e

echo "Cherry-picking a9fec267: Add file path limit configuration for entities and relations"
git cherry-pick -x a9fec267

echo "Cherry-picking e01c998e: Track placeholders in file paths for accurate source count display"
git cherry-pick -x e01c998e

echo "Cherry-picking 1248b3ab: Increase default limits for source IDs and file paths in metadata"
git cherry-pick -x 1248b3ab

echo "Cherry-picking 3ad616be: Change default source IDs limit method from KEEP to FIFO"
git cherry-pick -x 3ad616be

echo "Cherry-picking 3ed2abd8: Improve logging to show source ID ratios when skipping entities/edges"
git cherry-pick -x 3ed2abd8

echo "Cherry-picking 665f60b9: Refactor entity/relation merge to consolidate VDB operations within functions"
git cherry-pick -x 665f60b9

echo "Cherry-picking 80668aae: Improve file path truncation labels and UI consistency"
git cherry-pick -x 80668aae

echo "Cherry-picking a809245a: Preserve file path order by using lists instead of sets"
git cherry-pick -x a809245a

echo "Cherry-picking aee0afdd: Merge pull request #2240 from danielaskdd/limit-vdb-metadata-size"
git cherry-pick -x aee0afdd

echo "Cherry-picking be3d274a: Refactor node and edge merging logic with improved code structure"
git cherry-pick -x be3d274a

echo "Cherry-picking cd1c48be: Standardize placeholder format to use colon separator consistently"
git cherry-pick -x cd1c48be

echo "Cherry-picking fe890fca: Improve formatting of limit method info in rebuild functions"
git cherry-pick -x fe890fca

echo "Cherry-picking 04d9fe02: Merge branch 'HKUDS:main' into main"
git cherry-pick -x 04d9fe02

echo "Cherry-picking 06533fdb: Merge pull request #2248 from danielaskdd/preprocess-rayanything"
git cherry-pick -x 06533fdb

echo "Cherry-picking 20edd329: Merge pull request #2244 from danielaskdd/del-doc-cache"
git cherry-pick -x 20edd329

echo "Cherry-picking 3ba1d75c: Merge pull request #2243 from xiaojunxiang2023/main"
git cherry-pick -x 3ba1d75c

echo "Cherry-picking 8dc23eef: Fix RayAnything compatible problem"
git cherry-pick -x 8dc23eef

echo "Cherry-picking 904b1f46: Add entity name length truncation with configurable limit"
git cherry-pick -x 904b1f46

echo "Cherry-picking b76350a3: Fix linting"
git cherry-pick -x b76350a3

echo "Cherry-picking c92ab837: Fix linting"
git cherry-pick -x c92ab837

echo "Cherry-picking cf2174b9: Merge pull request #2245 from danielaskdd/entity-name-len"
git cherry-pick -x cf2174b9

echo "Cherry-picking d392db7b: Fix typo in 'equipment' in prompt.py"
git cherry-pick -x d392db7b

echo "Cherry-picking d7e2527e: Handle cache deletion errors gracefully instead of raising exceptions"
git cherry-pick -x d7e2527e

echo "Cherry-picking f24a2616: Allow users to provide keywords with QueryRequest"
git cherry-pick -x f24a2616

echo "Cherry-picking 78ad8873: Add cancellation check in delete loop"
git cherry-pick -x 78ad8873

echo "Cherry-picking 083b163c: Improve lock logging with consistent messaging and debug levels"
git cherry-pick -x 083b163c

echo "Cherry-picking 11f1f366: Merge pull request #2262 from danielaskdd/sort-edge"
git cherry-pick -x 11f1f366

echo "Cherry-picking 3ad4f12f: Merge pull request #2259 from danielaskdd/data-migration-problem"
git cherry-pick -x 3ad4f12f

echo "Cherry-picking 5ee9a2f8: Fix entity consistency in knowledge graph rebuilding and merging"
git cherry-pick -x 5ee9a2f8

echo "Cherry-picking 97a2ee4e: Rename rebuild function name and improve relationship logging format"
git cherry-pick -x 97a2ee4e

echo "Cherry-picking 9ed19695: Remove separate retry button and merge functionality into scan button"
git cherry-pick -x 9ed19695

echo "Cherry-picking a9bc3484: Remove enable_logging parameter from data init lock call"
git cherry-pick -x a9bc3484

echo "Cherry-picking c82485d9: Merge pull request #2253 from Mobious/main"
git cherry-pick -x c82485d9

echo "Cherry-picking 38559373: Fix entity merging to include target entity relationships"
git cherry-pick -x 38559373

echo "Cherry-picking 6015e8bc: Refactor graph utils to use unified persistence callback"
git cherry-pick -x 6015e8bc

echo "Cherry-picking 69b4cda2: Merge pull request #2265 from danielaskdd/edit-kg-new"
git cherry-pick -x 69b4cda2

echo "Cherry-picking bf1897a6: Normalize entity order for undirected graph consistency"
git cherry-pick -x bf1897a6

echo "Cherry-picking 11a1631d: Refactor entity edit and merge functions to support merge-on-rename"
git cherry-pick -x 11a1631d

echo "Cherry-picking 25f829ef: Enable editing of entity_type field in node properties"
git cherry-pick -x 25f829ef

echo "Cherry-picking 411e92e6: Fix vector deletion logging to show actual deleted count"
git cherry-pick -x 411e92e6

echo "Cherry-picking 5155edd8: feat: Improve entity merge and edit UX"
git cherry-pick -x 5155edd8

echo "Cherry-picking 8dfd3bf4: Replace global graph DB lock with fine-grained keyed locking"
git cherry-pick -x 8dfd3bf4

echo "Cherry-picking 97034f06: Add allow_merge parameter to entity update API endpoint"
git cherry-pick -x 97034f06

echo "Cherry-picking ab32456a: Refactor entity merging with unified attribute merge function"
git cherry-pick -x ab32456a

echo "Cherry-picking 29c4a91d: Move relationship ID sorting to before vector DB operations"
git cherry-pick -x 29c4a91d

echo "Cherry-picking 88d12bea: Add offline Swagger UI support with custom static file serving"
git cherry-pick -x 88d12bea

echo "Cherry-picking af6aff33: Merge pull request #2266 from danielaskdd/merge-entity"
git cherry-pick -x af6aff33

echo "Cherry-picking b32b2e8b: Refactor merge dialog and improve search history sync"
git cherry-pick -x b32b2e8b

echo "Cherry-picking d0be68c8: Merge pull request #2273 from danielaskdd/static-docs"
git cherry-pick -x d0be68c8

echo "Cherry-picking ea006bd3: Fix entity update logic to handle renaming operations"
git cherry-pick -x ea006bd3

echo "Cherry-picking 0fa2fc9c: Refactor systemd service config to use environment variables"
git cherry-pick -x 0fa2fc9c

echo "Cherry-picking 3fa79026: Fix Entity Source IDs Tracking Problem"
git cherry-pick -x 3fa79026

echo "Cherry-picking 4a46d39c: Replace GUNICORN_CMD_ARGS with custom LIGHTRAG_GUNICORN_MODE flag"
git cherry-pick -x 4a46d39c

echo "Cherry-picking 54c48dce: Fix z-index layering for GraphViewer UI panels"
git cherry-pick -x 54c48dce

echo "Cherry-picking 6489aaa7: Remove worker_exit hook and improve cleanup logging"
git cherry-pick -x 6489aaa7

echo "Cherry-picking 6dc027cb: Merge branch 'fix-exit-handler'"
git cherry-pick -x 6dc027cb

echo "Cherry-picking 72b29659: Fix worker process cleanup to prevent shared resource conflicts"
git cherry-pick -x 72b29659

echo "Cherry-picking 816feefd: Fix cleanup coordination between Gunicorn and UvicornWorker lifecycles"
git cherry-pick -x 816feefd

echo "Cherry-picking 8af8bd80: docs: add frontend build steps to server installation guide"
git cherry-pick -x 8af8bd80

echo "Cherry-picking a1cf01dc: Merge pull request #2280 from danielaskdd/fix-exit-handler"
git cherry-pick -x a1cf01dc

echo "Cherry-picking c5ad9982: Merge pull request #2281 from danielaskdd/restore-query-example"
git cherry-pick -x c5ad9982

echo "Cherry-picking d5bcd14c: Refactor service deployment to use direct process execution"
git cherry-pick -x d5bcd14c

echo "Cherry-picking 783e2f3b: Update uv.lock"
git cherry-pick -x 783e2f3b

echo "Cherry-picking 78ccc4f6: Refactor .gitignore"
git cherry-pick -x 78ccc4f6

echo "Cherry-picking 79a17c3f: Fix graph value handling for entity_id updates"
git cherry-pick -x 79a17c3f

echo "Cherry-picking 8145201d: Merge pull request #2284 from danielaskdd/fix-static-missiing"
git cherry-pick -x 8145201d

echo "Cherry-picking f610fdaf: Merge branch 'main' into Anush008/main"
git cherry-pick -x f610fdaf

echo "Cherry-picking 08b0283b: Merge pull request #2291 from danielaskdd/reload-popular-labels"
git cherry-pick -x 08b0283b

echo "Cherry-picking 2496d871: Add data/ directory to .gitignore"
git cherry-pick -x 2496d871

echo "Cherry-picking 3b48cf16: Merge pull request #2289 from danielaskdd/fix-pycrptodome-missing"
git cherry-pick -x 3b48cf16

echo "Cherry-picking 4cbd8761: feat: Update node color and legent after entity_type changed"
git cherry-pick -x 4cbd8761

echo "Cherry-picking 6b4514c8: Reduce logging verbosity in entity merge relation processing"
git cherry-pick -x 6b4514c8

echo "Cherry-picking 71b27ec4: Optimize property edit dialog to use trimmed value consistently"
git cherry-pick -x 71b27ec4

echo "Cherry-picking 7ccc1fdd: Add frontend rebuild warning indicator to version display"
git cherry-pick -x 7ccc1fdd

echo "Cherry-picking 94cdbe77: Merge pull request #2290 from danielaskdd/delete-residual-edges"
git cherry-pick -x 94cdbe77

echo "Cherry-picking afb5e5c1: Fix edge cleanup when deleting entities to prevent orphaned relationships"
git cherry-pick -x afb5e5c1

echo "Cherry-picking bda52a87: Merge pull request #2287 from danielaskdd/fix-ui"
git cherry-pick -x bda52a87

echo "Cherry-picking 728721b1: Remove redundant separator lines in gunicorn shutdown handler"
git cherry-pick -x 728721b1

echo "Cherry-picking bc8a8842: Merge pull request #2295 from danielaskdd/mix-query-without-kg"
git cherry-pick -x bc8a8842

echo "Cherry-picking ec2ea4fd: Rename function and variables for clarity in context building"
git cherry-pick -x ec2ea4fd

echo "Cherry-picking 363f3051: eval using open ai"
git cherry-pick -x 363f3051

echo "Cherry-picking 6975e69e: Merge pull request #2298 from anouar-bm/feat/langfuse-observability"
git cherry-pick -x 6975e69e

echo "Cherry-picking 9495778c: refactor: reorder Langfuse import logic for improved clarity"
git cherry-pick -x 9495778c

echo "Cherry-picking 9d69e8d7: fix(api): Change content field from string to list in query responses"
git cherry-pick -x 9d69e8d7

echo "Cherry-picking c9e1c6c1: fix(api): change content field to list in query responses"
git cherry-pick -x c9e1c6c1

echo "Cherry-picking e0966b65: Add BuildKit cache mounts to optimize Docker build performance"
git cherry-pick -x e0966b65

echo "Cherry-picking 6d61f70b: Clean up RAG evaluator logging and remove excessive separator lines"
git cherry-pick -x 6d61f70b

echo "Cherry-picking 7abc6877: Add comprehensive configuration and  compatibility fixes for RAGAS"
git cherry-pick -x 7abc6877

echo "Cherry-picking 451257ae: Doc: Update news with recent features"
git cherry-pick -x 451257ae

echo "Cherry-picking bd62bb30: Merge pull request #2314 from danielaskdd/ragas"
git cherry-pick -x bd62bb30

echo "Cherry-picking d803df94: Fix linting"
git cherry-pick -x d803df94

echo "Cherry-picking eb80771f: Merge pull request #2311 from danielaskdd/evalueate-cli"
git cherry-pick -x eb80771f

echo "Cherry-picking 0216325e: fix(ui): Remove dynamic import for i18n in settings store"
git cherry-pick -x 0216325e

echo "Cherry-picking 04ed709b: Optimize entity deletion by batching edge queries to avoid N+1 problem"
git cherry-pick -x 04ed709b

echo "Cherry-picking 0c47d1a2: Fix linting"
git cherry-pick -x 0c47d1a2

echo "Cherry-picking 155f5975: Fix node ID normalization and improve batch operation consistency"
git cherry-pick -x 155f5975

echo "Cherry-picking 3276b7a4: Fix linting"
git cherry-pick -x 3276b7a4

echo "Cherry-picking 5f49cee2: Merge branch 'main' into VOXWAVE-FOUNDRY/main"
git cherry-pick -x 5f49cee2

echo "Cherry-picking 678e17bb: Revert \"fix(ui): Remove dynamic import for i18n in settings store\""
git cherry-pick -x 678e17bb

echo "Cherry-picking 6e36ff41: Fix linting"
git cherry-pick -x 6e36ff41

echo "Cherry-picking 775933aa: Merge branch 'VOXWAVE-FOUNDRY/main'"
git cherry-pick -x 775933aa

echo "Cherry-picking 9d0012b0: Merge pull request #2321 from danielaskdd/fix-doc-del-slow"
git cherry-pick -x 9d0012b0

echo "Cherry-picking edf48d79: Merge pull request #2319 from danielaskdd/remove-deprecated-code"
git cherry-pick -x edf48d79

echo "Cherry-picking 366a1e0f: Merge pull request #2322 from danielaskdd/fix-delete"
git cherry-pick -x 366a1e0f

echo "Cherry-picking c14f25b7: Add mandatory dimension parameter handling for Jina API compliance"
git cherry-pick -x c14f25b7

echo "Cherry-picking c580874a: Remove depreced sample code"
git cherry-pick -x c580874a

echo "Cherry-picking d5362573: Merge pull request #2327 from huangbhan/patch-1"
git cherry-pick -x d5362573

echo "Cherry-picking d95efcb9: Fix linting"
git cherry-pick -x d95efcb9

echo "Cherry-picking 0f2c0de8: Fix linting"
git cherry-pick -x 0f2c0de8

echo "Cherry-picking 1864b282: Add colored output formatting to migration confirmation display"
git cherry-pick -x 1864b282

echo "Cherry-picking 1334b3d8: Update uv.lock"
git cherry-pick -x 1334b3d8

echo "Cherry-picking 37b71189: Fix table alignment and add validation for empty cleanup selections"
git cherry-pick -x 37b71189

echo "Cherry-picking 8859eaad: Merge pull request #2334 from danielaskdd/hotfix-opena-streaming"
git cherry-pick -x 8859eaad

echo "Cherry-picking a75efb06: Fix: prevent source data corruption by target upsert function"
git cherry-pick -x a75efb06

echo "Cherry-picking 1ffb5338: Update env.example"
git cherry-pick -x 1ffb5338

echo "Cherry-picking ff8f1588: Update env.example"
git cherry-pick -x ff8f1588

echo "Cherry-picking 477c3f54: Merge pull request #2345 from danielaskdd/remove-response-type"
git cherry-pick -x 477c3f54

echo "Cherry-picking 5127bf20: Lacombe Add support for environment variable fallback for API key and default host for cloud models"
git cherry-pick -x 5127bf20

echo "Cherry-picking 69ca3662: Merge pull request #2344 from danielaskdd/fix-josn-serialization-error"
git cherry-pick -x 69ca3662

echo "Cherry-picking 8c07c918: Remove deprecated response_type parameter from query settings"
git cherry-pick -x 8c07c918

echo "Cherry-picking 93a3e471: Remove deprecated response_type parameter from query settings"
git cherry-pick -x 93a3e471

echo "Cherry-picking f7432a26: Lacombe Add support for environment variable fallback for API key and default host for cloud models"
git cherry-pick -x f7432a26

echo "Cherry-picking 28fba19b: Merge pull request #2352 from danielaskdd/docling-gunicorn-multi-worker"
git cherry-pick -x 28fba19b

echo "Cherry-picking 297e4607: Merge branch 'main' into tongda/main"
git cherry-pick -x 297e4607

echo "Cherry-picking 2f2f35b8: Add macOS compatibility check for DOCLING with multi-worker Gunicorn"
git cherry-pick -x 2f2f35b8

echo "Cherry-picking 343d3072: Update env.example"
git cherry-pick -x 343d3072

echo "Cherry-picking 470e2fd1: Merge pull request #2350 from danielaskdd/reduce-dynamic-import"
git cherry-pick -x 470e2fd1

echo "Cherry-picking 4b31942e: refactor: move document deps to api group, remove dynamic imports"
git cherry-pick -x 4b31942e

echo "Cherry-picking 63510478: Improve error handling and logging in cloud model detection"
git cherry-pick -x 63510478

echo "Cherry-picking 67dfd856: Add a better regex"
git cherry-pick -x 67dfd856

echo "Cherry-picking 69a0b74c: refactor: move document deps to api group, remove dynamic imports"
git cherry-pick -x 69a0b74c

echo "Cherry-picking 72f68c2a: Update env.example"
git cherry-pick -x 72f68c2a

echo "Cherry-picking 746c069a: Implement lazy configuration initialization for API server"
git cherry-pick -x 746c069a

echo "Cherry-picking 76adde38: Merge pull request #2351 from danielaskdd/lazy-config-loading"
git cherry-pick -x 76adde38

echo "Cherry-picking 77ad906d: Improve error handling and logging in cloud model detection"
git cherry-pick -x 77ad906d

echo "Cherry-picking 7b7f93d7: Implement lazy configuration initialization for API server"
git cherry-pick -x 7b7f93d7

echo "Cherry-picking 844537e3: Add a better regex"
git cherry-pick -x 844537e3

echo "Cherry-picking 87659744: Merge branch 'tongda/main'"
git cherry-pick -x 87659744

echo "Cherry-picking 89e63aa4: Update edge keywords extraction in graph visualization"
git cherry-pick -x 89e63aa4

echo "Cherry-picking 8abc2ac1: Update edge keywords extraction in graph visualization"
git cherry-pick -x 8abc2ac1

echo "Cherry-picking c164c8f6: Merge branch 'main' of github.com:HKUDS/LightRAG"
git cherry-pick -x c164c8f6

echo "Cherry-picking cc031a3d: Add macOS compatibility check for DOCLING with multi-worker Gunicorn"
git cherry-pick -x cc031a3d

echo "Cherry-picking e6588f91: Update uv.lock"
git cherry-pick -x e6588f91

echo "Cherry-picking fa9206d6: Update uv.lock"
git cherry-pick -x fa9206d6

echo "Cherry-picking 1ccef2b9: Fix null reference errors in graph database error handling"
git cherry-pick -x 1ccef2b9

echo "Cherry-picking 399a23c3: Merge pull request #2356 from danielaskdd/improve-error-handling"
git cherry-pick -x 399a23c3

echo "Cherry-picking 423e4e92: Fix null reference errors in graph database error handling"
git cherry-pick -x 423e4e92

echo "Cherry-picking b88d7854: Merge branch 'HKUDS:main' into main"
git cherry-pick -x b88d7854

echo "Cherry-picking c6850ac5: Merge pull request #2358 from sleeepyin/main"
git cherry-pick -x c6850ac5

echo "Cherry-picking 4343db75: Add macOS fork safety check for Gunicorn multi-worker mode"
git cherry-pick -x 4343db75

echo "Cherry-picking 87221035: Update env.example"
git cherry-pick -x 87221035

echo "Cherry-picking 9a2ddcee: Merge pull request #2360 from danielaskdd/macos-gunicorn-numpy"
git cherry-pick -x 9a2ddcee

echo "Cherry-picking acae404f: Update env.example"
git cherry-pick -x acae404f

echo "Cherry-picking ec05d89c: Add macOS fork safety check for Gunicorn multi-worker mode"
git cherry-pick -x ec05d89c

echo "Cherry-picking 01814bfc: Fix missing function call parentheses in get_all_update_flags_status"
git cherry-pick -x 01814bfc

echo "Cherry-picking 1874cfaf: Fix linting"
git cherry-pick -x 1874cfaf

echo "Cherry-picking 90f52acf: Fix linting"
git cherry-pick -x 90f52acf

echo "Cherry-picking 98e964df: Fix initialization instructions in check_lightrag_setup function"
git cherry-pick -x 98e964df

echo "Cherry-picking c1ec657c: Fix linting"
git cherry-pick -x c1ec657c

echo "Cherry-picking 702cfd29: Fix document deletion concurrency control and validation logic"
git cherry-pick -x 702cfd29

echo "Cherry-picking 9c10c875: Fix linting"
git cherry-pick -x 9c10c875

echo "Cherry-picking fc9f7c70: Fix linting"
git cherry-pick -x fc9f7c70

echo "Cherry-picking b7de694f: Add comprehensive error logging across API routes"
git cherry-pick -x b7de694f

echo "Cherry-picking efbbaaf7: Merge pull request #2383 from danielaskdd/doc-table"
git cherry-pick -x efbbaaf7

echo "Cherry-picking 1d2f534f: Fix linting"
git cherry-pick -x 1d2f534f

echo "Cherry-picking 30e86fa3: Singh use deployment variable which extracted value from .env file or have default value"
git cherry-pick -x 30e86fa3

echo "Cherry-picking 72ece734: Remove obsolete config file and paging design doc"
git cherry-pick -x 72ece734

echo "Cherry-picking cc78e2df: Merge pull request #2395 from Amrit75/issue-2394"
git cherry-pick -x cc78e2df

echo "Cherry-picking ecea9399: Fix lingting"
git cherry-pick -x ecea9399

echo "Cherry-picking 66d6c7dd: Refactor main function to provide sync CLI entry point"
git cherry-pick -x 66d6c7dd

echo "Cherry-picking b46c1523: Fix linting"
git cherry-pick -x b46c1523

echo "Cherry-picking c9e1c86e: Refactor keyword extraction handling to centralize response format logic"
git cherry-pick -x c9e1c86e

echo "Cherry-picking 1b0413ee: Create copilot-setup-steps.yml"
git cherry-pick -x 1b0413ee

echo "Cherry-picking fa6797f2: Update env.example"
git cherry-pick -x fa6797f2

echo "Cherry-picking 16eb0d5b: Merge pull request #2409 from HKUDS/chaohuang-ai-patch-3"
git cherry-pick -x 16eb0d5b

echo "Cherry-picking 6d3bfe46: Merge pull request #2408 from HKUDS/chaohuang-ai-patch-2"
git cherry-pick -x 6d3bfe46

echo "Cherry-picking c233da63: Update copilot-setup-steps.yml"
git cherry-pick -x c233da63

echo "Cherry-picking 5b81ef00: Merge pull request #2410 from netbrah/create-copilot-setup-steps"
git cherry-pick -x 5b81ef00

echo "Cherry-picking 777c9179: Add Langfuse observability configuration to env.example"
git cherry-pick -x 777c9179

echo "Cherry-picking 8994c70f: fix:exception handling order error"
git cherry-pick -x 8994c70f

echo "Cherry-picking d2cd1c07: Merge pull request #2421 from EightyOliveira/fix_catch_order"
git cherry-pick -x d2cd1c07

echo "Cherry-picking 4f12fe12: Change entity extraction logging from warning to info level"
git cherry-pick -x 4f12fe12

echo "Cherry-picking 8eb63d9b: Merge pull request #2434 from cclauss/patch-1"
git cherry-pick -x 8eb63d9b

echo "Cherry-picking 90f341d6: Clauss Fix typos discovered by codespell"
git cherry-pick -x 90f341d6

echo "Cherry-picking d2ab7fb2: Clauss Add Python 3.13 and 3.14 to the testing"
git cherry-pick -x d2ab7fb2

echo "Cherry-picking 0aa77fdb: Merge pull request #2439 from HKUDS/chaohuang-ai-patch-1"
git cherry-pick -x 0aa77fdb

echo "Cherry-picking 607c11c0: Merge pull request #2443 from danielaskdd/fix-ktax"
git cherry-pick -x 607c11c0

echo "Cherry-picking 27805b9a: Merge pull request #2436 from cclauss/patch-2"
git cherry-pick -x 27805b9a

echo "Cherry-picking 2ecf77ef: Update help text to use correct gunicorn command with workers flag"
git cherry-pick -x 2ecf77ef

echo "Cherry-picking 4c775ec5: Merge pull request #2469 from danielaskdd/fix-track-id"
git cherry-pick -x 4c775ec5

echo "Cherry-picking 6fee81f5: Merge pull request #2435 from cclauss/patch-1"
git cherry-pick -x 6fee81f5

echo "Cherry-picking ed22e094: Merge branch 'main' into fix-track-id"
git cherry-pick -x ed22e094

echo "Cherry-picking 4fcae985: Update README.md"
git cherry-pick -x 4fcae985

echo "Cherry-picking a93c1661: Fix list formatting in README installation steps"
git cherry-pick -x a93c1661

echo "Cherry-picking efd50064: docs: improve Docker build documentation with clearer notes"
git cherry-pick -x efd50064

echo "Cherry-picking c18762e3: Simplify Docker deployment documentation and improve clarity"
git cherry-pick -x c18762e3

echo "Cherry-picking 88a45523: Increase default max file paths from 30 to 100 and improve documentation"
git cherry-pick -x 88a45523

echo "Cherry-picking 14a015d4: Restore query generation example and fix README path reference"
git cherry-pick -x 14a015d4

echo "Cherry-picking 1ad0bf82: feat: add RAGAS evaluation framework for RAG quality assessment"
git cherry-pick -x 1ad0bf82

echo "Cherry-picking 026bca00: fix: Use actual retrieved contexts for RAGAS evaluation"
git cherry-pick -x 026bca00

echo "Cherry-picking 0b5e3f9d: Use logger in RAG evaluation and optimize reference content joins"
git cherry-pick -x 0b5e3f9d

echo "Cherry-picking 77db0803: Merge remote-tracking branch 'lightrag-fork/feat/ragas-evaluation' into feat/ragas-evaluation"
git cherry-pick -x 77db0803

echo "Cherry-picking 36694eb9: fix(evaluation): Move import-time validation to runtime and improve documentation"
git cherry-pick -x 36694eb9

echo "Cherry-picking 5da709b4: moussa anouar Merge branch 'main' into feat/ragas-evaluation"
git cherry-pick -x 5da709b4

echo "Cherry-picking ad2d3c2c: Merge remote-tracking branch 'origin/main' into feat/ragas-evaluation"
git cherry-pick -x ad2d3c2c

echo "Cherry-picking debfa0ec: Merge branch 'feat/ragas-evaluation' of https://github.com/anouar-bm/LightRAG into feat/ragas-evaluation"
git cherry-pick -x debfa0ec

echo "Cherry-picking 41c26a36: feat: add command-line args to RAG evaluation script"
git cherry-pick -x 41c26a36

echo "Cherry-picking 4e4b8d7e: Update RAG evaluation metrics to use class instances instead of objects"
git cherry-pick -x 4e4b8d7e

echo "Cherry-picking a618f837: Merge branch 'new/ragas-evaluation'"
git cherry-pick -x a618f837

echo "Cherry-picking c358f405: Update evaluation defaults and expand sample dataset"
git cherry-pick -x c358f405

echo "Cherry-picking d4b8a229: Update RAGAS evaluation to use gpt-4o-mini and improve compatibility"
git cherry-pick -x d4b8a229

echo "Cherry-picking 06b91d00: Improve RAG evaluation progress eval index display with zero padding"
git cherry-pick -x 06b91d00

echo "Cherry-picking 2823f92f: Fix tqdm progress bar conflicts in concurrent RAG evaluation"
git cherry-pick -x 2823f92f

echo "Cherry-picking a73314a4: Refactor evaluation results display and logging format"
git cherry-pick -x a73314a4

echo "Cherry-picking d36be1f4: Improve RAGAS evaluation progress tracking and clean up output handling"
git cherry-pick -x d36be1f4

echo "Cherry-picking f490622b: Doc: Refactor evaluation README to improve clarity and structure"
git cherry-pick -x f490622b

echo "Cherry-picking 831e658e: Update readme"
git cherry-pick -x 831e658e

echo "Cherry-picking c12bc372: Update README"
git cherry-pick -x c12bc372

echo "Cherry-picking 3c85e488: Update README"
git cherry-pick -x 3c85e488

echo "Cherry-picking 37178462: Update README.md"
git cherry-pick -x 37178462

echo "Cherry-picking babbcb56: Update README.md"
git cherry-pick -x babbcb56

echo "Cherry-picking 5c964267: Update README.md"
git cherry-pick -x 5c964267
