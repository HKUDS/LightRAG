# LightRAG WebUI

LightRAG WebUI is a React-based web interface for interacting with the LightRAG system. It provides a user-friendly interface for querying, managing, and exploring LightRAG's functionalities.

## Installation

### Using Bun (recommended)

1. **Install Bun:**

    If you haven't already installed Bun, follow the official documentation: [https://bun.sh/docs/installation](https://bun.sh/docs/installation)

2. **Install Dependencies:**

    In the `lightrag_webui` directory, run the following command to install project dependencies:

    ```bash
    bun install --frozen-lockfile
    ```

3. **Build the Project:**

    Run the following command to build the project:

    ```bash
    bun run build
    ```

    This command will bundle the project and output the built files to the `lightrag/api/webui` directory.

### Using Node.js / npm (alternative)

If Bun is unavailable or the Bun build fails in your environment (e.g., older Linux distributions, restricted environments, or Bun version incompatibilities), you can use Node.js instead:

```bash
npm install
npm run build
```

> **Note:** Tests (`bun test`) still require Bun. All other scripts (`dev`, `build`, `preview`, `lint`) work with both Bun and Node.js/npm.

## Development

- **Start the Development Server:**

  ```bash
  # With Bun
  bun run dev

  # With Node.js/npm
  npm run dev
  ```

## Script Commands

The following are some commonly used script commands defined in `package.json`:

| Command | Description |
|---------|-------------|
| `bun run dev` / `npm run dev` | Starts the development server |
| `bun run build` / `npm run build` | Builds the project for production |
| `bun run lint` / `npm run lint` | Runs the linter |
| `bun run preview` / `npm run preview` | Previews the production build |
| `bun run build:bun` | Builds using Bun runtime explicitly |
| `bun test` | Runs tests (Bun only) |

## Troubleshooting

### `bun run build` fails silently or with exit code 1

This can happen due to Bun version incompatibilities or restricted environments. Try:

```bash
npm install
npm run build
```

### `could not open bin metadata file` / `corrupted node_modules directory` (WSL)

```
error: could not open bin metadata file

Bun failed to remap this bin to its proper location within node_modules.
This is an indication of a corrupted node_modules directory.
```

This is a [known Bun issue](https://github.com/oven-sh/bun/issues) that surfaces on WSL. Bun installs package binaries by remapping them into `node_modules/.bin`, and that step fails when `node_modules` lives on a Windows-mounted drive (a path under `/mnt/c`, `/mnt/d`, etc.). WSL exposes those drives through the `drvfs`/`9p` filesystem, which does not support the link/metadata operations Bun relies on, so the `.bin` entries end up corrupted even right after a successful `bun install`.

Fixes, in order of preference:

1. **Move the project into the Linux filesystem.** Clone/copy LightRAG somewhere under your WSL home (e.g. `~/LightRAG`) instead of `/mnt/c/...`, then reinstall:

    ```bash
    rm -rf node_modules
    bun install --frozen-lockfile
    bun run build
    ```

    This is the recommended fix — building from a Windows-mounted path is slow and fragile regardless of this specific error.

2. **If you must stay on the mounted drive, use Node.js/npm instead of Bun** (see *Using Node.js / npm* above):

    ```bash
    rm -rf node_modules
    npm install
    npm run build
    ```

3. **Make sure Bun is up to date** (`bun upgrade`) — older Bun releases hit this remapping bug more often.

### `Cannot find package '@/lib'`

This error occurred in older versions when the Vite config used a TypeScript path alias (`@/`) that only Bun could resolve at config load time. This has been fixed by using a relative import in `vite.config.ts`.
