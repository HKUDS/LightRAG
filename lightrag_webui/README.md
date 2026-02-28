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

### `Cannot find package '@/lib'`

This error occurred in older versions when the Vite config used a TypeScript path alias (`@/`) that only Bun could resolve at config load time. This has been fixed by using a relative import in `vite.config.ts`.
