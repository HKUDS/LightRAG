# LightRAG WebUI

LightRAG WebUI is a React-based web interface for interacting with the LightRAG system. It provides a user-friendly interface for querying, managing, and exploring LightRAG's functionalities.

## Installation

1.  **Install Bun:**

    If you haven't already installed Bun, follow the official documentation: [https://bun.sh/docs/installation](https://bun.sh/docs/installation)

2.  **Install Dependencies:**

    In the `lightrag_webui` directory, run the following command to install project dependencies:

    ```bash
    bun install --frozen-lockfile
    ```

3.  **Build the Project:**

    Before building, you need to adjust the constant `backendBaseUrl` in `lightrag_webui/src/lib/constants.ts`. For example:
    ```
    export const backendBaseUrl = 'http://127.0.0.1:9621'
    ```
    You must replace `http://127.0.0.1:9621` to the LightRAG API service you want to access.

    Then, run the following command to build the project:

    ```bash
    bun run build --emptyOutDir
    ```

    This command will bundle the project and output the built files to the `lightrag/api/webui` directory.

## Development

- **Start the Development Server:**

  If you want to run the WebUI in development mode, use the following command:

  ```bash
  bun run dev
  ```

## Script Commands

The following are some commonly used script commands defined in `package.json`:

- `bun install`: Installs project dependencies.
- `bun run dev`: Starts the development server.
- `bun run build`: Builds the project.
- `bun run lint`: Runs the linter.
