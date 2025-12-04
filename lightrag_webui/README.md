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

    Run the following command to build the project:

    ```bash
    bun run build
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
