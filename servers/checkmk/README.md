# Checkmk MCP Server

This document provides instructions on how to run the Checkmk MCP server.

## Prerequisites

Before running the server, make sure you have completed the installation steps in the root [README.md](../../README.md).

## Configuration

1.  **Create the `settings.yaml` file:**

    In the `servers/checkmk` directory, create a `settings.yaml` file with the following content:

    ```yaml
    checkmk_url: http://localhost:8080
    checkmk_username: your_username
    checkmk_token: ENCRYPTED_TOKEN_PLACEHOLDER
    ```

2.  **Generate the encryption key and encrypt the token:**

    Run the `encrypt_token.py` script to generate an encryption key and encrypt your Checkmk token. The script will save the key to `key.key` and print the encrypted token.

    ```bash
    uv python servers/checkmk/encrypt_token.py
    ```

3.  **Update `settings.yaml`:**

    Replace `ENCRYPTED_TOKEN_PLACEHOLDER` in `settings.yaml` with the encrypted token you obtained in the previous step.

## Running the Server

Once you have completed the configuration, you can run the server from the root of the repository:

```bash
uv python servers/checkmk/checkmk_mcp.py
```

## Running the Client

To run the client, first make sure you have an Ollama server running.
Then, from the root of the repository, run the client. Here is an example command:

```bash
uv python servers/checkmk/checkmk_mcp_client.py \
    --mcp-url "http://localhost:8000" \
    --ollama-model "gemma:2b"
```

### Arguments

| Argument       | Description                               | Default                |
| -------------- | ----------------------------------------- | ---------------------- |
| `--mcp-url`      | The URL of the Checkmk MCP server.        | `http://localhost:8000` |
| `--ollama-model` | The name of the Ollama model to use.      | `gemma:2b`             |