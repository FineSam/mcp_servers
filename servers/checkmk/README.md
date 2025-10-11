# Checkmk MCP Server

This document provides instructions on how to run the Checkmk MCP server.

## Prerequisites

Before running the server, make sure you have completed the installation steps in the root [README.md](../../README.md).

## Running the Server

To run the Checkmk MCP server, first set the required environment variables:

```bash
export CHECKMK_USERNAME="your_username"
export CHECKMK_PASSWORD="your_password"
```

Then, from the root of the repository, run the server:

```bash
uv python servers/checkmk/server.py
```
