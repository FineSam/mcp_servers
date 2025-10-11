# MCP Servers and Fine-Tuning

This repository contains a collection of MCP (Model Context Protocol) servers and the scripts to fine-tune models for them.

## Directory Structure

```
/
├── servers/
│   └── checkmk/
│       ├── checkmk_mcp_server.py
│       ├── checkmk_mcp_client.py
│       ├── encrypt_token.py
│       ├── settings.yaml
│       └── key.key
└── training/
    └── checkmk/
        ├── fine_tune.py
        ├── data/
        │   └── dataset.json
        └── adapters/
            └── checkmk-lora-adapter/
```

- **servers/**: Contains the MCP servers.
  - **checkmk/**: An MCP server for interacting with a Checkmk monitoring instance.
    - `checkmk_mcp_server.py`: The main server script.
    - `checkmk_mcp_client.py`: A client to interact with the Checkmk MCP server and an LLM.
    - `encrypt_token.py`: A script to generate an encryption key and encrypt the Checkmk token.
    - `settings.yaml`: Configuration file for the Checkmk server.
    - `key.key`: The encryption key (should be in .gitignore).
- **training/**: Contains the scripts and resources for fine-tuning models.
  - **checkmk/**: Contains the fine-tuning script, data, and adapters for the Checkmk server.
    - `fine_tune.py`: The script to fine-tune a model.
    - `data/dataset.json`: The dataset used for fine-tuning the model.
    - `adapters/`: The output directory for the trained model adapters.

## Installation

### 1. Install uv

`uv` is a fast Python package installer and resolver, written in Rust. To install it on Linux, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create a Virtual Environment

Create and activate a virtual environment with Python 3.11 using `uv`:

```bash
uv venv -p 3.11
source .venv/bin/activate
```

### 3. Install Dependencies

To install the dependencies, run the following command:

```bash
uv pip install -r requirements.txt
```

## Usage

For information on how to use the different components of this repository, please refer to the `README.md` files in the respective directories:

- [servers/checkmk](servers/checkmk)
- [training/checkmk](training/checkmk)

