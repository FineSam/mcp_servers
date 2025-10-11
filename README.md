# MCP Servers and Fine-Tuning

This repository contains a collection of MCP (Model Context Protocol) servers and the scripts to fine-tune models for them.

## Directory Structure

```
/
├── servers/
│   └── checkmk/
│       └── server.py
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
    - `server.py`: The main server script.
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

### Running the MCP Server

To run the Checkmk MCP server, first set the required environment variables:

```bash
export CHECKMK_USERNAME="your_username"
export CHECKMK_PASSWORD="your_password"
```

Then, run the server:

```bash
uv python servers/checkmk/server.py
```

### Fine-Tuning a Model

To fine-tune a model, run the `fine_tune.py` script. You can specify the dataset path using the `--dataset_path` argument.

```bash
uv python training/checkmk/fine_tune.py --dataset_path /path/to/your/dataset.json
```

If no dataset path is provided, it will use the default path: `training/checkmk/data/dataset.json`.

The script is optimized for running on an NVIDIA A100 GPU.

## Models

The fine-tuning script is configured to use the `google/gemma-2-9b-it` model.
