# MCP Servers and Fine-Tuning

This repository contains a collection of MCP (Multi-purpose Co-pilot) servers and the scripts to fine-tune models for them.

## Directory Structure

```
/
├── servers/
│   └── checkmk/
│       ├── server.py
│       └── data/
│           └── dataset.json
└── training/
    ├── fine_tune.py
    └── adapters/
        └── checkmk-lora-adapter/
```

- **servers/**: Contains the MCP servers.
  - **checkmk/**: An MCP server for interacting with a Checkmk monitoring instance.
    - `server.py`: The main server script.
    - `data/dataset.json`: The dataset used for fine-tuning the model for the Checkmk server.
- **training/**: Contains the scripts and resources for fine-tuning models.
  - `fine_tune.py`: The script to fine-tune a model.
  - `adapters/`: The output directory for the trained model adapters.

## Installation

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
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
python3 servers/checkmk/server.py
```

### Fine-Tuning a Model

To fine-tune a model, run the `fine_tune.py` script. You can specify the dataset path using the `--dataset_path` argument.

```bash
python3 training/fine_tune.py --dataset_path /path/to/your/dataset.json
```

If no dataset path is provided, it will use the default path: `servers/checkmk/data/dataset.json`.

The script is optimized for running on an NVIDIA A100 GPU.

## Models

The fine-tuning script is configured to use the `google/gemma-2-9b-it` model.
