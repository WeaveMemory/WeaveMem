# WeaveMem

## Setup

Use Python 3.12 and run commands from this directory.

```bash
uv sync --locked --extra dev
uv run python -m nltk.downloader punkt punkt_tab
cp .env.example .env
```

Fill in the API credentials and endpoints in `.env` before running model calls.

## Build a memory graph

Build from the included [example conversation](examples/sample_conversation.json):

```bash
export WEAVE_MEM_DATA_DIR="$PWD/runs/example-memory"

uv run python src/cli.py build \
  --input examples/sample_conversation.json
```

Inspect the generated nodes and edges:

```bash
uv run python -m json.tool "$WEAVE_MEM_DATA_DIR/mid_memories.json"
uv run python -m json.tool "$WEAVE_MEM_DATA_DIR/long_relations.json"
uv run python -m json.tool "$WEAVE_MEM_DATA_DIR/mid_relations.json"
```

Query the example graph:

```bash
uv run python src/cli.py search \
  --conversation conv-1 \
  --question "Where does Alice work now?"
```

## Run benchmarks

### LoCoMo

```bash
uv run python -m benchmarks.run locomo all \
  --input data/locomo/locomo10.json --output runs/locomo
```

### LongMemEval

```bash
uv run python -m benchmarks.run longmemeval all \
  --input data/longmemeval/longmemeval_s_cleaned.json --output runs/longmemeval
```

### PersonaMem

```bash
uv run python -m benchmarks.run personamem all \
  --input data/personamem --output runs/personamem
```

### BEAM

```bash
uv run python -m benchmarks.run beam all \
  --input data/beam --output runs/beam
```

### Run individual stages

```bash
uv run python -m benchmarks.run locomo prepare \
  --input data/locomo/locomo10.json --output runs/locomo-staged
uv run python -m benchmarks.run locomo build --output runs/locomo-staged
uv run python -m benchmarks.run locomo retrieve --output runs/locomo-staged
uv run python -m benchmarks.run locomo answer --output runs/locomo-staged
uv run python -m benchmarks.run locomo score --output runs/locomo-staged
uv run python -m benchmarks.run locomo summarize --output runs/locomo-staged
```

Repeat a stage with the same output directory to resume it. Use a new output
directory after changing inputs or configuration. Select a configuration with
`--config benchmarks/configs/<benchmark>.json`.

