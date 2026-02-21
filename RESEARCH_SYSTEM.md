# Research-Grade Evaluation & Safety System

This system implements a comprehensive wrapper around the SDLC Agent to provide metrics, safety, and evaluation capabilities without modifying the core logic.

## Components

### 1. Research Wrapper (`backend/research/wrapper.py`)
Wraps the `SDLCBuilder` to intercept all execution.
- **Metrics Logging**: Automatically captures task success/failure, execution time, token usage, and errors.
- **Protection**: Enforces daily token limits and caches responses to save costs.
- **Safety**: Catches all exceptions to prevent system crashes.

### 2. Metrics Logger (`backend/research/metrics.py`)
- Logs structured data to `research_logs/metrics.json` in each run directory.
- Captures 20+ data points including failure categories, project size, and static analysis stats.

### 3. Evaluation Engine (`backend/research/evaluate.py`)
- Runs benchmark tasks (Calculator, Todo App) automatically.
- Compares Agent vs Baseline (Single-shot Mistral).
- Generates statistical significance reports (t-test, confidence intervals).

### 4. Graph Generator (`backend/research/graphs.py`)
- Generates high-resolution plots for Success Rate and Execution Time.
- Saves images to `evaluation_results/`.

## Usage

### Run Benchmark
```bash
python backend/research/evaluate.py
```
Outputs:
- `evaluation_results/statistics.json`
- `evaluation_results/benchmark_summary.json`
- Graphs (`.png`)

### Normal Operation
The system is enabled by default in `backend/main.py`. Every user task is automatically logged and protected.

### Token Protection
- Daily Limit: 1M tokens (configurable in `protection.py`)
- Caches identical requests to `cache/` directory.

## Fixes Included
- **Open Output Button**: Fixed `open_folder` endpoint to support Windows/macOS/Linux robustly.
