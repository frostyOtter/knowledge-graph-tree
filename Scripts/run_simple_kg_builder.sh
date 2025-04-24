#!/bin/bash

# --- Configuration ---
# Default input path relative to the project root if no argument is provided
DEFAULT_INPUT_PATH="data"

# Path to the Python script relative to the project root
PYTHON_SCRIPT="src/build_graph/simple_kg_builder.py"
# --- End Configuration ---

python "$PYTHON_SCRIPT" "$INPUT_PATH"

