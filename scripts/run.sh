#!/bin/bash

# Make script exit on first error
set -e

# Default config path
CONFIG_PATH="config/config.yaml"

# Function to display usage
usage() {
    echo "Usage: $0 [-c config_path] <command>"
    echo "Commands:"
    echo "  all        - Run entire pipeline (prepare, train, evaluate)"
    echo "  prepare    - Prepare data only"
    echo "  train      - Train model only"
    echo "  evaluate   - Evaluate model only"
    echo "Options:"
    echo "  -c         - Path to config file (default: config/config.yaml)"
    exit 1
}

# Parse options
while getopts "c:h" opt; do
    case $opt in
        c) CONFIG_PATH="$OPTARG" ;;
        h) usage ;;
        \?) usage ;;
    esac
done

# Shift off the options and optional --
shift $((OPTIND-1))

# Get the command
COMMAND=$1

if [ -z "$COMMAND" ]; then
    usage
fi

# Ensure we're in the project root
cd "$(dirname "$0")/.."

case $COMMAND in
    "prepare")
        echo "Preparing data..."
        python -m src.cli prepare_data --config-path "$CONFIG_PATH"
        ;;
    "train")
        echo "Training model..."
        python -m src.cli run_train --config-path "$CONFIG_PATH"
        ;;
    "evaluate")
        echo "Evaluating model..."
        python -m src.cli run_evaluate --config-path "$CONFIG_PATH"
        ;;
    "all")
        echo "Running entire pipeline..."
        python -m src.cli prepare_data --config-path "$CONFIG_PATH"
        python -m src.cli run_train --config-path "$CONFIG_PATH"
        python -m src.cli run_evaluate --config-path "$CONFIG_PATH"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        usage
        ;;
esac
