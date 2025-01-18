#!/bin/bash

# Default log directory
LOG_DIR="lightning_logs"

# Function to display usage
usage() {
    echo "Usage: $0 [-l log_dir]"
    echo "Options:"
    echo "  -l         - Path to log directory (default: lightning_logs)"
    exit 1
}

# Parse options
while getopts "l:h" opt; do
    case $opt in
        l) LOG_DIR="$OPTARG" ;;
        h) usage ;;
        \?) usage ;;
    esac
    done

# Shift off the options and optional --
shift $((OPTIND-1))

# Start TensorBoard
echo "Starting TensorBoard with log directory: $LOG_DIR"
tensorboard --logdir="$LOG_DIR" --bind_all