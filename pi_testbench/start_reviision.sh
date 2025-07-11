#!/bin/bash
# ReViision Pi Test Bench Startup Script

# Exit on any error
set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Log startup
echo "$(date): Starting ReViision Pi Test Bench..." >> logs/startup.log

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "$(date): Virtual environment activated" >> logs/startup.log
else
    echo "$(date): ERROR - Virtual environment not found!" >> logs/startup.log
    exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "$(date): ERROR - main.py not found!" >> logs/startup.log
    exit 1
fi

# Start the application
echo "$(date): Starting main application..." >> logs/startup.log
exec python main.py 