#!/bin/bash

# Travel Guide External Data - Restore Script
# This script restores the backed up files to a new location

echo "Travel Guide External Data - Restore Script"
echo "=========================================="

# Check if target directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <target_directory>"
    echo "Example: $0 /path/to/restore/location"
    exit 1
fi

TARGET_DIR="$1"
BACKUP_DIR="$(dirname "${BASH_SOURCE[0]}")"

echo "Restoring from: $BACKUP_DIR"
echo "Restoring to: $TARGET_DIR"

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Restoring Python scripts..."
cp "$BACKUP_DIR/python_scripts/"* "$TARGET_DIR/" 2>/dev/null || true

echo "Restoring prompt templates..."
cp "$BACKUP_DIR/prompts/"* "$TARGET_DIR/" 2>/dev/null || true

echo "Restoring data files..."
cp "$BACKUP_DIR/data/"* "$TARGET_DIR/" 2>/dev/null || true

echo "Restoring configuration files..."
cp "$BACKUP_DIR/config/"* "$TARGET_DIR/" 2>/dev/null || true

# Create Old directory and restore old versions
if [ -d "$BACKUP_DIR/old_versions" ] && [ "$(ls -A "$BACKUP_DIR/old_versions" 2>/dev/null)" ]; then
    echo "Restoring old versions..."
    mkdir -p "$TARGET_DIR/Old"
    cp "$BACKUP_DIR/old_versions/"* "$TARGET_DIR/Old/" 2>/dev/null || true
fi

# Restore ChromaDB if it exists
if [ -d "$BACKUP_DIR/chromadb_db" ] && [ "$(ls -A "$BACKUP_DIR/chromadb_db" 2>/dev/null)" ]; then
    echo "Restoring ChromaDB database..."
    mkdir -p "$TARGET_DIR/chromadb_db"
    cp -r "$BACKUP_DIR/chromadb_db/"* "$TARGET_DIR/chromadb_db/" 2>/dev/null || true
fi

echo ""
echo "Restore completed successfully!"
echo "Project restored to: $TARGET_DIR"
echo ""
echo "Next steps:"
echo "1. Navigate to the restored directory: cd '$TARGET_DIR'"
echo "2. Create a virtual environment: python -m venv venv_excel"
echo "3. Activate the virtual environment: source venv_excel/bin/activate"
echo "4. Install dependencies: pip install -r requirements.txt (if available)"
echo "5. Configure your .env file with your API keys"
echo "6. Run the application: python itinerary_planner.py --help"
