#!/bin/bash
echo "Downloading CATH data..."

# Create data directory if it doesn't exist
mkdir -p ../data/

# Set file information
URL="https://data.ciirc.cvut.cz/public/projects/2025Flexpert/cath4.3/"
OUTPUT_DIR="../data/cath4.3"

# Download directory recursively
echo "Downloading CATH data..."
wget --no-check-certificate -r -np -nH --cut-dirs=3 --reject "index.html*" \
     --directory-prefix=${OUTPUT_DIR} ${URL}

echo "CATH data download completed."

