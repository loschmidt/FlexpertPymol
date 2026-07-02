echo "Downloading Flexpert weights..."

# Create the directory if it doesn't exist
mkdir -p models/weights

# Set file information for Flexpert weights
WEIGHTS_URL_3d="https://data.ciirc.cvut.cz/public/projects/2025Flexpert/flexpert-weights/flexpert_3d_weights.bin"
OUTPUT_FILE_3d="models/weights/flexpert_3d_weights.bin"

WEIGHTS_URL_SEQ="https://data.ciirc.cvut.cz/public/projects/2025Flexpert/flexpert-weights/flexpert_seq_weights.bin"
OUTPUT_FILE_SEQ="models/weights/flexpert_seq_weights.bin"

echo "Downloading Flexpert-3D weights..."
wget --no-check-certificate "${WEIGHTS_URL_3d}" -O ${OUTPUT_FILE_3d}

echo "Downloading Flexpert-Seq weights..."
wget --no-check-certificate "${WEIGHTS_URL_SEQ}" -O ${OUTPUT_FILE_SEQ}

echo "Flexpert weights download completed."
