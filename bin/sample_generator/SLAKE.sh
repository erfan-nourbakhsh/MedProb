#!/bin/bash

FILE_ID="1EZ0WpO5Z6BJUqC3iPBQJJS1INWSMsh7U"
DEST_DIR="Data/SLAKE"
OUTPUT_FILE="${DEST_DIR}/slake.zip"

echo "========================================"
echo "  Google Drive File Downloader - SLAKE"
echo "========================================"

echo "[*] Creating directory: $DEST_DIR"
mkdir -p "$DEST_DIR" || { echo "[✗] Failed to create $DEST_DIR"; exit 1; }

if command -v gdown &>/dev/null; then
    echo "[*] gdown found. Downloading..."
    gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUTPUT_FILE"
    echo "[✓] Download complete! File saved to: $OUTPUT_FILE"

    ZIP_FILE="$DEST_DIR/slake.zip"
    if [ -f "$ZIP_FILE" ]; then
        echo "[*] Unzipping $ZIP_FILE..."
        unzip -q "$ZIP_FILE" -d "$DEST_DIR"
        echo "[✓] Extraction complete!"
        rm -f "$ZIP_FILE"
        echo "[✓] Cleaned up: $ZIP_FILE removed."
    else
        echo "[!] Zip file not found at $ZIP_FILE — skipping extraction."
    fi

    python src/utils/preprocessing/slake.py
    echo "[✓] Preprocessing complete!"
    echo "[✓] Dataset statistics saved to: $DEST_DIR/dataset_statistics.txt"
    exit 0
fi

if command -v pip3 &>/dev/null || command -v pip &>/dev/null; then
    echo "[*] Installing gdown..."
    pip install -q gdown 2>/dev/null || pip3 install -q gdown 2>/dev/null

    if command -v gdown &>/dev/null; then
        echo "[*] Downloading file with gdown..."
        gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUTPUT_FILE"
        echo "[✓] Download complete! File saved to: $OUTPUT_FILE"

        ZIP_FILE="$DEST_DIR/slake.zip"
        if [ -f "$ZIP_FILE" ]; then
            echo "[*] Unzipping $ZIP_FILE..."
            unzip -q "$ZIP_FILE" -d "$DEST_DIR"
            echo "[✓] Extraction complete!"
            rm -f "$ZIP_FILE"
            echo "[✓] Cleaned up: $ZIP_FILE removed."
        else
            echo "[!] Zip file not found at $ZIP_FILE — skipping extraction."
        fi
        exit 0
    fi
fi

if command -v curl &>/dev/null; then
    echo "[*] Using curl to download..."
    CONFIRM=$(curl -sc /tmp/gcookie \
        "https://drive.google.com/uc?export=download&id=${FILE_ID}" | \
        grep -o 'confirm=[^&"]*' | head -1 | sed 's/confirm=//')

    if [ -n "$CONFIRM" ]; then
        echo "[*] Large file detected, using confirmation token..."
        curl -Lb /tmp/gcookie \
            "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
            -o "$OUTPUT_FILE"
    else
        curl -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
            -o "$OUTPUT_FILE"
    fi
    echo "[✓] Download complete! File saved to: $OUTPUT_FILE"
    exit 0
fi

if command -v wget &>/dev/null; then
    echo "[*] Using wget to download..."
    wget --no-check-certificate \
        "https://drive.google.com/uc?export=download&id=${FILE_ID}" \
        -O "$OUTPUT_FILE"
    echo "[✓] Download complete! File saved to: $OUTPUT_FILE"
    exit 0
fi

echo "[✗] Error: No suitable download tool found."
echo "    Please install one of: gdown, curl, or wget"
exit 1