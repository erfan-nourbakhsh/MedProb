#!/bin/bash

FOLDER_ID="1G2C2_FUCyYQKCkSeCRRiTTsLDvOAjFj5"
DEST_DIR="Data/PATH-VQA"

echo "========================================"
echo "  Google Drive Folder Downloader"
echo "========================================"

mkdir -p "$DEST_DIR"

if command -v pip3 &>/dev/null || command -v pip &>/dev/null; then
    echo "[*] Installing gdown..."
    pip install -q gdown 2>/dev/null || pip3 install -q gdown 2>/dev/null

    if command -v gdown &>/dev/null; then
        echo "[*] Downloading folder contents with gdown..."
        gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$DEST_DIR"
        echo "[✓] Download complete! Files saved to: $DEST_DIR"

        ZIP_FILE="$DEST_DIR/pvqa.zip"
        if [ -f "$ZIP_FILE" ]; then
            echo "[*] Unzipping $ZIP_FILE..."
            unzip -q "$ZIP_FILE" -d "$DEST_DIR"
            echo "[✓] Extraction complete!"
            rm -f "$ZIP_FILE"
            echo "[✓] Cleaned up: $ZIP_FILE removed."
        else
            echo "[!] Zip file not found at $ZIP_FILE — skipping extraction."
        fi

        python src/utils/preprocessing/path_vqa.py
        echo "[✓] Preprocessing complete!"
        echo "[✓] Dataset statistics saved to: $DEST_DIR/dataset_statistics.txt"
        exit 0
    fi
fi

if command -v rclone &>/dev/null; then
    echo "[*] rclone detected."
    echo "[!] Make sure you have configured a Google Drive remote named 'gdrive'."
    echo "    Run: rclone config  (if not done already)"
    read -rp "Enter your rclone remote name [default: gdrive]: " REMOTE
    REMOTE="${REMOTE:-gdrive}"
    echo "[*] Downloading folder with rclone..."
    rclone copy "${REMOTE}:" "$DEST_DIR" --drive-root-folder-id "$FOLDER_ID" -P
    echo "[✓] Download complete! Files saved to: $DEST_DIR"
    exit 0
fi

echo "[!] Neither gdown nor rclone found. Attempting curl fallback..."
echo "    Note: This only works for single public files, not entire folders."
FILE_URL="https://drive.google.com/uc?export=download&id=${FOLDER_ID}"
curl -L -o "${DEST_DIR}/downloaded_file" "$FILE_URL"
echo "[✓] Done. Check: $DEST_DIR"
