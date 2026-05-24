#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="89kps"
BASE_API="https://api.osf.io/v2"
OUTPUT_DIR="Data/VQA-RAD"

for cmd in curl jq; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' is required but not installed. Please install it and retry."
    exit 1
  fi
done

mkdir -p "$OUTPUT_DIR"
echo "Files will be saved to: $(pwd)/$OUTPUT_DIR"
echo ""

download_files() {
  local files_url="$1"
  local local_dir="$2"

  mkdir -p "$local_dir"

  while [[ -n "$files_url" ]]; do
    response=$(curl -fsSL \
      -H "Accept: application/vnd.api+json" \
      "$files_url")

    local count
    count=$(echo "$response" | jq '.data | length')

    for (( i=0; i<count; i++ )); do
      local kind name
      kind=$(echo "$response" | jq -r ".data[$i].attributes.kind")
      name=$(echo "$response" | jq -r ".data[$i].attributes.name")

      if [[ "$kind" == "folder" ]]; then

        local folder_url
        folder_url=$(echo "$response" | jq -r ".data[$i].relationships.files.links.related.href")
        echo "[FOLDER] $local_dir/$name"
        download_files "$folder_url" "$local_dir/$name"

      elif [[ "$kind" == "file" ]]; then
        local download_url
        download_url=$(echo "$response" | jq -r ".data[$i].links.download")
        local dest="$local_dir/$name"

        if [[ -f "$dest" ]]; then
          echo "[SKIP]   $dest (already exists)"
        else
          echo "[DL]     $dest"
          curl -fsSL --retry 3 --retry-delay 2 \
            -o "$dest" \
            "$download_url"
        fi
      fi
    done

    files_url=$(echo "$response" | jq -r '.links.next // empty')
  done
}

echo "Fetching storage providers for project $PROJECT_ID ..."
providers_url="${BASE_API}/nodes/${PROJECT_ID}/files/"
providers_resp=$(curl -fsSL -H "Accept: application/vnd.api+json" "$providers_url")
provider_count=$(echo "$providers_resp" | jq '.data | length')

echo "Found $provider_count provider(s)."
echo ""

for (( p=0; p<provider_count; p++ )); do
  provider_name=$(echo "$providers_resp" | jq -r ".data[$p].attributes.provider")
  provider_files_url=$(echo "$providers_resp" | jq -r ".data[$p].relationships.files.links.related.href")

  echo "=== Provider: $provider_name ==="
  provider_dir="$OUTPUT_DIR/$provider_name"
  download_files "$provider_files_url" "$provider_dir"
  echo ""
done

echo "Done! All files saved to: $(pwd)/$OUTPUT_DIR"

python src/utils/preprocessing/vqa_rad.py