#!/bin/bash

cd /mnt/h_h_public/lh/lz/Rare_rag/pipelines

JSONL_DIR="$1"
START_IDX="$2"
END_IDX="$3"

if [[ $# -lt 3 ]]; then
    echo "用法: $0 <jsonl_dir> <start_idx> <end_idx>"
    exit 1
fi

OUTPUT_FILE="embeddings_${START_IDX}_${END_IDX}.jsonl"
CONFIG_TMP="/mnt/h_h_public/lh/lz/Rare_rag/configs/text_rag_demo.yaml"

python3 /mnt/h_h_public/lh/lz/Rare_rag/pipelines/text_rag_wo_myscale_for_4096datasets.py "$CONFIG_TMP" "$OUTPUT_FILE" "$START_IDX" "$END_IDX" "$JSONL_DIR"

echo "Done! 结果已保存到 $OUTPUT_FILE"