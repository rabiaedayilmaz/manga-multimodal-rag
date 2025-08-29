import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pathlib import Path


def convert_jsonl_to_npy(jsonl_path, output_dir):
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Converting {jsonl_path.name}"):
            entry = json.loads(line)
            vol = str(entry["volume"]).zfill(2)
            page = str(entry["page_number"]).zfill(4)
            embedding = np.array(entry["embedding"], dtype=np.float32)

            out_path = output_dir / f"{vol}_{page}.npy"
            np.save(out_path, embedding)

if __name__ == "__main__":
    convert_jsonl_to_npy(
        jsonl_path="data/embeddings/jina_v4/vol_01.jsonl",
        output_dir="data/embeddings/jina_v4/reranker_patches"
    )
