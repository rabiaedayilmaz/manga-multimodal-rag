import os
import re
import json
import uuid
import argparse
import hashlib
from pathlib import Path
from typing import Iterable, Dict, Any, List

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


# helpers
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def detect_dim(example_file: Path) -> int:
    for obj in iter_jsonl(example_file):
        emb = obj.get("embedding")
        if emb:
            return len(emb)
    raise RuntimeError(f"No 'embedding' found in {example_file}")

def norm_vol(vol) -> str:
    return str(vol).zfill(2)

def norm_page(pg) -> str:
    return str(pg).zfill(4)

def make_point_id(vol: str, page: str, scheme: str = "uuid"):
    key = f"{norm_vol(vol)}_{norm_page(page)}"
    if scheme == "uuid":
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))  # deterministic UUID
    elif scheme == "int":
        h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
        return int(h, 16)  # 64-bit int
    else:
        raise ValueError("id-scheme must be 'uuid' or 'int'")

# qdrant
def ensure_qdrant_collection(client, name: str, dim: int, distance: str = "COSINE"):
    from qdrant_client.models import Distance, VectorParams
    # exists?
    try:
        cols = client.get_collections().collections
        if any(c.name == name for c in cols):
            print(f"Qdrant collection '{name}' exists.")
            return
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=getattr(Distance, distance.upper()))
    )
    print(f"[+] Created Qdrant collection '{name}' (dim={dim}, dist={distance})")

def qdrant_upsert_batch(client, collection: str, batch_points: List[dict]):
    from qdrant_client.models import PointStruct
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in batch_points
        ]
    )

def upload_to_qdrant(files: List[Path], collection: str, batch_size: int, id_scheme: str, distance: str):
    from qdrant_client import QdrantClient

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_key:
        raise SystemExit("set .env for qdrant")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=60)

    dim = detect_dim(files[0])
    ensure_qdrant_collection(client, collection, dim=dim, distance=distance)

    total = 0
    for fp in files:
        print(f"[+] Qdrant loading: {fp}")
        batch = []
        for obj in tqdm(iter_jsonl(fp), desc=f"Upserting {fp.name}"):
            emb = obj.get("embedding")
            if emb is None:
                continue
            vol = norm_vol(obj.get("volume", "00"))
            pg = norm_page(obj.get("page_number", "0000"))
            pid = make_point_id(vol, pg, scheme=id_scheme)
            batch.append({
                "id": pid,
                "vector": np.array(emb, dtype=np.float32),
                "payload": {
                    "manga": obj.get("manga"),
                    "volume": vol,
                    "page_number": pg,
                    "image_path": obj.get("image_path")
                }
            })
            if len(batch) >= batch_size:
                qdrant_upsert_batch(client, collection, batch)
                total += len(batch)
                print(f"    upserted total: {total}")
                batch.clear()
        if batch:
            qdrant_upsert_batch(client, collection, batch)
            total += len(batch)
            print(f"    upserted total: {total}")
    print(f"Qdrant done. Total points: {total}")


# cli to call
def main():
    ap = argparse.ArgumentParser(description="Upload manga panel embeddings to Qdrant or Pinecone.")
    ap.add_argument("--embeddings-glob", type=str, default="data/embeddings/jina_v4/vol_*.jsonl",
                    help="Glob pattern for sharded JSONL files")
    ap.add_argument("--db", type=str, default=os.getenv("VECTOR_DATABASE_TYPE", "qdrant"),
                    choices=["qdrant", "pinecone"], help="Vector DB backend")
    ap.add_argument("--collection", type=str, default="frieren-panels",
                    help="Qdrant collection name")
    ap.add_argument("--index-name", type=str, default=os.getenv("INDEX_NAME", "frieren-panels"),
                    help="Pinecone index name")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--id-scheme", type=str, default="uuid", choices=["uuid", "int"])
    ap.add_argument("--distance", type=str, default="COSINE", choices=["COSINE", "EUCLID", "DOT"],
                    help="Qdrant distance function")
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.embeddings_glob))
    if not files:
        raise SystemExit(f"[!] No files matched: {args.embeddings_glob}")

    if args.db == "qdrant":
        upload_to_qdrant(files, collection=args.collection, batch_size=args.batch_size,
                         id_scheme=args.id_scheme, distance=args.distance)

if __name__ == "__main__":
    main()
