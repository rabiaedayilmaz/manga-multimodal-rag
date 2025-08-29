import os
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.embed.jinav4_embedder import JinaEmbedder

load_dotenv()


class JinaV4QueryQdrant:

    def __init__(
        self,
        collection_name: str = "frieren-panels",
        truncate_dim: int = 1024,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        device: Optional[str] = None,
        max_side: int = 1536
    ):
        self.collection = collection_name
        self.truncate_dim = truncate_dim

        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise SystemExit("set .env for qdrant")

        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
        self.embedder = JinaEmbedder(truncate_dim=self.truncate_dim, max_side=max_side, device=device)

    def _embed_query_image(self, image_path: str) -> np.ndarray:
        vec = self.embedder.embed_image(Path(image_path))
        return vec.astype(np.float32)

    def _embed_query_text(self, text: str) -> np.ndarray:
        vec = self.embedder.embed_text(text)  
        return vec.astype(np.float32)

    def _qdrant_search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        must_volume: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        q_filter = None
        if must_volume:
            q_filter = Filter(
                must=[FieldCondition(key="volume", match=MatchValue(value=str(must_volume).zfill(2)))]
            )

        hits = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k,
            query_filter=q_filter,
        )
        results: List[Dict[str, Any]] = []
        for h in hits:
            p = h.payload or {}
            results.append({
                "score": float(h.score),
                "manga": p.get("manga"),
                "volume": p.get("volume"),
                "page_number": p.get("page_number"),
                "image_path": p.get("image_path"),
            })
        return results

    def search_image(self, image_path: str, top_k: int = 5, must_volume: Optional[str] = None):
        q = self._embed_query_image(image_path)
        return self._qdrant_search(q, top_k=top_k, must_volume=must_volume)

    def search_text(self, text: str, top_k: int = 5, must_volume: Optional[str] = None):
        q = self._embed_query_text(text)
        return self._qdrant_search(q, top_k=top_k, must_volume=must_volume)


def _print_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("[!] No results.")
        return
    for i, r in enumerate(results, 1):
        vol = r.get("volume", "??")
        pg = r.get("page_number", "????")
        print(f"[{i}] score={r['score']:.4f} | Vol {vol} Pg {pg}")
        print(f"    {r.get('image_path')}")


def main():
    ap = argparse.ArgumentParser(description="Search panels in Qdrant using Jina v4 embedder")
    ap.add_argument("--collection", type=str, default="frieren-panels")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--truncate-dim", type=int, default=1024)
    ap.add_argument("--volume", type=str, default=None, help="Optional filter: volume like '01'")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Query image path")
    g.add_argument("--text", type=str, help="Query text for text->image search")

    args = ap.parse_args()

    query = JinaV4QueryQdrant(collection_name=args.collection, truncate_dim=args.truncate_dim)

    if args.text:
        results = query.search_text(args.text, top_k=args.top_k, must_volume=args.volume)
    else:
        results = query.search_image(args.image, top_k=args.top_k, must_volume=args.volume)

    _print_results(results)


if __name__ == "__main__":
    main()
