from src.search.query import JinaV4QueryQdrant
from src.rerank.maxsim_reranker import MaxSimReranker
from typing import List, Dict


def search_and_rerank(
    query_image_path: str,
    query_text: str,
    top_k_retrieve: int = 10,
    top_k_rerank: int = 5,
    embedding_dir: str = "data/embeddings/patches/vol_01"
) -> List[Dict]:
    # 1. Initial retrieval from Qdrant
    retriever = JinaV4QueryQdrant()
    raw_hits = retriever.search_image(query_image_path, top_k=top_k_retrieve)

    candidates = []
    for hit in raw_hits:
        candidates.append({
            "volume": hit.get("volume"),
            "page_number": hit.get("page_number"),
            "image_path": hit.get("image_path"),
            "score": hit.get("score")
        })

    # 2. MaxSim reranking
    reranker = MaxSimReranker()
    reranked = reranker.rerank(query_text, candidates, embedding_dir=embedding_dir)

    return reranked[:top_k_rerank]


if __name__ == "__main__":
    query_img = "data/images/frieren/01/0081.png" # first search for similar panels
    query_text = "Frieren saw Himmel's statue." # then rerank by relevance to this text

    top_panels = search_and_rerank(
        query_image_path=query_img,
        query_text=query_text,
        top_k_retrieve=10,
        top_k_rerank=3
    )

    for i, panel in enumerate(top_panels, 1):
        print(f"[{i}] Vol {panel['volume']} - Page {panel['page_number']} | MaxSim: {panel['maxsim_score']:.3f}")
        print(f" → {panel['image_path']}\n")
