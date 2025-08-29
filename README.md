# Multimodal Panel Retrieval and Explanation over Manga PDFs - Frieren

## Features

* PDF-to-Image Conversion: Manga volumes are split into page-level image panels with metadata.

* Image/Text Retrieval: Qdrant is used for similarity search using jinaai/jina-embeddings-v4.

* MaxSim Reranking: Improves accuracy by reranking top-k results based on token-wise similarity (ColBERT, after patches with clip).

* Visual QA via VLM: Ask natural language questions to selected panels, powered by MiniCPM-V-4.

* Streamlit UI: Interactive app for uploading queries, viewing search results, and generating explanations.

---

## Installation

```bash
git clone https://github.com/<your-username>/manga-rag.git
cd manga-rag
pip install -r requirements.txt
```

Connect Qdrant and add QDRANT_API_KEY and QDRANT_URL to .env file.


## Data Processing and Vectorizing

1. Convert manga PDFs to Images
```bash
python src/extract/pdf_to_images.py
```

2. Embed Images with jinaai/jina-embeddings-v4
```bash
python src/embed/jinav4_embedder.py
```

3. Build Vector Index in Qdrant
```bash
python scripts/build_index.py --embeddings-glob "data/embeddings/jina_v4/vol_01.jsonl"
```

### Panel Search

* Simple Retrieval
```bash
python src/search/query.py
python src/search/query.py --image data/images/frieren/01/0033.png --top-k 5 --truncate-dim 1024
python src/search/query.py --image data/images/frieren/01/0033.png --volume 01

python src/search/query.py \
  --truncate-dim 1024 \
  --top-k 5 \
  --text "Frieren finds Himmel's statue."
```

* With MaxSim Reranker (ColBERT)
```bash
python src/search/query_with_rerank.py
```

* Question-Answering
```bash
python3 src/generate/vlm_response.py \
  --image_path data/images/frieren/01/0147.png \
  --prompt "Describe this manga panel in detail" \
  --temperature 0.2
```

## Launch 
Launch the app and test locally:
```bash
streamlit run app.py
```

## Used Models
| Purpose   | Model                       |
| --------- | --------------------------- |
| Retriever | `jinaai/jina-embeddings-v4` |
| Reranker  | `colbert-ir/colbertv2.0`    |
| VLM (QA)  | `openbmb/MiniCPM-V-4`       |
| Patching  |  `openai/clip-vit-base-patch16`|

For .env set:

QDRANT_API_KEY=
QDRANT_URL=
INDEX_NAME=
VECTOR_DATABASE_TYPE=qdrant