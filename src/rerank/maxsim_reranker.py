import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


class MaxSimReranker:
    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_text_tokens(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, dim)
        return token_embeddings.cpu().numpy()

    def compute_maxsim_score(
        self,
        text_tokens: np.ndarray,
        image_tokens: np.ndarray
    ) -> float:
        """MaxSim = avg over text tokens of their best image patch match"""
        text_tokens = text_tokens / np.linalg.norm(text_tokens, axis=1, keepdims=True)
        image_tokens = image_tokens / np.linalg.norm(image_tokens, axis=1, keepdims=True)

        sim_matrix = np.matmul(text_tokens, image_tokens.T)  # (T_text, T_image)
        max_sim_per_text_token = np.max(sim_matrix, axis=1)  # (T_text,)
        return float(np.mean(max_sim_per_text_token))        # scalar

    def rerank(
        self,
        query: str,
        candidates: list,
        embedding_dir: str = "data/embeddings/colbert/"
    ) -> list:

        query_embed = self.embed_text_tokens(query)
        reranked = []

        for entry in candidates:
            file_name = f"{entry['page_number']}.npy"
            emb_path = Path(embedding_dir) / file_name
            if not emb_path.exists():
                print(f"Skipping missing embedding: {emb_path}")
                continue

            image_embed = np.load(emb_path)
            score = self.compute_maxsim_score(query_embed, image_embed)
            entry["maxsim_score"] = score
            reranked.append(entry)

        return sorted(reranked, key=lambda x: x["maxsim_score"], reverse=True)
