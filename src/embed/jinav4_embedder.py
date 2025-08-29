from pathlib import Path
from PIL import Image, ImageOps
from transformers import AutoModel
import torch, json, re
from tqdm import tqdm

def extract_volume_from_meta(entry) -> str:
    vol = entry.get("volume") or "00"
    return str(vol).zfill(2)

def safe_open_and_resize(image_path: Path, max_side: int = 1536) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)  
    w, h = img.size
    if max(w, h) > max_side:
        img = ImageOps.contain(img, (max_side, max_side))
    return img

class JinaEmbedder:
    def __init__(self, model_name="jinaai/jina-embeddings-v4", device=None, truncate_dim=2048, max_side=1536):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32
        ).to(self.device)
        self.model.eval()
        self.truncate_dim = truncate_dim
        self.max_side = max_side

    def embed_image(self, image_path: Path):
        img = safe_open_and_resize(image_path, max_side=self.max_side)
        with torch.no_grad():
            vec = self.model.encode_image(
                images=[img],
                task="retrieval",
                truncate_dim=self.truncate_dim,
                return_multivector=False
            )
        return vec[0].detach().cpu().numpy()
    
    def embed_text(self, text: str):
        with torch.no_grad():
            vec = self.model.encode_text(
                texts=[text],
                task="retrieval",
                prompt_name="query",      
                truncate_dim=self.truncate_dim,
                return_multivector=False
            )
        return vec[0].detach().cpu().numpy()

def _load_seen_pages(shard_path: Path):
    seen = set()
    if shard_path.exists():
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    pn = str(obj.get("page_number", "")).zfill(4)
                    seen.add(pn)
                except Exception:
                    continue
    return seen

def batch_embed_sharded(
    metadata_path="data/metadata.jsonl",
    out_dir="data/embeddings/jina_v4",
    truncate_dim=2048,
    max_side=1536
):
    embedder = JinaEmbedder(truncate_dim=truncate_dim, max_side=max_side)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f_in:
        current_vol = None
        seen_pages = set()
        shard_fp = None

        for line in tqdm(f_in, desc="Embedding images (Jina v4, sharded)"):
            entry = json.loads(line)
            img_path = Path(entry["image_path"])
            vol = extract_volume_from_meta(entry)
            page_number = str(entry.get("page_number", "")).zfill(4)

            if vol != current_vol:
                if shard_fp:
                    shard_fp.close()
                    shard_fp = None

                shard_path = out_root / f"vol_{vol}.jsonl"
                seen_pages = _load_seen_pages(shard_path)
                shard_fp = open(shard_path, "a", encoding="utf-8")
                current_vol = vol
                print(f"\n[shard] Writing to {shard_path} (seen {len(seen_pages)} pages)")

            if page_number in seen_pages:
                continue

            if not img_path.exists():
                print(f"Missing image: {img_path}")
                continue

            try:
                vec = embedder.embed_image(img_path)
                entry["embedding"] = vec.tolist()
                shard_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                seen_pages.add(page_number)
            except Exception as e:
                print(f"Embed failed for {img_path}: {e}")
                continue

        if shard_fp:
            shard_fp.close()



if __name__ == "__main__":
    batch_embed_sharded(
        metadata_path="data/metadata.jsonl",
        out_dir="data/embeddings/jina_v4",
        truncate_dim=1024,  
        max_side=1536       
    )
