import re
import json
from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF

def extract_volume_from_name(name: str) -> str:
    m = re.search(r"(?:^|[\s_\-])vol(?:ume)?\s*(\d+)", name, flags=re.IGNORECASE)
    if m:
        return m.group(1).zfill(2)
    m2 = re.search(r"(?:^|[\s_\-])(\d{1,2})(?:\.pdf)?$", name, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).zfill(2)
    return "00"

def pdf_to_images(pdf_path: Path, out_root: Path, manga_name: str, dpi: int = 300) -> List[Dict]:
    volume = extract_volume_from_name(pdf_path.stem)
    out_dir = out_root / manga_name / volume
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    doc = fitz.open(pdf_path)
    metadata: List[Dict] = []
    total_pages = doc.page_count

    print(f" >> {pdf_path.name} | volume={volume} | pages={total_pages}")

    for i in range(total_pages):
        page_num = i + 1
        filename = f"{page_num:04d}.png"
        out_path = out_dir / filename

        if out_path.exists():
            metadata.append({
                "manga": manga_name,
                "volume": volume,
                "page_number": f"{page_num:04d}",
                "image_path": str(out_path)
            })
            continue

        try:
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(out_path))
            del pix
        except Exception as e:
            print(f"Render error at page {page_num}: {e}")
            continue

        metadata.append({
            "manga": manga_name,
            "volume": volume,
            "page_number": f"{page_num:04d}",
            "image_path": str(out_path)
        })

        if page_num % 25 == 0 or page_num == total_pages:
            print(f"    - saved {page_num}/{total_pages}")

    doc.close()
    return metadata

def main():
    manga_name = "frieren"
    pdf_dir = Path("data/raw_pdfs")
    out_images = Path("data/images")
    out_metadata = Path("data/metadata.jsonl")

    all_meta: List[Dict] = []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found under {pdf_dir.resolve()}")
        return

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        vol_meta = pdf_to_images(pdf_file, out_images, manga_name, dpi=300)
        all_meta.extend(vol_meta)

    out_metadata.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metadata, "w", encoding="utf-8") as f:
        for entry in all_meta:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(all_meta)} rows to {out_metadata}")

if __name__ == "__main__":
    main()
