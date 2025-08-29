#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import io

import streamlit as st
from PIL import Image

# Internal modules (your project)
from src.search.query import JinaV4QueryQdrant
from src.generate.vlm_response import MiniCPMV4Answerer
from src.search.query_with_rerank import search_and_rerank


# =========================
# UI Utilities
# =========================

def brand_header():
    st.set_page_config(
        page_title="Frieren Multimodal Search",
        layout="wide",
        page_icon="📚",
    )
    st.markdown(
        """
        <style>
        .muted { color: #6b7280; font-size: 0.9rem; }
        .pill { display:inline-block; padding: 2px 10px; border-radius: 999px; background:#eef2ff; color:#3730a3; font-weight:600; font-size:0.8rem; margin-right:6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("📚 Frieren Multimodal Search")



def show_query_image_small(img: Image.Image, caption: str = "Query image", width: int = 320) -> None:
    st.image(img, caption=caption, width=width)


def show_results_grid(results: List[Dict[str, Any]], title: str = "Results") -> None:
    st.subheader(title)
    if not results:
        st.info("No matches found.")
        return
    cols = st.columns(3)
    for i, r in enumerate(results):
        with cols[i % 3]:
            base_score = f"{r['score']:.3f}" if "score" in r and r["score"] is not None else "-"
            if "maxsim_score" in r and r["maxsim_score"] is not None:
                base_score += f" · MaxSim {r['maxsim_score']:.3f}"
            cap = f"Vol {r.get('volume','??')} · Pg {r.get('page_number','????')} · {base_score}"
            img_path = r.get("image_path")
            if img_path and Path(img_path).exists():
                st.image(img_path, use_container_width=True, caption=cap)
            else:
                st.caption(cap)
                if img_path:
                    st.code(img_path, language="text")


def safe_tmp_write(img: Image.Image, filename: str) -> Path:
    qdir = Path("data/tmp_queries")
    qdir.mkdir(parents=True, exist_ok=True)
    out = qdir / filename
    img.save(out)
    return out


# =========================
# Cached Resources
# =========================

@st.cache_resource(show_spinner=False)
def get_retriever(collection: str, truncate_dim: int):
    # Qdrant connection and embedder are configured internally.
    try:
        return JinaV4QueryQdrant(collection_name=collection, truncate_dim=truncate_dim)
    except Exception as e:
        raise RuntimeError("Search backend is currently unavailable.") from e


@st.cache_resource(show_spinner=False)
def get_vlm():
    try:
        return MiniCPMV4Answerer()
    except Exception as e:
        raise RuntimeError("Visual interpreter is currently unavailable.") from e


# =========================
# App
# =========================

def main():
    brand_header()

    # Sidebar – minimal knobs, no technical jargon
    with st.sidebar:
        st.markdown("**Search Settings**")
        collection_name = st.text_input("Collection", value="frieren-panels", help="Prebuilt collection name.")
        truncate_dim = st.selectbox("Embedding dimension", [512, 1024, 2048], index=1, help="Must match the indexed dimension.")
        #volume_filter = st.text_input("Volume filter (e.g., 01)", value="")
        st.markdown("**Appearance**")
        query_preview_w = st.slider("Query preview width", 160, 640, 320, 10)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 Image → Image", "🔎 Text → Image", "🪄 Image + Text Rerank", "🗣️ Visual Explanation"]
    )

    # ---------- Tab 1: Image → Image ----------
    with tab1:
        st.subheader("Find similar panels from an image")
        c1, c2 = st.columns([1, 2])
        with c1:
            uploaded = st.file_uploader("Upload a panel", type=["png", "jpg", "jpeg"])
            top_k = st.slider("Results", 1, 24, 8, 1)
            run = st.button("Search", type="primary", use_container_width=True)
        with c2:
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                show_query_image_small(img, width=query_preview_w)

        if run:
            if not uploaded:
                st.warning("Please upload an image.")
            else:
                img = Image.open(uploaded).convert("RGB")
                tmp = safe_tmp_write(img, "query_image.png")
                #must_volume = volume_filter.strip() or None
                try:
                    retriever = get_retriever(collection_name, truncate_dim)
                    with st.spinner("Searching…"):
                        results = retriever.search_image(str(tmp), top_k=top_k) #must_volume=must_volume
                except Exception:
                    st.error("Search is not available right now. Please try again later.")
                    results = []
                show_results_grid(results)

    # ---------- Tab 2: Text → Image ----------
    with tab2:
        st.subheader("Find panels similar to a text query")
        text_query = st.text_input("Query text", value="Frieren visits Himmel's grave.")
        c1, c2 = st.columns([1, 2])
        with c1:
            top_k_t = st.slider("Results", 1, 24, 8, 1, key="topk_text")
            run_t = st.button("Search", type="primary", use_container_width=True, key="btn_text")
        with c2:
            st.caption("Tip: mention characters, setting, action, or mood.")

        if run_t:
            if not text_query.strip():
                st.warning("Please type a query.")
            else:
                #must_volume = volume_filter.strip() or None
                try:
                    retriever = get_retriever(collection_name, truncate_dim)
                    with st.spinner("Searching…"):
                        results = retriever.search_text(text_query, top_k=top_k_t,) #must_volume=must_volume)
                except Exception:
                    st.error("Search is not available right now. Please try again later.")
                    results = []
                show_results_grid(results)

    # ---------- Tab 3: Image + Text Rerank ----------
    with tab3:
        st.subheader("Retrieve by image, then rerank by text (MaxSim)")
        c1, c2 = st.columns([1, 2])
        with c1:
            up_rerank = st.file_uploader("Upload a panel", type=["png", "jpg", "jpeg"], key="rerank_upload")
            text_rerank = st.text_input("Rerank text", value="Frieren saw Himmel's statue.")
            c3, c4 = st.columns(2)
            with c3:
                top_k_retrieve = st.number_input("Initial retrieve", 1, 50, 10, 1)
            with c4:
                top_k_rerank = st.number_input("Final top-k", 1, 20, 3, 1)
            embedding_dir = "data/embeddings/patches/vol_01"
            run_rr = st.button("Search + Rerank", type="primary", use_container_width=True)
        with c2:
            if up_rerank:
                img = Image.open(up_rerank).convert("RGB")
                show_query_image_small(img, width=query_preview_w)

        if run_rr:
            if not up_rerank:
                st.warning("Please upload an image.")
            elif not text_rerank.strip():
                st.warning("Please provide rerank text.")
            else:
                img = Image.open(up_rerank).convert("RGB")
                tmp = safe_tmp_write(img, "query_image_for_rerank.png")
                try:
                    with st.spinner("Retrieving and reranking…"):
                        results = search_and_rerank(
                            query_image_path=str(tmp),
                            query_text=text_rerank,
                            top_k_retrieve=int(top_k_retrieve),
                            top_k_rerank=int(top_k_rerank),
                            embedding_dir=embedding_dir,
                        )
                except Exception:
                    st.error("Rerank is not available right now. Please try again later.")
                    results = []
                show_results_grid(results, "Reranked Results")

    # ---------- Tab 4: Visual Explanation (VLM) ----------
    with tab4:
        st.subheader("Explain an image (MiniCPM-V-4)")
        c1, c2 = st.columns([1, 2])
        with c1:
            up_vlm = st.file_uploader("Upload a panel", type=["png", "jpg", "jpeg"], key="vlm_upload")
            sampling = True
            ca, cb = st.columns(2)
            with ca:
                temperature = st.number_input("Temperature", 0.0, 1.0, 0.0, 0.1)
            with cb:
                top_p = st.number_input("Top-p", 0.0, 1.0, 0.3, 0.05)
            run_v = st.button("Explain", type="primary", use_container_width=True)
        with c2:
            prompt_default = (
                "Describe this manga panel in detail: characters, setting, and events. "
                "If possible, guess the chapter/volume and justify briefly."
            )
            prompt = st.text_area("Prompt", value=prompt_default, height=140)

        if run_v:
            if not up_vlm:
                st.warning("Please upload an image.")
            else:
                img = Image.open(up_vlm).convert("RGB")
                show_query_image_small(img, caption="Image to explain", width=query_preview_w)
                tmp = safe_tmp_write(img, "vlm_query.png")
                try:
                    vlm = get_vlm()
                    with st.spinner("Generating response…"):
                        out = vlm.answer(
                            image_path=str(tmp),
                            prompt=prompt,
                            sampling=sampling,
                            temperature=temperature,
                            top_p=top_p,
                        )
                except Exception:
                    st.error("Visual explanation is not available right now. Please try again later.")
                    out = None

                if out:
                    st.success("Response")
                    st.write(out)
                else:
                    st.info("No response produced at this time.")

    st.markdown("<br><span class='muted'>© 2025 Rabia Eda Yılmaz. All rights reserved.</span>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
