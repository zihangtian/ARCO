"""Deploy the BM25 + SimCSE retrieval service for HotpotQA.

Starts an HTTP server at 127.0.0.1:2022 that accepts POST requests:
  {"entity": str, "data": [str, ...]}
and returns:
  {"response": str}   # top-k passages joined as text

Usage:
    python scripts/deploy_retriever.py \
        --model_path /data/pretrained_models/unsup-simcse-roberta-base \
        --port 2022
"""

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [retriever] %(message)s")


# ── SimCSE encoder ────────────────────────────────────────────────────────────

class SimCSEEncoder:
    def __init__(self, model_path: str, device: str = "cpu"):
        logging.info(f"Loading SimCSE from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        self.device = device

    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=512, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            hidden = self.model(**enc).last_hidden_state[:, 0]  # [CLS]
        vecs = hidden.cpu().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs / norms


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    def __init__(self, encoder: SimCSEEncoder, top_k: int = 3):
        self.encoder = encoder
        self.top_k = top_k

    def retrieve(self, query: str, passages: list[str]) -> str:
        if not passages:
            return ""
        # One forward pass per request is noticeably faster than encoding
        # query and passages separately.
        all_vecs = self.encoder.encode([query] + passages)
        q_vec = all_vecs[:1]                          # (1, d)
        p_vecs = all_vecs[1:]                         # (N, d)
        scores = (q_vec @ p_vecs.T)[0]               # (N,)
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        return "\n\n".join(passages[i] for i in top_idx)


# ── HTTP handler ──────────────────────────────────────────────────────────────

_retriever: Retriever = None
_request_slots = None


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        entity = body.get("entity", body.get("query", ""))
        data = body.get("data", [])
        with _request_slots:
            result = _retriever.retrieve(entity, data)
        resp = json.dumps({"response": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, fmt, *args):
        pass  # suppress per-request logs


# ── main ──────────────────────────────────────────────────────────────────────

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread."""
    daemon_threads = True
    request_queue_size = 128

def main():
    global _retriever, _request_slots
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hotpotqa/hotpotqa.yaml")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.misc import load_yaml
    cfg = load_yaml(args.config)["retriever"]

    model_path = cfg.get("model_path", "/data/pretrained_models/unsup-simcse-roberta-base")
    device     = cfg.get("device", "cpu")
    port       = cfg.get("port", 2022)
    top_k      = cfg.get("top_k", 3)
    max_concurrent = int(cfg.get("max_concurrent_requests", 32))
    max_concurrent = max(1, max_concurrent)

    encoder = SimCSEEncoder(model_path, device=device)
    _retriever = Retriever(encoder, top_k=top_k)
    _request_slots = threading.BoundedSemaphore(value=max_concurrent)

    server = ThreadedHTTPServer(("127.0.0.1", port), Handler)
    logging.info(
        "Retrieval service listening on 127.0.0.1:%s (max_concurrent_requests=%s)",
        port,
        max_concurrent,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down.")


if __name__ == "__main__":
    main()
