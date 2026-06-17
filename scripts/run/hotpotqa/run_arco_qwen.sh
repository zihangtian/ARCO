#!/usr/bin/env bash
# End-to-end ARCO pipeline for HotpotQA / Qwen3-4B.
#
# Stages:
#   0. Launch the SimCSE retriever service (background).
#   1. Collect warmup trajectories + step rubrics with a GPT teacher.
#   2. Warmup-SFT the policy pi.
#   3. Warmup-SFT the rubric model mu (dual-head).
#   4. Co-evolution RL.
#
# Configuration lives in configs/hotpotqa/arco_qwen.yaml and
# configs/hotpotqa/hotpotqa.yaml. Replace placeholders (API key,
# Qwen model path, SimCSE checkpoint) before running.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CFG_DATASET="configs/hotpotqa/hotpotqa.yaml"
CFG_ARCO="configs/hotpotqa/arco_qwen.yaml"

LOGDIR="logs/hotpotqa"
mkdir -p "${LOGDIR}"

# ---- 0. Retriever ----
RETRIEVER_LOG="${LOGDIR}/retriever.log"
if ! curl -s http://127.0.0.1:2022/retrieve \
  -X POST -H "Content-Type: application/json" \
  -d '{"query":"test","data":["test"]}' >/dev/null 2>&1; then
  echo "[stage 0] launching retriever -> ${RETRIEVER_LOG}"
  nohup python scripts/deploy_retriever.py --config "${CFG_DATASET}" > "${RETRIEVER_LOG}" 2>&1 &
  RETRIEVER_PID=$!
  echo "  retriever pid=${RETRIEVER_PID}"
  for _ in $(seq 1 60); do
    if curl -s http://127.0.0.1:2022/retrieve \
        -X POST -H "Content-Type: application/json" \
        -d '{"query":"test","data":["test"]}' >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
fi

# ---- 1. Warmup collection ----
echo "[stage 1] collect warmup trajectories"
python scripts/collect_warmup.py --config "${CFG_DATASET}"

# ---- 2. SFT pi ----
echo "[stage 2] warmup-SFT policy pi"
python scripts/sft_pi.py --config "${CFG_ARCO}"

# ---- 3. SFT mu ----
echo "[stage 3] warmup-SFT rubric model mu"
python scripts/sft_mu.py --config "${CFG_ARCO}"

# ---- 4. ARCO RL ----
echo "[stage 4] co-evolution RL"
python scripts/rl_train.py --config "${CFG_ARCO}"
