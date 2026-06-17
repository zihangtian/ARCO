# ARCO: Adaptive Rubric Co-Evolution for Multi-Step LLM Agents

ARCO trains a multi-step language agent with **interpretable, step-level
process rewards**. A single rubric model `μ` shares a backbone between a
generation head that emits `K=3` natural-language criteria per step and a score
head that returns rubric-conditioned step rewards. A trajectory-decomposition
constraint ties the sum of step rewards to the binary outcome reward, so `μ`
and the policy `π` can be co-optimized on the same on-policy rollouts without
an external judge.

This repository is the minimal reproduction kit for the HotpotQA / Qwen3-4B
setting reported in the paper. The framework code in `src/` is dataset- and
backbone-agnostic; the `configs/` and `prompts/` directories ship the
HotpotQA + Qwen recipe used for the main-table number `EM = 42.80`.

```
release_arco/
├── README.md
├── requirements.txt
├── src/                       # ARCO framework (agents / environments / training / utils)
├── scripts/
│   ├── collect_warmup.py      # GPT-teacher warmup collection
│   ├── sft_pi.py              # Warmup-SFT for the policy
│   ├── sft_mu.py              # Warmup-SFT for the dual-head rubric model
│   ├── rl_train.py            # Co-evolution RL trainer
│   ├── deploy_retriever.py    # SimCSE retriever service
│   └── run/hotpotqa/run_arco_qwen.sh    # End-to-end pipeline
├── configs/hotpotqa/
│   ├── hotpotqa.yaml          # Dataset / API / retriever paths
│   └── arco_qwen.yaml         # ARCO RL config (vLLM enabled, max_model_len=8192)
├── prompts/                   # Policy / rubric / shared prompts
└── data/hotpotqa/
    ├── splits/                # train_2k, dev_500 splits we used
    └── output/                # GPT-annotated warmup + π / μ SFT data
```

## 1. Environment

```bash
conda create -n arco python=3.10 -y
conda activate arco
pip install -r requirements.txt
```

A CUDA build of `torch` matching your driver is recommended. ARCO trains with
LoRA on `bf16`, so a single 80GB GPU is enough for Qwen3-4B; the dense stage
runs faster with 2 GPUs (one for `π`, one for `μ`).

## 2. Datasets and models

We do **not** ship the raw HotpotQA corpus. Place the original files at
`data/hotpotqa/`:

```bash
mkdir -p data/hotpotqa
wget -O data/hotpotqa/hotpot_train_v1.1.json \
  http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget -O data/hotpotqa/hotpot_dev_distractor_v1.json \
  http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

The 2k / 500 splits we trained on, together with the GPT-annotated warmup
trajectories and derived π / μ SFT data, are already included under
`data/hotpotqa/splits/` and `data/hotpotqa/output/`.

You will also need:
- A Qwen3-4B-Instruct-2507 checkpoint
  (replace `Qwen/Qwen3-4B-Instruct-2507` in the configs with your local path
  if you mirror the weights).
- The SimCSE encoder (`princeton-nlp/unsup-simcse-roberta-base`).
- An OpenAI-compatible API key (only needed to **regenerate** warmup data;
  not needed if you use the shipped warmup jsonl).

Set the API key in `configs/hotpotqa/hotpotqa.yaml -> api.api_key`.

## 3. Reproduce HotpotQA / Qwen

The end-to-end pipeline is one shell script:

```bash
bash scripts/run/hotpotqa/run_arco_qwen.sh
```

Stages:

| Stage | Script | Output |
|------:|--------|--------|
| 0 | `scripts/deploy_retriever.py` | SimCSE service on `localhost:2022` |
| 1 | `scripts/collect_warmup.py` | GPT-annotated warmup trajectories |
| 2 | `scripts/sft_pi.py`         | π SFT adapter |
| 3 | `scripts/sft_mu.py`         | μ SFT adapter (dual-head) |
| 4 | `scripts/rl_train.py`       | ARCO co-evolution RL run |

If you only want to reproduce the RL run on top of the included warmup +
SFT data, skip stages 1–3 and call the last command directly:

```bash
python scripts/rl_train.py --config configs/hotpotqa/arco_qwen.yaml
```

## 4. Configuration knobs

`configs/hotpotqa/arco_qwen.yaml`:

- `rl.epochs`, `rl.dense_start_epoch`: sparse-to-dense schedule.
- `rl.dense_baseline_mode = position_bucket`: position-bucketed advantage.
- `rl.dense_signal_mode = sum_return`: reward-to-go from rubric scores.
- `rl.mu_kl_coef`, `rl.pi_kl_coef`: KL strength for `μ` and `π`.
- `vllm.enabled = true`: rollouts use vLLM; `max_model_len = 8192`.

`configs/hotpotqa/hotpotqa.yaml` controls dataset paths, the GPT teacher
endpoint, and the retriever URL.

## 5. Citation

If you find ARCO useful, please cite the paper. A BibTeX entry will be added
once the camera-ready version is finalized.
