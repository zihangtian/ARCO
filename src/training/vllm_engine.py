"""vLLM-based generation engine for RL rollouts.

Wraps vllm.LLM to handle lifecycle and LoRA adapter management for π and μ.
The engine is kept alive for an entire epoch; only LoRA paths are updated
between batches to pick up freshly-trained weights.
"""

import gc
import os
from dataclasses import dataclass

import torch


@dataclass
class GenerateResult:
    """Per-sample output from vLLM generate."""
    prompt_token_ids: list[int]
    resp_token_ids: list[int]
    resp_text: str


class VllmRolloutEngine:
    """Manages a persistent vLLM LLM instance with hot-swappable LoRA adapters."""

    def __init__(self, base_model_path: str,
                 max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85,
                 max_lora_rank: int = 64,
                 device: str = "cuda:0",
                 pi_only: bool = False):
        self.base_model_path = base_model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_lora_rank = max_lora_rank
        self.device = device
        self.pi_only = pi_only
        self.llm = None
        self._pi_lora_path = None
        self._mu_lora_path = None
        # Monotonically increasing IDs so vLLM never serves stale cached LoRA
        self._pi_lora_id = 0
        self._mu_lora_id = 0

    @property
    def serves_mu(self) -> bool:
        return not self.pi_only

    def start(self, pi_lora_path: str | None = None,
              mu_lora_path: str | None = None):
        """Create vLLM LLM instance with LoRA support.

        Narrows CUDA_VISIBLE_DEVICES so vLLM's spawn subprocess lands on
        the desired physical GPU. The env var stays narrowed until stop()
        to prevent the subprocess from seeing other GPUs.
        """
        from vllm import LLM

        if self.pi_only:
            mu_lora_path = None  # μ does not run on this engine
        has_lora = pi_lora_path is not None or mu_lora_path is not None
        self._pi_lora_path = pi_lora_path
        self._mu_lora_path = mu_lora_path
        self._pi_lora_id = 1
        self._mu_lora_id = 2

        # Map logical cuda:N → physical GPU id via CUDA_VISIBLE_DEVICES
        self._orig_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        logical_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
        if self._orig_cvd:
            physical_gpus = self._orig_cvd.split(",")
            target_gpu = physical_gpus[logical_idx]
        else:
            target_gpu = str(logical_idx)
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu
        # Cross-family (pi_only) mode triggers a vLLM v1 spawn bug that hides
        # errors as "EngineCore: 255" when CUDA is initialised in the parent
        # on a different GPU than the vLLM target. Force in-process mode in
        # that case. Same-family runs keep the default (multiprocessing on)
        # so they behave exactly as before.
        if self.pi_only:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        # Disable usage telemetry (its background thread crashes on some
        # conda envs with cpuinfo PermissionError; harmless but noisy).
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("DO_NOT_TRACK", "1")
        print(f"[vLLM] Starting on physical GPU {target_gpu} (logical {self.device})")

        max_loras = 1 if self.pi_only else (2 if has_lora else 0)
        self.llm = LLM(
            model=self.base_model_path,
            dtype="bfloat16",
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enable_lora=has_lora,
            max_loras=max_loras,
            max_lora_rank=self.max_lora_rank if has_lora else 0,
            trust_remote_code=True,
            enforce_eager=True,
        )

    def stop(self):
        """Destroy vLLM instance and free GPU memory, restore CUDA_VISIBLE_DEVICES.

        vLLM v1 InprocClient holds onto KV cache pool, LoRA workspace, and the
        distributed parallel state. A bare `del self.llm` does not free those
        on the next start, so we explicitly tear them down before returning.
        """
        if self.llm is not None:
            try:
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )
                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception as e:
                print(f"[vLLM] destroy_model_parallel failed: {e}")
            del self.llm
            self.llm = None
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Restore original CUDA_VISIBLE_DEVICES
        if hasattr(self, "_orig_cvd"):
            os.environ["CUDA_VISIBLE_DEVICES"] = self._orig_cvd
            print(f"[vLLM] Restored CUDA_VISIBLE_DEVICES={self._orig_cvd}")

    def update_lora(self, pi_lora_path: str | None = None,
                    mu_lora_path: str | None = None):
        """Update LoRA adapter paths and bump IDs so vLLM loads fresh weights."""
        self._pi_lora_path = pi_lora_path
        self._mu_lora_path = None if self.pi_only else mu_lora_path
        # Bump IDs by 2 (pi uses odd, mu uses even) to avoid collisions
        self._pi_lora_id += 2
        self._mu_lora_id += 2

    def generate_pi(self, prompts: list[str], temperature: float = 0.7,
                    max_tokens: int = 64) -> list[GenerateResult]:
        """Generate π actions via vLLM with π LoRA adapter."""
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampling = SamplingParams(
            temperature=temperature if temperature > 0 else 0,
            max_tokens=max_tokens,
        )
        lora_req = None
        if self._pi_lora_path:
            lora_req = LoRARequest(
                lora_name=f"pi_v{self._pi_lora_id}",
                lora_int_id=self._pi_lora_id,
                lora_path=self._pi_lora_path,
            )
        outputs = self.llm.generate(prompts, sampling, lora_request=lora_req, use_tqdm=False)
        return self._parse_outputs(outputs)

    def generate_mu(self, prompts: list[str],
                    max_tokens: int = 128) -> list[GenerateResult]:
        """Generate μ criteria via vLLM with μ LoRA adapter (greedy).

        Raises if pi_only mode — caller should fall back to HF generate.
        """
        if self.pi_only:
            raise RuntimeError("vLLM engine is in pi_only mode; μ generation must use HF")
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampling = SamplingParams(temperature=0, max_tokens=max_tokens)
        lora_req = None
        if self._mu_lora_path:
            lora_req = LoRARequest(
                lora_name=f"mu_v{self._mu_lora_id}",
                lora_int_id=self._mu_lora_id,
                lora_path=self._mu_lora_path,
            )
        outputs = self.llm.generate(prompts, sampling, lora_request=lora_req, use_tqdm=False)
        return self._parse_outputs(outputs)

    @staticmethod
    def _parse_outputs(outputs) -> list[GenerateResult]:
        results = []
        for out in outputs:
            completion = out.outputs[0]
            results.append(GenerateResult(
                prompt_token_ids=list(out.prompt_token_ids),
                resp_token_ids=list(completion.token_ids),
                resp_text=completion.text,
            ))
        return results
