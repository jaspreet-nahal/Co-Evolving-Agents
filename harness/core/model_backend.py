"""Production model backend: lazy Hugging Face loading, status inspection, preflight.

This module is the single place that knows the four frozen models' exact
Hugging Face ids and how to load/unload them. It performs no network
activity at import time -- every model-touching operation (`load()`,
`check_model_status()` with cache probing, `preflight()`) is an explicit
call, and only `load()` itself may trigger a download.

Credentials are never stored here or anywhere in the repo: HF_TOKEN (or the
older HUGGINGFACE_HUB_TOKEN / HUGGING_FACE_HUB_TOKEN) is read from the
environment at call time and handed to `transformers`/`huggingface_hub`
verbatim; it is never logged, written to disk, or echoed back in any
status/report structure.
"""
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# Exact, verified Hugging Face model ids. Do not substitute or invent ids.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Muse-Glimmer-30B": {
        "model_id": "meta-models/Muse-Glimmer-30B",
        "requires_auth": True,
    },
    "Qwen3.8-27B": {
        "model_id": "Qwen/Qwen3.8-27B",
        "requires_auth": False,
    },
    "Gemma-4-26B-A4B-it": {
        "model_id": "google/gemma-4-26B-A4B-it",
        "requires_auth": True,
    },
    "Gemma-4-31B-it": {
        "model_id": "google/gemma-4-31B-it",
        "requires_auth": True,
    },
}

# Whether a given model publisher's family is typically gated on Hugging
# Face is a property of the model, not something this module can verify
# without contacting the Hub (see check_model_status, which does contact
# the Hub for an authoritative answer). "requires_auth" above is a
# configured *hint* used only when the Hub cannot be reached; the
# authoritative signal at runtime is always the Hub response itself.


class ModelStatus(str, Enum):
    AVAILABLE_IN_CONFIG = "AVAILABLE_IN_CONFIG"   # id is registered; nothing checked yet
    CACHED = "CACHED"                             # weights already present in local HF cache
    DOWNLOAD_REQUIRED = "DOWNLOAD_REQUIRED"        # reachable on the Hub, not cached locally
    AUTH_REQUIRED = "AUTH_REQUIRED"                # Hub says this repo needs credentials we don't have
    LOAD_FAILED = "LOAD_FAILED"                    # an actual load attempt raised
    READY = "READY"                                # currently loaded in this process


class ModelBackendError(RuntimeError):
    """Raised for any model load/auth/download failure. Never caught to
    silently substitute a different policy -- callers must let this
    propagate or handle it explicitly."""


class AuthenticationRequiredError(ModelBackendError):
    pass


class ModelDownloadError(ModelBackendError):
    pass


def _hf_token() -> Optional[str]:
    """Read an HF auth token from the environment only. Never persisted."""
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value
    return None


def _hf_cache_dir() -> Optional[str]:
    """Resolve the effective HF cache dir the same way huggingface_hub does,
    without importing it, purely for reporting purposes."""
    return os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")


def resolve_model_entry(model_name: str) -> Dict[str, Any]:
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise ModelBackendError(f"Unknown model '{model_name}'; not present in MODEL_REGISTRY.")
    return entry


def is_cached_locally(model_id: str) -> bool:
    """True if a local Hugging Face cache already has this repo's weights.

    Uses huggingface_hub's own cache scanner -- this never touches the
    network. Any exception scanning the cache is treated as "not cached"
    rather than propagated, since a broken/missing cache dir is not a
    reason to fail the whole status check.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        return any(repo.repo_id == model_id for repo in cache_info.repos)
    except Exception:
        return False


def check_model_status(model_name: str, contact_hub: bool = True) -> Dict[str, Any]:
    """Inspect (without downloading) whether `model_name` can be loaded.

    Returns a dict with at least: model_name, model_id, status (a
    ModelStatus value), cached (bool), hf_reachable (bool or None if not
    checked), auth_required (bool or None), detail (str).

    `contact_hub` controls whether this performs a lightweight metadata
    call (HEAD-style repo info request) to distinguish DOWNLOAD_REQUIRED
    from AUTH_REQUIRED. It never downloads model weights.
    """
    entry = resolve_model_entry(model_name)
    model_id = entry["model_id"]
    result: Dict[str, Any] = {
        "model_name": model_name,
        "model_id": model_id,
        "cached": False,
        "hf_reachable": None,
        "auth_required": None,
        "status": ModelStatus.AVAILABLE_IN_CONFIG.value,
        "detail": "",
    }

    cached = is_cached_locally(model_id)
    result["cached"] = cached
    if cached:
        result["status"] = ModelStatus.CACHED.value
        result["detail"] = "Weights found in local Hugging Face cache."
        return result

    if not contact_hub:
        result["status"] = ModelStatus.DOWNLOAD_REQUIRED.value
        result["detail"] = "Not cached locally; Hub connectivity not checked (contact_hub=False)."
        return result

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        api = HfApi()
        token = _hf_token()
        try:
            api.model_info(model_id, token=token)
            result["hf_reachable"] = True
            result["auth_required"] = False
            result["status"] = ModelStatus.DOWNLOAD_REQUIRED.value
            result["detail"] = "Reachable on the Hub; weights will be downloaded on first load."
        except GatedRepoError:
            result["hf_reachable"] = True
            result["auth_required"] = True
            if token:
                result["status"] = ModelStatus.AUTH_REQUIRED.value
                result["detail"] = "Repo is gated and the configured HF_TOKEN was not accepted or lacks access."
            else:
                result["status"] = ModelStatus.AUTH_REQUIRED.value
                result["detail"] = "Repo is gated; no HF_TOKEN found in the environment."
        except RepositoryNotFoundError:
            result["hf_reachable"] = True
            result["auth_required"] = None
            result["status"] = ModelStatus.LOAD_FAILED.value
            result["detail"] = f"Repository '{model_id}' was not found on the Hub with the current credentials."
    except Exception as error:
        result["hf_reachable"] = False
        result["status"] = ModelStatus.DOWNLOAD_REQUIRED.value
        result["detail"] = f"Could not reach the Hugging Face Hub to verify: {error}"

    return result


@dataclass
class GpuInfo:
    cuda_available: bool
    device_count: int
    devices: List[Dict[str, Any]] = field(default_factory=list)


def inspect_gpu() -> GpuInfo:
    """Discover the runtime's GPU situation without hard-coding any device.

    Never assumes any specific GPU hardware or a fixed device count --
    it asks torch/CUDA directly.
    """
    try:
        import torch
    except Exception:
        return GpuInfo(cuda_available=False, device_count=0, devices=[])

    if not torch.cuda.is_available():
        return GpuInfo(cuda_available=False, device_count=0, devices=[])

    count = torch.cuda.device_count()
    devices = []
    for index in range(count):
        try:
            name = torch.cuda.get_device_name(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            devices.append({
                "index": index,
                "name": name,
                "total_memory_gb": round(total_bytes / (1024 ** 3), 2),
                "free_memory_gb": round(free_bytes / (1024 ** 3), 2),
            })
        except Exception as error:
            devices.append({"index": index, "error": str(error)})

    return GpuInfo(cuda_available=True, device_count=count, devices=devices)


def preflight(models: Optional[List[str]] = None, contact_hub: bool = True) -> Dict[str, Any]:
    """Inspect environment + per-model loadability without downloading weights.

    This is the function backing `scripts/preflight.py`. It never calls
    `TransformersModelBackend.load()` -- only status/metadata checks.
    """
    models = models or list(MODEL_REGISTRY)
    gpu = inspect_gpu()

    model_reports = []
    for model_name in models:
        status = check_model_status(model_name, contact_hub=contact_hub)
        ready = status["status"] in (ModelStatus.CACHED.value, ModelStatus.DOWNLOAD_REQUIRED.value)
        status["ready_or_blocked"] = "READY" if ready else "BLOCKED"
        model_reports.append(status)

    return {
        "hf_home": _hf_cache_dir(),
        "hf_token_configured": _hf_token() is not None,
        "cuda_available": gpu.cuda_available,
        "gpu_device_count": gpu.device_count,
        "gpu_devices": gpu.devices,
        "models": model_reports,
    }


class TransformersModelBackend:
    """Lazy-loading Hugging Face model + tokenizer/processor for one model.

    - No network access happens until `load()` is called.
    - `load()` reuses local cache if present, otherwise downloads via
      `from_pretrained`, honoring HF_HOME/TRANSFORMERS_CACHE (read by
      transformers/huggingface_hub directly from the environment; this
      class does not override them) and HF_TOKEN (read here and passed as
      the `token=` kwarg, never stored on the instance longer than the
      call).
    - `device_map="auto"` and `dtype="auto"` are the defaults per the
      infra requirements; both are overridable via constructor kwargs (or
      environment, see `from_env`), and no GPU/model is hard-coded.
    - `unload()` releases the model/tokenizer references and clears the
      CUDA cache so a subsequent model can load cleanly.
    """

    def __init__(self, model_name: str, device_map: str = "auto", dtype: str = "auto",
                 max_new_tokens: int = 512, quantization: Optional[str] = None):
        self.model_name = model_name
        self.entry = resolve_model_entry(model_name)
        self.model_id = self.entry["model_id"]
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        # Quantization is never invented implicitly; it is only applied if
        # the caller explicitly asks for it (e.g. via HarnessConfig or env).
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self.status = ModelStatus.AVAILABLE_IN_CONFIG

    @classmethod
    def from_env(cls, model_name: str) -> "TransformersModelBackend":
        return cls(
            model_name=model_name,
            device_map=os.environ.get("HARNESS_MODEL_DEVICE_MAP", "auto"),
            dtype=os.environ.get("HARNESS_MODEL_DTYPE", "auto"),
            max_new_tokens=int(os.environ.get("HARNESS_MODEL_MAX_NEW_TOKENS", "512")),
            quantization=os.environ.get("HARNESS_MODEL_QUANTIZATION") or None,
        )

    def load(self) -> None:
        if self.model is not None:
            self.status = ModelStatus.READY
            return

        # `requires_auth` on the registry entry is only a configured HINT
        # (see model_backend.py module docstring / MODEL_REGISTRY comment).
        # It is not used to pre-emptively block loading: the real
        # from_pretrained call below is authoritative, and only an actual
        # auth-shaped failure from the Hub is classified as
        # AuthenticationRequiredError. Preflight's check_model_status()
        # separately contacts the Hub to give an accurate hint in advance.
        token = _hf_token()

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)

            load_kwargs: Dict[str, Any] = {"device_map": self.device_map, "token": token}
            load_kwargs["dtype"] = self.dtype
            if self.quantization:
                load_kwargs["quantization_config"] = self._build_quantization_config()

            try:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
            except TypeError:
                # Older transformers versions use torch_dtype instead of dtype.
                load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

            self.status = ModelStatus.READY
        except AuthenticationRequiredError:
            raise
        except Exception as error:
            self.status = ModelStatus.LOAD_FAILED
            message = str(error)
            if "gated" in message.lower() or "authoriz" in message.lower() or "401" in message or "403" in message:
                raise AuthenticationRequiredError(
                    f"Failed to load '{self.model_name}' ({self.model_id}): authentication "
                    f"appears to be required or the token was rejected. Original error: {error}"
                ) from error
            raise ModelDownloadError(
                f"Failed to load '{self.model_name}' ({self.model_id}): {error}"
            ) from error

    def _build_quantization_config(self) -> Any:
        # Quantization is opt-in only (Part D): construct a config purely
        # from the explicitly requested scheme, never inferred.
        from transformers import BitsAndBytesConfig
        if self.quantization == "4bit":
            return BitsAndBytesConfig(load_in_4bit=True)
        if self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ModelBackendError(f"Unsupported quantization scheme '{self.quantization}'.")

    def unload(self) -> None:
        """Release model/tokenizer references and clear CUDA cache.

        Safe to call whether or not the model was ever loaded. After this,
        `load()` can be called again (e.g. for a different HarnessConfig)
        without stale GPU allocations from this instance.
        """
        self.model = None
        self.tokenizer = None
        self.status = ModelStatus.AVAILABLE_IN_CONFIG
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, prompt: str) -> str:
        """Generate raw text from `prompt` using the loaded model.

        Callers (agent_policy.py, answer_generator.py) are responsible for
        prompt construction and for parsing the returned text; this method
        performs no JSON parsing or retry logic itself.
        """
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
