"""
LLM *models* registry (PropView.md terminology).

Models propose hypotheses via:
  - Claude Code CLI (`claude -p`) — subscription, not paid API keys
  - Grok Build CLI (`grok --single`) — session auth, not XAI API keys
  - DeepSeek on remote host via SSH → local llama-server OpenAI-compatible endpoint
  - offline_heuristic fallback

Do **not** call expensive cloud APIs (Anthropic/xAI/DeepSeek SaaS) from this registry.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMModel:
    """An LLM that can propose hypotheses."""

    model_id: str
    display_name: str
    provider: str  # claude_cli | grok_cli | ssh_openai | offline_heuristic
    # provider-specific
    cli_bin: str = ""
    ssh_host: str = ""  # user@host
    remote_base_url: str = "http://127.0.0.1:8080/v1"
    api_model: str = ""
    temperature: float = 0.3
    max_tokens: int = 600
    timeout_sec: int = 180
    notes: str = ""
    proposal_count: int = 0
    last_error: str = ""

    def is_available(self) -> bool:
        if self.provider == "offline_heuristic":
            return True
        if self.provider == "claude_cli":
            return bool(shutil.which(self.cli_bin or "claude") or os.path.exists(self.cli_bin))
        if self.provider == "grok_cli":
            path = self.cli_bin or "grok"
            return bool(shutil.which(path) or os.path.exists(path))
        if self.provider == "ssh_openai":
            return bool(self.ssh_host)
        return False


def default_registry(cfg: dict | None = None) -> list[LLMModel]:
    cfg = cfg or {}
    custom = cfg.get("llm_models")
    if custom:
        return [LLMModel(**m) for m in custom]

    deepseek_ssh = (
        cfg.get("deepseek_ssh_host")
        or os.environ.get("PROPVIEW_DEEPSEEK_SSH")
        or "anaclast@100.73.106.98"
    )
    deepseek_model = (
        cfg.get("deepseek_model")
        or os.environ.get("PROPVIEW_DEEPSEEK_MODEL")
        or "deepseek-v4-flash-abliterated"
    )
    deepseek_url = (
        cfg.get("deepseek_remote_base_url")
        or os.environ.get("PROPVIEW_DEEPSEEK_BASE_URL")
        or "http://127.0.0.1:8080/v1"
    )
    grok_bin = (
        cfg.get("grok_cli_bin")
        or os.environ.get("PROPVIEW_GROK_BIN")
        or (shutil.which("grok") or os.path.expanduser("~/.grok/bin/grok"))
    )
    claude_bin = (
        cfg.get("claude_cli_bin")
        or os.environ.get("PROPVIEW_CLAUDE_BIN")
        or (shutil.which("claude") or "claude")
    )

    return [
        LLMModel(
            model_id="claude",
            display_name="Claude (CLI)",
            provider="claude_cli",
            cli_bin=claude_bin,
            notes="Claude Code CLI -p (subscription OAuth, not ANTHROPIC_API_KEY)",
            timeout_sec=180,
        ),
        LLMModel(
            model_id="grok",
            display_name="Grok (CLI)",
            provider="grok_cli",
            cli_bin=grok_bin,
            notes="Grok Build CLI --single (session auth, not XAI_API_KEY)",
            timeout_sec=180,
        ),
        LLMModel(
            model_id="deepseek",
            display_name="DeepSeek (SSH llama-server)",
            provider="ssh_openai",
            ssh_host=deepseek_ssh,
            remote_base_url=deepseek_url,
            api_model=deepseek_model,
            notes=f"SSH {deepseek_ssh} → {deepseek_url} model={deepseek_model}",
            timeout_sec=240,
            max_tokens=800,
        ),
        LLMModel(
            model_id="offline_heuristic",
            display_name="Offline heuristic proposer",
            provider="offline_heuristic",
            notes="Fallback if CLIs/SSH fail",
        ),
    ]


def select_available_models(registry: list[LLMModel], prefer: list[str] | None = None) -> list[LLMModel]:
    prefer = prefer or []
    available = [m for m in registry if m.is_available()]
    ordered: list[LLMModel] = []
    for pid in prefer:
        for m in available:
            if m.model_id == pid and m not in ordered:
                ordered.append(m)
    for m in available:
        if m not in ordered:
            ordered.append(m)
    # Prefer live CLIs/SSH over offline when both present
    live = [m for m in ordered if m.provider != "offline_heuristic"]
    offline = [m for m in ordered if m.provider == "offline_heuristic"]
    return live + offline if live else offline


def _run(cmd: list[str], timeout: int) -> str:
    # Prefer unaliased binaries; shell aliases can inject flags.
    env = os.environ.copy()
    # Ensure interactive auth still works for CLIs
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()[:500]
        raise RuntimeError(f"cmd failed ({r.returncode}): {err}")
    out = (r.stdout or "").strip()
    if not out:
        raise RuntimeError("empty stdout from CLI")
    return out


def _claude_chat(model: LLMModel, system: str, user: str) -> str:
    # Do not use --bare (skips OAuth). Subscription login via `claude auth`.
    prompt = (
        f"{system.strip()}\n\n"
        f"--- USER REQUEST ---\n{user.strip()}\n\n"
        "Respond with ONLY the JSON object required. No tools. No prose."
    )
    bin_path = model.cli_bin or "claude"
    # Resolve alias issues: if which finds it, use full path
    resolved = shutil.which(bin_path) or bin_path
    cmd = [resolved, "-p", prompt, "--output-format", "text"]
    return _run(cmd, model.timeout_sec)


def _grok_chat(model: LLMModel, system: str, user: str) -> str:
    prompt = (
        f"{system.strip()}\n\n"
        f"--- USER REQUEST ---\n{user.strip()}\n\n"
        "Respond with ONLY the JSON object required. No tools. No prose."
    )
    bin_path = model.cli_bin or "grok"
    resolved = bin_path if os.path.exists(bin_path) else (shutil.which(bin_path) or bin_path)
    # --single prints response and exits; no API key
    cmd = [resolved, "--single", prompt]
    return _run(cmd, model.timeout_sec)


def _ssh_openai_chat(model: LLMModel, system: str, user: str) -> str:
    """SSH to remote host and call local OpenAI-compatible /v1/chat/completions."""
    payload = {
        "model": model.api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": model.temperature,
        "max_tokens": model.max_tokens,
    }
    payload_json = json.dumps(payload)
    # Escape for single-quoted remote shell: end quote, escaped quote, restart
    remote_json = payload_json.replace("'", "'\"'\"'")
    url = model.remote_base_url.rstrip("/") + "/chat/completions"
    remote_cmd = (
        f"curl -sS --max-time {max(30, model.timeout_sec - 10)} "
        f"'{url}' "
        f"-H 'Content-Type: application/json' "
        f"-d '{remote_json}'"
    )
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        model.ssh_host,
        remote_cmd,
    ]
    raw = _run(cmd, model.timeout_sec)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"SSH OpenAI non-JSON response: {raw[:400]}") from e
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"empty choices: {raw[:400]}")
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    # Some DeepSeek builds put draft text in reasoning_content
    if not content:
        content = (msg.get("reasoning_content") or "").strip()
    if not content:
        raise RuntimeError(f"empty content/reasoning: {raw[:400]}")
    return content


def chat_completion(model: LLMModel, system: str, user: str, timeout: int | None = None) -> str:
    """Dispatch to CLI or SSH backend."""
    if timeout is not None:
        model = LLMModel(**{**model.__dict__, "timeout_sec": timeout})

    if model.provider == "offline_heuristic":
        raise RuntimeError("offline_heuristic has no chat backend")
    if model.provider == "claude_cli":
        return _claude_chat(model, system, user)
    if model.provider == "grok_cli":
        return _grok_chat(model, system, user)
    if model.provider == "ssh_openai":
        return _ssh_openai_chat(model, system, user)
    raise RuntimeError(f"unknown provider {model.provider}")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse first JSON object from model text."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                text = p
                break
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def probe_backends(registry: list[LLMModel] | None = None) -> list[dict]:
    """Quick health check for wiring tests."""
    registry = registry or default_registry()
    results = []
    for m in registry:
        row = {
            "model_id": m.model_id,
            "provider": m.provider,
            "available": m.is_available(),
            "ok": False,
            "detail": "",
        }
        if not m.is_available():
            row["detail"] = "not available"
            results.append(row)
            continue
        if m.provider == "offline_heuristic":
            row["ok"] = True
            row["detail"] = "always"
            results.append(row)
            continue
        try:
            out = chat_completion(
                m,
                "Output only JSON.",
                'Return exactly {"ok":true}',
                timeout=min(90, m.timeout_sec),
            )
            obj = extract_json_object(out)
            row["ok"] = bool(obj.get("ok") is True or "ok" in obj)
            row["detail"] = out[:120].replace("\n", " ")
        except Exception as e:
            row["ok"] = False
            row["detail"] = str(e)[:200]
            m.last_error = row["detail"]
        results.append(row)
    return results
