# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""LLMManager that drives Claude through the local Claude Code CLI (``claude -p``).

This backend uses the Claude Code subscription auth (Pro / Max) that is already
configured on the machine — it needs **no** ``ANTHROPIC_API_KEY`` and **no**
OpenAI key. Each suggestion is an independent ``claude -p`` subprocess, run in
parallel, so the Eureka loop keeps its "n parallel samples per iteration"
behaviour even though the Anthropic surface has no ``n`` parameter.

Environment overrides:
    CLAUDE_CLI          Path to the claude binary (default: ``claude``).
    CLAUDE_MODEL        Model alias/id to use when the caller passes a non-Claude
                        model such as ``gpt-4`` (default: ``opus`` -> claude-opus-4-8).
    CLAUDE_CLI_TIMEOUT  Per-call timeout in seconds (default: ``1200``).
"""

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from isaaclab_eureka.managers.llm_manager import LLMManager

# Caller-supplied model strings that are clearly *not* Claude models -> remap.
_NON_CLAUDE_PREFIXES = ("gpt", "o1", "o3", "o4", "text-", "davinci", "azure")


class ClaudeCodeLLMManager(LLMManager):
    """LLMManager backed by the Claude Code CLI (subscription auth, no API key)."""

    def __init__(self, gpt_model: str, num_suggestions: int, temperature: float,
                 system_prompt: str, resume: bool = False):
        # Set parent attributes manually; skip super().__init__() so that no
        # OpenAI client is created and no API key is required.
        self._gpt_model = self._resolve_model(gpt_model)
        self._num_suggestions = num_suggestions
        self._temperature = temperature  # unused: the claude CLI has no temperature knob
        self._system_prompt = system_prompt
        self._prompts = [{"role": "system", "content": system_prompt}]
        self._debug = False
        self.resume = resume
        self._client = None

        self._cli = os.environ.get("CLAUDE_CLI", "claude")
        self._timeout = int(os.environ.get("CLAUDE_CLI_TIMEOUT", "1200"))
        # Neutral working dir so the agent can't read the repo / CLAUDE.md and wander.
        self._workdir = tempfile.mkdtemp(prefix="eureka_claude_")
        print(f"[ClaudeCodeLLMManager] cli={self._cli} model={self._gpt_model} "
              f"n={self._num_suggestions} timeout={self._timeout}s workdir={self._workdir}")

    @staticmethod
    def _resolve_model(gpt_model: str) -> str:
        """Map the caller's model string to a Claude model the CLI understands."""
        m = (gpt_model or "").strip().lower()
        if m.startswith("claude") or m in ("opus", "sonnet", "haiku"):
            return gpt_model
        if not m or any(m.startswith(p) for p in _NON_CLAUDE_PREFIXES):
            return os.environ.get("CLAUDE_MODEL", "opus")
        return gpt_model

    def _flatten_conversation(self) -> str:
        """Flatten the message history (minus the system prompt) into one prompt.

        The system prompt is passed via ``--system-prompt``; everything else is
        rendered as a readable transcript so multi-turn feedback survives.
        """
        parts = []
        for msg in self._prompts[1:]:
            parts.append(f"[{msg['role'].upper()}]\n{msg['content']}")
        return "\n\n".join(parts)

    def _call_once(self, prompt_text: str) -> str:
        """Run a single ``claude -p`` completion and return its raw text."""
        cmd = [
            self._cli, "-p",
            "--model", self._gpt_model,
            "--system-prompt", self._system_prompt,
            "--output-format", "json",
            # Pure text completion: disable agentic tools so it can't touch the FS.
            "--disallowed-tools",
            "Bash", "Edit", "Write", "Read", "Glob", "Grep",
            "WebFetch", "WebSearch", "Task", "TodoWrite",
        ]
        try:
            proc = subprocess.run(
                cmd, input=prompt_text, capture_output=True, text=True,
                cwd=self._workdir, timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return "[ClaudeCodeLLMManager] ERROR: call timed out"
        except FileNotFoundError:
            raise RuntimeError(
                f"Claude Code CLI not found ('{self._cli}'). Install it or set CLAUDE_CLI."
            )
        if proc.returncode != 0:
            return f"[ClaudeCodeLLMManager] ERROR rc={proc.returncode}: {proc.stderr[:500]}"
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return proc.stdout  # fall back to raw stdout if not JSON
        if data.get("is_error"):
            return f"[ClaudeCodeLLMManager] ERROR: {str(data.get('result', 'unknown'))[:500]}"
        return data.get("result", "")

    def prompt(self, user_prompt: str, assistant_prompt: str = None) -> dict:
        if assistant_prompt is not None:
            self._prompts.append({"role": "assistant", "content": self._sanitize(assistant_prompt)})
        self._prompts.append({"role": "user", "content": self._sanitize(user_prompt)})

        # Consume the resume flag; the hardcoded shortcut in LLMManager is
        # task-specific and not replicated here (same as FileLLMManager).
        if self.resume:
            print("[ClaudeCodeLLMManager] resume=True — ignoring hardcoded shortcut.")
            self.resume = False

        # Keep parent's history-trimming (only last round of feedback).
        if len(self._prompts) == 6:
            self._prompts.pop(2)
            self._prompts.pop(2)

        prompt_text = self._flatten_conversation()

        # Fire num_suggestions independent completions in parallel.
        with ThreadPoolExecutor(max_workers=max(1, self._num_suggestions)) as ex:
            raw_outputs = list(ex.map(lambda _: self._call_once(prompt_text),
                                      range(self._num_suggestions)))

        reward_strings = [self.extract_code_from_response(r) for r in raw_outputs]
        return {"reward_strings": reward_strings, "raw_outputs": raw_outputs}
