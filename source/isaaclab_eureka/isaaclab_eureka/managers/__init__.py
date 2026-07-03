# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os
from shutil import which

from .claude_code_llm_manager import ClaudeCodeLLMManager
from .eureka_task_manager import EurekaTaskManager
from .eureka_task_manager_sequential_vlm import EurekaTaskManagerSequentialVLM
from .llm_manager import LLMManager
from .replayrunner import ReplayRunner


def make_llm_manager(gpt_model, num_suggestions, temperature, system_prompt, resume=False):
    """Pick the LLM backend automatically.

    Selection (override with ``EUREKA_LLM_BACKEND``):
      - ``claude`` : Claude Code CLI, uses the local subscription (no API key).
      - ``openai`` : native/Azure OpenAI API (needs OPENAI/AZURE key).
      - ``auto``   : prefer the Claude Code CLI if it's on PATH, else OpenAI.
    """
    backend = os.environ.get("EUREKA_LLM_BACKEND", "auto").lower()
    cli = os.environ.get("CLAUDE_CLI", "claude")

    if backend in ("claude", "claude_code", "claude-code"):
        use_claude = True
    elif backend == "openai":
        use_claude = False
    else:  # auto
        use_claude = which(cli) is not None

    if use_claude:
        return ClaudeCodeLLMManager(
            gpt_model=gpt_model,
            num_suggestions=num_suggestions,
            temperature=temperature,
            system_prompt=system_prompt,
            resume=resume,
        )
    return LLMManager(
        gpt_model=gpt_model,
        num_suggestions=num_suggestions,
        temperature=temperature,
        system_prompt=system_prompt,
        resume=resume,
    )
