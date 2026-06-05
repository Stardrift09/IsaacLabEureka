import json
import os
import time
import uuid
from pathlib import Path

from isaaclab_eureka.managers.llm_manager import LLMManager


class FileLLMManager(LLMManager):
    """LLMManager that writes prompts to files and polls for responses.

    Used when the compute environment has no internet access (e.g., BSC MareNostrum).
    A watcher script on a machine with internet reads the request files, calls OpenAI,
    and writes back response files.

    Configuration via environment variables:
        FILE_LLM_REQUEST_DIR   Path where request/response files are written (default: /tmp/llm_requests)
        FILE_LLM_POLL_INTERVAL Seconds between polls for a response (default: 60)
    """

    def __init__(self, gpt_model: str, num_suggestions: int, temperature: float,
                 system_prompt: str, resume: bool):
        # Set parent attributes manually — skip super().__init__() to avoid requiring
        # an API key or creating an OpenAI client (no internet on BSC compute nodes).
        self._gpt_model = gpt_model
        self._num_suggestions = num_suggestions
        self._temperature = temperature
        self._prompts = [{"role": "system", "content": system_prompt}]
        self._debug = False
        self.resume = resume
        self._client = None

        request_dir = os.environ.get("FILE_LLM_REQUEST_DIR", "/tmp/llm_requests")
        self._request_dir = Path(request_dir)
        self._poll_interval = int(os.environ.get("FILE_LLM_POLL_INTERVAL", "60"))
        self._request_dir.mkdir(parents=True, exist_ok=True)
        print(f"[FileLLMManager] Request dir: {self._request_dir}  poll interval: {self._poll_interval}s")

    def prompt(self, user_prompt: str, assistant_prompt: str = None) -> dict:
        if assistant_prompt is not None:
            self._prompts.append({"role": "assistant", "content": self._sanitize(assistant_prompt)})
        self._prompts.append({"role": "user", "content": self._sanitize(user_prompt)})

        # Consume the resume flag; fall through to file-based call
        # (the hardcoded resume string in LLMManager is task-specific and not replicated here)
        if self.resume:
            print("[FileLLMManager] resume=True — ignoring hardcoded shortcut, using file-based call.")
            self.resume = False

        # Keep same history-trimming logic as parent
        if len(self._prompts) == 6:
            self._prompts.pop(2)
            self._prompts.pop(2)

        request_id = uuid.uuid4().hex
        request_file = self._request_dir / f"{request_id}_request.json"
        response_file = self._request_dir / f"{request_id}_response.json"

        request_data = {
            "id": request_id,
            "model": self._gpt_model,
            "messages": list(self._prompts),
            "temperature": self._temperature,
            "n": self._num_suggestions,
        }
        request_file.write_text(json.dumps(request_data, indent=2))
        print(f"[FileLLMManager] Request written: {request_file.name}")
        print(f"[FileLLMManager] Waiting for: {response_file.name}  (poll every {self._poll_interval}s)")

        while not response_file.exists():
            time.sleep(self._poll_interval)
            print(f"[FileLLMManager] Still waiting... ({response_file.name})")

        response_data = json.loads(response_file.read_text())
        raw_outputs = response_data["raw_outputs"]
        reward_strings = [self.extract_code_from_response(r) for r in raw_outputs]
        print(f"[FileLLMManager] Got {len(raw_outputs)} outputs.")
        return {"reward_strings": reward_strings, "raw_outputs": raw_outputs}
