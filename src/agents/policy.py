"""Policy agent π: state → action.

Prompt files:
  hotpotqa/policy/system.txt        static system prompt
  hotpotqa/policy/turn.txt          normal turn template
                                    placeholders: {question}, {history}, {searches_remaining}
  hotpotqa/policy/forced_finish.txt forced-submission turn (search budget exhausted)
                                    placeholders: {question}, {history}

max_steps in config = number of Search actions allowed.
When searches_remaining hits 0, act_finish() is called instead of act().
"""

from pathlib import Path

from ..utils.api_client import APIClient
from ..utils.prompt_builders import policy_fields_from_state
from ..environments.base import BaseState


class PolicyAgent:
    def __init__(self, client: APIClient,
                 system_prompt: str,
                 user_template: str,
                 finish_template: str,
                 dataset: str = "hotpotqa",
                 parser_fn=None,
                 temperature: float = 0.0,
                 max_tokens: int = 128):
        self.client = client
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.finish_template = finish_template
        self.dataset = dataset
        if parser_fn is None:
            from ..environments.registry import get_dataset
            parser_fn = get_dataset(dataset).parse_output
        self.parser_fn = parser_fn
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _build_messages(self, state: BaseState, searches_remaining: int, force_finish: bool = False) -> list[dict]:
        template = self.finish_template if force_finish else self.user_template
        user_content = template.format(**policy_fields_from_state(state, self.dataset, searches_remaining))
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_content},
        ]

    def act_with_trace(self, state: BaseState, searches_remaining: int,
                       use_cache: bool = True) -> tuple[str, str, list[dict]]:
        messages = self._build_messages(state, searches_remaining, force_finish=False)
        output = self.client.chat(
            messages, temperature=self.temperature,
            max_tokens=self.max_tokens, use_cache=use_cache,
        )
        _, action = self.parser_fn(output)
        return action, output, messages

    def act_with_trace_safe(self, state: BaseState, searches_remaining: int,
                            use_cache: bool = True) -> tuple[str, str, list[dict], bool]:
        """Normal turn, returning raw invalid output instead of raising."""
        messages = self._build_messages(state, searches_remaining, force_finish=False)
        output = self.client.chat(
            messages, temperature=self.temperature,
            max_tokens=self.max_tokens, use_cache=use_cache,
        )
        try:
            _, action = self.parser_fn(output)
            return action, output, messages, True
        except ValueError:
            return output.strip() or "<EMPTY>", output, messages, False

    def act(self, state: BaseState, searches_remaining: int,
            use_cache: bool = True) -> str:
        """Normal turn — agent may Search or Finish."""
        action, _, _ = self.act_with_trace(state, searches_remaining, use_cache=use_cache)
        return action

    def act_finish(self, state: BaseState,
                   use_cache: bool = True) -> str:
        """Forced final turn — search budget exhausted, agent must submit."""
        action, _, _ = self.act_finish_with_trace(state, use_cache=use_cache)
        return action

    def act_finish_with_trace(self, state: BaseState,
                              use_cache: bool = True) -> tuple[str, str, list[dict]]:
        """Forced final turn with raw output and prompt trace."""
        messages = self._build_messages(state, searches_remaining=0, force_finish=True)
        output = self.client.chat(
            messages, temperature=self.temperature,
            max_tokens=self.max_tokens, use_cache=use_cache,
        )
        _, action = self.parser_fn(output)
        return action, output, messages

    def act_finish_with_trace_safe(self, state: BaseState,
                                   use_cache: bool = True) -> tuple[str, str, list[dict], bool]:
        """Forced final turn, returning raw invalid output instead of raising."""
        messages = self._build_messages(state, searches_remaining=0, force_finish=True)
        output = self.client.chat(
            messages, temperature=self.temperature,
            max_tokens=self.max_tokens, use_cache=use_cache,
        )
        try:
            _, action = self.parser_fn(output)
            return action, output, messages, True
        except ValueError:
            return output.strip() or "<EMPTY>", output, messages, False

    @classmethod
    def from_prompt_dir(cls, client: APIClient, prompt_dir: str | Path,
                        **kwargs) -> "PolicyAgent":
        d = Path(prompt_dir)
        system        = (d / "policy" / "system.txt").read_text(encoding="utf-8").strip()
        user_tmpl     = (d / "policy" / "turn.txt").read_text(encoding="utf-8").strip()
        finish_path   = d / "policy" / "forced_finish.txt"
        finish_tmpl   = finish_path.read_text(encoding="utf-8").strip() if finish_path.exists() else ""
        return cls(client=client, system_prompt=system,
                   user_template=user_tmpl, finish_template=finish_tmpl, **kwargs)
