"""mlx_lm.lora, but chat prompts render with enable_thinking=False.

Qwen3's template puts an empty <think></think> block before the answer when
thinking is off, which is how we query the model. mlx-lm renders the training
prompt without that flag, so its masked loss covered those fixed tokens and
drowned out the one-letter answer. Here both sides render like inference does.

  uv run --extra train scripts/train_lora.py --config runs/lora-qwen3-0.6b.yaml
"""

import sys

from mlx_lm.tuner import datasets


def _process(self, d):
    messages = d[self.chat_key]
    render = dict(return_dict=False, enable_thinking=False)
    tokens = self.tokenizer.apply_chat_template(messages, **render)
    if not self.mask_prompt:
        return tokens, 0
    prompt = self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **render)
    return tokens, len(prompt)


datasets.ChatDataset.process = _process

from mlx_lm.lora import main  # noqa: E402  (after the patch)

if __name__ == "__main__":
    sys.exit(main())
