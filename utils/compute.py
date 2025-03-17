import gc
import torch
from tqdm import tqdm
from typing import Union
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)


def extract_hidden_states(raw_output) -> dict:
    processed = {}

    assert hasattr(raw_output, "hidden_states")
    cpu_hidden = []
    for layer_output in raw_output.hidden_states:
        layer_tensors = []
        for tensor in layer_output:
            assert isinstance(tensor, torch.Tensor)
            layer_tensors.append(tensor.to("cpu"))
        cpu_hidden.append(layer_tensors)
    processed["hidden_states"] = cpu_hidden

    return processed


def compute_refusals(
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    harmful_list: list[str],
    harmless_list: list[str],
    layer_fraction: float = 0.6,
) -> torch.Tensor:
    harmful_tokens = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for inst in harmful_list
    ]
    harmless_tokens = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for inst in harmless_list
    ]

    torch.cuda.empty_cache()
    gc.collect()

    layer_idx = int(len(model.model.layers) * layer_fraction)
    pos = -1

    harmful_sum = None
    harmful_count = 0
    for token in tqdm(harmful_tokens, desc="Generating harmful outputs"):
        raw_output = model.generate(
            token.to(model.device),
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        cpu_output = extract_hidden_states(raw_output)
        current_hidden = cpu_output["hidden_states"][0][layer_idx][:, pos, :]
        assert isinstance(current_hidden, torch.Tensor)
        if harmful_sum is None:
            harmful_sum = current_hidden.sum(dim=0)
        else:
            harmful_sum += current_hidden.sum(dim=0)
        harmful_count += current_hidden.size(dim=0)
        del raw_output, cpu_output, current_hidden
        torch.cuda.empty_cache()
        gc.collect()

    harmless_sum = None
    harmless_count = 0
    for token in tqdm(harmless_tokens, desc="Generating harmless outputs"):
        raw_output = model.generate(
            token.to(model.device),
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        cpu_output = extract_hidden_states(raw_output)
        current_hidden = cpu_output["hidden_states"][0][layer_idx][:, pos, :]
        assert isinstance(current_hidden, torch.Tensor)
        if harmless_sum is None:
            harmless_sum = current_hidden.sum(dim=0)
        else:
            harmless_sum += current_hidden.sum(dim=0)
        harmless_count += current_hidden.size(dim=0)
        del raw_output, cpu_output, current_hidden
        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    assert harmful_sum is not None
    assert harmless_sum is not None
    harmful_mean = harmful_sum / harmful_count
    harmless_mean = harmless_sum / harmless_count
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    print(refusal_dir)
    return refusal_dir
