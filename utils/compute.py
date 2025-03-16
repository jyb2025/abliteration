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

    harmful_outputs = []
    harmless_outputs = []

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
        harmful_outputs.append(cpu_output)
        del raw_output
        torch.cuda.empty_cache()
        gc.collect()

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
        harmless_outputs.append(cpu_output)
        del raw_output
        torch.cuda.empty_cache()
        gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    layer_idx = int(len(model.model.layers) * layer_fraction)
    pos = -1
    harmful_hidden = [
        output["hidden_states"][0][layer_idx][:, pos, :] for output in harmful_outputs
    ]
    harmless_hidden = [
        output["hidden_states"][0][layer_idx][:, pos, :] for output in harmless_outputs
    ]

    harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
    harmless_mean = torch.stack(harmless_hidden).mean(dim=0)
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    print(refusal_dir)
    return refusal_dir
