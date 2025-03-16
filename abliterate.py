import gc
import torch
import random
import pandas
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from utils.arguments import parser
from utils.compute import compute_refusals
from utils.apply import apply_abliteration


if __name__ == "__main__":
    args = parser.parse_args()
    assert args.skip_begin >= 1, "Do not mess with the first layer!"
    assert (
        args.layer_fraction >= 0.0 and args.layer_fraction <= 1.0
    ), "Invalid layer fraction"
    torch.inference_mode()
    torch.set_grad_enabled(False)

    if args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    else:
        precision = torch.float32
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    df = pandas.read_parquet("./data/harmless.parquet")
    harmless_list = df["text"].tolist()
    df = pandas.read_parquet("./data/harmful.parquet")
    harmful_list = df["text"].tolist()
    if args.deccp:
        deccp_list = load_dataset("augmxnt/deccp", split="censored")
        harmful_list += deccp_list["text"]

    if args.num_calibs > 0:
        harmful_list = random.sample(harmful_list, args.num_calibs)
        harmless_list = random.sample(harmless_list, args.num_calibs)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=precision,
        low_cpu_mem_usage=True,
        device_map=args.device,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )
    model.requires_grad_(False)

    assert args.skip_begin + args.skip_end < len(
        model.model.layers
    ), "Too many layers to skip"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, device_map=args.device
    )

    if args.input_refusal is not None:
        print(f"Loading refusal tensor from {args.input_refusal}...")
        refusal_dir = torch.load(args.input_refusal)
    else:
        print("Computing refusal tensor...")
        refusal_dir = compute_refusals(
            model, tokenizer, harmful_list, harmless_list, args.layer_fraction
        )
    if args.output_refusal is not None:
        print(f"Saving refusal tensor to {args.output_refusal}...")
        torch.save(refusal_dir, args.output_refusal)
    print("Applying refusal tensor...")

    if args.load_in_4bit or args.load_in_8bit:
        print("Reloading model with bf16 precision...")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

    model = apply_abliteration(
        model, refusal_dir, args.skip_begin, args.skip_end, args.scale_factor
    )
    print(f"Saving abliterated model to {args.output}...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
