import argparse

parser = argparse.ArgumentParser(description="Make abliterated models")
parser.add_argument(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Your model directory or huggingface model ID",
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    choices=["auto", "cuda", "cpu"],
    default="auto",
    help="Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU",
)
parser.add_argument(
    "--precision",
    "-p",
    type=str,
    choices=["fp16", "bf16", "fp32"],
    default="bf16",
    help="Precision to use for ablation, default is bf16",
)
parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
parser.add_argument(
    "--skip-begin",
    type=int,
    default=1,
    help="Number of layers to skip at the beginning. Defaults to 1 to avoid messing with the first layer",
)
parser.add_argument(
    "--skip-end", type=int, default=0, help="Number of layers to skip at the end"
)
parser.add_argument(
    "--layer-fraction",
    type=float,
    default=1,
    help="Fraction of layers to use for refusal_dir calculation",
)
parser.add_argument(
    "--scale-factor",
    type=float,
    default=1.0,
    help="Scale factor for ablation. Use a negative scale-factor to encourage refusal. If abliteration is not good, try to increase it a little bit",
)
parser.add_argument(
    "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
)
parser.add_argument(
    "--data-harmful", "-dhf", type=str, default=None, help="Harmful prompts file"
)
parser.add_argument(
    "--data-harmless", "-dhl", type=str, default=None, help="Harmless prompts file"
)
parser.add_argument(
    "--deccp",
    action="store_true",
    default=False,
    help="For Chinese models, in specific topics",
)
parser.add_argument(
    "--num-harmful", "-nhf", type=int, default=-1, help="Number of harmful calibrations"
)
parser.add_argument(
    "--num-harmless",
    "-nhl",
    type=int,
    default=-1,
    help="Number of harmless calibrations",
)

refusals = parser.add_mutually_exclusive_group()
refusals.add_argument(
    "--input-refusal", "-ir", type=str, default=None, help="Input tensor for refusal"
)
refusals.add_argument(
    "--output-refusal", "-or", type=str, default=None, help="Output tensor for refusal"
)

quant = parser.add_mutually_exclusive_group()
quant.add_argument(
    "--load-in-4bit",
    action="store_true",
    default=False,
    help="Load model in 4-bit precision using bitsandbytes",
)
quant.add_argument(
    "--load-in-8bit",
    action="store_true",
    default=False,
    help="Load model in 8-bit precision using bitsandbytes",
)
