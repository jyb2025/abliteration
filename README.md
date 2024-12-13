# Abliteration

## Usage

**Clone the repositoty**:

```shell
git clone https://github.com/Orion-zhen/abliteration.git
cd abliteration
```

**Install dependencies**:

```shell
pip install -r requirements.txt
```

**Make your abliterations**:

```shell
python abliteration.py -m <path_to_your_model> -o <output_dir>
```

Now your model will be abliterated and saved to `<output_dir>`. Once it finishes, you can immediately chat with your abliterated model in the terminal.

**Full arguments**:

```shell
usage: abliterate.py [-h] --model MODEL [--device DEVICE] --output OUTPUT [--no-chat] [--load-in-4bit | --load-in-8bit | --awq]

Make abliterated models

options:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Your model directory or huggingface model ID
  --device DEVICE, -d DEVICE
                        Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU
  --output OUTPUT, -o OUTPUT
                        Output directory
  --no-chat             Do not chat with model after abliteration
  --load-in-4bit        Load model in 4-bit precision using bitsandbytes
  --load-in-8bit        Load model in 8-bit precision using bitsandbytes
  --awq                 Load awq model
```

## Credits

[Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
