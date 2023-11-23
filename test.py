import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_tts.model as module_model

from hw_tts.trainer import Trainer
from hw_tts.utils import ROOT_PATH
from hw_tts.utils.object_loading import get_dataloaders
from hw_tts.utils import MetricTracker
from hw_tts.utils.parse_config import ConfigParser
from hw_tts.utils.util import get_WaveGlow
from hw_tts.text import text_to_sequence
from hw_tts.waveglow.inference import inference

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"
sr = 22050

def main(config, txt_path, out_dir):
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    waveglow = get_WaveGlow()

    # prepare model for testing
    print(f"DEVICE: {device}")
    model = model.to(device)
    model.eval()

    with open(txt_path, 'r', encoding="utf-8") as f:
        texts = list(map(lambda x: x.strip(), f.readlines()))
    
    cleaners = ["english_cleaners"]
    encoded_texts = [text_to_sequence(text, cleaners) for text in texts]

    with torch.no_grad():
        for i, tokenized_text in enumerate(encoded_texts):
            for a, b, g in config["scale"]:
                text = torch.tensor(tokenized_text, device=device).unsqueeze(0)
                src_pos = torch.tensor([i + 1 for i in range(len(tokenized_text))], device=device).unsqueeze(0)
                outputs = model(text=text, 
                                src_pos=src_pos,
                                alpha=a, 
                                beta=b, 
                                gamma=g)
                
                inference(outputs["mel_predicted"].transpose(1, 2), waveglow, os.path.join(out_dir, f"audio_{i + 1}_a_{a}_b_{b}_g_{g}.wav"), sampling_rate=sr)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="./",
        type=str,
        help="output dir",
    )
    args.add_argument(
        "-t",
        "--txt",
        default="input.txt",
        type=str,
        help="path to input txt file",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set

    main(config, args.txt, args.output)
