"""ELYZAを利用したスクリプトです.

Notes
-----
実行方法: `elyza.py --device=cuda -vv --loop -n=7b-fast`
"""
import logging
import random
import sys
from argparse import ArgumentParser
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import internal.data_directory as data_directory
import internal.log_settings as log_settings

_logger = logging.getLogger(__name__)
# 利用想定しているモデル名とパス
_MODEL_ID = {
    "7b": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    "7b-fast": "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
}
# デフォルトの設定
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"


class _DeviceOption(Enum):
    """デバイスの選択肢."""

    CPU = "cpu"
    CUDA = "cuda"


class _RunConfig(BaseModel):
    """スクリプト実行のためのオプション."""

    prompt: str  # 画像生成に利用するプロンプト
    loop: bool  # 単一処理ではなく繰り返し実行する

    network_name: str  # 画像生成に利用するモデル名
    seed: int  # 生成に利用するシード値
    device: _DeviceOption  # 利用するデバイス

    verbose: int  # ログレベル


def _main() -> None:
    """スクリプトのエントリポイント."""
    # 実行時引数の読み込み
    config = _parse_args()

    # ログ設定
    loglevel = log_settings.get_loglevel_from_verbosity(config.verbose)
    script_filepath = Path(__file__)
    log_filepath = data_directory.get_interim_dir() / f"{script_filepath.stem}.log"
    log_filepath.parent.mkdir(exist_ok=True)
    log_settings.setup_lib_logger(log_filepath, loglevel=loglevel)
    log_settings.setup_logger(_logger, log_filepath, loglevel=loglevel)
    _logger.info(config)

    # モデルの設定
    model_name = _MODEL_ID[config.network_name]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    if torch.cuda.is_available() and (config.device == _DeviceOption.CUDA):
        model = model.to(_DeviceOption.CUDA.value)

    loop_num = 100 if config.loop else 1
    for _ in range(loop_num):
        # プロンプトの準備
        query = config.prompt
        if config.loop:
            query = input("Enter your question (or 'quit' to stop): ")
            if query.lower() == "quit":
                break
        prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
            bos_token=tokenizer.bos_token,
            b_inst=B_INST,
            system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
            prompt=query,
            e_inst=E_INST,
        )

        if config.seed > 0:
            _set_seed(config.seed)
        with torch.no_grad():
            token_ids = tokenizer.encode(
                prompt, add_special_tokens=False, return_tensors="pt"
            )
            output_ids = model.generate(
                token_ids.to(model.device),
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True
        )
        _logger.info(output)


def _parse_args() -> _RunConfig:
    """スクリプト実行のための引数を読み込む."""
    parser = ArgumentParser(description="Text2Imageを実行する.")

    parser.add_argument(
        "-p",
        "--prompt",
        default="quit",
        help="画像生成に利用するプロンプト.",
    )
    parser.add_argument(
        "-l",
        "--loop",
        action="store_true",
        help="繰り返し行う.",
    )

    parser.add_argument(
        "-n",
        "--network-name",
        choices=list(_MODEL_ID.keys()),
        default=list(_MODEL_ID.keys())[0],
        help="利用するモデル名.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=-1,
        help="ランダムシード値. `-1`の場合はランダムとなる.",
    )
    parser.add_argument(
        "--device",
        default=_DeviceOption.CPU.value,
        choices=[v.value for v in _DeviceOption],
        help="利用するデバイス.",
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="詳細メッセージのレベルを設定."
    )

    args = parser.parse_args()
    config = _RunConfig(**vars(args))

    return config


def _set_seed(seed_value: int):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        _logger.exception(e)
        sys.exit(1)
