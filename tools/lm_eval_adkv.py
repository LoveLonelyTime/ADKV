import torch
import argparse
import logging
from pathlib import Path
from transformers import AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from adkv.train.train_model import ADKVTrainModel
from adkv.hf.llama_adkv_eval import LlamaForCausalLM_ADKV, ADKVConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str, calib_dir: str):
    calib_dir = Path(calib_dir)

    model = LlamaForCausalLM_ADKV.from_pretrained(
        model_path,
        dtype=torch.float16,
        trust_remote_code=True
    )

    adkv_config = []
    layer_num = model.config.num_hidden_layers
    hidden_size = model.config.num_key_value_heads * model.config.head_dim

    # Create default ADKV parameters
    for layer_idx in range(layer_num):
        init_param = torch.randn(hidden_size)
        adkv_config_layer = (ADKVConfig(init_param, init_param, init_param, init_param, init_param),
                             ADKVConfig(init_param, init_param, init_param, init_param, init_param))
        adkv_config.append(adkv_config_layer)

    # Load ADKV parameters
    for layer_idx in range(layer_num):
        k_model = ADKVTrainModel(hidden_size, truncation=32).to('cpu')
        k_model.load_state_dict(torch.load(
            calib_dir / f"ckpt_k_{layer_idx}.pt"))

        v_model = ADKVTrainModel(hidden_size, truncation=32).to('cpu')
        v_model.load_state_dict(torch.load(
            calib_dir / f"ckpt_v_{layer_idx}.pt"))

        # Key
        adkv_config[layer_idx][0].zp_init = k_model.init_zp.detach()
        adkv_config[layer_idx][0].scale_init = k_model.get_sc().detach()
        adkv_config[layer_idx][0].alpha = k_model.get_alpha().detach()
        adkv_config[layer_idx][0].beta = k_model.get_beta().detach()
        adkv_config[layer_idx][0].quant_target = k_model.target.detach()

        # Value
        adkv_config[layer_idx][1].zp_init = v_model.init_zp.detach()
        adkv_config[layer_idx][1].scale_init = v_model.get_sc().detach()
        adkv_config[layer_idx][1].alpha = v_model.get_alpha().detach()
        adkv_config[layer_idx][1].beta = v_model.get_beta().detach()
        adkv_config[layer_idx][1].quant_target = v_model.target.detach()

    model.load_adkv_config(adkv_config)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect calibration data for ADKV."
    )

    parser.add_argument('--model', type=str,
                        default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--calib_dir", type=str, default="calib")
    parser.add_argument("--task", type=str, default='winogrande')
    parser.add_argument(
        "--prefill_full_precision", action="store_true",
        help="Attend with full-precision K/V during prefill (contract A). "
             "Default: attend from the quantized cache (contract B).")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(args.model, args.calib_dir)

    # lm-eval's HFLM calls `model(input_ids)` directly, so the flag cannot be
    # passed as a forward kwarg.  LlamaForCausalLM_ADKV.forward falls back to
    # this config attribute when prefill_full_precision is not given.
    model.config.prefill_full_precision = args.prefill_full_precision
    # Make sure the ADKV cache is actually built during evaluation.
    model.config.use_cache = True

    model.to(device)
    model.eval()

    lm = HFLM(pretrained=model, tokenizer=tokenizer)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[args.task]
    )

    logger.info(make_table(results))


if __name__ == "__main__":
    main()
