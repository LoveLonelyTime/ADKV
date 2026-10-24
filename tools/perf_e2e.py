# Llama model with ADKV
import argparse
import logging
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer

from adkv.train.train_model import ADKVTrainModel
from adkv.hf.llama_adkv import LlamaForCausalLM_ADKV, ADKVConfig

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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--prompt_lenth", type=int, default=160)
    parser.add_argument("--output_length", type=int, default=338)
    parser.add_argument("--num_repeats", type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_model(args.model, args.calib_dir)
    model.cuda().eval()

    # Prepare prompts
    context = []
    for _ in range(args.batch_size):
        string = 't,' * (args.prompt_lenth // 2)
        context.append(string[:-1])

    inputs = tokenizer(context, return_tensors="pt").to('cuda')
    input_ids = inputs['input_ids']

    logger.info(
        f"bs: {args.batch_size}, seqlen: {input_ids.shape[1]} + {args.output_length}, model:{args.model}")

    # Run
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        torch.cuda.synchronize()
        st = time.time()
        for i in range(args.num_repeats):
            outputs = model.generate(
                **inputs, max_new_tokens=args.output_length)
        torch.cuda.synchronize()
        logger.info(
            f'used time: {(time.time() - st) / args.num_repeats * 1000} ms')
        used_mem = torch.cuda.max_memory_allocated()
        logger.info(f'peak mem: {used_mem / 1024 ** 3} GB')


if __name__ == "__main__":
    main()
