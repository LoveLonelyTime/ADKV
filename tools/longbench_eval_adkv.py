import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import ray
import toml
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from adkv.hf.llama_adkv_eval import ADKVConfig, LlamaForCausalLM_ADKV
from adkv.train.train_model import ADKVTrainModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_chat(tokenizer, prompt, model_name):
    if "Instruct" in model_name:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # else:
    #     prompt = f"[INST]{prompt}[/INST]"
    return prompt


def post_process(response, model_name):
    if "Instruct" in model_name:
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    # else:
    #     response = (
    #         response.split("(Document")[0]
    #         .split("\n\nQuestion")[0]
    #         .split("\n\nAnswer")[0]
    #         .split("(Passage")[0]
    #         .strip()
    #     )
    return response

# Ray remote worker
#   - requests exactly 1 GPU
#   - processes a contiguous slice of the dataset
#   - accumulates results in memory and returns a list of dicts


@ray.remote(num_gpus=1)
def pred_worker(
    worker_id: int,
    data_slice: list,
    model_name: str,
    calib_dir: str,
    task_info: dict
) -> list:
    """
    Run inference on *data_slice* using the single GPU assigned by Ray.

    Dependencies are imported at module level; only per-worker runtime state
    is initialised here.

    Returns a list of result dicts (held in memory).
    The caller is responsible for writing them to disk.
    """

    _SEED = 114514

    # Seed
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed(_SEED)
    np.random.seed(_SEED)
    random.seed(_SEED)

    # Kept in a nested function: if these attribute accesses appear in
    # pred_worker's top-level code, Ray's serializer resolves
    # torch.backends.cudnn as a submodule and fails to pickle it.
    def _configure_cudnn():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    _configure_cudnn()
    torch.cuda.manual_seed_all(_SEED)

    # Device
    # Ray sets CUDA_VISIBLE_DEVICES to expose only the assigned GPU,
    # so the logical device is always cuda:0 inside this worker process.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Model
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

    model, tokenizer = load_model(model_name, calib_dir)
    # Generation Task
    model.config.prefill_full_precision = True
    model.to(device)
    model.eval()

    max_length = model.config.max_position_embeddings

    # Inference loop
    results = []
    for json_obj in tqdm(data_slice, desc=f"worker-{worker_id}"):
        # Xnhyacinth/LongBench pre-splits the official prompt into
        # context + question + answer_prefix (it has no `input` column).
        # For samsum/triviaqa the answer prefix is already in `question`.
        prompt = json_obj["context"] + json_obj["question"]
        answer_prefix = json_obj["answer_prefix"]
        if answer_prefix and not prompt.rstrip().endswith(answer_prefix.rstrip()):
            prompt += answer_prefix

        # Truncate in the middle to fit max_length
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = (
                tokenizer.decode(
                    tokenized_prompt[:half], skip_special_tokens=True)
                + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            )
        if task_info["name"] not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompt,
            truncation=False,
            return_tensors="pt"
            # padding=True,
            # pad_to_multiple_of=256
        ).to(device)

        context_length = inputs.input_ids.shape[-1]

        if task_info["name"] == "samsum":
            # Prevent the model from endlessly repeating "\nDialogue"
            output = model.generate(
                **inputs,
                max_new_tokens=task_info["max_gen"],
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
            )[0]
        else:
            output = model.generate(
                **inputs,
                max_new_tokens=task_info["max_gen"],
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]

        pred = tokenizer.decode(
            output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        results.append({
            "pred": pred,
            "answers": json_obj["answers"],
            "all_classes": json_obj["all_classes"],
            "length": json_obj["length"],
        })

    return results


def run_task(model_name: str, calib_dir: str, task_info: dict):
    """
    Submit one (model_info, task_info) job.

    The dataset is split into contiguous chunks, one chunk per available GPU,
    and each chunk is processed by an independent Ray worker that owns one GPU.
    Each worker returns its results in memory.
    This function merges them and writes the final .jsonl file.
    """
    # Load data
    data = load_dataset('Xnhyacinth/LongBench',
                        task_info["name"], split='test')
    out_dir = f"longbench-pred/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{task_info['name']}.jsonl"

    data_all = list(data)
    num_gpus = int(ray.cluster_resources().get("GPU", 1))
    # Clamp so we never launch more workers than samples
    num_workers = min(num_gpus, len(data_all))

    # Contiguous slices: worker i gets data_all[start:end]
    chunk_size = (len(data_all) + num_workers - 1) // num_workers
    data_slices = [
        data_all[i * chunk_size: (i + 1) * chunk_size]
        for i in range(num_workers)
        if i * chunk_size < len(data_all)
    ]

    logger.info(
        f"[run_task] model={model_name} task={task_info['name']}\n"
        f"  total={len(data_all)} workers={len(data_slices)} chunk={chunk_size}"
    )

    # Submit Ray tasks
    futures = [
        pred_worker.remote(
            worker_id,
            data_slice,
            model_name,
            calib_dir,
            task_info
        )
        for worker_id, data_slice in enumerate(data_slices)
    ]

    # Collect results (in memory) and write once
    all_results = []
    for worker_results in ray.get(futures):  # list[list[dict]]
        all_results.extend(worker_results)

    with open(out_path, "w", encoding="utf-8") as f_out:
        for record in all_results:
            json.dump(record, f_out, ensure_ascii=False)
            f_out.write("\n")

    logger.info(f"[run_task] Done -> {out_path}  ({len(all_results)} records)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect calibration data for ADKV."
    )

    parser.add_argument('--model', type=str,
                        default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--calib_dir", type=str, default="calib")
    parser.add_argument("--task", type=str, default='narrativeqa')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ray.init()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tasks = toml.load(os.path.join(script_dir, "config/longbench_config.toml"))
    model_name = args.model

    os.makedirs("longbench-pred", exist_ok=True)

    run_task(model_name, args.calib_dir, tasks[args.task])

    ray.shutdown()
