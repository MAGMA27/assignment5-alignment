import os
import torch
import torch.nn.utils as utils
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
import wandb
import json
import tqdm
from typing import Callable
import torch.multiprocessing as mp
from queue import Empty
import datetime

from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------- dataset ----------
class TokenSequenceDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        first_key = next(iter(self.data))
        return self.data[first_key].shape[0]

    def __getitem__(self, idx):
        input_ids = self.data['input_ids'][idx, :]
        labels = self.data['labels'][idx, :]
        response_mask = self.data['response_mask'][idx, :]
        return input_ids, labels, response_mask

# ---------- evaluate ----------
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truth: list[str],
    eval_sampling_params: SamplingParams,
) -> tuple[float, float, float]:
    format_reward = 0.0
    answer_reward = 0.0
    reward = 0.0
    for prompt, truth in tqdm(zip(prompts, ground_truth), desc="evaluating"):
        output = vllm_model.generate([prompt], eval_sampling_params)
        generated_text = output[0].outputs[0].text
        reward_dict = reward_fn(generated_text, truth)
        format_reward += reward_dict['format_reward']
        answer_reward += reward_dict['answer_reward']
        reward += reward_dict['reward']
    n = len(prompts)
    return format_reward / n, answer_reward / n, reward / n

# ---------- vLLM ----------
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_state_dict_into_vllm(state_dict_cpu, llm):
    # 去掉 '_orig_mod.' 前缀
    cleaned_state_dict = {}
    for key, value in state_dict_cpu.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
        else:
            new_key = key
        cleaned_state_dict[new_key] = value
    
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(cleaned_state_dict.items())
# ---------- evaluation_process ----------
def evaluation_process(
    model_path: str,
    val_prompts: list[str],
    val_ground_truths: list[str],
    sampling_params: SamplingParams,
    ckpt_queue: mp.Queue,     # (step, state_dict_cpu)
    result_queue: mp.Queue,   # (step, format_reward, answer_reward, reward)
    device: str = "cuda:1",
    seed: int = 2026,
):
    # os.environ["CUDA_VISIBLE_DEVICES"] = device.split(':')[-1]
    llm = init_vllm(model_path, device, seed, gpu_memory_utilization=0.8)
    print("[Eval Process] vLLM initialized, waiting for checkpoints...")
    while True:
        item = ckpt_queue.get()
        if item == "STOP":
            break
        step, state_dict_cpu = item
        print(f"[Eval Process] Received state_dict for step {step}")
        load_state_dict_into_vllm(state_dict_cpu, llm)
        fmt_r, ans_r, rew = evaluate_vllm(llm, r1_zero_reward_fn, val_prompts, val_ground_truths, sampling_params)
        result_queue.put((step, fmt_r, ans_r, rew))
        print(f"[Eval Process] Step {step} evaluated: reward={rew:.4f}")

# ---------- training ----------
def main():
    mp.set_start_method('spawn', force=True)

    config = {
        "n_sft_steps": 200,
        "learning_rate": 1e-5,
        "train_batch_size": 256,
        "gradient_accumulation_steps": 128,
    }
    model_path = r'Qwen/Qwen2.5-Math-1.5B'
    output_dir = r'results/sft_qwen_2p5_math'
    os.makedirs(output_dir, exist_ok=True)

    # device
    device_train = "cuda:0"
    device_eval = "cuda:1"
    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device_train)
    model = torch.compile(model)
    model.gradient_checkpointing_enable()
    model.train()
    # init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # ---------- data preparation ----------
    data_path = r'data/MATH/sft.jsonl'
    prompt_lst, response_lst = [], []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt_lst.append(item['prompt'])
            response_lst.append(item['response'])
    # tokenize
    data = tokenize_prompt_and_output(prompt_lst, response_lst, tokenizer)
    dataset = TokenSequenceDataset(data)
    micro_batch_size = config['train_batch_size'] // config['gradient_accumulation_steps']
    dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True)

    # ---------- validation data preparation ----------
    prop_path = r'cs336_alignment/prompts/r1_zero.prompt'
    with open(prop_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    val_path = r'data/MATH/validation.jsonl'
    val_prompts, val_ground_truths= [], []
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question = item['problem']
            prompt = prompt_template.replace('{question}', question)
            val_prompts.append(prompt)
            val_ground_truths.append(item['answer'])

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True
    )

    # creating queue
    ckpt_queue = mp.Queue()
    result_queue = mp.Queue()

    # start evaluation process
    eval_proc = mp.Process(
        target=evaluation_process,
        args=(model_path, val_prompts, val_ground_truths, sampling_params,
              ckpt_queue, result_queue, device_eval, 2026)
    )
    eval_proc.start()

    # ---------- wandb init ----------
    wandb.login()
    wandb.init(project='sft_qwen_2p5_math', config=config)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    train_step = 0
    eval_step = 0
    step_loss = 0
    it = 0

    # ---------- training loop ----------
    for ep in range(100):
        for input_ids, labels, response_mask in dataloader:
            input_ids = input_ids.to(device_train)
            labels = labels.to(device_train)
            response_mask = response_mask.to(device_train)
    
            # forward pass
            log_probs = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
            loss, metadata = sft_microbatch_train_step(log_probs['log_probs'], response_mask, config['gradient_accumulation_steps'])
            wandb.log({"it": it, "training_loss": loss})
            print(f'timestamp:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, it:{it}, training_loss:{loss}')
            step_loss += loss
    
            # gradient accumulation
            if (it + 1) % config['gradient_accumulation_steps'] == 0:
                # gradient clipping with value 1
                utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                optimizer.step()
                optimizer.zero_grad()
    
                wandb.log({
                    "train_step": train_step,
                    "step_loss": step_loss
                })
                print(f'timestamp:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, train_step:{train_step}, step_loss:{step_loss}')
    
                if (train_step + 1) % 50 == 0:
                    # 将模型 state_dict 移到 CPU（减少 GPU 内存占用并便于序列化）
                    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                    # 将检查点放入队列（传递 state_dict）
                    ckpt_queue.put((train_step, state_dict_cpu))
                    print(f"[Main] Sent checkpoint step {train_step} to eval process.")
    
                    # 非阻塞检查是否有评估结果返回
                    try:
                        step, fmt_r, ans_r, rew = result_queue.get_nowait()
                        eval_step += 1
                        wandb.log({
                            "eval_step": eval_step,
                            "format_reward": fmt_r,
                            "answer_reward": ans_r,
                            "reward": rew
                        })
                        print(f"[Main] Received eval result for step {step}: reward={rew:.4f}")
                    except Empty:
                        pass
    
                    # checkpointing
                    ckpt_path = os.path.join(output_dir, f"sft_ckpt_step{train_step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    model.save_pretrained(save_directory=ckpt_path)
                    tokenizer.save_pretrained(save_directory=ckpt_path)
                    print(f"saving checkpoint to {ckpt_path}")
    
                train_step += 1
                step_loss = 0
            it += 1
            if train_step >= config['n_sft_steps']:
                break
        if train_step >= config['n_sft_steps']:
            break

    # 训练结束，发送停止信号并等待评估进程结束
    ckpt_queue.put("STOP")
    eval_proc.join()

    final_path = os.path.join(output_dir, "latest")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(save_directory=final_path)
    tokenizer.save_pretrained(save_directory=final_path)
    print(f"Training finished. Final model saved to {final_path}")
    wandb.finish()

if __name__ == "__main__":
    main()