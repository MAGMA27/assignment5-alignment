import torch
from torch.utils.data import Dataset, DataLoader
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
import json
import wandb


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
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

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


class TokenSequenceDataset(Dataset):
    def __init__(self, data:dict):
        self.data = data

    def __len__(self):
        return next(iter(data.items())).shape[0]

    def __getitem__(self, idx):
        input_ids = data['input_ids'][idx, :]
        labels = data['labels'][idx, :]
        response_mask = data['response_mask'][idx, :]
        return input_ids, labels, response_mask


if __name__ == '__main__':
    wandb.login()

    project ='grpo_qwen_2p5_math'
    device = 'cuda'
    run_idx = 1

    config = {
        "n_sft_steps": 200,
        "learning_rate": 1e-5,
        "train_batch_size": 256, # On-policy
        "gradient_accumulation_steps": 128, # microbatch size is 2, will fit on H100
    }

    model_path = r'models/qwen2p5_math'
    output_dir = r'results/grpo_qwen_2p5_math'

    # load model
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # data preparation
    data_path = r'data/MATH/sft.jsonl'
    prompt_lst = []
    response_lst = []
    ground_truth_lst = []
    with open(data_path, 'r', encoding='utf_8') as f:
        for line in f:
            line = line.strip() 
            if not line:
                continue
            # parsing
            item = json.loads(line)
            # operating
            prompt_lst.append(item['prompt'])
            response_lst.append(item['response'])
            ground_truth_lst.append(item['ground_truth'])

    data = tokenize_prompt_and_output(prompt_lst, response_lst, tokenizer)
    dataset = TokenSequenceDataset(data)
    micro_batch_size = config['train_batch_size']//config['gradient_accumulation_steps']
    data_loader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True)

    wandb.init(
            project=project,
            config=config
        )
    # Setup wandb metrics
    wandb.define_metric("train_step") # the x‑axis for training
    # wandb.define_metric("eval_step") # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    # wandb.define_metric("eval/*", step_metric="eval_step")
    train_step = 0
    model.train()
    for it, (input_ids, labels, response_mask) in enumerate(data_loader):
        # Forward pass.
        log_probs = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
        loss, metadata = sft_microbatch_train_step(log_probs['log_probs'], response_mask, config['gradient_accumulation_steps'])

        if (it + 1) % config['gradient_accumulation_steps'] == 0:
            # Update weights every `gradient_accumulation_steps` batches.
            optimizer.step()
            # Zero gradients every `gradient_accumulation_steps` batches.
            optimizer.zero_grad()

            # Save the model weights
            if (train_step + 1) % 50 == 0:
                output_dir_ckpt = f'{output_dir}\\{project}_run{run_idx}_step{train_step}.pt'
                model.save_pretrained(save_directory=output_dir_ckpt)
                tokenizer.save_pretrained(save_directory=output_dir_ckpt)

            wandb.log({"train_step":train_step, "training_loss":loss, "token_entropy":log_probs['token_entropy']})

            if train_step+1 >= config['n_sft_steps']:
                break

            train_step += 1

    output_latest_dir = f'{output_dir}\\latest\\{project}_run{run_idx}_step{train_step}.pt'
    model.save_pretrained(save_directory=output_latest_dir)
    tokenizer.save_pretrained(save_directory=output_latest_dir)