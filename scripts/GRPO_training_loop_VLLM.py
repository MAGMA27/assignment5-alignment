import torch
import os
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.utils as utils
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from unittest.mock import patch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import datetime
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.grpo_microbatch_train_step import grpo_microbatch_train_step

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

def load_policy_into_vllm_instance(policy, llm):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# ---------- main ----------
def main(project, config, device_train, device_rollout, sampling_params):
    # load model
    model_train = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device_train)
    model_train.train()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # init vllm
    llm = init_vllm(model_path, device_rollout, 2026, gpu_memory_utilization=config["gpu_memory_utilization"])

    # init optimizer
    optimizer = torch.optim.AdamW(
        model_train.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    # init wandb
    wandb.login()
    wandb.init(project=project, config=config)

    # data preparation
    prompt_path = r'cs336_alignment/prompts/r1_zero.prompt'
    with open(prompt_path, 'r', encoding='utf_8') as f:
        prompt_template = f.read()

    data_path = r'data/MATH/train.jsonl'
    prompt_lst = []
    # problem_lst = []
    ground_truth_lst = []
    with open(data_path, 'r', encoding='utf_8') as f:
        for line in f:
            line = line.strip() 
            if not line:
                continue
            # parsing
            item = json.loads(line)
            question = item['problem']
            # operating
            prompt = prompt_template.replace('{question}', question)
            prompt_lst.append(prompt)
            ground_truth_lst.append(item['answer'])
            # problem_lst.append(item['problem'])
    n_prompts_per_rollout_batch = config["rollout_batch_size"] // config["group_size"]
    micro_batch_size = config["train_batch_size"] // config["gradient_accumulation_steps"]

    # training loop implementation
    for it in tqdm(range(config["n_grpo_steps"]), desc="training"):
        # sample a batch of questions
        sample_idx = torch.randint(0, len(prompt_lst), (n_prompts_per_rollout_batch, )).tolist()
        repeated_prompt_lst = [prompt_lst[i] for i in sample_idx for _ in range(config["group_size"])]
        repeated_ground_truth_lst = [ground_truth_lst[i] for i in sample_idx for _ in range(config["group_size"])]
        response_lst = []
        
        # # iterative generating microbatch
        # for i in tqdm(range(n_prompts_per_rollout_batch), desc=f"generating rollouts for it {it}"):
        #     prompt = repeated_prompt_lst[i*config["group_size"]:(i+1)*config["group_size"]]
        #     outputs = llm.generate(prompt, sampling_params)

        #     for seq in outputs:
        #         generated_text = seq.outputs[0].text
        #         response_lst.append(generated_text)

        # generating all of a microbatch
        outputs = llm.generate(repeated_prompt_lst, sampling_params)
        for seq in outputs:
            generated_text = seq.outputs[0].text
            response_lst.append(generated_text)
            
        # tokenize_prompt_and_output
        tk_prop_out = tokenize_prompt_and_output(
            repeated_prompt_lst, 
            response_lst, 
            tokenizer
        )
        # compute advantages
        advantages, _, _ = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            response_lst,
            repeated_ground_truth_lst,
            config["group_size"],
            config["advantage_eps"],
            config["use_std_normalization"]
        )
        # n_train_steps_per_rollout_batch
        for ep in range(config["epochs_per_rollout_batch"]):
            loss = 0
            # shuffle
            perm = torch.randperm(len(response_lst))
            shuffled_ids = tk_prop_out['input_ids'][perm]
            shuffled_labels = tk_prop_out['labels'][perm]
            shuffled_mask = tk_prop_out['response_mask'][perm]
            shuffled_advantages = advantages[perm]
            for i in range(config["gradient_accumulation_steps"]):
                input_ids = shuffled_ids[micro_batch_size*i:micro_batch_size*(i+1), ...].to(device_train)
                input_labels = shuffled_labels[micro_batch_size*i:micro_batch_size*(i+1), ...].to(device_train)
                input_response_mask = shuffled_mask[micro_batch_size*i:micro_batch_size*(i+1), ...].to(device_train)
                input_advantages = shuffled_advantages[micro_batch_size*i:micro_batch_size*(i+1)].to(device_train)
                # forward pass
                outputs = model_train(input_ids=input_ids)
                log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                policy_log_probs = log_probs.gather(2, input_labels.unsqueeze(-1)).squeeze(-1)
                pgloss, _ = grpo_microbatch_train_step(
                    policy_log_probs,
                    input_response_mask,
                    config["gradient_accumulation_steps"],
                    config["loss_type"],
                    advantages=input_advantages
                )
                loss += pgloss
                
            utils.clip_grad_value_(model_train.parameters(), clip_value=1.0)
            optimizer.step()
            optimizer.zero_grad()
        wandb.log({"it": it , "loss": loss})
        print(f"timestamp:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, it: {it}, loss: {loss}")
        load_policy_into_vllm_instance(model_train, llm) 
            
        if it%50 == 0:
            # Save the model weights
            ckpt_path = os.path.join(output_dir, f"grpo_ckpt_it{it}")
            os.makedirs(ckpt_path, exist_ok=True)
            model_train.save_pretrained(save_directory=ckpt_path)
            tokenizer.save_pretrained(save_directory=ckpt_path)

    final_path = os.path.join(output_dir, "latest")
    os.makedirs(final_path, exist_ok=True)
    model_train.save_pretrained(save_directory=final_path)
    tokenizer.save_pretrained(save_directory=final_path)
    print(f"Training finished. Final model saved to {final_path}")
    wandb.finish()

if __name__ == "__main__":
    config = {
        "n_grpo_steps": 200,
        "learning_rate": 1e-5,
        "advantage_eps": 1e-6,
        "rollout_batch_size": 256,
        "group_size": 8,
        "sampling_temperature": 1.0,
        "sampling_min_tokens": 4, # As in Expiter, disallow empty string responses
        "sampling_max_tokens": 1024,
        "epochs_per_rollout_batch": 1, # On-policy
        "train_batch_size": 256, # On-policy
        "gradient_accumulation_steps": 128, # microbatch size is 2, will fit on H100
        "gpu_memory_utilization": 0.85,
        "loss_type": "reinforce_with_baseline",
        "use_std_normalization": True,
    }

    model_path = r'Qwen/Qwen2.5-Math-1.5B'
    output_dir = r'results/grpo_qwen_2p5_math'

    project ='grpo_qwen_2p5_math'
    device_train = 'cuda:0'
    device_rollout = 'cuda:1'

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024,
        stop=["</answer>"], include_stop_str_in_output=True
    )

    main(project, config, device_train, device_rollout, sampling_params)