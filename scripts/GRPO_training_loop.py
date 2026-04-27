import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output
from cs336_alignment.compute_group_normalized_rewards import compute_group_normalized_rewards
from cs336_alignment.grpo_microbatch_train_step import grpo_microbatch_train_step


wandb.login()

project ='grpo_qwen_2p5_math'
device = 'cuda'

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
with wandb.init(project=project, config=config) as run:
    model.eval()
    # training loop implementation
    for it in tqdm(range(config["n_grpo_steps"]), desc="training"):
        # sample a batch of questions
        sample_idx = torch.randint(0, len(prompt_lst), (n_prompts_per_rollout_batch, )).tolist()
        repeted_prompt_lst = [prompt_lst[i] for i in sample_idx for _ in range(config["group_size"])]
        repeted_ground_truth_lst = [ground_truth_lst[i] for i in sample_idx for _ in range(config["group_size"])]
        response_lst = []
        # # get response
        # inputs = tokenizer(repeted_prompt_lst, return_tensors='pt', padding=True).to(device)
        # outputs = model.generate(
        #     inputs['input_ids'],
        #     attention_mask=inputs['attention_mask'],
        #     max_new_tokens=config["sampling_max_tokens"],
        #     min_new_tokens=config["sampling_min_tokens"],
        #     temperature=config["sampling_temperature"],
        #     top_p=1.0,
        #     do_sample=True,
        #     pad_token_id=tokenizer.pad_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        # )
        # # decode
        # response_lst = [tokenizer.decode(o[len(p):], skip_special_tokens=False) 
        #             for o, p in zip(outputs, inputs['input_ids'])]
        for prompt in repeted_prompt_lst:
            # get response
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=config["sampling_max_tokens"],
                min_new_tokens=config["sampling_min_tokens"],
                temperature=config["sampling_temperature"],
                top_p=1.0,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # decode
            response_lst.append(tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False))
        # tokenize_prompt_and_output
        tk_prop_out = tokenize_prompt_and_output(
            repeted_prompt_lst, 
            response_lst, 
            tokenizer
        )
        # compute advantages
        advantages, _, _ = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            response_lst,
            repeted_ground_truth_lst,
            config["group_size"],
            config["advantage_eps"],
            config["use_std_normalization"]
        )

        model.train()
        # n_train_steps_per_rollout_batch
        for ep in range(config["epochs_per_rollout_batch"]):
            # shuffle
            perm = torch.randperm(len(response_lst))
            shuffled_ids = tk_prop_out['input_ids'][perm]
            shuffled_labels = tk_prop_out['labels'][perm]
            shuffled_mask = tk_prop_out['response_mask'][perm]
            shuffled_advantages = advantages[perm]
            for i in range(config["gradient_accumulation_steps"]):
                input_ids = shuffled_ids[micro_batch_size*i:micro_batch_size*(i+1), ...].to(device)
                input_labels = shuffled_labels[micro_batch_size*i:micro_batch_size*(i+1), ...].to(device)
                input_response_mask = shuffled_mask[micro_batch_size*i:micro_batch_size*(i+1), ...].to(device)
                input_advantages = shuffled_advantages[micro_batch_size*i:micro_batch_size*(i+1)].to(device)
                # forward pass
                outputs = model(input_ids=input_ids)
                log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                policy_log_probs = log_probs.gather(2, input_labels.unsqueeze(-1)).squeeze(-1)
                _, _ = grpo_microbatch_train_step(
                    policy_log_probs,
                    input_response_mask,
                    config["gradient_accumulation_steps"],
                    config["loss_type"],
                    advantages=input_advantages
                )
            optimizer.step()
            optimizer.zero_grad()
        if it%50 == 0:
            # Save the model weights
            model.save_pretrained(save_directory=output_dir)
            tokenizer.save_pretrained(save_directory=output_dir)

