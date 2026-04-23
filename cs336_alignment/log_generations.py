import torch
from vllm import LLM, SamplingParams
from typing import Callable
import json
import tqdm

def log_generations(
        vllm_model: LLM,
        prompts: list[str],
        ground_truth: list[float],
        problems: list[str],
        eval_sampling_params: SamplingParams,
        reward_fn: Callable[[str, str], dict[str, float]],
        training_name: str,
        step: int
) -> None:
    '''
    an be used to log generations from your model
    '''
    result_path = f'results/{training_name}_step{step}.jsonl'
    accu_txt_len = 0
    cor_len = 0
    cor_num = 0
    incor_len = 0
    incor_num = 0

    with open(result_path, 'a', encoding='utf-8') as r:
        for prompt, truth, problem in tqdm(zip(prompts, ground_truth, problems)):
            # inferring
            output = vllm_model.generate(prompt, eval_sampling_params)
            # parsing
            generated_text = output[0].outputs[0].text
            tokens_len = len(output.outputs[0].token_ids)
            # entropy
            for step_output in output.outputs[0].token_probs:
                logprob_entry = list(step_output.values())[0]
                log_prob = logprob_entry.logprob
                entropy += -log_prob.exp() * log_prob / tokens_len
            # recording
            reward = reward_fn(generated_text, truth)
            reward['problem'] = problem
            reward['output'] = generated_text
            reward['truth'] = truth
            reward['avg_token_entropy'] = entropy

            accu_txt_len += len(generated_text)
            if reward["reward"] == 1:
                cor_len += reward['avg_txt_len']
                cor_num += 1
            else:
                incor_len += reward['avg_txt_len']
                incor_num += 1

            # saving results
            json_str = json.dumps(reward, ensure_ascii=False) + '\n'
            r.write(json_str)
        
        summary_data = {
            "avg_txt_len": accu_txt_len / len(prompts),
            "avg_cor_txt_len": cor_len / cor_num,
            "avg_incor_txt_len": incor_len / incor_num,
        }
        # saving results
        json_str = json.dumps(summary_data, ensure_ascii=False) + '\n'
        r.write(json_str)