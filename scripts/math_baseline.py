from vllm import LLM, SamplingParams
from typing import Callable
import json
from tqdm import tqdm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truth: list[str],
    problems: list[str],
    eval_sampling_params: SamplingParams,
    result_path: str
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    with open(result_path, 'a', encoding='utf-8') as r:
        for prompt, truth, problem in tqdm(zip(prompts, ground_truth, problems), desc="处理结果"):
            # inferring
            output = vllm_model.generate(prompt, eval_sampling_params)
            # print(output)
            # parsing
            generated_text = output[0].outputs[0].text
            # recording
            reward = reward_fn(generated_text, truth)
            reward['problem'] = problem
            reward['output'] = generated_text
            reward['truth'] = truth
            # saving results
            json_str = json.dumps(reward, ensure_ascii=False) + '\n'
            r.write(json_str)
            # break


if __name__ == '__main__':
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, 
        stop=["</answer>"], include_stop_str_in_output=True
    )

    # Create an LLM.
    llm = LLM(model=r'models/qwen2p5_math')

    # Sample prompts.
    prop_path = r'cs336_alignment/prompts/r1_zero.prompt'
    with open(prop_path, 'r', encoding='utf_8') as f:
        prompt_template = f.read()

    # load the MATH validation examples 
    val_path = r'data/MATH/validation.jsonl'
    res_path = r'results/result_baseline.jsonl'

    prompt_lst = []
    problem_lst = []
    ground_truth_lst = []
    with open(val_path, 'r', encoding='utf_8') as f:
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
            problem_lst.append(item['problem'])

    # inferring
    evaluate_vllm(llm, r1_zero_reward_fn, 
                  prompt_lst, ground_truth_lst, problem_lst, 
                  sampling_params, res_path)
