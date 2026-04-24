import torch


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    '''
    Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
    reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
        the ground truths, producing a dict with keys "reward", "format_reward", and
        "answer_reward".
    rollout_responses: list[str] Rollouts from the policy. The length of this list is
        rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    repeated_ground_truths: list[str] The ground truths for the examples. The length of this
        list is rollout_batch_size, because the ground truth for each example is repeated
        group_size times.
    group_size: int Number of responses per question (group).
    advantage_eps: float Small constant to avoid division by zero in normalization.
    normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
        subtract only the group mean.

    Returns:
    tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
            response.
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    '''
    raw_rewards_dict = [reward_fn(response, truth) for response, truth in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([re_dict['reward'] for re_dict in raw_rewards_dict])
    raw_rewards = raw_rewards.reshape(-1, group_size) # [n_prompts_per_rollout_batch, group_size]

    advantages = raw_rewards - torch.mean(raw_rewards, -1, keepdim=True)
    if normalize_by_std:
        advantages = advantages / torch.std(raw_rewards, -1, keepdim=True)

    advantages = (advantages + advantage_eps).reshape(-1, )

    return advantages, raw_rewards.reshape(-1, ), {"normalize_by_std": normalize_by_std}