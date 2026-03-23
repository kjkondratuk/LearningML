# Module 04: Alignment -- RLHF & DPO

## Learning Objectives

- Understand the Bradley-Terry preference model and reward modeling
- Implement PPO for language model alignment
- Derive and implement Direct Preference Optimization (DPO)
- Understand the KL-constrained optimization framework connecting RLHF and DPO

## Key Papers

- **Ouyang et al. (2022).** "Training language models to follow instructions with human feedback" (InstructGPT). arXiv:2203.02155
- **Rafailov et al. (2023).** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290
- **Schulman et al. (2017).** "Proximal Policy Optimization Algorithms." arXiv:1707.06347

## Theoretical Background

### Bradley-Terry Model

Given two completions y_w (preferred) and y_l (rejected), the probability that y_w is
preferred under the Bradley-Terry model is:

    P(y_w > y_l) = sigma(r(y_w) - r(y_l))

### Reward Model Training

The reward model is trained to maximize:

    L_RM = -E[log sigma(r(y_w) - r(y_l))]

### From RLHF to DPO

RLHF optimizes: max_pi E[r(y)] - beta * KL(pi || pi_ref)

The optimal policy is: pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(y)/beta)

DPO eliminates the reward model by substituting the optimal policy back:

    L_DPO = -E[log sigma(beta * (log pi(y_w)/pi_ref(y_w) - log pi(y_l)/pi_ref(y_l)))]

> **Math Callout:** Bradley-Terry preference model, policy gradient theorem (Sutton &
> Barto ch. 13), KL-constrained optimization. Resource: Anthropic "Constitutional AI"
> paper, Hugging Face RLHF blog.

## Exercises

1. `bradley_terry_probability` -- Preference probability
2. `reward_model_loss` -- Reward model training loss
3. `ppo_clipped_objective` -- PPO clipped surrogate objective
4. `dpo_loss` -- Direct Preference Optimization loss
5. `kl_penalty` -- KL divergence penalty for RLHF
