import os
import argparse
import numpy as np
import json
from tqdm import tqdm


def get_transition_probabilities(price_history: np.ndarray):

    returns = np.divide(
        price_history[1:] - price_history[:-1], 
        price_history[:-1], 
        out=np.zeros_like(price_history[1:]), 
        where=price_history[:-1]!=0
    )

    capital_losses = np.abs(returns[returns < 0]) * 100
    
    if len(capital_losses) == 0:
        return np.array([]), np.array([]), 1.0


    max_loss = np.ceil(np.max(capital_losses))
    bins = np.arange(0, max_loss + 2)
    hist, _ = np.histogram(capital_losses, bins=bins)
    

    probabilities = hist / len(capital_losses)
    loss_values = np.arange(len(probabilities))
    
    prob_no_loss = 1.0 - np.sum(probabilities)

    return loss_values, probabilities, prob_no_loss

def run_value_iteration(
    train_data_path: str,
    output_dir: str,
    total_budget: float,
    n_components: int,
    capital_levels: int = 101,
    budget_levels: int = 51,
    discount_factor: float = 0.99,
    conv_threshold: float = 1e-4
):

    print("--- orcale ---")
    

    train_v = np.load(train_data_path)
    if n_components > train_v.shape[1]:
        print(f" n_components ({n_components}) ")
        n_components = train_v.shape[1]
    
    all_returns = []
    for i in range(n_components):

        price_series = train_v[:, i]
        valid_prices = price_series[price_series > 0]
        if len(valid_prices) < 2:
            continue
        returns = (valid_prices[1:] - valid_prices[:-1]) / valid_prices[:-1]
        all_returns.append(returns)
        
    if not all_returns:
        raise ValueError("no enough")

    avg_returns = np.concatenate(all_returns)
    loss_values, loss_probs, prob_no_loss = get_transition_probabilities(avg_returns)

    print("step2...")
    value_function = np.zeros((capital_levels, budget_levels))
    policy = np.zeros((capital_levels, budget_levels), dtype=int)
    budget_step = total_budget / (budget_levels - 1) if budget_levels > 1 else 0
    repair_cost = 10.0 
    repair_cost_steps = int(np.round(repair_cost / budget_step)) if budget_step > 0 else budget_levels


    print("step3")
    iteration = 0
    while True:
        iteration += 1
        delta = 0
        v_old = value_function.copy()
        for k in range(1, capital_levels):
            for c in range(budget_levels):
                expected_future_value_defer = 0
                for loss, prob in zip(loss_values, loss_probs):
                    next_k = max(0, k - int(loss))
                    expected_future_value_defer += prob * v_old[next_k, c]
                expected_future_value_defer += prob_no_loss * v_old[k, c]
                q_value_defer = 1.0 + discount_factor * expected_future_value_defer

                if c >= repair_cost_steps:
                    next_c_repair = c - repair_cost_steps
                    expected_future_value_repair = v_old[capital_levels - 1, next_c_repair]
                    q_value_repair = 1.0 + discount_factor * expected_future_value_repair
                else:
                    q_value_repair = -np.inf
                
                best_value = max(q_value_defer, q_value_repair)
                value_function[k, c] = best_value
                delta = max(delta, abs(v_old[k, c] - best_value))
        
        if delta < conv_threshold:
            print(f"{iteration} ")
            break
        if iteration > 2000:
            print("max steps")
            break


    print("step4")
    for k in range(1, capital_levels):
        for c in range(budget_levels):
            expected_future_value_defer = 0
            for loss, prob in zip(loss_values, loss_probs):
                next_k = max(0, k - int(loss))
                expected_future_value_defer += prob * value_function[next_k, c]
            expected_future_value_defer += prob_no_loss * value_function[k, c]
            q_value_defer = 1.0 + discount_factor * expected_future_value_defer

            if c >= repair_cost_steps:
                next_c_repair = c - repair_cost_steps
                expected_future_value_repair = value_function[capital_levels - 1, next_c_repair]
                q_value_repair = 1.0 + discount_factor * expected_future_value_repair
            else:
                q_value_repair = -np.inf

            policy[k, c] = 0 if q_value_defer >= q_value_repair else 2

    
    print(f"output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    policy_path = os.path.join(output_dir, "oracle_policy.npy")
    params_path = os.path.join(output_dir, "oracle_params.json")
    
    np.save(policy_path, policy)
    
 
    derived_threshold = 0
    if policy.shape[1] > 0:
        for k in range(1, capital_levels):
            # 检查预算最充足时的情况
            if policy[k, -1] == 0: 
                derived_threshold = k
                break
    print(f"{derived_threshold}")
    

    params = {
        "best_repair_threshold": float(derived_threshold),
        "budget_step": budget_step, 
        "total_budget": total_budget
    }
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
        
    print(f"✅  {policy_path}")
    print(f"✅  {params_path} (best_repair_threshold)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an oracle policy using Value Iteration.")
    parser.add_argument('--train_data_path', type=str, default="data/V_train.npy")
    parser.add_argument('--output_dir', type=str, default="results/oracle_policy")
    parser.add_argument('--n_components', type=int, default=50, help="Number of components from training data to use for estimating transitions.")
    parser.add_argument('--total_budget', type=float, default=200.0, help="The maximum budget for a single component, used for discretization.")
    
    args = parser.parse_args()
    
    run_value_iteration(
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        total_budget=args.total_budget,
        n_components=args.n_components
    )