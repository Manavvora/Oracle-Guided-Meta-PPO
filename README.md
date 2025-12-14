# Oracle-Guided Meta-PPO

A scalable two-stage reinforcement learning framework for multi-agent budget-constrained decision making under partial observability. This repository contains the implementation for our paper currently under review at TMLR.

## Overview

Oracle-Guided Meta-PPO is a novel approach that addresses the challenge of training reinforcement learning policies for large-scale multi-agent systems with budget constraints and partial observability (POMDPs). The key insight is to leverage computationally tractable oracle policies (computed via value iteration on a surrogate MDP) to guide the training of a meta-policy that generalizes across diverse agent configurations.

### Key Contributions

1. **Two-Stage Framework**: Decomposes the problem into budget allocation and policy learning
2. **Scalability**: Demonstrates computational efficiency scaling to 100,000+ agents
3. **Generalization**: Meta-policy trained on a small subset of agents generalizes to unseen configurations
4. **Oracle Guidance**: Uses MDP-based oracle policies to accelerate POMDP policy learning

## Method

The proposed approach consists of three main stages:

### Stage 1: Budget Allocation via Random Forest
A Random Forest regressor learns to predict optimal per-agent budget allocations based on agent-specific features (degradation dynamics, costs, etc.). The model is trained on data generated from solving individual agent optimization problems.

### Stage 2: Oracle Policy Generation
For each agent-budget pair, an oracle policy is computed via value iteration on a surrogate MDP. This oracle provides guidance on when to take maintenance/recapitalization actions given full state observability.

### Stage 3: Oracle-Guided Meta-PPO Training
A PPO-based meta-policy is trained across multiple agent configurations. The key innovation is that the agent chooses between:
- **Action 0**: Follow the oracle policy's recommendation
- **Action 1**: Take an inspection action to reduce uncertainty

This hierarchical action space allows the learned policy to focus on the core challenge of POMDPs: deciding when to gather information.

## Repository Structure

```
Oracle-Guided-Meta-PPO/
├── infra_env/                          # Infrastructure Management Scenario
│   ├── env/                            # Environment definitions
│   │   ├── component_mdp_repair.py     # MDP formulation for components
│   │   ├── component_pomdp_repair.py   # POMDP formulation with belief tracking
│   │   ├── meta_ppo_env.py             # Meta-PPO environment wrapper
│   │   └── baseline_env.py             # Baseline environment
│   └── pomdp_solver/                   # Main algorithms
│       ├── random_forest.py            # RF model for budget prediction
│       ├── random_forest_budget_split.py   # Budget allocation via RF
│       ├── generate_oracle_policies.py     # Value iteration oracle generation
│       ├── oracle_guided_meta_ppo_train.py # Oracle-Guided Meta-PPO training
│       ├── oracle_guided_meta_ppo_test.py  # Oracle-Guided Meta-PPO testing
│       ├── oracle_guided_meta_ppo_optimal_budget_split.py  # Full pipeline
│       ├── vanilla_meta_ppo_train.py   # Baseline: Vanilla Meta-PPO
│       ├── vanilla_meta_ppo_test.py    # Baseline: Vanilla Meta-PPO testing
│       ├── realistic_baseline.py       # Baseline: Rule-based policy
│       ├── oracle_policy_test.py       # Baseline: Oracle-only policy
│       └── time_complexity.py          # Scalability experiments
│
├── etf_env/                            # ETF Risk Capital Management Scenario
│   ├── env/                            # Environment definitions
│   │   ├── etf_env.py                  # Multi-asset ETF environment
│   │   └── sub_etf_env.py              # Sub-environment definitions
│   ├── models/                         # Machine learning models
│   │   ├── random_forest.py            # Budget split model
│   │   └── random_forest_budget_split.py
│   ├── etf_oracle_guided_meta_ppo.py   # Oracle-Guided Meta-PPO for ETF
│   ├── etf_oracle_policy.py            # Oracle policy generation
│   ├── oracle_guided_meta_ppo_train_refactored.py
│   ├── oracle_guided_meta_ppo_test_refactored.py
│   ├── vanilla_meta_ppo_train.py       # Baseline comparison
│   ├── vanilla_meta_ppo_test.py
│   ├── generate_sp500_data.py          # Data generation utilities
│   └── baselinefin.py                  # Financial baseline configuration
│
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Oracle-Guided-Meta-PPO.git
cd Oracle-Guided-Meta-PPO

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Infrastructure Management Scenario

```bash
cd infra_env/pomdp_solver

# Step 1: Train Random Forest for budget prediction
python random_forest.py

# Step 2: Compute budget split for components
python random_forest_budget_split.py --num_components 1000

# Step 3: Generate oracle policies
python generate_oracle_policies_optimal_budget_split.py --num_components 1000

# Step 4: Train Oracle-Guided Meta-PPO
python oracle_guided_meta_ppo_train.py

# Step 5: Test the trained model
python oracle_guided_meta_ppo_optimal_budget_split.py --num_components 1000
```

### ETF Risk Capital Management Scenario

```bash
cd etf_env

# Train Oracle-Guided Meta-PPO
python etf_oracle_guided_meta_ppo.py --timesteps 100000

# Test the model
python oracle_guided_ppo_test.py
```

### Time Complexity Analysis

```bash
cd infra_env/pomdp_solver

# Run scalability experiments
python time_complexity.py
```

## Experiments

### Baselines

The following baselines are implemented for comparison:

1. **Vanilla Meta-PPO**: Standard meta-PPO without oracle guidance
2. **Oracle Policy**: Directly applying the MDP oracle policy to the POMDP
3. **Realistic Baseline**: Rule-based inspection and replacement policy
4. **Equal Budget Split**: Uniform budget allocation across agents

### Metrics

- **Time-to-Failure (TTF)**: Average operational lifetime of components/portfolios
- **Cost Incurred**: Total maintenance/action costs
- **Action Distribution**: Frequency of inspect/replace/no-action decisions

## Application Domains

### 1. Infrastructure Management
- **Problem**: Managing degradation of infrastructure components (bridges, roads, machinery)
- **State**: Component health (partially observable) and budget consumed
- **Actions**: No action, Inspect (reveals true health), Replace (restores health)
- **Objective**: Maximize component lifetime while staying within budget

### 2. ETF Risk Capital Management
- **Problem**: Managing risk capital across multiple financial assets
- **State**: Asset prices (observable) and risk capital levels (partially observable)
- **Actions**: No action, Inspect (reveals risk capital), Recapitalize (restores capital)
- **Objective**: Maximize portfolio survival while managing action budget

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon:
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) for environment interfaces

