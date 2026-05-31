#!/bin/bash
# train_rsl_rl.sh
# Script to train RSL_RL tasks with multiple seeds

# ---------- CONFIGURATION ----------
TASK="TestPickItUp"
MAX_ITERS=1500        # maximum training iterations
SEEDS=(0 4 5 6 7 8 42 123 456)    # list of seeds to run

# ---------- TRAINING LOOP ----------
for SEED in "${SEEDS[@]}"; do
    echo "===== Training with seed $SEED ====="
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task="$TASK" \
        --headless \
        --max_iterations=$MAX_ITERS \
        --seed=$SEED \
        --resume \
        --checkpoint=/home/shaotongchen/workspace_eureka/IsaacLabEureka/IsaacLab/logs/rsl_rl/test_pick_it_up/2026-03-20_22-21-02/model_800.pt
    echo "===== Finished training with seed $SEED ====="
done

echo "===== All training runs finished ====="