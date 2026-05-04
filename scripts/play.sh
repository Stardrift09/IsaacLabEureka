#!/bin/bash

TASK="PickItUp"
NUM_ENVS=20
VIDEO_LENGTH=1500

CHECKPOINTS=(
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_17-27-55_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_18-00-37_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_18-33-49_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_19-06-18_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_19-38-53_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_20-10-49_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_20-43-24_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_21-15-58_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_21-48-45_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_22-22-04_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_22-54-56_Run-0_tueilsy-st-13/model_1999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/pick_it_up/2026-04-14_23-28-39_Run-0_tueilsy-st-13/model_1550.pt"
)

# ---------- PLAY LOOP ----------
for CKPT in "${CHECKPOINTS[@]}"; do
    echo "===== Playing checkpoint: $CKPT ====="

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task="$TASK" \
        --num_envs=$NUM_ENVS \
        --video \
        --video_length=$VIDEO_LENGTH \
        --checkpoint="$CKPT"

    echo "===== Finished: $CKPT ====="
done

echo "===== All runs finished ====="