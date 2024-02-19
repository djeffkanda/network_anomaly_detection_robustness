#!bash

ROOT_LOG=logs
LOGPATH=$ROOT_LOG/$MODEL/${DATASET}/
LOG_F_NAME=log_${ROB_METHOD}_${CONTAMINATION}_${ROB}_${WARMUP}

mkdir -p ${LOGPATH}
echo "log path created if not exists"

RUN_ARGS="-m=$MODEL -e=$EPOCH --dataset=$DATASET --dataset-path=$DATASET_PATH --n-runs=$N_RUN --batch-size=$BATCH_SIZE --batch-size-test=$BATCH_SIZE_TEST -lat=$LATENT --pct=$DATASET_PCT --rho=$CONTAMINATION --hold_out=$HOLDOUT --test_pct=$TESTPCT --lr=$LR --reg_n=$REG_N --reg_a=$REG_A --num_clusters=$N_CLUSTER --seed=$SEED $ROB --rob_method=$ROB_METHOD --warmup=$WARMUP $EVAL_TEST"

echo "running main.py with the following arguments:\n $RUN_ARGS"
nohup python ./main.py $RUN_ARGS > ${LOGPATH}/${LOG_F_NAME}.out 2> ${LOGPATH}/${LOG_F_NAME}.err < /dev/null &