if [ ! -d "./run_log" ]; then
    mkdir ./run_log
fi
if [ ! -d "./run_log/log_test_win" ]; then
    mkdir ./run_log/log_test_win
fi
if [ ! -d "./run_log/log_test_win/ETTm1" ]; then
    mkdir ./run_log/log_test_win/ETTm1
fi
if [ ! -d "./run_log/log_test_win/ETTh1" ]; then
    mkdir ./run_log/log_test_win/ETTh1
fi
if [ ! -d "./run_log/log_test_win/ETTm2" ]; then
    mkdir ./run_log/log_test_win/ETTm2
fi

if [ ! -d "./run_log/log_test_win/ETTh2" ]; then
    mkdir ./run_log/log_test_win/ETTh2
fi
if [ ! -d "./run_log/log_test_win/electricity" ]; then
    mkdir ./run_log/log_test_win/electricity
fi

if [ ! -d "./run_log/log_test_win/Exchange" ]; then
    mkdir ./run_log/log_test_win/Exchange
fi

if [ ! -d "./run_log/log_test_win/Solar" ]; then
    mkdir ./run_log/log_test_win/Solar
fi

if [ ! -d "./run_log/log_test_win/weather" ]; then
    mkdir ./run_log/log_test_win/weather
fi

if [ ! -d "./run_log/log_test_win/Traffic" ]; then
    mkdir ./run_log/log_test_win/Traffic
fi

if [ ! -d "./run_log/log_test_win/PEMS03" ]; then
    mkdir ./run_log/log_test_win/PEMS03
fi

if [ ! -d "./run_log/log_test_win/PEMS04" ]; then
    mkdir ./run_log/log_test_win/PEMS04
fi

if [ ! -d "./run_log/log_test_win/PEMS07" ]; then
    mkdir ./run_log/log_test_win/PEMS07
fi
if [ ! -d "./run_log/log_test_win/PEMS08" ]; then
    mkdir ./run_log/log_test_win/PEMS08
fi



device=0

# finetune
python run.py \
    --is_training 1 \
    --load_pretrain 1 \
    --transfer_data ETTh2 \
    --transfer_root_path ./dataset/ETT-small/ \
    --transfer_data_path ETTh2.csv \
    --data ETTh2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --dropout 0.0 \
    --learning_rate 0.0001 \
    --gpu $device \
    > ./run_log/log_test_win/ETTh2/'GPHT_finetune'0.01.log 2>&1


# evalution
for pred in 96 192 336 720; do
  python run.py \
    --is_training 0 \
    --transfer_data ETTh2 \
    --transfer_root_path ./dataset/ETT-small/ \
    --transfer_data_path ETTh2.csv \
    --data ETTh2 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --ar_pred_len $pred \
    --gpu $device \
    > ./run_log/log_test_win/ETTh2/'GPHT_evalution'$pred_len'_'_0.01.log 2>&1

  done