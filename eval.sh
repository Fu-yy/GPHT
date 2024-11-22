device=0
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
if [ ! -d "./run_log/log_test_win/pretrain" ]; then
    mkdir ./run_log/log_test_win/pretrain
fi
if [ ! -d "./run_log/log_test_win/eval" ]; then
    mkdir ./run_log/log_test_win/eval
fi
device=0

for data in ETTh1 ETTh2 ETTm1 ETTm2; do
  for pred in 96 192 336 720; do
    echo $data
    python run.py \
      --is_training 0 \
      --transfer_data $data \
      --transfer_root_path dataset/ETT-small/ \
      --transfer_data_path $data.csv \
      --ar_pred_len $pred \
      --gpu $device \
      > ./run_log/log_test_win/eval/'GPTH_eval_'$data'_'$pred'_'0.01.log 2>&1

  done
done

for data in exchange_rate weather traffic electricity; do
  for pred in 96 192 336 720; do
    echo $data
    python run.py \
      --is_training 0 \
      --transfer_data $data \
      --transfer_root_path dataset/$data/ \
      --transfer_data_path $data.csv \
      --ar_pred_len $pred \
      --gpu $device \
      > ./run_log/log_test_win/eval/'GPTH_eval_'$data'_'$pred'_'0.01.log 2>&1

  done
done
