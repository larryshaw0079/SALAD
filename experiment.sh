#!/bin/bash

data_files=$(ls ./data/kpi/series*)
label_rates=(0.0 0.1 1.0)
seeds=(2016 2017 2018 2019 2020)

specified_label_rate=0
specified_seed=0

function usage() {
  echo '-m [Model]'
  echo '-l [Label rate] Specify the label rate'
  echo '-s [Seed] Specify the random seed'
  echo '-h [Print this help information]'
}

run_experiment() {
  python3 run_baseline.py --data $1 --model $2 --label $3 --seed $4
}

while getopts m:l:s:h OPT; do
  case $OPT in
  m)
    model=$OPTARG
    ;;
  l)
    specified_label_rate=1
    label_rate=$OPTARG
    ;;
  s)
    specified_seed=1
    seed=$OPTARG
    ;;
  h)
    usage
    exit 0
    ;;
  \?)
    usage
    exit 1
    ;;
  esac
done

echo '============================================================='
echo '|                      Session Start                        |'
echo '============================================================='

echo '[INFO] Model: '$model
echo '[INFO] Specified label rate: '$specified_label_rate
echo '[INFO] Specified seed: '$specified_seed

for dir in ${data_files[*]}; do
  echo '[INFO] Training '$model 'on '${dir##*/}' ...'
  if [[ $specified_label_rate == 1 ]]; then
    echo '[INFO] Using specified label rate '$label_rate
    if [[ $specified_seed == 1 ]]; then
      echo '[INFO] Using specified random seed '$seed
      run_experiment $dir $model $label_rate $seed
    else
      for seed in ${seeds[*]}; do
        echo '[INFO] Enumerate random seeds, current is '$seed
        run_experiment $dir $model $label_rate $seed
      done
    fi
  else
    for label_rate in ${label_rates[*]}; do
      echo '[INFO] Enumerate label rates, current is '$label_rate
      if [[ $specified_seed == 1 ]]; then
        echo '[INFO] Using specified random seed '$seed
        run_experiment $dir $model $label_rate $seed
      else
        for seed in ${seeds[*]}; do
          echo '[INFO] Enumerate random seeds, current is '$seed
          run_experiment $dir $model $label_rate $seed
        done
      fi
    done
  fi
done

echo '============================================================='
echo '|                        Session End                        |'
echo '============================================================='
