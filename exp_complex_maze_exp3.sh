#!/bin/bash
export PYTHONPATH="$PATH:$PWD"

for i in {1234..1238}
do
  nohup python experiments_exploration/complex_maze/compute_estimates_along_path.py -lr 0.0005 -l_int 0.0 -exp_int 2.0 -exp_f 5 -p 10 -e_nb 3 -s $i &
  sleep 1
done

for i in {1234..1238}
do
  nohup python experiments_exploration/complex_maze/compute_estimates_along_path.py -lr 0.0005 -l_int 2.0 -exp_int 2.0 -exp_f 5 -p 10 -e_nb 3 -s $i &
  sleep 1
done
