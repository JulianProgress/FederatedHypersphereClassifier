#CUDA_VISIBLE_DEVICES=0,1,2,7 python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 50 -r 100 -ad 'unbalanced2' -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose

CUDA_VISIBLE_DEVICES=0,1,2,6 python3 ./RHSCFedAvg_v2.py --seed 999 -or 0.1 -tr 0.2 -n 25 -b 128 -e 40 -r 100 -ad 'ood1tr2' -ed 'cuda:0' -gi -1 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose
CUDA_VISIBLE_DEVICES=0,1,2,6 python3 ./RHSCFedAvg_v2.py --seed 999 -or 0.1 -tr 0.3 -n 25 -b 128 -e 40 -r 100 -ad 'ood1tr3' -ed 'cuda:0' -gi -1 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose
CUDA_VISIBLE_DEVICES=0,1,2,6 python3 ./RHSCFedAvg_v2.py --seed 999 -or 0.1 -tr 0.5 -n 25 -b 128 -e 40 -r 100 -ad 'ood1tr5' -ed 'cuda:0' -gi -1 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose
CUDA_VISIBLE_DEVICES=0,1,2,6 python3 ./RHSCFedAvg_v2.py --seed 999 -or 0.3 -tr 0.2 -n 25 -b 128 -e 40 -r 100 -ad 'ood3tr2' -ed 'cuda:0' -gi -1 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose
CUDA_VISIBLE_DEVICES=0,1,2,6 python3 ./RHSCFedAvg_v2.py --seed 999 -or 0.3 -tr 0.3 -n 25 -b 128 -e 40 -r 100 -ad 'ood3tr3' -ed 'cuda:0' -gi -1 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose
CUDA_VISIBLE_DEVICES=0,1,2,6 python3 ./RHSCFedAvg_v2.py --seed 999 -or 0.3 -tr 0.5 -n 25 -b 128 -e 40 -r 100 -ad 'ood3tr5' -ed 'cuda:0' -gi -1 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose
