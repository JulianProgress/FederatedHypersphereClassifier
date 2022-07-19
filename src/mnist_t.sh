CUDA_VISIBLE_DEVICES=3,4,5,7 python3 ./RHSCFedAvg_v2.py --seed 999 -or 0.3 -tr 0.1 -n 25 -b 128 -e 5 -r 100 -ad 'e5ood3tr1gt' -ed 'cuda:0' -gi 1 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose












#python3 ./FedAvg.py --seed 999 -n 8 -b 128 -e 5 -r 100 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5
#python3 ./FedAvg.py --seed 999 -n 8 -b 128 -e 40 -r 10 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5
#python3 ./FedAvg.py --seed 999 -n 8 -b 128 -e 5 -r 40 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 --include_pert
#python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 10 -r 50 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.05
#python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 10 -r 50 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.15
#python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 10 -r 50 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.2

#python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 30 -r 20 -rr 19 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.35
#python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 10 -r 50 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0

#CUDA_VISIBLE_DEVICES=0,1,2,7 python3 ./RHSCFedAvg.py --seed 999 -n 4 -b 128 -e 50 -r 100 -ad 'best' -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0.1 --verbose



#CUDA_VISIBLE_DEVICES=3,4,5,6 python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 20 -r 100 -ad 'unbalanced' -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose
#CUDA_VISIBLE_DEVICES=3,4,5,6 python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 40 -r 100 -ad 'unbalanced' -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose
#CUDA_VISIBLE_DEVICES=3,4,5,7 python3 ./RHSCFedAvg_v2.py --seed 999 -n 25 -b 128 -e 5 -r 100 -ad 'unbalanced2' -ed 'cuda:2' -gi -1 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose
#CUDA_VISIBLE_DEVICES=3,4,5,7 python3 ./RHSCFedAvg_v2.py --seed 999 -n 25 -b 128 -e 40 -r 100 -ad 'best5updatec' -ed 'cuda:2' -gi 1 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose



#CUDA_VISIBLE_DEVICES=3,4,5,6 python3 ./RHSCFedAvg.py --seed 999 -n 4 -b 128 -e 50 -r 100 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose
#CUDA_VISIBLE_DEVICES=3,4,5,6 python3 ./RHSCFedAvg.py --seed 999 -n 4 -b 128 -e 80 -r 50 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0.05 --verbose
#CUDA_VISIBLE_DEVICES=3,4,5,6 python3 ./RHSCFedAvg.py --seed 999 -n 4 -b 128 -e 80 -r 50 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0.1 --verbose

#python3 ./RHSCFedAvg_noMulti.py --seed 999 -n 4 -b 128 -e 10 -r 50 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.