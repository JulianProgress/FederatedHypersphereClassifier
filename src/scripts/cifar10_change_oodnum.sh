#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.3 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial23' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 10 -rg 0.99 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best

CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' -no 2 \
-dp 1 -tr 0.3 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial40_2' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
-rms 10 -rg 0.99 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' -no 3 \
#-dp 1 -tr 0.3 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial40_3' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 10 -rg 0.99 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best
#
#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' -no 4 \
#-dp 1 -tr 0.3 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial40_4' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 10 -rg 0.99 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best

CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' -no 5 \
-dp 1 -tr 0.3 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial40_5' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
-rms 10 -rg 0.99 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' -no 6 \
#-dp 1 -tr 0.3 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial40_6' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 10 -rg 0.99 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best



#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ./RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.3 -or 0.1 -v 0 -e 20 -r 150 -ad 'trial10' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --return_best --update_c