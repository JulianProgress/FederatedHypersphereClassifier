
#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.4 -or 0.1 -v 0 -e 20 -r 150 -ad 'trial10' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --return_best --update_c

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.4 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial11' -l 0. -ed 'cuda:0' -gi 10 -rlr 5e-4 \
#-clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --return_best --update_c
#
#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.4 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial12' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-4 \
#-clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --return_best --update_c

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.4 -or 0.1 -v 0 -e 20 -r 150 -ad 'trial13' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --return_best --update_c

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.4 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial14' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --return_best --update_c

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.4 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial15' -l 0. -ed 'cuda:0' -gi 10 -rlr 5e-4 \
#-clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --return_best --update_c

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.4 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial16' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 10 -rg 0.9 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --return_best --update_c

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.4 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial17' -l 0. -ed 'cuda:0' -gi -1 -rlr 1e-3 \
#-rms 10 -rg 0.9 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --return_best --update_c
#
#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.4 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial18' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 10 -rg 0.9 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.5 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial19' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 10 -rg 0.9 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.5 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial20' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 15 -rg 0.9 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.2 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial21' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 10 -rg 0.9 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.2 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial22' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 20 -rg 0.9 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.5 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial23' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 10 -rg 0.95 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.5 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial24' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 5 -rg 0.99 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best

#CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
#-dp 1 -tr 0.1 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial22' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
#-rms 20 -rg 0.9 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best

CUDA_VISIBLE_DEVICES=1,2,3,4 python3 ../RHSCFedAvg_v3.py --seed 999 -m 'cifar10' \
-dp 1 -tr 0.1 -or 0.1 -v 0 -e 30 -r 150 -ad 'trial23' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-3 \
-rms 10 -rg 0.99 -clr 1e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-4 -i 5 -g 0. --verbose --update_c --return_best
