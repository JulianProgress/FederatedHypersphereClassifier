# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./RHSCFedAvg_v3.py --seed 999 -m 'cicids' -dp 1.0 -tr 0.3 -or 0.1 -v 0 -e 20 -r 50 -ad 'trial30' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-4 -clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0. --verbose --update_c
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./RHSCFedAvg_v3_bk.py --seed 999 -m 'cicids' -dp 1.0 -tr 0.3 -or 0.1 -v 0 -e 20 -r 50 -ad 'trial31' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-5 -clr 1e-5 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0. --verbose --update_c
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./RHSCFedAvg_v3_bk.py --seed 999 -m 'cicids' -dp 1.0 -tr 0.3 -or 0.1 -v 0 -e 20 -r 50 -ad 'trial32' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-4 -clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0. —verbose —update_c -return_best

#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../RHSCFedAvg_v3_bk.py --seed 999 -m 'cicids' -dp 1.0 -or 0.1 \
#-v 0 -e 20 -r 100 -ad 'trial4' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-4 -clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 \
#-lr 1e-5 -i 5 -g 0.0 --update_c -tr 0.2 -nc 3
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../RHSCFedAvg_v3_bk.py --seed 999 -m 'cicids' -dp 1.0 -or 0.1 \
#-v 0 -e 20 -r 100 -ad 'trial4' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-4 -clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 \
#-lr 1e-5 -i 5 -g 0.0 --update_c -tr 0.4 -nc 3
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../RHSCFedAvg_v3_bk.py --seed 999 -m 'cicids' -dp 1.0 -or 0.1 \
#-v 0 -e 20 -r 100 -ad 'trial4' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-4 -clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 \
#-lr 1e-5 -i 5 -g 0.0 --update_c -tr 0.8 -nc 3
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../RHSCFedAvg_v3_bk.py --seed 999 -m 'cicids' -dp 1.0 -or 0.1 \
#-v 0 -e 20 -r 100 -ad 'trial4' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-4 -clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 \
#-lr 1e-5 -i 5 -g 0.0 --update_c -tr 1.0 -nc 3

#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../RHSCFedAvg_v3_bk.py --seed 999 -m 'cicids' -dp 1.0 -or 0.1 \
#-v 0 -e 20 -r 100 -ad 'trial2' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-4 -clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 \
#-lr 1e-5 -i 5 -g 0.0 --update_c -tr 0.6 -nc 3
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../RHSCFedAvg_v3_bk.py --seed 999 -m 'cicids' -dp 1.0 -or 0.1 \
#-v 0 -e 20 -r 100 -ad 'trial3' -l 0. -ed 'cuda:0' -gi 10 -rlr 5e-4 -clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 \
#-lr 1e-5 -i 5 -g 0.0 --update_c -tr 0.6 -nc 3

#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../RHSCFedAvg_v3_bk.py --seed 999 -m 'cicids' -dp 1.0 -or 0.1 \
#-v 0 -e 20 -r 100 -ad 'trial5' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-4 -clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 \
#-lr 1e-5 -i 5 -g 0.0 --update_c -tr 1.0 -nc 3

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../RHSCFedAvg_v3_bk.py --seed 999 -m 'cicids' -dp 1.0 -or 0.1 \
-v 0 -e 20 -r 100 -ad 'trial5' -l 0. -ed 'cuda:0' -gi 10 -rlr 1e-4 -clr 5e-4 -p 10 -ps 10 -pss 0.001 -pd 2 \
-lr 1e-5 -i 5 -g 0.0 --update_c -tr 0.6 -nc 3

