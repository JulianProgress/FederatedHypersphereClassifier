#python3 ./FedAvg.py --seed 999 -n 8 -b 128 -e 5 -r 100 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5
#python3 ./FedAvg.py --seed 999 -n 8 -b 128 -e 40 -r 10 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5
#python3 ./FedAvg.py --seed 999 -n 8 -b 128 -e 5 -r 40 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 --include_pert
python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 5 -r 100 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.1

#python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 5 -r 100 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.1
#python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 5 -r 100 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.1
#python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 5 -r 100 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.1
#python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 5 -r 100 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.1
#python3 ./RHSCFedAvg.py --seed 999 -n 8 -b 128 -e 5 -r 100 -ed 'cuda:2' -p 10 -ps 10 -pss 0.001 -pd 2 -lr 1e-5 -i 5 -g 0.1


