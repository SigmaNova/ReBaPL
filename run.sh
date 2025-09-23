trainer=cocoop
dataset=$1
seed=$2

./scripts/${trainer}/base2new_train.sh $dataset $seed;
./scripts/${trainer}/base2new_test.sh $dataset $seed;