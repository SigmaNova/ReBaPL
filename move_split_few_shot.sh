cd ~/datasets/split_fewshot
for dataset in eurosat caltech-101 food-101 dtd
do 
    mkdir -p ${dataset}
    cp -r ~/datasets/${dataset}/split_fewshot ${dataset}/

done