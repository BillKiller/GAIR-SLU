word_embeding_list=(256 128 64)
topk_list=(5 4 3)
dropout=(0.4 0.1 0.2 0.3)
for w in ${word_embeding_list[@]};
do
        for d in ${dropout[@]};
        do
                for k in ${topk_list[@]};
                do
                        for x  in {1..3};
                        do
                            python train.py -wed $w -ehd 256 -aod 128  -dd=data/cais/ --topk $k -rs $x -bs 8  -dr $d -sd save_wed-${w}-topk${k}-drop${d}-seed${x} >> log-SMP-wed=${w}-topk=${k}-drop=${d}-seed=${x}.log
                        done
                done
        done
done
