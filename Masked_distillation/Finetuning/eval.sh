torchrun --standalone --nproc_per_node 4 main_finetune.py \
    --model arm_base_pz16 --finetune your_ckpt \
    --global_pool True \
    --eval --dist_eval \
    --ft_scan v3 \
    --batch_size 256 --ema_decay 0.99992 \
    --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --data_path your_data_path \