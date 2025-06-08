# # For MambaMLP-B Finetune
# torchrun --standalone --nproc_per_node 4 main_finetune.py \
#     --model arm_base_pz16 \
#     --finetune your_ckpt \
#     --ft_scan v3 \
#     --epochs 60 --global_pool True \
#     --batch_size 256 \
#     --blr 3e-4 --layer_decay 0.65 --ema_decay 0.9999 \
#     --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#     --weight_decay 0.05 --drop_path 0.1 --ssr 0. \
#     --dist_eval --data_path your_data_path \
#     --output_dir your_work_dir  \


torchrun --standalone --nproc_per_node 4 main_finetune.py \
    --model arm_large_pz16 --finetune your_ckpt \
    --ft_scan v3 \
    --epochs 50 --global_pool True \
    --batch_size 128 \
    --blr 3e-4 --layer_decay 0.75 --ema_decay 0.9999 \
    --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --weight_decay 0.05 --drop_path 0.3 --ssr 0. \
    --dist_eval --data_path your_data_path \
    --output_dir your_work_dir \

# # For MambaMLP-B Finetune
# torchrun --standalone --nproc_per_node 4 main_finetune.py \
#     --model arm_huge_pz16 --finetune your_ckpt \
#     --ft_scan v3 \
#     --epochs 50 --global_pool True \
#     --batch_size 128 \
#     --blr 3e-4 --layer_decay 0.75 --ema_decay 0.9999 \
#     --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#     --weight_decay 0.05 --drop_path 0.3 --ssr 0.0 \
#     --dist_eval --data_path your_data_path \
#     --output_dir your_work_dir \