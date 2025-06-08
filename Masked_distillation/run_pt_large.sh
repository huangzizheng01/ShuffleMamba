# # For MambaMLP-B pretraining
# torchrun --standalone --nproc_per_node=8 main_pretrain.py \
#     --batch_size 128 \
#     --ssr 0.4 \
#     --model arm_base_pz16 \
#     --norm_pix_loss \
#     --epochs 300 \
#     --warmup_epochs 30 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path your_data_path \
#     --output_dir your_work_dir \


torchrun --standalone --nproc_per_node 8 main_pretrain.py \
    --batch_size 128 \
    --ssr 0.5 \
    --model arm_large_pz16 \
    --norm_pix_loss \
    --epochs 300 \
    --warmup_epochs 30 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path your_data_path \
    --output_dir your_work_dir \

# # For MambaMLP-H pretraining
# torchrun --standalone --nproc_per_node=8 main_pretrain.py \
#     --batch_size 128 \
#     --ssr 0.6 \
#     --model arm_huge_pz16 \
#     --norm_pix_loss \
#     --epochs 300 \
#     --warmup_epochs 30 \
#     --blr 1.5e-4 --weight_decay 0.05 \
#     --data_path your_data_path \
#     --output_dir your_work_dir \