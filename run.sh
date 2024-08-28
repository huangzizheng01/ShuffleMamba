python -m torch.distributed.launch --nproc_per_node=8 --use_env main_shuffle.py \
  --model mambar_base_patch16_224 \
  --num_workers 12 \
  --batch 256 --lr 5e-4 --weight-decay 0.1 \
  --clip-grad 3.0 --opt-betas 0.9 0.95 \
  --warmup-lr 5e-7 --min-lr 5e-6 --warmup-epochs 30 \
  --data-path your_data_path \
  --output_dir your_work_dir \
  --epochs 300 --input-size 224 --drop-path 0.5 --ssr 0.5 --dist-eval \
  --model-ema --model-ema-decay 0.9999 \
