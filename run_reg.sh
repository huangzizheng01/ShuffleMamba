python -m torch.distributed.launch --nproc_per_node=8 --use_env main_reg.py \
  --model mambar_base_patch16_224 \
  --num_workers 16 \
  --batch 128 --lr 5e-4 --weight-decay 0.05 \
  --data-path your_data_path \
  --output_dir your_work_dir/128pt \
  --reprob 0.0 --smoothing 0.0 --repeated-aug --ThreeAugment \
  --epochs 300 --input-size 128 --drop-path 0.1 --ssr 0.1 --dist-eval \

python -m torch.distributed.launch --nproc_per_node=8 --use_env main_reg.py \
  --model mambar_base_patch16_224 \
  --batch 128 --lr 2e-4 --weight-decay 0.05 \
  --data-path your_data_path \
  --finetune your_work_dir/128pt/checkpoint.pth \
  --output_dir your_work_dir/224mid \
  --reprob 0.0 --smoothing 0.0  --repeated-aug --ThreeAugment \
  --epochs 100 --input-size 224 --drop-path 0.4 --ssr 0.1 --dist-eval

python -m torch.distributed.launch --nproc_per_node=8 --use_env main_reg.py \
  --model mambar_base_patch16_224 \
  --batch 64 --lr 1e-5 --weight-decay 0.1 \
  --data-path your_data_path \
  --finetune your_work_dir/224mid \
  --output_dir your_work_dir/224ft \
  --reprob 0.0 --smoothing 0.1 --no-repeated-aug \
  --aa rand-m9-mstd0.5-inc1 --eval-crop-ratio 1.0 \
  --epochs 20 --input-size 224 --drop-path 0.4 --ssr 0.1 --dist-eval
