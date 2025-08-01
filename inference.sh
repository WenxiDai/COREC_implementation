CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nproc_per_node 1 scripts/inference.py