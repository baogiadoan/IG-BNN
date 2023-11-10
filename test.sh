CUDA_VISIBLE_DEVICES=0 python test_PGD.py \
  --model ./path_to_checkpoint.pth \
  --dataset stl \
  --num_particles 10 \
  --batch_size 50 \
  --PGD_steps 20
