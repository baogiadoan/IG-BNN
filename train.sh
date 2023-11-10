dataset=stl  # choose the dataset cifar, stl
batchsize=100  # choose the batch size
ensemble=3  # choose the number of parameter particles
root=/data/datasets/CV  # the root for dataset
ig=5  # hyperparameter for IG
epochs=1000 # the number of training epochs
optim=adam  # choose the optimizer adam, sgd
lr=0.001  # learning rate
checkpoint=./checkpoint/stl_robust/  # the path to save or resume the checkpoint

CUDA_VISIBLE_DEVICES=0 python main.py \
  --batch_size ${batchsize} \
  --epochs ${epochs} \
  --dataset ${dataset} \
  --ensemble ${ensemble} \
  --root ${root} \
  --checkpoint ${checkpoint} \
  --wig ${ig} \
  --optim ${optim} \
  --lr ${lr} \
#   --resume  # this will resume the checkpoint for training