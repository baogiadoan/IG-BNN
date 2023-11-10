# Evaluation

Evaluation code for the submitted paper

# Dependencies

- Pytorch 1.7.1
- torchvision 0.8.2

Above two packages can be installed from Pytorch website: https://pytorch.org, 

- cleverhans 4.0.0
  install using `pip`
  `pip install clerverhans`

# Train 

- Users might need to change the mode to run the bash script:
  `sudo chmod +x ./train.sh`
- Run the script:
  `bash ./train.sh`



# Test

- Users might need to change the mode to run the bash script:
  `sudo chmod +x ./test.sh`
- Run the script:
  `bash ./test.sh`

content inside:
```
CUDA_VISIBLE_DEVICES=0 python test_PGD.py \
  --model ./path_to_checkpoint.pth \  # need to set this to the path of the saved checkpoint
  --dataset stl \
  --num_particles 10 \
  --batch_size 50 \
  --PGD_steps 20
```

