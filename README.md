This is released source code for the paper

[Bayesian Learning with Information Gain Provably Bounds Risk for a Robust Adversarial Defense](https://proceedings.mlr.press/v162/doan22a.html)

>@inproceedings{doan2022bayesian,
>  title={Bayesian Learning with Information Gain Provably Bounds Risk for a Robust Adversarial Defense},
>  author={Doan, Bao Gia and Abbasnejad, Ehsan M and Shi, Javen Qinfeng and Ranasinghe, Damith C},
>  booktitle={International Conference on Machine Learning},
>  pages={5309--5323},
>  year={2022},
>  organization={PMLR}
>}


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

