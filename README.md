# Pytorch IF project
First run src/train.py to train your model, it will be saved to checkpoints
Then Run main.py to compute IF

## structure of project
├── src
│   ├── dataset.py
│   ├── hvp_grad.py
│   ├── model.py
│   ├── solver.py
│   ├── tracIn.py
│   ├── train.py
│   ├── utils.py
├── checkpoints
│   ├── config_dic
│   ├── 10000 ... 90000
│   ├── last
├── data
│   ├── cifar
├── README.md


## problems tracIn
- What should the save_step, max_iter be?

## problems Hvp
- r, recursion_depth, scale, damp in hvp_grad need to be fixed
    - see https://github.com/ryokamoi/pytorch_influence_functions/issues/2
