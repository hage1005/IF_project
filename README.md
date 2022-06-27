# Pytorch IF project
First run src/train.py to train your model, it will be saved to checkpoints\
Then Run main.py to compute IF


## problems with tracIn
- What should the save_step, max_iter be?

## problemswith  Hvp
- r, recursion_depth, scale, damp in hvp_grad need to be fixed
    - see https://github.com/ryokamoi/pytorch_influence_functions/issues/2

## Debug commands
x = torch.cuda.FloatTensor([[-1],[0],[1],[-1],[0],[1]])
y = torch.cuda.LongTensor([0,0,0,1,1,1])
self._influence_model(x,y)

import numpy as np
idx = np.argwhere(train_labels.cpu() == 3)
self._influence_model(train_inputs, train_labels)[idx]
np.where(self._influence_model(train_inputs, train_labels).squeeze().cpu() > 0.2)
