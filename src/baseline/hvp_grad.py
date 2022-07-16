#! /usr/bin/env python3
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torch.autograd import grad
from torch.autograd.functional import vhp
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from .model import Net
from .utils import (
    load_weights,
    make_functional,
    calc_loss,
    grad_z
)


def s_test(
        x_test,
        y_test,
        model,
        i,
        samples_loader,
        device,
        damp=0.01,
        scale=25.0,
        loss_func="cross_entropy"):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, stochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        x_test: torch tensor, test data points, such as test images
        y_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        i: the sample number
        samples_loader: torch DataLoader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor

    Returns:
        h_estimate: list of torch tensors, s_test"""

    v = grad_z(x_test, y_test, model, device, loss_func=loss_func)
    h_estimate = v

    params, names = make_functional(model)
    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in params)

    # TODO: Dynamically set the recursion depth so that iterations stop once
    # h_estimate stabilises
    progress_bar = tqdm(samples_loader, desc=f"IHVP sample {i}")
    for i, (x_train, y_train) in enumerate(progress_bar):

        x_train, y_train = x_train.to(device), y_train.to(device)

        def f(*new_params):
            load_weights(model, names, new_params)
            out = model(x_train)
            loss = calc_loss(out, y_train, loss_func=loss_func)
            return loss

        hv = vhp(f, params, tuple(h_estimate), strict=True)[1]

        # Recursively calculate h_estimate
        with torch.no_grad():
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)
            ]

            if i % 100 == 0:
                norm = sum([h_.norm() for h_ in h_estimate])
                progress_bar.set_postfix({"est_norm": norm.item()})

    with torch.no_grad():
        load_weights(model, names, params, as_params=True)

    return h_estimate


def s_test_sample(
    model,
    x_test,
    y_test,
    train_loader,
    device,
    damp=0.01,
    scale=25,
    recursion_depth=5000,
    r=1,
    loss_func="cross_entropy",
):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        x_test: test image
        y_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""

    inverse_hvp = [
        torch.zeros_like(
            params,
            dtype=torch.float) for params in model.parameters()]

    for i in range(r):

        hessian_loader = DataLoader(
            train_loader.dataset,
            sampler=torch.utils.data.RandomSampler(
                train_loader.dataset, True, num_samples=recursion_depth
            ),
            batch_size=1,
            num_workers=4,
        )

        cur_estimate = s_test(
            x_test,
            y_test,
            model,
            i,
            hessian_loader,
            device,
            damp=damp,
            scale=scale,
            loss_func=loss_func,
        )

        with torch.no_grad():
            inverse_hvp = [
                old + (cur / scale) for old, cur in zip(inverse_hvp, cur_estimate)
            ]

    with torch.no_grad():
        inverse_hvp = [component / r for component in inverse_hvp]

    return inverse_hvp


def calc_influence_single(
    ckpt_dir,
    x_test,
    y_test,
    train_loader,
    device,
    recursion_depth,
    r,
    damp=0.01,
    scale=25,
    s_test_vec=None,
    time_logging=False,
    loss_func="cross_entropy",
):
    """Calculates the influences of all training data points on a single
    test dataset image.
    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated
    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided
    model = Net()
    checkpoint = torch.load(os.path.join(ckpt_dir, 'last'),
                            map_location='cpu')
    model.load_state_dict(checkpoint['model_states']['net'])
    model = model.to(device)

    if s_test_vec is None:
        s_test_vec = s_test_sample(
            model,
            x_test,
            y_test,
            train_loader,
            device,
            recursion_depth=recursion_depth,
            r=r,
            damp=damp,
            scale=scale,
            loss_func=loss_func,
        )

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in tqdm(range(train_dataset_size)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])

        grad_z_vec = grad_z(z, t, model, device)
        with torch.no_grad():
            tmp_influence = (
                -sum(
                    [
                        ####################
                        # TODO: potential bottle neck, takes 17% execution time
                        # torch.sum(k * j).data.cpu().numpy()
                        ####################
                        torch.sum(k * j).data
                        for k, j in zip(grad_z_vec, s_test_vec)
                    ]
                )
                / train_dataset_size
            )

        influences.append(tmp_influence)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]
    infl = [x.cpu().numpy().tolist() for x in influences]

    return infl, harmful.tolist(), helpful.tolist()
