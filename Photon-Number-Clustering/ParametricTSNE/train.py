import datetime
import json
from math import log2
from typing import Optional
import os

import torch

from torch import Tensor, tensor, eye, device, ones, isnan, zeros, log, nn
from torch import max as torch_max
from numpy import array
from numpy import save as np_save
from tqdm import tqdm

from .utils import get_random_string, entropy, distance_functions
from .model import Network





def get_optimized_p_cond(X_high: tensor,
                        target_entropy : float,
                        EPS : Tensor,
                        dist_func : str = 'eucl',
                        tol : float = 1e-5,
                        max_iter : int = 300,
                        min_allowed_sig_sq : float = 0,
                        max_allowed_sig_sq : float = 1000,
                        dev : str = 'cpu') -> Optional[tensor]:
    """
    Calculate conditional probability matrix optimized by binary search.
    

    Parameters
    ----------
    X_high : tensor 
        High dimensional data (initial samples).
    target_entropy : float
        Target entropy to define sigma_i associated to evey sample.
    dist_func : str
        Name of the distance metric used in distance evaluation. 
    tol : float
        Tolerance used in definition of sigma_i^2. 
        Iterations stop when the difference between the expected entropy and the computed entropy is less than `tol`.
    max_iter : int
        Maximum number of iteration to find sigma_i^2. 
    min_allowed_sig_sq : float
        Minimum value of sigma_i^2 evaluated.
    max_allowed_sig_sq : float
        Maximum value of sigma_i^2 evaluated.
    dev : str
        Name of the device used in torch ('cpu', 'cuda', ...).


    Returns
    -------
    p_cond : tensor
        Conditional probability matrix.
    """
    
    n_points = X_high.size(0)

    # Calculating distance matrix with the given distance function
    dist_f = distance_functions[dist_func]
    distances = dist_f(X_high)
    diag_mask = (1 - eye(n_points)).to(device(dev))

    # Initializing sigmas
    min_sigma_sq = (min_allowed_sig_sq + 1e-20) * ones(n_points).to(device(dev))
    max_sigma_sq = max_allowed_sig_sq * ones(n_points).to(device(dev))
    sq_sigmas = (min_sigma_sq + max_sigma_sq) / 2

    # Computing conditional probability matrix from distance matrix
    p_cond = get_p_cond(distances = distances, 
                        sigmas_sq = sq_sigmas, 
                        mask = diag_mask,
                        EPS = EPS)

    # Making a vector of differences between target entropy and entropies for all rows in p_cond
    ent_diff = entropy(p_cond) - target_entropy

    # Binary search ends when all entropies match the target entropy
    finished = ent_diff.abs() < tol

    curr_iter = 0
    while not finished.all().item():

        if curr_iter >= max_iter:
            print("Warning! Exceeded max_iter.", flush=True)
            return p_cond
        
        pos_diff = (ent_diff > 0).float()
        neg_diff = (ent_diff <= 0).float()

        max_sigma_sq = pos_diff * sq_sigmas + neg_diff * max_sigma_sq
        min_sigma_sq = pos_diff * min_sigma_sq + neg_diff * sq_sigmas

        sq_sigmas = finished.logical_not() * (min_sigma_sq + max_sigma_sq) / 2 + finished * sq_sigmas
        p_cond = get_p_cond(distances = distances, 
                            sigmas_sq = sq_sigmas, 
                            mask = diag_mask,
                            EPS = EPS)
        ent_diff = entropy(p_cond) - target_entropy
        finished = ent_diff.abs() < tol
        curr_iter += 1

    if isnan(ent_diff.max()):
        print("Warning! Entropy is nan. Discarding batch", flush=True)
        return
    
    return p_cond


def get_p_cond(distances : tensor, 
               sigmas_sq : tensor, 
               mask : tensor,
               EPS : Tensor) -> tensor:
    """
    Calculates conditional probability distribution given distances and squared sigmas.

    Parameters
    ----------
    distances : tensor 
        Matrix of squared distances ||x_i - x_j||^2
    sigmas_sq : tensor
        Row vector of squared sigma for each row in distances
    mask : tensor
        A mask tensor to set diagonal elements to zero 

    Returns
    -------
    Conditional probability matrix : tensor
        Conditional probability matrix
    """
    logits = -distances / (2 * torch_max(sigmas_sq, EPS).view(-1, 1))
    logits.exp_()
    masked_exp_logits = logits * mask
    normalization = torch_max(masked_exp_logits.sum(1), EPS).unsqueeze(1)
    return masked_exp_logits / normalization + 1e-10


def get_q_joint(X_low: tensor, 
                dist_func: str, 
                EPS : Tensor,
                alpha: int = 1) -> tensor:
    """
    Calculates the joint probability matrix in embedding space (low-dimensional space).

    Parameters
    ----------
    distr_cond : tensor 
        Conditional probability matrix.
    dist_func : str
        Name of the distance metric used in distance evaluation. 
    alpha : int
        Number of degrees of freedom in t-distribution

    Returns
    -------
    Joint distribution matrix : tensor
        Joint probability matrix in low-dimensional space. All values in it sum up to 1. 
    """
    n_points = X_low.size(0)
    mask = (-eye(n_points) + 1).to(X_low.device)
    dist_f = distance_functions[dist_func]
    distances = dist_f(X_low) / alpha
    q_joint = (1 + distances).pow(-(1 + alpha) / 2) * mask
    q_joint /= q_joint.sum()
    return torch_max(q_joint, EPS)


def get_sym_joint(cond_prob: tensor, 
                  EPS : Tensor) -> tensor:
    """
    Make a joint probability distribution out of conditional distribution based on symmetric SNE.
    Small values are set to fixed epsilon.

    Parameters
    ----------
    distr_cond : tensor 
        Conditional probability matrix.

    Returns
    -------
    Joint distribution matrix : tensor
        Joint probability matrix. All values in it sum up to 1.    
    """
    n_points = cond_prob.size(0)
    distr_joint = (cond_prob + cond_prob.t()) / (2 * n_points)
    return torch_max(distr_joint, EPS)


def loss_function_KL(p_joint : Tensor, 
                    q_joint : Tensor,
                    EPS : Tensor) -> Tensor:
    """
    Calculates KLDiv between joint distributions in original and embedding space.

    Parameters
    ----------
    p_joint : tensor 
        Joint probability matrix of high-dimensional space.
    q_joint : tensor
        Joint probability matrix of low-dimensional space.

    Returns
    -------
    KLDiv : Tensor
        Kullback-Leibler divergence
    """
    return (p_joint * log((p_joint + EPS) / (q_joint + EPS))).sum()


def loss_function_MSE(X_high : Tensor,
                    X_reconst : Tensor):
    loss = nn.MSELoss()
    return loss(X_high, X_reconst) / torch.norm(X_high)
    

def loss_total(loss_KL,
            loss_MSE,
            alpha_loss : float = 1):

    return loss_MSE +  alpha_loss * loss_KL

    


def train_ptsne(X_high,
                dim_emb : int,
                perplexity : Optional[int],
                n_epochs : int,
                dev : str = 'cpu',
                save_dir_path : str = '',
                early_exaggeration : int = 4,
                early_exaggeration_constant : int = 12,
                learning_rate : float = 0.002,
                alpha_loss : float = 1,
                batch_size : int = 1024,
                dist_func_name : str = 'euc',
                bin_search_tol : float = 1e-5,
                bin_search_max_iter : int = 300,
                min_allowed_sig_sq : float = 0,
                max_allowed_sig_sq : float = 1000,
                verbose : bool = False) -> None:
    """
    Fits a parametric t-SNE model and optionally saves it to the desired directory.
    Fits either regular or multi-scale t-SNE

    Parameters
    ----------
    X_high : np.array
        Dataset used to train the network.
    dim_emb : int
        Number of dimension of the low-dimensional representation.
    perplexity : int | None
        Perplexity of a model. If passed None, multi-scale parametric t-SNE
        model will be trained
    n_epochs : int
        Number of epochs for training.
    dev : str
        Device to run torch tensors ('cpu', 'cuda', ...)
    save_dir_path : str
        Path where to save the results of the training. 
    early_exaggeration : int
        Number of first training cycles in which exaggeration will be applied.
    early_exaggeration_constant : int
        Constant by which p_joint is multiplied in early exaggeration.
    learning_rate : float
        Learning rate used in the training process.
    batch_size : int
        Batch size for training (number of sample per batch)
    dist_func_name : str
        Name of distance function for distance matrix.
    bin_search_tol : float
        Tolerance used in binary search to define sigma_i^2. 
        Iterations stop when the difference between the expected entropy and the computed entropy is less than `tol`.
    bin_search_max_iter : int
        Maximum number of iteration in binary search to define sigma_i^2.
    min_allowed_sig_sq : int
        Minimum allowed value for the spread of any distribution in conditional probability matrix
    max_allowed_sig_sq : int
        Maximum allowed value for the spread of any distribution in conditional probability matrix

    Returns
    -------
    Joint distribution matrix : tensor
        Joint probability matrix. All values in it sum up to 1.    
    """

    EPS = tensor([1e-10]).to(device(dev))

    n_samples, input_dimens = X_high.shape

    assert n_samples % batch_size == 0, \
        f'The number of samples should be divisible by the batch number (got {n_samples}/{batch_size})'

    net = Network
    model = net(dim_input = input_dimens, dim_emb = dim_emb).to(torch.device(dev))
    opt = torch.optim.Adam(model.parameters(), lr = learning_rate)

    model.train()
    batches_passed = 0
    model_name = get_random_string(6)
    epoch_losses = []

    train_dl = torch.from_numpy(X_high).view(n_samples, input_dimens).float().to(torch.device(dev))
    train_dl = (train_dl - train_dl.min()) / (train_dl.max() - train_dl.min())

    for epoch in range(n_epochs):
        train_loss_KL = 0
        train_loss_MSE = 0
        train_loss_tot = 0
        epoch_start_time = datetime.datetime.now()

        len_ = train_dl.size(0)
        index = torch.randperm(len_)
        index_split = torch.split(index, batch_size)

        # For every batch
        for batch_index in tqdm(index_split, disable = not verbose):

            #orig_points_batch, _ = list_with_batch
            orig_points_batch = train_dl[batch_index]
            
            # Calculate conditional probability matrix in higher-dimensional space for the batch

            # Regular parametric t-SNE
            if perplexity is not None:
                target_entropy = log2(perplexity)
                p_cond_in_batch = get_optimized_p_cond(X_high = orig_points_batch,
                                                        target_entropy = target_entropy,
                                                        EPS = EPS,
                                                        dist_func = dist_func_name,
                                                        tol = bin_search_tol,
                                                        max_iter = bin_search_max_iter,
                                                        min_allowed_sig_sq = min_allowed_sig_sq,
                                                        max_allowed_sig_sq = max_allowed_sig_sq,
                                                        dev = dev)

                if p_cond_in_batch is None:
                    continue
                p_joint_in_batch = get_sym_joint(cond_prob = p_cond_in_batch, 
                                                EPS = EPS)

            # Multiscale parametric t-SNE
            else:
                max_entropy = round(log2(batch_size / 2))
                mscl_p_joint_in_batch = zeros(batch_size, batch_size).to(device(dev))
                
                for n_different_entropies, h in enumerate(range(1, max_entropy)):
                    p_cond_for_h = get_optimized_p_cond(X_high = orig_points_batch.view(-1, input_dimens),
                                                        target_entropy = h,
                                                        EPS = EPS,
                                                        dist_func = dist_func_name,
                                                        tol = bin_search_tol,
                                                        max_iter = bin_search_max_iter,
                                                        min_allowed_sig_sq = min_allowed_sig_sq,
                                                        max_allowed_sig_sq = max_allowed_sig_sq,
                                                        dev = dev)
                    
                    if p_cond_for_h is None:
                        continue

                    p_joint_for_h = get_sym_joint(cond_prob = p_cond_for_h, 
                                                EPS = EPS)

                    # TODO This fails if the last batch doesn't match the shape of mscl_p_joint_in_batch
                    mscl_p_joint_in_batch += p_joint_for_h

                p_joint_in_batch = mscl_p_joint_in_batch / n_different_entropies

            # Apply early exaggeration to the conditional probability matrix
            if early_exaggeration:
                p_joint_in_batch *= early_exaggeration_constant
                early_exaggeration -= 1

            batches_passed += 1
            

            #for index, sample in enumerate(orig_points_batch):

            opt.zero_grad()

            # Calculate joint probability matrix in lower-dimensional space for the batch
            X_low, X_reconst_batch = model(orig_points_batch, both = True)

            q_joint_in_batch = get_q_joint(X_low = X_low.view(-1, dim_emb),
                                            dist_func = dist_func_name,
                                            EPS = EPS)
            
            # Calculate loss
            loss_KL_ = loss_function_KL(p_joint = p_joint_in_batch, 
                                        q_joint = q_joint_in_batch,
                                        EPS = EPS)
            
            loss_MSE_ = loss_function_MSE(X_high = orig_points_batch,
                                        X_reconst = X_reconst_batch.view(-1, input_dimens))
            
            loss = loss_total(loss_KL = loss_KL_,
                            loss_MSE = loss_MSE_,
                            alpha_loss = alpha_loss)


            train_loss_KL += loss_KL_.item()
            train_loss_MSE += loss_MSE_.item()
            train_loss_tot += loss.item()

            # Make an optimization step
            loss.backward()
            opt.step()

        epoch_end_time = datetime.datetime.now()
        time_elapsed = epoch_end_time - epoch_start_time

        # Report loss for epoch
        average_loss_KL = train_loss_KL / batches_passed
        average_loss_MSE = train_loss_MSE / batches_passed
        average_loss_tot = train_loss_tot / batches_passed
        epoch_losses.append(array([average_loss_MSE, average_loss_KL, average_loss_tot]))
        #print(f'Epoch: {epoch + 1}. Time {time_elapsed}. Average loss: {average_loss_tot:.4f}', flush=True)

    #save_path_model = os.path.join(save_dir_path, f"{save_dir_path}/models/{model_name}")
    #save_path_config = os.path.join(save_dir_path, f"{save_dir_path}/config/{model_name}")
    #save_path_loss = os.path.join(save_dir_path, f"{save_dir_path}/loss/{model_name}")

    #{save_dir_path}/

    config = {
            'dim_emb' : dim_emb,
            'perplexity' : perplexity,
            'n_epochs' : n_epochs,
            'dev' : dev,
            'early_exaggeration' : early_exaggeration,
            'early_exaggeration_constant' : early_exaggeration_constant,
            'learning_rate' : learning_rate,
            'alpha_loss' : alpha_loss,
            'batch_size' : batch_size
            }

    torch.save(model, f'{save_dir_path}/models/{model_name}.pt')

    with open(f'{save_dir_path}/config/{model_name}.json', "w") as here:
        json.dump(config, here)

    np_save(f'{save_dir_path}/loss/{model_name}.npy', array(epoch_losses))

    print(f'Model saved as {save_dir_path} -> {model_name}', flush=True)
