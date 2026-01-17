import numpy as np
import torch
from scipy.optimize import minimize

"""
This script implements a simplified version of the Hyper Conflict Averse aggregation mechanism proposed by:
Lu, Y., Huang, S., Yang, Y., Sirejiding, S., Ding, Y., & Lu, H. (2024). 
FedHCA2: Towards Hetero-Client Federated Multi-Task Learning. 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5599-5609).

The functions below are a subset of functions developed by the authors above. Only minor modifications have taken place.
As such credits of the code below goes to the authors above!
Their source code is accesible here: https://github.com/innovator-zero/FedHCA2
"""


def get_delta_dict_list(param_dict_list, last_param_dict_list):
    """Get the difference between current and last parameters

    Args:
        param_dict_list ([{}]): Current Parameter Dictionaries
        last_param_dict_list ([{}]): Last Parameter Dictionaries

    Returns:
        [{}]: Returns the difference between the current and last parameter dictionaries.
    """

    # a list of length N, each element is a dict of delta parameters
    delta_dict_list = []
    layers = param_dict_list[0].keys()
    for i in range(len(param_dict_list)):
        delta_dict_list.append({})
        for layer in layers:
            delta_dict_list[i][layer] = (
                param_dict_list[i][layer] - last_param_dict_list[i][layer]
            )

    return delta_dict_list


def new_get_encoder_params(all_clients):
    """Get encoder parameters from each client's current checkpoint. (Note: Encoder translates to backbone as part of this project.)

    Args:
        all_clients ([{}]): list of client dictionaries.

    Returns:
        []: encoder_param_dict_list, all_name_keys, layers, shapes
    """

    # Assuming that 'current_checkpoint' contains a dictionary with 'model_state_dict'
    first_client_state_dict = all_clients[0]
    all_name_keys = list(first_client_state_dict.keys())

    encoder_param_dict_list = []
    layers = []
    shapes = []

    for client in all_clients:
        param_dict = {}
        model_state_dict = client
        for key in all_name_keys:
            prefix, layer = key.split(".", 1)
            param_dict[layer] = model_state_dict[key]
        encoder_param_dict_list.append(param_dict)

    # Get layers and shapes (same for all encoders)
    for key in all_name_keys:
        layer = key.split(".", 1)[1]
        layers.append(layer)
        shapes.append(model_state_dict[key].shape)

    return encoder_param_dict_list, all_name_keys, layers, shapes


def get_ca_delta(flatten_delta_list, alpha, rescale=1):
    """Solve for aggregated conflict-averse delta

    Args:
        flatten_delta_list ([]): A list of the flatted deltas
        alpha (float): Is a scaling factor that controls the influence of the norm of the gradients
            on the objective function during the optimization process, helping to balance the magnitude of the update.
        rescale (int, optional): Modifies the impact of alpha on the update.

    Returns:
        final_update: The calculated hyper conflict averse update.
    """

    print(alpha)
    N = len(flatten_delta_list)
    grads = torch.stack(flatten_delta_list).t()  # [d , N]

    # Check for NaN or Inf in gradients
    if torch.isnan(grads).any() or torch.isinf(grads).any():
        print("ERROR: NaN or Inf detected in gradients before HCA aggregation")
        return torch.zeros_like(flatten_delta_list[0])

    # Compute gradient norms for numerical stability check
    grad_norms = torch.stack([g.norm() for g in flatten_delta_list])
    max_norm = grad_norms.max()
    min_norm = grad_norms.min()

    # Check for extreme gradient norm ratios (numerical instability indicator)
    if max_norm > 0 and (max_norm / (min_norm + 1e-10)) > 100:
        print(f"WARNING: Extreme gradient norm ratio detected: {max_norm:.2e}/{min_norm:.2e}")
        print(f"  Gradient norms: {grad_norms.tolist()}")
        # Normalize gradients to prevent numerical issues - use max instead of mean for stronger effect
        grads = grads / (grad_norms.max() + 1e-8)
        print("  Applied gradient normalization for stability")

    GG = grads.t().mm(grads).cpu()  # [N, N]
    assert not torch.isnan(GG).any(), "NaN detected in GG matrix"
    g0_norm = (GG.mean() + 1e-8).sqrt()

    x_start = np.ones(N) / N
    bnds = tuple((0, 1) for x in x_start)
    cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
    A = GG.numpy()
    assert not torch.isnan(GG).any(), "NaN detected in GG matrix"
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, -1).dot(A).dot(b.reshape(-1, 1))
            + c * np.sqrt(x.reshape(1, -1).dot(A).dot(x.reshape(-1, 1)) + 1e-8)
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)

    # Check if optimization succeeded
    if not res.success:
        print(f"WARNING: Optimization failed: {res.message}")
        print("  Falling back to simple averaging")
        return grads.mean(1)

    ww = torch.Tensor(res.x).to(grads.device)

    # Check for NaN in optimization weights
    if torch.isnan(ww).any() or torch.isinf(ww).any():
        print("ERROR: NaN or Inf in optimization weights, using simple averaging")
        return grads.mean(1)

    gw = (grads * ww.reshape(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw

    if rescale == 0:
        final_update = g
    elif rescale == 1:
        final_update = g / (1 + alpha**2 + 0.5)
    else:
        final_update = g / (1 + alpha)

    # Final NaN check before returning
    if torch.isnan(final_update).any() or torch.isinf(final_update).any():
        print("ERROR: NaN or Inf in final update, using simple averaging")
        return grads.mean(1)

    return final_update


def flatten_param(param_dict_list, layers):
    """
    Flattens a list of parameter dictionaries into a list of 1D tensors.

    Args:
        param_dict_list ([{}]): A list of dictionaries where each dictionary contains model parameters
                                        organized by layer names as keys and their corresponding tensors as values.
        layers ([str]): A list of layer names.

    Returns:
        [torch.Tensor]: Returns a list of 1D tensors where each tensor is the flattened and concatenated representation
                              of the selected parameters from each dictionary.
    """

    flatten_list = [
        torch.cat([param_dict_list[idx][layer].flatten() for layer in layers])
        for idx in range(len(param_dict_list))
    ]
    assert len(flatten_list[0].shape) == 1

    return flatten_list


def unflatten_param(flatten_list, shapes, layers):
    """Reconstructs a list of parameter dictionaries from flattened tensors.

    Args:
        flatten_list ([torch.Tensor]): A list of 1D tensors where each tensor contains the flattened parameters
                                             of a model.
        shapes ([tuple]): A list of tuples representing the original shapes of each parameter tensor in the same
                                order as they were flattened.
        layers ([str]): A list of layer names corresponding to the flattened parameters.

    Returns:
        [{}]: Returns a list of dictionaries where each dictionary maps layer names to their reconstructed parameter
                      tensors.
    """

    param_dict_list = []
    for model_idx in range(len(flatten_list)):
        start = 0
        param_dict_list.append({})
        for layer, shape in zip(layers, shapes):
            end = start + int(np.prod(shape))
            param_dict_list[model_idx][layer] = flatten_list[model_idx][
                start:end
            ].reshape(shape)
            start = end

    return param_dict_list


def conflict_averse(curr_backbones_dicts, prev_backbones_dicts, ca_c):
    """Aggregates model parameters using a conflict-averse update strategy.

    Args:
        curr_backbones_dicts ([dict]): A list of dictionaries. Each contains the current model parameters.
        prev_backbones_dicts ([dict]): A list of dictionaries. Each contains the previous model parameters.
        ca_c (float): Conflict-averse hyperparameter that controls the impact of the aggregated update to reduce
                      conflicts between the current and previous model states.

    Returns:
        [{}]: Returns a list of updated parameter dictionaries for each client.
    """

    N = len(curr_backbones_dicts)

    # update_ckpt = copy.deepcopy(save_ckpt)  # store updated parameters

    # Get encoder parameter list
    encoder_param_list, encoder_keys, enc_layers, enc_shapes = new_get_encoder_params(
        curr_backbones_dicts
    )

    # Encoder agg
    last_encoder_param_list, _, _, _ = new_get_encoder_params(prev_backbones_dicts)
    encoder_delta_list = get_delta_dict_list(
        encoder_param_list, last_encoder_param_list
    )

    # Flatten
    flatten_last_encoder = flatten_param(last_encoder_param_list, enc_layers)
    del last_encoder_param_list
    flatten_encoder_delta = flatten_param(encoder_delta_list, enc_layers)
    del encoder_delta_list

    # Check if all deltas are zero (happens in first round when prev_backbone is None)
    all_deltas_zero = all(torch.norm(delta) < 1e-10 for delta in flatten_encoder_delta)

    if all_deltas_zero:
        # Skip HCA aggregation, just return current backbones unchanged
        print("Warning: All gradients are zero, skipping HCA aggregation")
        flatten_delta_update = torch.zeros_like(flatten_encoder_delta[0])
    else:
        # Solve for aggregated conflict-averse delta
        flatten_delta_update = get_ca_delta(flatten_encoder_delta, ca_c)  # flattened tensor

    for idx, client_encoder in enumerate(flatten_last_encoder):
        update = flatten_encoder_delta[idx] + 1 * flatten_delta_update

        # Gradient clipping to prevent explosion
        max_update_norm = 5.0
        update_norm = update.norm()
        if update_norm > max_update_norm:
            update = update * (max_update_norm / update_norm)
            print(f"[Client {idx}] Clipping update norm from {update_norm:.2f} to {max_update_norm}")

        # Check for NaN before applying update
        if torch.isnan(update).any():
            print(f"Warning: NaN detected in update for client {idx}, skipping update")
            continue
        client_encoder.add_(update)
    flatten_new_encoders = flatten_last_encoder

    # Unflatten with original keys to preserve parameter structure
    new_encoders_with_correct_keys = []
    for model_idx in range(len(flatten_new_encoders)):
        start = 0
        param_dict = {}
        for original_key, shape in zip(encoder_keys, enc_shapes):
            end = start + int(np.prod(shape))
            param_dict[original_key] = flatten_new_encoders[model_idx][start:end].reshape(shape)
            start = end
        new_encoders_with_correct_keys.append(param_dict)

    # Final NaN check
    for idx, encoder in enumerate(new_encoders_with_correct_keys):
        for key, param in encoder.items():
            if torch.isnan(param).any():
                print(f"ERROR: NaN detected in encoder {idx}, key {key}")
                raise ValueError(f"NaN in aggregated parameters for client {idx}")

    return new_encoders_with_correct_keys
