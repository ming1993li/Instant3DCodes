import torch


def rho(x, type='mae'):
    if type == 'mae':
        return torch.abs(x)
    else:
        raise NotImplementedError('Unknown rho type {}!'.format(type))


def t(d):
    """ d: [b, hw]
    """
    return torch.median(d, dim=1, keepdim=True).values


def s(d):
    """ d: [b, hw]
    """
    return torch.mean(torch.abs(d - t(d)), dim=1, keepdim=True)


def hat(d):
    return (d - t(d)) / (s(d) + 1e-6)


def L_ssi_trim(d:torch.Tensor, d_star:torch.Tensor, u_m=0.8):
    """ d: [b, 1, h, w]
        d_star: [b, 1, h, w]
    """
    d_hat = hat(d.flatten(1))
    d_star_hat = hat(d_star.flatten(1))

    # Compute rho_mae values
    rho_mae_values = rho(d_hat - d_star_hat, type='mae')   # [b, hw]
    rho_mae_values_sorted, _ = torch.sort(rho_mae_values, dim=-1, descending=False)

    U_m = int(rho_mae_values_sorted.shape[-1] * u_m)
    return torch.mean(rho_mae_values_sorted[:, :U_m])