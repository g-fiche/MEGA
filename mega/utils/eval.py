import torch

H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]


def dist3d(data, data_recon, reduction=True, dim=0):
    if reduction:
        return torch.norm(data - data_recon, dim=-1).mean()
    else:
        return torch.norm(data - data_recon, dim=-1).mean(dim=dim)


def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, _, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def v2v(gt_v, pred_v, reduction=True, dim=0):
    pred_v = pred_v - torch.mean(pred_v, axis=1, keepdims=True)
    return dist3d(gt_v, pred_v, reduction, dim)


def pa_v2v(gt_v, pred_v, reduction=True, dim=0):
    pred_sym = batch_compute_similarity_transform_torch(pred_v, gt_v)
    return dist3d(gt_v, pred_sym, reduction, dim)


def mpjpe(gt_v, pred_v, joints_reg, reduction=True, dim=0):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_v)
    pred_v = pred_v - torch.mean(pred_v, axis=1, keepdims=True)
    gt_joints = torch.matmul(J_regressor_batch, gt_v)
    pred_joints = torch.matmul(J_regressor_batch, pred_v)
    return dist3d(gt_joints, pred_joints, reduction, dim)


def pa_mpjpe(gt_v, pred_v, joints_reg, reduction=True, dim=0):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_v)
    gt_joints = torch.matmul(J_regressor_batch, gt_v)
    pred_joints = torch.matmul(J_regressor_batch, pred_v)
    return pa_v2v(gt_joints, pred_joints, reduction, dim)


def average_pairwise_distance(x):
    n = len(x)
    apd = 0
    total = 0
    for i in range(n - 1):
        diff = x[i + 1 :] - x[i]
        mean_dist = torch.norm(diff, dim=-1).mean()
        num_samples = n - (i + 1)
        apd = (total * apd + num_samples * mean_dist) / (total + num_samples)
        total += num_samples
    return apd


def mesh_variance(mesh_list, mesh, reduction=True, dim=0):
    if mesh.shape[0] != 1:
        mesh = mesh.unsqueeze(0).repeat(mesh_list.shape[0], 1, 1)
    else:
        mesh = mesh.repeat(mesh_list.shape[0], 1, 1)
    if reduction:
        return torch.norm(mesh_list - mesh, dim=-1).mean()
    else:
        return torch.norm(mesh_list - mesh, dim=-1).mean(dim=dim)
