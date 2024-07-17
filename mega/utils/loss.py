import torch
import torch.nn.functional as F
from einops import rearrange

H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]


def masked_cross_entropy(pred, labels, ignore_index=None, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed."""
    # print(pred.shape, labels.shape) #torch.Size([64, 1028, 55]) torch.Size([64, 55])
    # print(pred.shape, labels.shape) #torch.Size([64, 1027, 55]) torch.Size([64, 55])
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), "a ... b -> a b ...")[
            :, :n_class
        ]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (
            n_class - 1
        )
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss


def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d


def reprojection_loss(gt_2d, pred_v, pred_cam, joints_reg):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    l1_loss = torch.nn.L1Loss(reduction="mean")
    return l1_loss(pred_2d, gt_2d)


def reprojection_loss_conf(gt_2d, pred_v, pred_cam, joints_reg):
    J_regressor_batch = joints_reg[None, :].expand(pred_v.shape[0], -1, -1).to(gt_2d)
    pred_3dkpt = torch.matmul(J_regressor_batch, pred_v)
    pred_3dkpt = pred_3dkpt[:, H36M_TO_J17]
    pred_2d = orthographic_projection(pred_3dkpt, pred_cam)
    l1_loss = torch.nn.L1Loss(reduction="none")
    loss = l1_loss(pred_2d, gt_2d[:, :, :2]).mean(dim=-1) * gt_2d[:, :, -1]
    return loss.mean()
