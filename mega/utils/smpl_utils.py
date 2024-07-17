import torch
import os
import sys

sys.path.append("./")
from mega.utils.body_model import BodyModel, SMPLH_PATH
import numpy as np

MAX_LEN = 1000


def get_body_model_sequence(
    gender,
    num_frames,
    betas,
    pose_body,
    root_orient=None,
    trans=None,
    pose_hand=None,
    num_betas=16,
    normalize=False,
    cpu=False,
):
    bm_path = os.path.join(SMPLH_PATH, gender + "/model.npz")
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")

    out_mesh = np.zeros((num_frames, 6890, 3))

    last_b = 0

    if num_frames > MAX_LEN:
        last_b = num_frames % MAX_LEN
        num_b = num_frames // MAX_LEN
        bm_last = BodyModel(bm_path=bm_path, num_betas=num_betas, batch_size=last_b).to(
            device
        )
        num_frames = MAX_LEN
    else:
        num_b = 1

    bm = BodyModel(bm_path=bm_path, num_betas=num_betas, batch_size=num_frames).to(
        device
    )

    sidx = 0
    eidx = num_frames
    for _ in range(num_b):
        betas_b = torch.Tensor(
            np.repeat(betas[:num_betas][np.newaxis], num_frames, axis=0)
        ).to(device)
        root_orient_b = None
        if root_orient is not None:
            root_orient_b = torch.Tensor(root_orient[sidx:eidx]).to(device)
        trans_b = None
        if trans is not None:
            trans_b = torch.Tensor(trans[sidx:eidx]).to(device)
        pose_body_b = torch.Tensor(pose_body[sidx:eidx]).to(device)
        pose_hand_b = None
        if pose_hand is not None:
            pose_hand_b = torch.Tensor(pose_hand[sidx:eidx]).to(device)
        body = bm(
            pose_body=pose_body_b,
            pose_hand=pose_hand_b,
            betas=betas_b,
            root_orient=root_orient_b,
            trans=trans_b,
        )
        out_mesh[sidx:eidx] = body.v.clone().detach().cpu().numpy()
        sidx += num_frames
        eidx += num_frames

    if num_b >= 1 and last_b != 0:
        betas_b = torch.Tensor(
            np.repeat(betas[:num_betas][np.newaxis], last_b, axis=0)
        ).to(device)
        if root_orient is not None:
            root_orient_b = torch.Tensor(root_orient[-last_b:]).to(device)
        if trans is not None:
            trans_b = torch.Tensor(trans[-last_b:]).to(device)
        pose_body_b = torch.Tensor(pose_body[-last_b:]).to(device)
        if pose_hand is not None:
            pose_hand_b = torch.Tensor(pose_hand[-last_b:]).to(device)
        body = bm_last(
            pose_body=pose_body_b,
            pose_hand=pose_hand_b,
            betas=betas_b,
            root_orient=root_orient_b,
            trans=trans_b,
        )
        out_mesh[-last_b:] = body.v.clone().detach().cpu().numpy()

    if normalize:
        out_mesh = out_mesh - np.mean(out_mesh, axis=1, keepdims=True)

    return out_mesh


if __name__ == "__main__":
    gender = "neutral"
    num_frames = 1
    betas = torch.zeros((16))
    pose_body = torch.zeros((1, 63))
    out_mesh = get_body_model_sequence(gender, num_frames, betas, pose_body)
    np.save("body_models/tpose.npy", out_mesh)
