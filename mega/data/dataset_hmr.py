"""Inspired from BEDLAM and FastMetro"""

import sys

sys.path.append("./")

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
import cv2
import numpy as np
from mega.utils.augmentations import (
    generate_image_patch_cv2,
    trans_point2d,
    rot_aa,
    flip_pose,
    flip_kp,
    flip_17,
)
import albumentations as A
import torch

from mega.utils.smpl_utils import get_body_model_sequence
import random

import matplotlib.pyplot as plt
from mega.utils import renderer
from matplotlib.gridspec import GridSpec

import cv2

J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]


class DatasetHMR(Dataset):
    def __init__(
        self,
        dataset_file: str,
        subset: str = "train",
        augment: bool = True,
        flip: bool = True,
        proportion: float = 1.0,
    ):
        super().__init__()

        self.dataset_file = dataset_file.split("/")[-1]

        self.data = np.load(dataset_file, allow_pickle=True)

        self.imgname = self.data["imgname"]

        full_len = len(self.imgname)

        new_len = full_len
        sampled_indices = [x for x in range(full_len)]
        if proportion != 1:
            new_len = int(proportion * full_len)
            sampled_indices = random.sample(range(full_len), new_len)
            self.imgname = self.data["imgname"][sampled_indices]

        self.scale = self.data["scale"][sampled_indices]
        self.center = self.data["center"][sampled_indices]

        if "pose_cam" in self.data:
            self.full_pose = self.data["pose_cam"][sampled_indices]
        elif "pose" in self.data:
            self.full_pose = self.data["pose"][sampled_indices]
        else:
            root_orient = self.data["root_orient"][sampled_indices]
            pose_body = self.data["pose_body"][sampled_indices]
            self.full_pose = np.concatenate([root_orient, pose_body], axis=-1)

        if "gender" in self.data:
            self.gender = self.data["gender"][sampled_indices]
        else:
            self.gender = ["neutral" for _ in range(new_len)]

        if "j2d" in self.data:
            self.j2d = self.data["j2d"][sampled_indices]
        else:
            self.j2d = self.data["part"][sampled_indices]

        if "betas" in self.data:
            self.betas = self.data["betas"][sampled_indices]
        else:
            self.betas = self.data["shape"][sampled_indices]

        self.is_train = subset in ["train", "all"] and augment

        print(f"{len(self.imgname)} training samples found")

        self.rot_factor = 30
        self.noise_factor = 0.4
        self.scale_factor = 0.25
        self.flip = flip

        self.normalize_vqvae = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.normalize_resnet = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.imgname)

    def augm_params(self):
        """Get augmentation parameters."""
        rot = 0  # rotation
        sc = 1  # scaling
        flip = 0
        if self.is_train:
            if np.random.uniform() <= 0.5 and self.flip:
                flip = 1

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(
                2 * self.rot_factor,
                max(-2 * self.rot_factor, np.random.randn() * self.rot_factor),
            )

            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(
                1 + self.scale_factor,
                max(1 - self.scale_factor, np.random.randn() * self.scale_factor + 1),
            )
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0

        return flip, rot, sc

    def rgb_processing(self, rgb_img, center, scale, sc, rot, flip):
        if self.is_train:
            aug_comp = [
                A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                A.ImageCompression(20, 100, p=0.1),
                A.RandomRain(blur_value=4, p=0.1),
                A.MotionBlur(blur_limit=(3, 15), p=0.2),
                A.Blur(blur_limit=(3, 10), p=0.1),
                A.RandomSnow(
                    brightness_coeff=1.5, snow_point_lower=0.2, snow_point_upper=0.4
                ),
            ]
            aug_mod = [
                A.CLAHE((1, 11), (10, 10), p=0.2),
                A.ToGray(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.MultiplicativeNoise(
                    multiplier=[0.5, 1.5], elementwise=False, per_channel=True, p=0.2
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    always_apply=False,
                    p=0.2,
                ),
                A.Posterize(p=0.1),
                A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                A.Equalize(mode="cv", p=0.1),
            ]
            albumentation_aug = A.Compose(
                [A.OneOf(aug_comp, p=0.3), A.OneOf(aug_mod, p=0.3)]
            )
            rgb_img = albumentation_aug(image=rgb_img)["image"]

        rgb_img, trans = generate_image_patch_cv2(
            rgb_img,
            center[0],
            center[1],
            200 * scale,
            200 * scale,
            256,
            256,
            flip,
            sc,
            rot,
        )

        return rgb_img, trans

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype("float32")
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype("float32")
        return pose

    def j2d_processing(self, kp, flip, trans, width):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        if flip:
            kp[:, 0] = width - kp[:, 0] - 1
            if kp.shape[0] == 24:
                kp = flip_kp(kp)
            elif kp.shape[0] == 17:
                kp = flip_17(kp)

        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i, 0:2] = trans_point2d(kp[i, 0:2], trans)
        # convert to normalized coordinates
        kp[:, :-1] = kp[:, :-1] / 256 - 0.5
        kp = kp.astype("float32")
        return kp

    def __getitem__(self, index):
        imgname = self.imgname[index]
        is_3dpw = "3DPW" in imgname
        if not is_3dpw:
            img_path = f"datasets/{self.dataset_file[:-4]}/{imgname}"
        else:
            img_path = imgname
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width = img.shape[0]
        if "closeup" in img_path:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        center = self.center[index]
        if not is_3dpw:
            scale = self.scale[index]
        else:
            scale = 1 / self.scale[index]

        flip, rot, sc = self.augm_params()
        img, trans = self.rgb_processing(img, center, scale, sc, rot, flip)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float() / 255
        resnet_img = self.normalize_resnet(img)

        j2d = self.j2d[index][:24]
        if not is_3dpw:
            j2d = j2d[J24_TO_J17]
            j2d = self.j2d_processing(j2d, flip, trans, width)
            j2d_full = np.zeros((24, 3))
            j2d_full[:17] = j2d
        else:
            j2d_full = self.j2d_processing(j2d, flip, trans, width)

        betas = self.betas[index][:10]
        gender = str(self.gender[index])
        full_pose = self.full_pose[index][:66]
        pose = self.pose_processing(full_pose, rot, flip)

        root_orient = pose[:3]
        pose_body = pose[3:]

        root_orient = np.expand_dims(root_orient, 0)
        pose_body = np.expand_dims(pose_body, 0)

        local_mesh = get_body_model_sequence(
            gender=gender,
            num_frames=1,
            betas=betas,
            pose_body=pose_body,
            num_betas=10,
            normalize=True,
            cpu=True,
        )

        global_mesh = get_body_model_sequence(
            gender=gender,
            num_frames=1,
            betas=betas,
            pose_body=pose_body,
            root_orient=root_orient,
            num_betas=10,
            normalize=True,
            cpu=True,
        )

        item = {}
        item["img"] = resnet_img
        item["imgname"] = img_path
        item["raw_img"] = img
        item["j2d"] = j2d
        item["gender"] = gender
        item["mesh"] = torch.from_numpy(global_mesh[0]).float()
        item["local_mesh"] = torch.from_numpy(local_mesh[0]).float()
        item["rotation"] = torch.from_numpy(root_orient[0]).float()
        item["smpl_theta"] = pose_body.reshape(21, 3)
        item["smpl_beta"] = torch.from_numpy(betas).float()
        item["is_3dpw"] = is_3dpw

        return item


def plot_meshes_(meshes, faces, show: bool = True, save: str = None):
    images = renderer(meshes, faces, device="cpu")
    fig = plt.figure(figsize=(10, 10))
    if len(meshes) == 16:
        nrows = 4
        ncols = 4
    elif len(meshes) == 4:
        nrows = 2
        ncols = 2
    else:
        ncols = len(meshes)
        nrows = 1

    gs = GridSpec(ncols=ncols, nrows=nrows)
    i = 0
    for line in range(nrows):
        for col in range(ncols):
            ax = fig.add_subplot(gs[line, col])
            if images[i].shape[0] == 1:
                ax.imshow(images[i][0, :, :].cpu().detach().numpy())
            else:
                ax.imshow(images[i].cpu().detach().numpy())
            plt.axis("off")
            i = i + 1
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()


def plot_images_(images, show: bool = True, save: str = None):
    fig = plt.figure(figsize=(10, 10))
    if len(images) == 16:
        nrows = 4
        ncols = 4
    elif len(images) == 4:
        nrows = 2
        ncols = 2
    else:
        ncols = len(images)
        nrows = 1

    gs = GridSpec(ncols=ncols, nrows=nrows)
    i = 0
    for line in range(nrows):
        for col in range(ncols):
            ax = fig.add_subplot(gs[line, col])
            if images[i].shape[0] == 1:
                ax.imshow(images[i][0, :, :].cpu().detach().numpy())
            else:
                ax.imshow(images[i].permute(1, 2, 0).cpu().detach().numpy())
            plt.axis("off")
            i = i + 1
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
        plt.close()


if __name__ == "__main__":
    ref_bm_path = "body_models/smplh/neutral/model.npz"
    ref_bm = np.load(ref_bm_path)
    faces = torch.from_numpy(ref_bm["f"].astype(np.int32))

    folder = "datasets/3DPW/3DPW_validation_coco.npz"
    dataset = DatasetHMR(
        folder,
        augment=True,
    )
    train_data = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, data in enumerate(train_data):
        plot_images_(data["raw_img"], show=False, save=f"test_viz/{i}_img.png")
