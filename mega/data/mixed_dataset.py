"""Inspired from Bedlam"""

import torch
import numpy as np
import sys

sys.path.append("./")
from mega.data.dataset_hmr_square import DatasetHMRSquare
from mega.data.dataset_hmr import DatasetHMR

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from mega.utils import renderer
from matplotlib.gridspec import GridSpec


class MixedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file: str,
        list_ratios: list = None,
        augment: bool = True,
        flip: bool = True,
        proportion: float = 1,
        square=True,
    ):
        dataset_file = open(file, "r")
        data = dataset_file.read()
        self.dataset_list = data.split("\n")
        if self.dataset_list[-1] == "":
            self.dataset_list = self.dataset_list[:-1]

        self.proportion = proportion

        if list_ratios is not None:
            self.dataset_ratios = list_ratios
        else:
            self.dataset_ratios = [1 / len(self.dataset_list)] * len(self.dataset_list)

        assert len(self.dataset_list) == len(
            self.dataset_ratios
        ), "Number of datasets and ratios should be equal"

        print(len(self.dataset_list))

        if not square:
            self.datasets = [
                DatasetHMR(
                    ds,
                    augment=augment,
                    flip=flip,
                    proportion=self.proportion,
                )
                for ds in self.dataset_list
            ]
        else:
            self.datasets = [
                DatasetHMRSquare(
                    ds,
                    augment=augment,
                    flip=flip,
                    proportion=self.proportion,
                )
                for ds in self.dataset_list
            ]
        self.length = max([len(ds) for ds in self.datasets])

        self.partition = []

        for idx, (ds_name, ds_ratio) in enumerate(
            zip(self.dataset_list, self.dataset_ratios)
        ):
            r = ds_ratio
            self.partition.append(r)

        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length


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

    dataset = MixedDataset(
        "configs/config_mesh_regressor/datasets_list_full.txt",
        augment=False,
        flip=False,
    )
    train_data = DataLoader(dataset, batch_size=4)
    for i, data in enumerate(train_data):
        plot_images_(data["raw_img"], show=False, save=f"viz_test/{i}_img.png")
        plot_meshes_(
            data["mesh"], faces=faces, show=False, save=f"viz_test/{i}_mesh.png"
        )
