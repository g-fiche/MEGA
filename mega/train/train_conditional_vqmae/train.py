from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
from ...base import Train
from ...model import CVQMAE
from mesh_vq_vae import MeshVQVAE, get_colors_from_diff_pc
from .follow_up_mae import Follow
import math
from math import sqrt
from .idr_torch import IDR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
)
from ...utils.loss import *
from ...utils.eval import *
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from ...utils.mesh_render import renderer
from ...utils.img_renderer import visualize_reconstruction_pyrender, PyRender_Renderer
import pandas as pd
import random


class CVQMAE_Train(Train):
    def __init__(
        self,
        mae: CVQMAE,
        vqvae: MeshVQVAE,
        training_data: Dataset,
        validation_data: Dataset,
        config_training: dict = None,
        multigpu_bool: bool = False,
        faces=None,
        joints_regressor=None,
        joints_regressor_smpl=None,
        vit_backbone=False,
    ):
        super().__init__()
        self.f = faces
        if multigpu_bool:
            self.idr = IDR()
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.idr.size,
                rank=self.idr.rank,
            )
            torch.cuda.set_device(self.idr.local_rank)
        self.device = torch.device(config_training["device"])
        """ Model """
        self.model = mae
        self.vqvae = vqvae
        self.model.to(self.device)
        self.vqvae.to(self.device)

        self.vit_backbone = vit_backbone

        if multigpu_bool:
            self.model = DDP(
                self.model,
                device_ids=[self.idr.local_rank],
                find_unused_parameters=True,
            )
            self.vqvae = DDP(
                self.vqvae,
                device_ids=[self.idr.local_rank],
                find_unused_parameters=True,
            )

        """ Dataloader """
        if multigpu_bool:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                training_data,
                num_replicas=self.idr.size,
                rank=self.idr.rank,
                shuffle=True,
                drop_last=True,
            )
            self.training_loader = torch.utils.data.DataLoader(
                dataset=training_data,
                batch_size=config_training["batch"] // self.idr.size,
                shuffle=False,
                num_workers=config_training["workers"],
                pin_memory=True,
                drop_last=True,
                sampler=train_sampler,
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                validation_data,
                num_replicas=self.idr.size,
                rank=self.idr.rank,
                shuffle=True,
            )
            self.validation_loader = torch.utils.data.DataLoader(
                dataset=validation_data,
                batch_size=config_training["batch"] // self.idr.size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=val_sampler,
                drop_last=True,
                prefetch_factor=2,
            )
        else:
            self.training_loader = DataLoader(
                training_data,
                batch_size=config_training["batch"],
                shuffle=True,
                num_workers=config_training["workers"],
                drop_last=True,
            )
            self.validation_loader = DataLoader(
                validation_data,
                batch_size=config_training["batch"],
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )

        """ Optimizer """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config_training["lr"] * config_training["batch"] / 256,
            betas=(0.9, 0.95),
            weight_decay=config_training["weight_decay"],
        )
        lr_func = lambda epoch: min(
            (epoch + 1) / (config_training["warmup_epoch"] + 1e-8),
            0.5 * (math.cos(epoch / config_training["total_epoch"] * math.pi) + 1),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_func, verbose=True
        )

        """ Loss """
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.mse = torch.nn.MSELoss(reduction="mean")

        """ Config """
        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0
        self.parameters = dict()

        self.joints_reg = joints_regressor
        self.joints_reg_smpl = joints_regressor_smpl

        self.multigpu_bool = multigpu_bool

        self.train_cross = []
        self.train_rot = []
        self.train_2d = []
        self.train_loss = []
        self.train_v2v = []
        self.train_mpjpe = []
        self.train_pampjpe = []

        self.val_cross = []
        self.val_rot = []
        self.val_2d = []
        self.val_loss = []
        self.val_v2v = []
        self.val_mpjpe = []
        self.val_pampjpe = []

        """ Follow """
        self.follow = Follow(
            "cvqmae",
            dir_save="checkpoint",
            multigpu_bool=multigpu_bool,
        )

        self.inference_batch = config_training["batch"]

    def one_epoch(self, epoch):
        self.model.train()
        losses = []
        for data in tqdm(iter(self.training_loader)):
            mesh = data["local_mesh"]
            self.optimizer.zero_grad()
            self.step_count += 1
            mesh = mesh.to(self.device)
            with torch.no_grad():
                indices = self.vqvae.get_codebook_indices(
                    mesh.to(self.device),
                )
                img_features = data["img"].to(self.device)

            if self.vit_backbone:
                predicted_indices, pred_rot, pred_cam, mask = self.model(
                    indices, img_features[:, :, :, 32:-32]
                )
            else:
                predicted_indices, pred_rot, pred_cam, mask = self.model(
                    indices, img_features
                )

            with torch.no_grad():
                _, mesh_indices = torch.max(predicted_indices.data, -1)
                mesh_indices = (
                    mesh_indices * mask + indices * (~mask.to(torch.bool))
                ).type(torch.int64)
                mesh_canonical = self.vqvae.decode(mesh_indices).cpu()
            rotmat = rotation_6d_to_matrix(pred_rot)
            pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

            loss = 0

            cross_entropy = self.criterion(
                predicted_indices.flatten(0, 1)[mask.flatten(0).to(torch.bool)],
                indices.flatten(0)[mask.flatten(0).to(torch.bool)].to(torch.long),
            )
            if not torch.isnan(cross_entropy):
                self.train_cross.append(cross_entropy.item())
                loss += cross_entropy

            rot_loss = self.mse(
                rotmat,
                axis_angle_to_matrix(data["rotation"]),
            ).mean()
            self.train_rot.append(rot_loss.item())
            loss += rot_loss

            is_3dpw = data["is_3dpw"] == True
            not_3dpw = data["is_3dpw"] == False
            reproj_loss = 0
            if is_3dpw.any():
                reproj_loss += reprojection_loss(
                    data["j2d"][is_3dpw][:, :, :2],
                    pred_mesh[is_3dpw],
                    pred_cam[is_3dpw],
                    self.joints_reg_smpl,
                )
            if not_3dpw.any():
                reproj_loss += reprojection_loss_conf(
                    data["j2d"][not_3dpw],
                    pred_mesh[not_3dpw],
                    pred_cam[not_3dpw],
                    self.joints_reg,
                )
            self.train_2d.append(reproj_loss.item())
            loss += reproj_loss

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            self.train_loss.append(loss.item())

            pa_mpjpe_err = pa_mpjpe(data["mesh"], pred_mesh, self.joints_reg)

            mpjpe_err = mpjpe(data["mesh"], pred_mesh, self.joints_reg)

            v2v_err = v2v(
                data["mesh"],
                pred_mesh,
            )

            self.train_pampjpe.append(1000 * pa_mpjpe_err.item())
            self.train_mpjpe.append(1000 * mpjpe_err.item())
            self.train_v2v.append(1000 * v2v_err.item())

        self.plot_meshes_(
            pred_mesh[:4],
            show=False,
            rot=True,
            save=f"{self.follow.path_samples_train}/{epoch}-reconstruction.png",
        )
        self.plot_meshes_(
            data["mesh"][:4],
            show=False,
            rot=True,
            save=f"{self.follow.path_samples_train}/{epoch}-real.png",
        )

        pred_v = pred_mesh.detach()
        cam = pred_cam
        raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
        self.plot_reproj_(
            raw_img[:4],
            pred_v[:4],
            cam[:4],
            show=False,
            save=f"{self.follow.path_samples_train}/{epoch}_reprojection.png",
        )
        return losses

    def fit(self):
        for e in range(self.config_training["total_epoch"]):
            if self.multigpu_bool:
                self.training_loader.sampler.set_epoch(e)
                self.validation_loader.sampler.set_epoch(e)
            losses = self.one_epoch(epoch=e)
            with torch.no_grad():
                losses_val = self.eval(epoch=e)
            self.lr_scheduler.step()
            self.parameters = dict(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.lr_scheduler.state_dict(),
                epoch=e,
                loss=mean(self.val_loss[-len(self.validation_loader) :]),
                pampjpe=mean(self.val_pampjpe[-len(self.validation_loader) :]),
                v2v=mean(self.val_v2v[-len(self.validation_loader) :]),
            )
            self.follow(
                epoch=e,
                loss_train=mean(self.train_loss[-len(self.training_loader) :]),
                loss_validation=mean(self.val_loss[-len(self.validation_loader) :]),
                loss_cross_train=mean(self.train_cross[-len(self.training_loader) :]),
                loss_cross_validation=mean(
                    self.val_cross[-len(self.validation_loader) :]
                ),
                loss_rot_train=mean(self.train_rot[-len(self.training_loader) :]),
                loss_rot_validation=mean(self.val_rot[-len(self.validation_loader) :]),
                loss_2d_train=mean(self.train_2d[-len(self.training_loader) :]),
                loss_2d_validation=mean(self.val_2d[-len(self.validation_loader) :]),
                v2v_train=mean(self.train_v2v[-len(self.training_loader) :]),
                v2v_validation=mean(self.val_v2v[-len(self.validation_loader) :]),
                mpjpe_train=mean(self.train_mpjpe[-len(self.training_loader) :]),
                mpjpe_validation=mean(self.val_mpjpe[-len(self.validation_loader) :]),
                pampjpe_train=mean(self.train_pampjpe[-len(self.training_loader) :]),
                pampjpe_validation=mean(
                    self.val_pampjpe[-len(self.validation_loader) :]
                ),
                parameters=self.parameters,
            )

    def plot_train(self):
        pass

    def plot_meshes_(
        self,
        meshes,
        show: bool = True,
        save: str = None,
        rot: bool = False,
        colors=None,
    ):
        images = renderer(
            meshes,
            self.f,
            "cpu",
            rot=rot,
            colors=colors,
        )
        fig = plt.figure(figsize=(20, 20))
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

    def plot_reproj_(
        self,
        images,
        meshes,
        cameras,
        show: bool = True,
        save: str = None,
        vertex_colors=None,
    ):
        rendered_img = []
        render_reproj = PyRender_Renderer(faces=self.f)
        if vertex_colors is not None:
            for img, vertices, camera, texture in zip(
                images, meshes, cameras, vertex_colors
            ):
                rendered_img.append(
                    visualize_reconstruction_pyrender(
                        img, vertices, camera, render_reproj, vertex_colors=texture
                    )
                )
        else:
            for img, vertices, camera in zip(images, meshes, cameras):
                rendered_img.append(
                    visualize_reconstruction_pyrender(
                        img, vertices, camera, render_reproj
                    )
                )
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
                if rendered_img[i].shape[0] == 1:
                    ax.imshow(rendered_img[i][0, :, :])
                else:
                    ax.imshow(rendered_img[i])
                plt.axis("off")
                i = i + 1
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()

    def plot_img_(self, images, show: bool = True, save: str = None):
        rendered_img = []
        for img in images:
            img = (img * 255).astype(np.uint8)
            rendered_img.append(img)
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
                if rendered_img[i].shape[0] == 1:
                    ax.imshow(rendered_img[i][0, :, :])
                else:
                    ax.imshow(rendered_img[i])
                plt.axis("off")
                i = i + 1
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save)
            plt.close()

    def eval(self, epoch):
        self.model.eval()
        losses = []
        for data in tqdm(iter(self.validation_loader)):
            mesh = data["local_mesh"]
            mesh = mesh.to(self.device)
            indices = self.vqvae.get_codebook_indices(mesh)
            img_features = data["img"].to(self.device)

            if self.vit_backbone:
                predicted_indices, pred_rot, pred_cam, mask = self.model(
                    indices, img_features[:, :, :, 32:-32], fixed_ratio=1
                )
            else:
                predicted_indices, pred_rot, pred_cam, mask = self.model(
                    indices, img_features, fixed_ratio=1
                )

            _, mesh_indices = torch.max(predicted_indices.data, -1)
            mesh_indices = (
                mesh_indices * mask + indices * (~mask.to(torch.bool))
            ).type(torch.int64)
            mesh_canonical = self.vqvae.decode(mesh_indices).cpu()
            rotmat = rotation_6d_to_matrix(pred_rot)
            pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

            loss = 0

            cross_entropy = self.criterion(
                predicted_indices.flatten(0, 1)[mask.flatten(0).to(torch.bool)],
                indices.flatten(0)[mask.flatten(0).to(torch.bool)].to(torch.long),
            )
            if not torch.isnan(cross_entropy):
                self.val_cross.append(cross_entropy.item())
                loss += cross_entropy

            rot_loss = self.mse(
                rotmat,
                axis_angle_to_matrix(data["rotation"]),
            ).mean()
            self.val_rot.append(rot_loss.item())
            loss += rot_loss

            is_3dpw = data["is_3dpw"] == True
            not_3dpw = data["is_3dpw"] == False
            reproj_loss = 0
            if is_3dpw.any():
                reproj_loss += reprojection_loss(
                    data["j2d"][is_3dpw][:, :, :2],
                    pred_mesh[is_3dpw],
                    pred_cam[is_3dpw],
                    self.joints_reg_smpl,
                )
            if not_3dpw.any():
                reproj_loss += reprojection_loss_conf(
                    data["j2d"][not_3dpw],
                    pred_mesh[not_3dpw],
                    pred_cam[not_3dpw],
                    self.joints_reg,
                )
            self.val_2d.append(reproj_loss.item())
            loss += reproj_loss

            losses.append(loss.item())
            self.val_loss.append(loss.item())

            pa_mpjpe_err = pa_mpjpe(data["mesh"], pred_mesh, self.joints_reg)

            mpjpe_err = mpjpe(data["mesh"], pred_mesh, self.joints_reg)

            v2v_err = v2v(
                data["mesh"],
                pred_mesh,
            )

            self.val_pampjpe.append(1000 * pa_mpjpe_err.item())
            self.val_mpjpe.append(1000 * mpjpe_err.item())
            self.val_v2v.append(1000 * v2v_err.item())

        self.plot_meshes_(
            pred_mesh[:4].to(self.device),
            show=False,
            rot=True,
            save=f"{self.follow.path_samples}/{epoch}-reconstruction.png",
        )
        self.plot_meshes_(
            data["mesh"].to(self.device)[:4],
            show=False,
            rot=True,
            save=f"{self.follow.path_samples}/{epoch}-real.png",
        )

        pred_v = pred_mesh
        cam = pred_cam
        raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
        self.plot_reproj_(
            raw_img[:4],
            pred_v[:4],
            cam[:4],
            show=False,
            save=f"{self.follow.path_samples}/{epoch}_reprojection.png",
        )

        return losses

    def eval_deterministic(self, visualize=True):
        self.model.eval()
        with torch.no_grad():
            lpampjpe = []
            lmpjpe = []
            lv2v = []
            limgname = []
            count = 0
            for data in tqdm(iter(self.validation_loader)):
                mesh = data["local_mesh"]
                mesh = mesh.to(self.device)
                indices = self.vqvae.get_codebook_indices(mesh)
                img_features = data["img"].to(self.device)
                limgname.extend(data["imgname"])

                if self.vit_backbone:
                    predicted_indices, pred_rot, pred_cam, mask = self.model(
                        indices, img_features[:, :, :, 32:-32], fixed_ratio=1
                    )
                else:
                    predicted_indices, pred_rot, pred_cam, mask = self.model(
                        indices, img_features, fixed_ratio=1
                    )

                _, mesh_indices = torch.max(predicted_indices.data, -1)
                mesh_indices = (
                    mesh_indices * mask + indices * (~mask.to(torch.bool))
                ).type(torch.int64)
                mesh_canonical = self.vqvae.decode(mesh_indices).cpu()
                rotmat = rotation_6d_to_matrix(pred_rot)
                pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

                pa_mpjpe_err = pa_mpjpe(data["mesh"], pred_mesh, self.joints_reg)

                mpjpe_err = mpjpe(data["mesh"], pred_mesh, self.joints_reg)

                v2v_err = v2v(
                    data["mesh"],
                    pred_mesh,
                )

                lpampjpe.append(1000 * pa_mpjpe_err.item())
                lmpjpe.append(1000 * mpjpe_err.item())
                lv2v.append(1000 * v2v_err.item())

                if visualize:
                    count += 1
                    self.plot_meshes_(
                        data["mesh"][:4],
                        show=False,
                        rot=True,
                        save=f"{self.follow.path_samples}/{count}_gt.png",
                    )
                    raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
                    self.plot_img_(
                        raw_img[:4],
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_img.png",
                    )
                    self.plot_reproj_(
                        raw_img[:4],
                        pred_mesh[:4],
                        pred_cam[:4],
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_reprojection.png",
                    )
                    self.plot_meshes_(
                        pred_mesh[:4],
                        show=False,
                        rot=True,
                        save=f"{self.follow.path_samples}/{count}_reconstructed.png",
                    )

            print(f"V2V: {mean(lv2v)}, MPJPE: {mean(lmpjpe)}, PAMPJPE {mean(lpampjpe)}")

            dict_results = {
                "imgname": limgname,
                "pampjpe": lpampjpe,
                "mpjep": lmpjpe,
                "v2v": lv2v,
            }
            df = pd.DataFrame(dict_results)
            df.to_csv(f"{self.follow.path}/results.csv", index=False)

        return v2v

    def eval_stochastic(self, steps=5, temp=1, sample_size=25, visualise=True):
        self.model.eval()
        with torch.no_grad():
            lpampjpe = []
            lmpjpe = []
            lv2v = []
            limgname = []
            count = 0
            for data in tqdm(iter(self.validation_loader)):
                limgname.append(data["imgname"])

                img_features = data["img"].to(self.device)
                img_features = img_features.repeat(sample_size, 1, 1, 1)

                if self.vit_backbone:
                    mesh_indices, pred_rot, _ = self.model.generate(
                        img_features[:, :, :, 32:-32], nb_steps=steps, gen_temp=temp
                    )
                else:
                    mesh_indices, pred_rot, _ = self.model.generate(
                        img_features, nb_steps=steps, gen_temp=temp
                    )
                mesh_canonical = self.vqvae.decode(mesh_indices).cpu()
                rotmat = rotation_6d_to_matrix(pred_rot).cpu()
                pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1)

                pa_mpjpe_err = pa_mpjpe(
                    data["mesh"].repeat(sample_size, 1, 1),
                    mesh_canonical,
                    self.joints_reg,
                    reduction=False,
                    dim=1,
                )
                list_pampjpe = (1000 * pa_mpjpe_err).tolist()
                lpampjpe.append(min(list_pampjpe))

                mpjpe_err = mpjpe(
                    data["mesh"].repeat(sample_size, 1, 1),
                    pred_mesh,
                    self.joints_reg,
                    reduction=False,
                    dim=1,
                )
                list_mpjpe = (1000 * mpjpe_err).tolist()
                lmpjpe.append(min(list_mpjpe))

                v2v_err = v2v(
                    data["mesh"].repeat(sample_size, 1, 1),
                    pred_mesh,
                    reduction=False,
                    dim=1,
                )
                list_v2v = (1000 * v2v_err).tolist()
                lv2v.append(min(list_v2v))

                if visualise:
                    count += 1
                    self.plot_meshes_(
                        data["mesh"],
                        show=False,
                        rot=True,
                        save=f"{self.follow.path_samples}/{count}_gt.png",
                    )
                    raw_img = data["raw_img"].cpu().numpy().transpose(0, 2, 3, 1)
                    self.plot_img_(
                        raw_img,
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_img.png",
                    )

                    if self.vit_backbone:
                        mesh_indices, _, pred_cam = self.model.generate(
                            img_features[:, :, :, 32:-32], nb_steps=1, gen_temp=0
                        )
                    else:
                        mesh_indices, _, pred_cam = self.model.generate(
                            img_features, nb_steps=1, gen_temp=0
                        )

                    deterministic_mesh = self.vqvae.decode(mesh_indices).cpu()

                    rotmat = rotation_6d_to_matrix(pred_rot).cpu()
                    deterministic_mesh_oriented = (
                        rotmat @ deterministic_mesh.transpose(2, 1)
                    ).transpose(2, 1)[:1]

                    var = mesh_variance(
                        pred_mesh,
                        torch.mean(pred_mesh, dim=0, keepdim=True),
                        reduction=False,
                    ).unsqueeze(0)
                    mesh_colors = get_colors_from_diff_pc(
                        diff_pc=var, min_error=0, max_error=0.03
                    )

                    self.plot_reproj_(
                        raw_img,
                        deterministic_mesh_oriented,
                        pred_cam,
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_reprojection.png",
                    )
                    self.plot_meshes_(
                        deterministic_mesh[:1].cpu(),
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_{steps}_{temp}_uncertainty.svg",
                        colors=torch.from_numpy(mesh_colors),
                    )
                    self.plot_meshes_(
                        deterministic_mesh_oriented.cpu(),
                        show=False,
                        save=f"{self.follow.path_samples}/{count}_{steps}_{temp}_uncertainty_oriented.svg",
                        colors=torch.from_numpy(mesh_colors),
                        rot=True,
                    )

            print(f"V2V: {mean(lv2v)}, MPJPE: {mean(lmpjpe)}, PAMPJPE {mean(lpampjpe)}")

            dict_results = {
                "imgname": limgname,
                "pampjpe": lpampjpe,
                "mpjep": lmpjpe,
                "v2v": lv2v,
            }
            df = pd.DataFrame(dict_results)
            df.to_csv(f"{self.follow.path}/results.csv", index=False)

    def load(self, path: str = "", optimizer: bool = True):
        print("LOAD [", end="")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        self.load_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(
            f"model: ok  | optimizer:{optimizer}  |  loss: {loss}  |  epoch: {self.load_epoch}]"
        )
