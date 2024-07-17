from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
from ...base import Train
from ...model import VQMAE
from mesh_vq_vae import MeshVQVAE
import matplotlib.pyplot as plt
from .follow_up_mae import Follow
import math
from einops import rearrange
from math import sqrt
from .idr_torch import IDR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib.gridspec import GridSpec
from ...utils.mesh_render import renderer
from ...utils.eval import average_pairwise_distance
from pytorch3d.io import save_obj


class VQMAE_Train(Train):
    def __init__(
        self,
        mae: VQMAE,
        vqvae: MeshVQVAE,
        training_data: Dataset,
        validation_data: Dataset,
        config_training: dict = None,
        multigpu_bool: bool = False,
        faces=None,
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

        if multigpu_bool:
            self.model = DDP(
                self.model,
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

        """ Config """
        self.config_training = config_training
        self.load_epoch = 0
        self.step_count = 0
        self.parameters = dict()

        self.multigpu_bool = multigpu_bool

        """ Follow """
        self.follow = Follow(
            "vqmae",
            dir_save="checkpoint",
            multigpu_bool=multigpu_bool,
        )

        self.inference_batch = config_training["batch"]

    def one_epoch(self):
        self.model.train()
        losses = []
        for mesh in tqdm(iter(self.training_loader)):
            self.optimizer.zero_grad()
            self.step_count += 1
            mesh = mesh.to(self.device)
            with torch.no_grad():
                indices = self.vqvae.get_codebook_indices(
                    mesh.to(self.device),
                )
            predicted_indices, mask = self.model(indices)
            loss = self.criterion(
                predicted_indices.flatten(0, 1)[mask.flatten(0).to(torch.bool)],
                indices.flatten(0)[mask.flatten(0).to(torch.bool)].to(torch.long),
            )
            if not torch.isnan(loss):
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
        return losses

    def fit(self):
        for e in range(self.config_training["total_epoch"]):
            if self.multigpu_bool:
                self.training_loader.sampler.set_epoch(e)
                self.validation_loader.sampler.set_epoch(e)
            losses = self.one_epoch()
            with torch.no_grad():
                losses_val = self.eval(e)
            self.lr_scheduler.step()
            avg_loss_train = sum(losses) / len(losses)
            avg_loss_val = sum(losses_val) / len(losses_val)
            self.parameters = dict(
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.lr_scheduler.state_dict(),
                epoch=e,
                loss=avg_loss_train,
            )
            print(
                f"In epoch {e}, average traning loss is {avg_loss_train}. and average validation loss is {avg_loss_val}"
            )
            self.follow(
                epoch=e,
                loss_train=avg_loss_train,
                loss_validation=avg_loss_val,
                parameters=self.parameters,
            )

    def plot_train(self):
        pass

    def plot_meshes_(
        self,
        meshes,
        show: bool = True,
        save: str = None,
    ):
        images = renderer(meshes, self.f, meshes.device, rot=False)
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

    def eval(self, epoch):
        self.model.eval()
        losses = []
        for mesh in tqdm(iter(self.validation_loader)):
            mesh = mesh.to(self.device)
            indices = self.vqvae.get_codebook_indices(mesh)
            predicted_indices, mask = self.model(indices)
            loss = self.criterion(
                predicted_indices.flatten(0, 1)[mask.flatten(0).to(torch.bool)],
                indices.flatten(0)[mask.flatten(0).to(torch.bool)].to(torch.long),
            )
            if not torch.isnan(loss):
                losses.append(loss.item())
        _, predicted_indices = torch.max(predicted_indices.data, -1)
        predicted_indices = (
            predicted_indices * mask + indices * (~mask.to(torch.bool))
        ).type(torch.int64)
        self.plot_meshes_(
            mesh[:4].to(self.device),
            show=False,
            save=f"{self.follow.path_samples}/{epoch}_original.png",
        )
        predicted_mesh = self.vqvae.decode(predicted_indices)
        self.plot_meshes_(
            predicted_mesh[:4].to(self.device),
            show=False,
            save=f"{self.follow.path_samples}/{epoch}_reconstructed.png",
        )

        generated_indices = self.model.generate(batch_size=self.inference_batch)
        generated_meshes = self.vqvae.decode(generated_indices)
        self.plot_meshes_(
            generated_meshes[:4].to(self.device),
            show=False,
            save=f"{self.follow.path_samples}/{epoch}_generation",
        )
        return losses

    def eval_final(self):
        self.model.eval()

        number_steps = [20]
        starting_temperature = [1.3]
        generation_epochs = 10
        for epoch in range(generation_epochs):
            for steps in number_steps:
                for temp in starting_temperature:
                    generated_indices = self.model.generate(
                        batch_size=self.inference_batch, nb_steps=steps, gen_temp=temp
                    )
                    generated_meshes = self.vqvae.decode(generated_indices)
                    self.plot_meshes_(
                        generated_meshes.to("cpu"),
                        show=False,
                        save=f"{self.follow.path_samples}/generation_{steps}_{temp}_{epoch}.svg",
                    )

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
