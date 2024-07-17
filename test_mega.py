from mega import (
    CVQMAE,
    CVQMAE_Train,
    MixedDataset,
    set_seed,
    hrnet_w48,
    vit,
)
import mesh_vq_vae
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import torch
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-p", "--path", type=str, help="Path to H5", default="datasets")

args = parser.parse_args()
path = args.path
print(os.listdir(args.path))


@hydra.main(
    config_path="configs/config_cvqmae",
    config_name="config_hrnet",
    version_base=None,
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    set_seed()

    ref_bm_path = "body_models/smplh/neutral/model.npz"
    ref_bm = np.load(ref_bm_path)

    """Data"""
    test_data = MixedDataset(
        cfg.test_data.file,
        augment=False,
        flip=False,
        proportion=1,
    )

    """ Backbone """

    if cfg.backbone.type == "resnet":
        resnet_checkpoints = models.ResNet50_Weights.DEFAULT
        resnet_model = models.resnet50(weights=resnet_checkpoints)
        backbone = torch.nn.Sequential(*list(resnet_model.children())[:-2])
    elif cfg.backbone.type == "hrnet":
        pretrained_ckpt_path = cfg.backbone.pretrained
        backbone = hrnet_w48(
            pretrained_ckpt_path=pretrained_ckpt_path,
            downsample=True,
            use_conv=True,
        )
    else:
        backbone = vit()
        backbone.load_state_dict(
            torch.load(cfg.backbone.pretrained, map_location="cpu")["state_dict"]
        )

    """ ConvMesh VQVAE model """
    convmesh_model = mesh_vq_vae.FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = mesh_vq_vae.MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model="checkpoint/MESH_VQVAE/mesh_vqvae_54")
    convmesh_model.init_test_mode()
    pytorch_total_params = sum(
        p.numel() for p in mesh_vqvae.parameters() if p.requires_grad
    )
    print(f"Mesh-VQVAE: {pytorch_total_params}")

    """ MeshRegressor model """
    mesh_regressor = CVQMAE(
        backbone=backbone,
        **cfg.model,
    )
    pytorch_total_params = sum(
        p.numel() for p in mesh_regressor.parameters() if p.requires_grad
    )
    # Load the VQMAE pretrained on motion capture data
    mesh_regressor.load("checkpoint/CVQMAE/mega_hrnet")
    print(f"Regressor: {pytorch_total_params}")

    """Joint regressor"""
    J_regressor = torch.from_numpy(np.load("body_models/J_regressor_h36m.npy")).float()

    J_regressor_24 = torch.from_numpy(np.load("body_models/J_regressor_24.npy")).float()

    """ Training """
    pretrain_mesh_regressor = CVQMAE_Train(
        mesh_regressor,
        mesh_vqvae,
        test_data,
        test_data,
        cfg.train,
        faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
        joints_regressor=J_regressor,
        joints_regressor_smpl=J_regressor_24,
    )

    pretrain_mesh_regressor.eval_deterministic()
    # pretrain_mesh_regressor.eval_stochastic(sample_size=25)


if __name__ == "__main__":
    main()
