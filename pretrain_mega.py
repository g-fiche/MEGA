from mega import (
    VQMAE,
    VQMAE_Train,
    set_seed,
)
from mesh_vq_vae import (
    MeshVQVAE,
    FullyConvAE,
    DatasetMeshFromSmpl,
    DatasetMeshTest,
)
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import torch

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-p", "--path", type=str, help="Path to H5", default="datasets")

args = parser.parse_args()
path = args.path
print(os.listdir(args.path))


@hydra.main(
    config_path="configs/config_vqmae",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    set_seed()

    """Data"""
    training_data = DatasetMeshFromSmpl(folder=cfg.training_data.file)
    validation_data = DatasetMeshTest(
        dataset_file=cfg.validation_data.file, proportion=0.1
    )

    """ ConvMesh VQVAE model """
    convmesh_model = FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model="checkpoint/MESH_VQVAE/mesh_vqvae_54")
    convmesh_model.init_test_mode()
    pytorch_total_params = sum(
        p.numel() for p in mesh_vqvae.parameters() if p.requires_grad
    )
    print(f"VQVAE-Mesh: {pytorch_total_params}")

    """ MeshRegressor model """
    vqmae = VQMAE(
        **cfg.model,
    )
    pytorch_total_params = sum(p.numel() for p in vqmae.parameters() if p.requires_grad)
    print(f"Regressor: {pytorch_total_params}")

    ref_bm_path = "body_models/smplh/neutral/model.npz"
    ref_bm = np.load(ref_bm_path)

    """ Training """
    pretrain_mesh_regressor = VQMAE_Train(
        vqmae,
        mesh_vqvae,
        training_data,
        validation_data,
        cfg.train,
        faces=torch.from_numpy(ref_bm["f"].astype(np.int32)),
    )
    pretrain_mesh_regressor.fit()

    # vqmae.load("checkpoint/VQMAE/mega_pretrained")
    # pretrain_mesh_regressor.eval_final()


if __name__ == "__main__":
    main()
