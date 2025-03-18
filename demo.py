import os
import torch
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import hydra
from omegaconf import DictConfig
from mega import (
    CVQMAE,
    hrnet_w48,
)
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
)
import mesh_vq_vae
from mega.utils.demo_renderer import Renderer, cam_crop_to_full
from mega.data.dataset_demo import DemoDataset

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


@hydra.main(config_path="configs/config_cvqmae", config_name="config_demo", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load YOLOv8 Human Detector
    weights_path = "body_models/"
    if not os.path.exists(weights_path):
        from ultralytics.utils.downloads import download
        download("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt", weights_path)

    # Load YOLOv8 model from the specified path
    yolo_model = YOLO(os.path.join(weights_path,"yolov8x.pt"))

    # Load SMPL
    ref_bm_path = "body_models/smplh/neutral/model.npz"
    ref_bm = np.load(ref_bm_path)

    # Load HMR Model
    os.chdir(hydra.utils.get_original_cwd())

    """ Load Backbone """
    backbone = hrnet_w48(pretrained_ckpt_path=cfg.backbone.pretrained, downsample=True, use_conv=True)

    """ Load MeshRegressor Model """
    mega = CVQMAE(backbone=backbone, **cfg.model)
    mega.load("checkpoint/CVQMAE/mega_hrnet")  # Load pre-trained weights
    mega.to(device).eval()

    convmesh_model = mesh_vq_vae.FullyConvAE(cfg.modelconv, test_mode=True)
    mesh_vqvae = mesh_vq_vae.MeshVQVAE(convmesh_model, **cfg.vqvaemesh)
    mesh_vqvae.load(path_model="checkpoint/MESH_VQVAE/mesh_vqvae_54")
    mesh_vqvae.to(device)
    convmesh_model.init_test_mode()

    # Create output directory
    os.makedirs("demo_out", exist_ok=True)
    
    input_path = "demo_data/"

    renderer = Renderer(1000, 224, faces=ref_bm["f"].astype(np.int32))

    if os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if f.lower().endswith((".jpg", ".png"))]
        for file in tqdm(files, desc="Processing"):
            file_path = os.path.join(input_path, file)
            img_cv2 = cv2.imread(file_path)

            boxes = yolo_model.predict(img_cv2, 
                                device='cuda', 
                                classes=00, 
                                conf=0.5, 
                                save=False, 
                                verbose=False
                                    )[0].boxes.xyxy.detach().cpu().numpy()
            
            dataset = DemoDataset(img_cv2, boxes)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

            with torch.no_grad():
                all_verts = []
                all_cam_t = []
                for batch in dataloader:  
                    # Run MEGA
                    img_fn, _ = os.path.splitext(os.path.basename(file_path))
                    img = batch["img"].to(device)
                    indices = torch.zeros((img.shape[0],54), dtype=torch.int).to(device)
                    predicted_indices, pred_rot, pred_cam, mask = mega(
                                indices, img, fixed_ratio=1
                            )
                    _, mesh_indices = torch.max(predicted_indices.data, -1)
                    mesh_indices = (
                        mesh_indices * mask + indices * (~mask.to(torch.bool))
                    ).type(torch.int64)
                    mesh_canonical = mesh_vqvae.decode(mesh_indices)[:img.shape[0]].cpu()
                    rotmat = rotation_6d_to_matrix(pred_rot)
                    pred_mesh = (rotmat @ mesh_canonical.transpose(2, 1)).transpose(2, 1).squeeze(0).detach().cpu().numpy()
                    all_verts.append(pred_mesh)

                    # Convert the predictions from the frame to the camera coordinates system
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()
                    scaled_focal_length = 1000 / 224 * img_size.max()
                    cam_t = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).squeeze(0).detach().cpu().numpy()
                    all_cam_t.append(cam_t)


                if len(all_verts) > 0:
                    misc_args = dict(
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        focal_length=scaled_focal_length,
                    )
                    cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[0], **misc_args)
    
                    # Overlay image
                    input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                    input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                    input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
    
                    cv2.imwrite(os.path.join("demo_out", f'{img_fn}_all.png'), 255*input_img_overlay[:, :, ::-1])
    

if __name__ == "__main__":
    main()













