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
import tempfile
import gradio as gr
from PIL import Image

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

    renderer = Renderer(1000, 224, faces=ref_bm["f"].astype(np.int32))

    def infer(in_pil_img, in_threshold=0.8, out_pil_img=None):
        # Convert RGB to BGR
        img_cv2 = np.array(in_pil_img)[:, :, ::-1].copy()

        boxes = yolo_model.predict(img_cv2, 
                            device='cuda', 
                            classes=00, 
                            conf=in_threshold, 
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

                out_pil_img =  Image.fromarray((input_img_overlay*255).astype(np.uint8))
                return out_pil_img
            else:
                return None
    
    with gr.Blocks(title="MEGA", css=".gradio-container") as demo:

        gr.HTML("""<div style="font-weight:bold; text-align:center; color:royalblue;">MEGA</div>""")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input image", type="pil")
            with gr.Column():
                output_image = gr.Image(label="Reconstructions", type="pil")

        gr.HTML("""<br/>""")

        with gr.Row():
            threshold = gr.Slider(0, 1.0, value=0.6, label='Detection Threshold')
            send_btn = gr.Button("Infer")
            send_btn.click(fn=infer, inputs=[input_image, threshold], outputs=[output_image])

        gr.Examples([
                ['demo_data/pexels-labonheure-13020855.jpg'], 
                ['demo_data/pexels-marcus-aurelius-6787357.jpg'],
                ['demo_data/pexels-polina-tankilevitch-6739044.jpg']
            ], 
            inputs=[input_image, 0.6])

    demo.launch()


if __name__ == "__main__":
    main()













