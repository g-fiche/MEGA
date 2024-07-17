import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes

import torch
import numpy as np


def renderer(vertices, faces, device, colors=None, rot=True):
    R, T = look_at_view_transform(2.0, 0, 0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=256, blur_radius=0.0, faces_per_pixel=1, bin_size=-1
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    # create mesh from vertices
    if colors is None:
        verts_rgb = torch.ones_like(vertices)  # (1, V, 3)
    else:
        verts_rgb = colors
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    if rot:
        Rx = (
            torch.from_numpy(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            .float()
            .to(device)
        )
        vertices = vertices.to(device).reshape(-1, 3) @ Rx
    vertices = vertices.reshape(-1, 6890, 3)

    meshes = Meshes(
        vertices,
        faces.unsqueeze(0).repeat(len(vertices), 1, 1).to(vertices),
        textures=textures,
    )
    images = renderer(meshes)
    return images[:, :, :, :-1]
