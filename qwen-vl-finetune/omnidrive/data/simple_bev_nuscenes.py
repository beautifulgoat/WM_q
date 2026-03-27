import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


class SimpleBEVNuScenesBuilder:
    """
    将你当前 JSON 中的三帧多相机路径:
        item["images"]["past_2" / "past_1" / "current"]
    转成 Simple-BEV 风格的相机输入张量。

    返回:
        bev_imgs:     [3, 6, 3, H, W]
        bev_rots:     [3, 6, 3, 3]
        bev_trans:    [3, 6, 3]
        bev_intrins:  [3, 6, 4, 4]
        bev_ego_pose: [3, 4, 4]
    """

    DEFAULT_CAMERA_ORDER = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ]

    DEFAULT_TIME_KEYS = ["past_2", "past_1", "current"]

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-trainval",
        final_dim: Tuple[int, int] = (448, 800),   # (H, W)
        camera_order: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.dataroot = dataroot
        self.version = version
        self.final_dim = final_dim
        self.camera_order = camera_order or self.DEFAULT_CAMERA_ORDER
        self.to_tensor = transforms.ToTensor()

        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.dataroot,
            verbose=verbose
        )

        self.filename_to_sd_token = self._build_filename_index()

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------
    def build_triplet(self, images_dict: Dict[str, Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        images_dict 对应你 JSON 里的 item["images"]，例如:
        {
            "past_2": {...6 cams...},
            "past_1": {...6 cams...},
            "current": {...6 cams...},
            "next_1": {...}
        }
        """
        frames = []
        for time_key in self.DEFAULT_TIME_KEYS:
            if time_key not in images_dict:
                raise KeyError(f"Missing time key '{time_key}' in images_dict.")
            frames.append(self._build_one_timestamp(images_dict[time_key]))

        bev_imgs = torch.stack([x["imgs"] for x in frames], dim=0)          # [3, 6, 3, H, W]
        bev_rots = torch.stack([x["rots"] for x in frames], dim=0)          # [3, 6, 3, 3]
        bev_trans = torch.stack([x["trans"] for x in frames], dim=0)        # [3, 6, 3]
        bev_intrins = torch.stack([x["intrins"] for x in frames], dim=0)    # [3, 6, 4, 4]
        bev_ego_pose = torch.stack([x["ego_pose"] for x in frames], dim=0)  # [3, 4, 4]

        return {
            "bev_imgs": bev_imgs.float(),
            "bev_rots": bev_rots.float(),
            "bev_trans": bev_trans.float(),
            "bev_intrins": bev_intrins.float(),
            "bev_ego_pose": bev_ego_pose.float(),
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _build_filename_index(self) -> Dict[str, str]:
        """
        建立:
            'samples/CAM_FRONT/xxx.jpg' -> sample_data_token
        的索引。
        """
        index = {}
        for sd in self.nusc.sample_data:
            if sd.get("sensor_modality") != "camera":
                continue
            rel = self._normalize_relpath(sd["filename"])
            index[rel] = sd["token"]
        return index

    def _normalize_relpath(self, path: str) -> str:
        path = path.replace("\\", "/").strip()
        if path.startswith("./"):
            path = path[2:]

        # 如果传进来的是绝对路径，则转成相对 dataroot 的路径
        if os.path.isabs(path):
            path = os.path.relpath(path, self.dataroot)
            path = path.replace("\\", "/")
        return path

    def _lookup_sample_data(self, rel_or_abs_path: str) -> Dict:
        key = self._normalize_relpath(rel_or_abs_path)

        token = self.filename_to_sd_token.get(key, None)
        if token is not None:
            return self.nusc.get("sample_data", token)

        # 兜底：有些情况下路径可能只给了尾部，尝试 endswith 匹配
        matches = [
            tok for rel, tok in self.filename_to_sd_token.items()
            if rel.endswith(key)
        ]
        if len(matches) == 1:
            return self.nusc.get("sample_data", matches[0])

        raise KeyError(
            f"Cannot map image path to nuScenes sample_data: {rel_or_abs_path}"
        )

    def _build_one_timestamp(self, cam_paths: Dict[str, str]) -> Dict[str, torch.Tensor]:
        imgs = []
        rots = []
        trans = []
        intrins = []

        ego_pose_4x4 = None

        for cam in self.camera_order:
            if cam not in cam_paths:
                raise KeyError(f"Camera '{cam}' missing in timestamp dict.")

            sd = self._lookup_sample_data(cam_paths[cam])

            img_rel = self._normalize_relpath(sd["filename"])
            img_abs = os.path.join(self.dataroot, img_rel)

            img = Image.open(img_abs).convert("RGB")
            W, H = img.size

            calib = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            intrin_4x4 = self._camera_intrinsic_to_4x4(calib["camera_intrinsic"])
            rot = torch.tensor(
                Quaternion(calib["rotation"]).rotation_matrix,
                dtype=torch.float32
            )
            tran = torch.tensor(calib["translation"], dtype=torch.float32)

            # 与官方 val/test 路线一致：固定 resize，无随机 crop
            img_resized, intrin_4x4 = self._resize_and_update_intrinsics(
                img, intrin_4x4, orig_hw=(H, W), final_hw=self.final_dim
            )

            imgs.append(self.to_tensor(img_resized))     # [3, H, W], range [0,1]
            rots.append(rot)                             # [3, 3]
            trans.append(tran)                           # [3]
            intrins.append(intrin_4x4)                   # [4, 4]

            if ego_pose_4x4 is None:
                ego = self.nusc.get("ego_pose", sd["ego_pose_token"])
                ego_pose_4x4 = torch.tensor(
                    transform_matrix(
                        ego["translation"],
                        Quaternion(ego["rotation"]),
                        inverse=False
                    ),
                    dtype=torch.float32
                )

        return {
            "imgs": torch.stack(imgs, dim=0),           # [6, 3, H, W]
            "rots": torch.stack(rots, dim=0),           # [6, 3, 3]
            "trans": torch.stack(trans, dim=0),         # [6, 3]
            "intrins": torch.stack(intrins, dim=0),     # [6, 4, 4]
            "ego_pose": ego_pose_4x4,                   # [4, 4]
        }

    def _camera_intrinsic_to_4x4(self, K_3x3) -> torch.Tensor:
        K = torch.eye(4, dtype=torch.float32)
        K[:3, :3] = torch.tensor(K_3x3, dtype=torch.float32)
        return K

    def _resize_and_update_intrinsics(
        self,
        img: Image.Image,
        intrin_4x4: torch.Tensor,
        orig_hw: Tuple[int, int],
        final_hw: Tuple[int, int],
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        仿照 Simple-BEV val/test 路线：
        - resize 到 final_dim
        - 不做随机 crop
        - 只缩放 intrinsics
        """
        orig_h, orig_w = orig_hw
        final_h, final_w = final_hw

        sx = float(final_w) / float(orig_w)
        sy = float(final_h) / float(orig_h)

        out = img.resize((final_w, final_h), Image.BILINEAR)

        K = intrin_4x4.clone()
        K[0, 0] *= sx   # fx
        K[1, 1] *= sy   # fy
        K[0, 2] *= sx   # cx
        K[1, 2] *= sy   # cy

        return out, K