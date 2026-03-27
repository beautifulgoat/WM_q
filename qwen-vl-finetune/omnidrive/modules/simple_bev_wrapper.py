import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class SimpleBEVWrapper(nn.Module):
    """
    只负责：
    1) 加载原始 simple_bev 仓库
    2) 加载 camera-only checkpoint
    3) 接收 batch 里的 bev_* 张量
    4) 提取 past_2 / past_1 / current 三帧的 BEV 特征

    不负责：
    - 时序对齐
    - residual fusion
    - token 化
    - 注入 VLM
    """

    def __init__(
        self,
        simple_bev_root: str,
        ckpt_path: str,
        device: Optional[str] = None,
        encoder_type: str = "res101",
        rand_flip: bool = False,
        freeze: bool = True,
        use_pre_decoder_feat: bool = True,
        voxel_size: Tuple[int, int, int] = (200, 8, 200),  # X, Y, Z-ish placeholder config
        bounds: Tuple[float, float, float, float, float, float] = (-50.0, 50.0, -5.0, 5.0, -50.0, 50.0),
        debug: bool = False,
    ):
        super().__init__()

        self.simple_bev_root = simple_bev_root
        self.ckpt_path = ckpt_path
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_type = encoder_type
        self.rand_flip = rand_flip
        self.freeze = freeze
        self.use_pre_decoder_feat = use_pre_decoder_feat
        self.debug = debug

        # --------------------------------------------------
        # 1. 注入 simple_bev repo root 到 sys.path
        # --------------------------------------------------
        if self.simple_bev_root not in sys.path:
            sys.path.insert(0, self.simple_bev_root)

        # 延迟导入，避免路径问题
        from nets.segnet import Segnet
        import saverloader
        import utils.geom
        import utils.vox
        import utils.basic

        self._Segnet = Segnet
        self._saverloader = saverloader
        self._geom = utils.geom
        self._vox = utils.vox
        self._basic = utils.basic

        # --------------------------------------------------
        # 2. BEV 网格配置
        # --------------------------------------------------
        # 这里给一个和常见 nuScenes BEV 兼容的默认值；
        # 若你之后发现和 checkpoint 对不上，只需要改这里，不动其余主逻辑。
        self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX = bounds

        # 注意：Simple-BEV 内部常用 (Z, Y, X) 排布
        X, Y, Z = voxel_size
        self.X = X
        self.Y = Y
        self.Z = Z

        # --------------------------------------------------
        # 3. 构建原始 Segnet
        # --------------------------------------------------
        self.model = self._Segnet(
            self.Z,
            self.Y,
            self.X,
            vox_util=None,              # forward 时再传
            use_radar=False,
            use_lidar=False,
            use_metaradar=False,
            do_rgbcompress=True,
            encoder_type=self.encoder_type,
            rand_flip=self.rand_flip,
        )

        # --------------------------------------------------
        # 4. hook pre-decoder BEV feature
        # --------------------------------------------------
        self._hooked_feat_bev = None

        def _save_bev_feat(module, inp, out):
            self._hooked_feat_bev = out

        # 官方 forward 里 feat_bev = self.bev_compressor(feat_bev_)，很适合做 hook
        self._bev_hook_handle = None
        if self.use_pre_decoder_feat and hasattr(self.model, "bev_compressor"):
            self._bev_hook_handle = self.model.bev_compressor.register_forward_hook(_save_bev_feat)

        # --------------------------------------------------
        # 5. 加载 checkpoint
        # --------------------------------------------------
        self._load_checkpoint(self.ckpt_path)

        # --------------------------------------------------
        # 6. freeze
        # --------------------------------------------------
        # self.model.to(self.device_name)
        self.model = self.model.float()
        self.model.to(self.device_name)
        self.model.eval()

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    # ======================================================
    # checkpoint loading
    # ======================================================
    def _find_latest_ckpt_file(self, ckpt_dir: str) -> str:
        ckpt_files = []
        for name in os.listdir(ckpt_dir):
            if name.endswith(".pth") and name.startswith("model-"):
                ckpt_files.append(os.path.join(ckpt_dir, name))

        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint .pth found in directory: {ckpt_dir}")

        ckpt_files = sorted(ckpt_files)
        return ckpt_files[-1]


    def _load_checkpoint(self, ckpt_path: str):
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Simple-BEV checkpoint not found: {ckpt_path}")

        # A. 如果是目录，自己找最新 ckpt 文件
        if os.path.isdir(ckpt_path):
            ckpt_file = self._find_latest_ckpt_file(ckpt_path)
            if self.debug:
                print(f"[SimpleBEVWrapper] Loading checkpoint dir via manual CPU load: {ckpt_file}")
        else:
            ckpt_file = ckpt_path
            if self.debug:
                print(f"[SimpleBEVWrapper] Loading checkpoint file via manual CPU load: {ckpt_file}")

        # B. 一律 CPU 加载，避免多卡 map_location 到错误 cuda
        ckpt = torch.load(ckpt_file, map_location="cpu")

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if self.debug:
            print("[SimpleBEVWrapper] Missing keys:", missing)
            print("[SimpleBEVWrapper] Unexpected keys:", unexpected)

    # ======================================================
    # geometry helpers
    # ======================================================
    def _build_vox_util(self, B: int, device: torch.device):
        scene_centroid = torch.zeros(B, 3, device=device, dtype=torch.float32)

        vox_util = self._vox.Vox_util(
            Z=self.Z,
            Y=self.Y,
            X=self.X,
            scene_centroid=scene_centroid,
            bounds=(self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX),
            assert_cube=False,
        )
        return vox_util

    def _prepare_single_timestep_inputs(
        self,
        imgs: torch.Tensor,       # [B, 6, 3, H, W], [0,1]
        rots: torch.Tensor,       # [B, 6, 3, 3]
        trans: torch.Tensor,      # [B, 6, 3]
        intrins: torch.Tensor,    # [B, 6, 4, 4]
    ):
        """
        按官方 train_nuscenes.py 的风格准备：
        - rgb_camXs
        - pix_T_cams
        - cam0_T_camXs
        - vox_util
        """
        device = imgs.device
        B, S, C, H, W = imgs.shape

        # 官方做法：rgb_camXs = imgs.float().to(device) - 0.5
        rgb_camXs = imgs - imgs.new_tensor(0.5)
        pix_T_cams = intrins

        # 官方 train_nuscenes.py 里会把 intrins merge/split 后得到 pix_T_cams

        # 官方里 velo_T_cams = merge_rtlist(rots, trans)
        velo_T_cams = self._geom.merge_rtlist(rots, trans)
        

        # 官方里 cam0_T_camXs = get_camM_T_camXs(velo_T_cams, ind=0)
        cam0_T_camXs = self._geom.get_camM_T_camXs(velo_T_cams, ind=0)

        vox_util = self._build_vox_util(B=B, device=device)

        return rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util

    # ======================================================
    # core forward
    # ======================================================
    @torch.no_grad()
    def forward_single_timestep(
        self,
        imgs: torch.Tensor,       # [B, 6, 3, H, W]
        rots: torch.Tensor,       # [B, 6, 3, 3]
        trans: torch.Tensor,      # [B, 6, 3]
        intrins: torch.Tensor,    # [B, 6, 4, 4]
    ) -> Dict[str, torch.Tensor]:
        if next(self.model.parameters()).dtype != torch.float32:
            self.model.float()
        device = next(self.model.parameters()).device

        imgs = imgs.to(device=device, dtype=torch.float32, non_blocking=True)
        rots = rots.to(device=device, dtype=torch.float32, non_blocking=True)
        trans = trans.to(device=device, dtype=torch.float32, non_blocking=True)
        intrins = intrins.to(device=device, dtype=torch.float32, non_blocking=True)

        rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util = self._prepare_single_timestep_inputs(
            imgs=imgs,
            rots=rots,
            trans=trans,
            intrins=intrins,
        )

        print("param dtype:", next(self.model.parameters()).dtype)
        print("imgs dtype:", imgs.dtype)
        print("rgb_camXs dtype:", rgb_camXs.dtype)
        print("cam0_T_camXs dtype:", cam0_T_camXs.dtype)

        self._hooked_feat_bev = None
        with torch.cuda.amp.autocast(enabled=False):
            raw_e, feat_e, seg_e, center_e, offset_e = self.model(
                rgb_camXs=rgb_camXs,
                pix_T_cams=pix_T_cams,
                cam0_T_camXs=cam0_T_camXs,
                vox_util=vox_util,
                rad_occ_mem0=None,
            )

        out = {
            "raw_e": raw_e,
            "feat_e": feat_e,
            "seg_e": seg_e,
            "center_e": center_e,
            "offset_e": offset_e,
        }

        if self.use_pre_decoder_feat and self._hooked_feat_bev is not None:
            out["bev_feat"] = self._hooked_feat_bev
        else:
            # 退化备选：至少保证你能拿到一个可用 BEV 特征
            out["bev_feat"] = feat_e

        return out

    @torch.no_grad()
    def forward(
        self,
        bev_imgs: torch.Tensor,        # [B, 3, 6, 3, H, W]
        bev_rots: torch.Tensor,        # [B, 3, 6, 3, 3]
        bev_trans: torch.Tensor,       # [B, 3, 6, 3]
        bev_intrins: torch.Tensor,     # [B, 3, 6, 4, 4]
        bev_ego_pose: Optional[torch.Tensor] = None,   # [B, 3, 4, 4]
    ) -> Dict[str, torch.Tensor]:
        """
        返回 past_2 / past_1 / current 三帧特征。
        时间顺序与你 data builder 保持一致:
            0 -> past_2
            1 -> past_1
            2 -> current
        """
        assert bev_imgs.dim() == 6, f"bev_imgs shape error: {bev_imgs.shape}"
        assert bev_rots.dim() == 5, f"bev_rots shape error: {bev_rots.shape}"
        assert bev_trans.dim() == 4, f"bev_trans shape error: {bev_trans.shape}"
        assert bev_intrins.dim() == 5, f"bev_intrins shape error: {bev_intrins.shape}"

        feats = []
        aux = []

        T = bev_imgs.shape[1]
        if T != 3:
            raise ValueError(f"Expected 3 timesteps [past_2, past_1, current], but got {T}")

        for t in range(T):
            out_t = self.forward_single_timestep(
                imgs=bev_imgs[:, t],
                rots=bev_rots[:, t],
                trans=bev_trans[:, t],
                intrins=bev_intrins[:, t],
            )
            feats.append(out_t["bev_feat"])
            aux.append(out_t)

        res = {
            "bev_feat_t2": feats[0],
            "bev_feat_t1": feats[1],
            "bev_feat_t0": feats[2],
            "bev_feat_seq": torch.stack(feats, dim=1),  # [B, 3, C, Hb, Wb]
        }

        if bev_ego_pose is not None:
            res["ego_pose_seq"] = bev_ego_pose.to(feats[0].device, non_blocking=True)

        # 调试时保留原输出
        if self.debug:
            res["aux_t2"] = aux[0]
            res["aux_t1"] = aux[1]
            res["aux_t0"] = aux[2]

        return res


    def _move_model_to_device(self):
        has_meta = any(p.device.type == "meta" for p in self.model.parameters())
        if has_meta:
            raise RuntimeError(
                "Simple-BEV model is still on meta device. "
                "Do not initialize it inside OmniDriveVLA.__init__. "
                "Call init_simple_bev_sidecar() after from_pretrained instead."
            )
        self.model.to(self.device_name)


    def close(self):
        if self._bev_hook_handle is not None:
            self._bev_hook_handle.remove()
            self._bev_hook_handle = None