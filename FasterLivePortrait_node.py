import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from .src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline


class LivePortraitModelLoader():
    base_path = os.getcwd()
    config_dir = os.path.join(base_path,
                              "custom_nodes",
                              "FasterLivePortraitWithAnimal-custom",
                              "configs")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            },
            "optional": {
                 "config": (
                    os.listdir(LivePortraitModelLoader.config_dir),
                    {"default": 'trt_infer.yaml'}),
                "dsize": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "srx_scale": ("FLOAT", {"default": 2.3, "min": 1.0, "max": 4.0, "step": 0.01}),
                "srx_vx_ratio": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "srx_vy_ratio": ("FLOAT", {"default": -0.125, "min": -1.0, "max": 1.0, "step": 0.01}),
                "dri_scale": ("FLOAT", {"default": 2.2, "min": 1.0, "max": 4.0, "step": 0.01}),
                "dri_vx_ratio": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "dri_vy_ratio": ("FLOAT", {"default": -0.1, "min": -1.0, "max": 1.0, "step": 0.01}),
                "flag_normalize_lip": ("BOOLEAN", {"default": True}),
                "lip_normalize_threshold": ("FLOAT", {"default": 0.03, "min": 0, "max": 1.0, "step": 0.01}),

                "driving_type": (
                    [
                        'human',
                        'animal',
                    ], {
                        "default": 'human'
                    }),
            }
        }

    RETURN_TYPES = ("LIVEPORTRAITPIPE",)
    RETURN_NAMES = ("live_portrait_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "FasterLivePortrait"

    def loadmodel(self, config='trt_infer.yaml',
                  dsize=512,
                  srx_scale=2.3,
                  srx_vx_ratio=0,
                  srx_vy_ratio=-0.125,
                  dri_scale=2.2,
                  dri_vx_ratio=0,
                  dri_vy_ratio=-0.1,
                  flag_normalize_lip=True,
                  lip_normalize_threshold=0.03,
                  driving_type='human',
                  ):
        # 加载配置文件
        infer_cfg = OmegaConf.load(os.path.join(LivePortraitModelLoader.config_dir, config))
        infer_cfg['crop_params']['src_dsize'] = dsize
        infer_cfg['crop_params']['src_scale'] = srx_scale
        infer_cfg['crop_params']['src_vx_ratio'] = srx_vx_ratio
        infer_cfg['crop_params']['src_vy_ratio'] = srx_vy_ratio
        infer_cfg['crop_params']['dri_scale'] = dri_scale
        infer_cfg['crop_params']['dri_vx_ratio'] = dri_vx_ratio
        infer_cfg['crop_params']['dri_vy_ratio'] = dri_vy_ratio
        infer_cfg['infer_params']['flag_normalize_lip'] = flag_normalize_lip
        infer_cfg['infer_params']['lip_normalize_threshold'] = lip_normalize_threshold
        is_animal = True if driving_type == 'animal' else False
        # 加载LivePortrait模型管道
        pipeline = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=is_animal)
        return (pipeline, )


class FasterLivePortraitProcess():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "pipeline": ("LIVEPORTRAITPIPE",),
            "source_image": ("IMAGE",),
            "driving_images": ("IMAGE",),
            "batch_mode": ("BOOLEAN", {"default": False}),
            "face_index": ("INT", {"default": 0, "min": 0, "max": 100}),
            "max_number": ("INT", {"default": 3, "min": 1, "max": 5})
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("cropped_images", "full_images",)
    FUNCTION = "process"
    CATEGORY = "FasterLivePortrait"

    def process(self, pipeline, source_image, driving_images,
                batch_mode=False, face_index=0, max_number=3):
        source_image_np = (source_image * 255).byte().numpy()
        driving_images_np = (driving_images * 255).byte().numpy()

        if source_image_np.ndim > 3:
            source_image_np = source_image_np[0]

        pipeline.init_vars()
        cropped_images, full_images = self.run(pipeline,
                                               source_image_np,
                                               driving_images_np,
                                               batch_mode,
                                               face_index,
                                               max_number)
        cropped_images = cropped_images.astype(np.float32) / 255.0
        cropped_images = torch.from_numpy(cropped_images)
        full_images = full_images.astype(np.float32) / 255.0
        full_images = torch.from_numpy(full_images)

        return (cropped_images, full_images)

    def run(self, pipeline, src_image, driving_images,
            batch_mode=False, face_index=0, max_number=3):
        src_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2BGR)
        ret = pipeline.prepare_source(src_image, realtime=False)
        if not ret:
            raise Exception("no face in src_image exit!")

        if batch_mode:
            src_infos = pipeline.src_infos[0]
            if max_number < 1 or max_number > len(src_infos):
                max_number = len(src_infos)
            src_infos = src_infos[:max_number]
        else:
            src_infos = pipeline.src_infos[0][face_index]
            src_infos = [src_infos]

        # 逐帧处理
        cropped_images = []
        full_images = []
        with tqdm(driving_images, desc="infer") as pbar:
            for frame in pbar:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, out_crop, out_org = dri_crop, out_crop, out_org = pipeline.run(frame,
                                                                                  pipeline.src_imgs[0],
                                                                                  src_infos)
                if out_crop is not None:
                    cropped_images.append(out_crop)
                    full_images.append(out_org)
        cropped_images = np.array(cropped_images)
        full_images = np.array(full_images)
        return cropped_images, full_images