import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import torch

from utils.data_utils import generate_color_list
from utils.load_models import load_clip_model, load_dino_model, load_sam_model
from utils.func_utils import dino_detection, sam_masks_from_dino_boxes, clipseg_segmentation, \
    clip_and_shrink_preds, sample_points_based_on_preds, sam_mask_from_points, preds_to_semantic_inds


class PSAM(object):
    def __init__(self, model_cfg, category_cfg, gpu_ids):
        # 初始化GroundingDINO、SAM、CLIPSeg模型
        self.device = torch.device("cuda:%s" % gpu_ids[0] if torch.cuda.is_available() and len(gpu_ids) > 0 else "cpu")
        self.groundingdino_model = load_dino_model(model_cfg["DINO_CFG_PATH"], model_cfg["DINO_WEIGHT_PATH"], self.device)
        self.sam_predictor = load_sam_model(model_cfg["SAM_WEIGHT_PATH"], self.device)
        self.clipseg_processor, self.clipseg_model = load_clip_model(model_cfg["CLIP_WEIGHT_DIR"], self.device)

        self.landcover_categories = category_cfg["landcover_prompts"]
        self.cityobject_categories = category_cfg["cityobject_prompts"]
        self.category_names = ["background"] + self.landcover_categories + self.cityobject_categories
        self.category_name_to_id = {
            category_name: i for i, category_name in enumerate(self.category_names)
        }
        self.category_id_to_name = {
            i: category_name for i, category_name in enumerate(self.category_names)
        }
        self.color_map = generate_color_list(len(self.category_names))

        self.landcover_categories_cn = category_cfg["landcover_prompts_cn"]
        self.cityobject_categories_cn = category_cfg["cityobject_prompts_cn"]
        self.category_names_cn = ["背景"] + self.landcover_categories_cn + self.cityobject_categories_cn
        self.category_id_to_name_cn = {
            i: category_name for i, category_name in enumerate(self.category_names_cn)
        }


    def load_image(self, img_path):
        # 读取图像并进行SAM的图像编码
        image = Image.open(img_path)
        self.image = image.convert("RGB")
        self.image_array = np.asarray(self.image)
        self.sam_predictor.set_image(self.image_array)


    def generate_panoptic_mask(self, dino_box_threshold=0.2,
                               dino_text_threshold=0.20,
                               segmentation_background_threshold=0.1,
                               shrink_kernel_size=10,
                               num_samples_factor=300
    ):
        # 1.基于DINO的城市目标检测，并结合SAM进行分割
        cityobject_category_ids = []
        cityobject_masks = torch.empty(0)
        cityobject_boxes = []
        if len(self.cityobject_categories) > 0:
            cityobject_boxes, cityobject_category_ids, _ = dino_detection(
                self.groundingdino_model,
                self.image,
                self.cityobject_categories,
                self.category_name_to_id,
                dino_box_threshold,
                dino_text_threshold,
                self.device,
            )
            if len(cityobject_boxes) > 0:
                cityobject_masks = sam_masks_from_dino_boxes(
                    self.sam_predictor, self.image_array, cityobject_boxes, self.device
                )

        # 2.基于CLIP的地物分类，并结合SAM进行分割
        if len(self.landcover_categories) > 0:
            clipseg_preds, clipseg_semantic_inds = clipseg_segmentation(
                self.clipseg_processor,
                self.clipseg_model,
                self.image,
                self.landcover_categories,
                segmentation_background_threshold,
                self.device,
            )
            clipseg_semantic_inds_without_cityobject = clipseg_semantic_inds.clone()
            if len(cityobject_boxes) > 0:
                combined_cityobject_mask = torch.any(cityobject_masks, dim=0)
                clipseg_semantic_inds_without_cityobject[combined_cityobject_mask[0]] = 0
            clipsed_clipped_preds, relative_sizes = clip_and_shrink_preds(
                clipseg_semantic_inds_without_cityobject,
                clipseg_preds,
                shrink_kernel_size,
                len(self.landcover_categories) + 1,
            )
            sam_preds = torch.zeros_like(clipsed_clipped_preds)
            for i in range(clipsed_clipped_preds.shape[0]):
                clipseg_pred = clipsed_clipped_preds[i]
                num_samples = int(relative_sizes[i] * num_samples_factor)
                if num_samples == 0:
                    continue
                points = sample_points_based_on_preds(
                    clipseg_pred.cpu().numpy(), num_samples
                )
                if len(points) == 0:
                    continue
                pred = sam_mask_from_points(self.sam_predictor, self.image_array, points)

                sam_preds[i] = pred

            sam_semantic_inds = preds_to_semantic_inds(
                sam_preds, segmentation_background_threshold
            )

        # 3.结合城市目标和地物分类的掩码结果
        if len(self.landcover_categories) > 0:
            # 进行开闭运算
            self.panoptic_inds = sam_semantic_inds.clone().cpu().numpy().astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            self.panoptic_inds = cv2.morphologyEx(self.panoptic_inds, cv2.MORPH_OPEN, kernel)
            self.panoptic_inds = cv2.morphologyEx(self.panoptic_inds, cv2.MORPH_CLOSE, kernel)
        else:
            self.panoptic_inds = np.zeros((self.image_array.shape[0], self.image_array.shape[1]), dtype=np.uint8)

        for mask_cid in range(cityobject_masks.shape[0]):
            ind = cityobject_category_ids[mask_cid]
            mask_bool = cityobject_masks[mask_cid].squeeze(dim=0).cpu().numpy()
            self.panoptic_inds[mask_bool] = ind

        return self.panoptic_inds

    def plt_draw_image(self, cn_style=False, font_style_path=None, save_file_path =None):
        # 是否使用中文显示
        if cn_style==True and font_style_path is not None:
            cn_style = True
            font = FontProperties(fname=font_style_path)
            id_to_name = self.category_id_to_name_cn
        else:
            cn_style = False
            font = FontProperties()
            id_to_name = self.category_id_to_name

        # 使用unique函数和return_counts参数计算每种类别的占用像素个数
        unique_values, counts = np.unique(self.panoptic_inds, return_counts=True)
        count_map = {}
        bar_colors = []  # 储存每种类别的颜色
        for value, count in zip(unique_values, counts):
            count_map[id_to_name[value]] = count
            r, g, b = self.color_map[value]
            r = r / 255
            g = g / 255
            b = b / 255
            bar_colors.append((r, g, b, 1.0))
        x = list(count_map.keys())
        y = list(count_map.values())

        # 创建子图
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

        # 绘制原图
        axes[0, 0].imshow(self.image)

        # 绘制掩码图
        cm = [list(t) for t in self.color_map]
        cm = np.array(cm).astype('uint8')
        label_img = cm[self.panoptic_inds]
        axes[0, 1].imshow(Image.fromarray(label_img))

        # 绘制合并图
        draw_image = cv2.addWeighted(np.array(self.image), 0.7, label_img, 0.3, 0)
        axes[1, 0].imshow(Image.fromarray(draw_image))

        # 绘制柱状图
        axes[1, 1].bar(range(len(x)), y, label=x, color=bar_colors)
        # 添加数值标签
        for a, b in zip(range(len(x)), y):
            axes[1, 1].text(a, b, b, ha='center', va='bottom', fontproperties=font)
        # 添加标题和横纵坐标含义
        if cn_style:
            axes[1, 1].set_title('统计每个类别占用的像素', fontproperties=font)
            axes[1, 1].set_xlabel('类别', fontproperties=font)
            axes[1, 1].set_ylabel('像素', fontproperties=font)
        else:
            axes[1, 1].set_title('Pixel Count', fontproperties=font)
            axes[1, 1].set_xlabel('Category', fontproperties=font)
            axes[1, 1].set_ylabel('Pixel', fontproperties=font)
        axes[1, 1].set_xticklabels([])
        # 添加图例
        axes[1, 1].legend(prop=font)

        # 调整子图间距
        plt.subplots_adjust(wspace=0.15, hspace=0.2)
        #保存图像
        plt.savefig(save_file_path)
        # # 显示图形
        # plt.show()
