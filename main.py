from inference import PSAM

# 模型配置文件和权重文件
model_cfg = {
    "DINO_WEIGHT_PATH": "weights/GSA_weights/groundingdino_swinb_cogcoor.pth",
    "DINO_CFG_PATH": "groundingdino/config/GroundingDINO_SwinB.py",
    "SAM_WEIGHT_PATH": "weights/GSA_weights/sam_vit_h_4b8939.pth",
    "CLIP_WEIGHT_DIR": "weights/CLIP_weights/"
}

# prompts提示，可自定义类别列表
# 模型会根据不同的prompts提示，生成不同的掩码
# category_cfg = {
#     "landcover_prompts": ['building', 'low vegetation', 'tree', 'water', 'shed', 'road', 'lake', 'bare soil',],
#     "landcover_prompts_cn": ['建筑', '低矮植被', '树木', '水体', '棚屋', '道路', '湖泊', '裸土'],
#     "cityobject_prompts": ['car', 'truck', 'bus', 'train', 'ship', 'boat'],
#     "cityobject_prompts_cn": ['轿车', '卡车', '巴士', '列车', '船(舰)', '船(舶)']
# }
category_cfg = {
    "landcover_prompts": [ 'building', 'water', 'tree', 'road','shed', 'cropland','grassland', 'Agricultural Fields','bare soil'],
    "landcover_prompts_cn": ['建筑', '水体', '树木', '道路', '棚屋', '农田', '草地', '农用地','裸土'],
    "cityobject_prompts": ['car', 'truck','train'],
    "cityobject_prompts_cn": ['轿车', '货车','火车']
}

gpus = ["1"]

# matplotlib使用中文绘制
cn_style = False # 是否使用中文
font_style_path = '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc' # 中文字体路径，可通过fc-list命令查看系统中所安装的字体

if __name__ == "__main__":
    psam = PSAM(model_cfg, category_cfg, gpus)

    # img_path = "/home/piesat/data/无人机全景图/panorama01-04/match_imgs/CD_dataset/01->03/A_B/A/100_right_0_1_hw(2701,672).png"
    # img_path = "/home/piesat/media/ljh/pycharm_project__ljh/panorama_sam/photos/c1.png"
    file_path = '/home/piesat/data/无人机全景图/panorama01-04/match_imgs/CD_dataset/cwptys_tmp/A'
    save_path = '/home/piesat/media/ljh/pycharm_project__ljh/panorama_sam/photos/croplands/'

    import os
    files = []
    for root, dirs, filenames in os.walk(file_path):
        for filename in filenames:
            in_img_path = os.path.join(root, filename)
            out_img_path = os.path.join(save_path, filename)
            psam.load_image(in_img_path)
            panoptic_inds = psam.generate_panoptic_mask()
            psam.plt_draw_image(cn_style, font_style_path, out_img_path)
            print(panoptic_inds.shape) # panoptic_inds：单通道掩码图像

