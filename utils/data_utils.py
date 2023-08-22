import random
from supervision.draw.color import ColorPalette

from groundingdino.datasets import transforms as T


def load_image_for_dino(image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dino_image, _ = transform(image, None)
    return dino_image


def generate_color_list(num_colors):
    colors = [(0, 0, 0)]  # 第一个颜色为黑色
    for c in ColorPalette.default().colors:
        colors.append(c.as_rgb())
    color_set = set(colors)
    if len(colors) >= num_colors:
        return colors[:num_colors]
    while len(colors) < num_colors:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = (r, g, b)
        if color not in color_set:
            colors.append(color)
            color_set.add(color)
    return colors