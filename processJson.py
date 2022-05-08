import json
import numpy as np
import os
import PIL.Image
import PIL.ImageDraw
import cv2

def create_def(path):
    if not os.path.exists(path):
        os.mkdir(path)

def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    #['shapes'][0]['points']
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    k = 0
    for i in range(len(points['shapes'])):
        k = points['shapes'][i]['points']
        xy = [tuple(point) for point in k]

        assert len(xy) > 2, "Polygon must have points more than 2"
        if points['shapes'][i]['label'] == "atopic dermatitis":
            draw.polygon(xy=xy, outline=1, fill=1)
            k = 1
        elif points['shapes'][i]['label'] == "psoriasis":
            draw.polygon(xy=xy, outline=1, fill=2)
            k = 2
        elif points['shapes'][i]['label'] == "wart":
            draw.polygon(xy=xy, outline=1, fill=3)
            k = 3

    mask = np.array(mask, dtype=int)
    if k == 1:
        mask[mask == 2] = 1
        mask[mask == 3] = 1
    if k == 2:
        mask[mask == 1] = 2
        mask[mask == 3] = 2
    if k == 3:
        mask[mask == 2] = 3
        mask[mask == 1] = 3
    return mask



def jsonToMask(path):
    with open(path, "r",encoding="utf-8") as f:
        dj = json.load(f)
    mask = shape_to_mask((dj['imageHeight'],dj['imageWidth']), dj, shape_type=None,line_width=1, point_size=1)
    mask_img = mask.astype(np.int)
    newPath = 'dataset-mask/masks/' + path.split('/')[1].split('.')[0] + '.png'
    cv2.imwrite(newPath, mask_img)

def imgToImg(path):
    img = cv2.imread(path)
    newPath = 'dataset-mask/images/' + path.split('/')[1].split('.')[0] + '.jpg'
    cv2.imwrite(newPath, img)


create_def('dataset-mask')
create_def('dataset-mask/images')
create_def('dataset-mask/masks')
path = 'dataset-json'
for i in os.listdir(path):
    if '.json' in i:
        jsonToMask(path+'/'+i)
    else:
        imgToImg(path+'/'+i)



