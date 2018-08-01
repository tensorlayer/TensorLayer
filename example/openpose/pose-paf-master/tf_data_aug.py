import math
import random
from data_process import PoseInfo
import cv2
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from tf_cal_maps import get_heatmap,get_vectormap
from faster_map_cal import get_vectormap as get_vectormap2
from faster_map_cal import get_heatmap as get_heatmap2
from numpy import linalg as LA
from pycocotools.coco import maskUtils
from tensorpack.dataflow.imgaug.geometry import RotationAndCropValid
import time
import tensorflow as tf
def crop_meta_image(image,annos,mask):
    _target_height=368
    _target_width =368
    if len(np.shape(image))==2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    height,width,_=np.shape(image)
    # print("the size of original img is:", height, width)
    if height<=width:
        ratio=_target_height/height
        new_width=int(ratio*width)
        if height==width:
            new_width=_target_height

        image,annos,mask=_resize_image(image,annos,mask,new_width,_target_height)

        for i in annos:
            if len(i) is not 19:
                print('Joints of person is not 19 ERROR FROM RESIZE')

        if new_width>_target_width:
            crop_range_x=np.random.randint(0, new_width-_target_width)
        else:
            crop_range_x=0
        image= image[:, crop_range_x:crop_range_x + 368,:]
        mask = mask[:, crop_range_x:crop_range_x + 368]
        joint_list= []
        new_joints = []
        #annos-pepople-joints (must be 19 or [])
        for people in annos:
            # print("number of keypoints is", np.shape(people))
            new_keypoints = []
            for keypoints in people:
                if keypoints[0] < -10 or keypoints[1] < -10:
                    new_keypoints.append((-1000, -1000))
                    continue
                top=crop_range_x+367
                if keypoints[0]>=crop_range_x and keypoints[0]<= top:
                    # pts = (keypoints[0]-crop_range_x, keypoints[1])
                    pts = (int(keypoints[0] - crop_range_x),int(keypoints[1]))
                else:
                    pts= (-1000,-1000)
                new_keypoints.append(pts)

            new_joints.append(new_keypoints)
            if len(new_keypoints) != 19:
                print('1:The Length of joints list should be 0 or 19 but actually:', len(new_keypoints))
        annos = new_joints

    if height>width:
        ratio = _target_width / width
        new_height = int(ratio * height)
        image,annos,mask = _resize_image(image,annos,mask,_target_width, new_height)

        for i in annos:
            if len(i) is not 19:
                print('Joints of person is not 19 ERROR')

        if new_height > _target_height:
            crop_range_y = np.random.randint(0, new_height - _target_height)

        else:
            crop_range_y = 0
        image = image[crop_range_y:crop_range_y + 368, :, :]
        mask = mask[crop_range_y:crop_range_y + 368, :]
        new_joints = []

        for people in annos:
            new_keypoints = []
            for keypoints in people:

                # case orginal points are not usable
                if keypoints[0] < 0 or keypoints[1] < 0:
                    new_keypoints.append((-1000, -1000))
                    continue
                # y axis coordinate change
                bot = crop_range_y + 367
                if keypoints[1] >= crop_range_y and keypoints[1] <= bot:
                    # pts = (keypoints[0], keypoints[1]-crop_range_y)
                    pts = (int(keypoints[0]), int (keypoints[1] - crop_range_y))
                    # if pts[0]>367 or pts[1]>367:
                    #     print('Error2')
                else:
                    pts =(-1000,-1000)

                new_keypoints.append(pts)

            new_joints.append(new_keypoints)
            if len(new_keypoints) != 19:
                print('2:The Length of joints list should be 0 or 19 but actually:', len(new_keypoints))

        annos = new_joints

    # mask = cv2.resize(mask, (46, 46), interpolation=cv2.INTER_AREA)
    return image,annos,mask

def _resize_image(image,annos,mask,_target_width,_target_height):
    # _target_height=368
    # _target_width =368

    #original image
    y,x,_=np.shape(image)

    ratio_y= _target_height/y
    ratio_x= _target_width/x

    new_joints=[]
    # update meta
    # meta.height=_target_height
    # meta.width =_target_width
    for people in annos:
        new_keypoints=[]
        for keypoints in people:
            if keypoints[0]<0 or keypoints[1]<0:
                new_keypoints.append((-1000, -1000))
                continue
            pts = (int(keypoints[0] * ratio_x+0.5), int(keypoints[1] * ratio_y+0.5))
            if pts[0] > _target_width-1 or pts[1] > _target_height-1:
                new_keypoints.append((-1000, -1000))
                continue

            new_keypoints.append(pts)
        new_joints.append(new_keypoints)
    annos=new_joints

    new_image = cv2.resize(image, (_target_width, _target_height), interpolation=cv2.INTER_AREA)
    new_mask = cv2.resize(mask, (_target_width, _target_height), interpolation=cv2.INTER_AREA)
    return new_image,annos,new_mask

def _rotate_coord(shape, newxy, point, angle):
    angle = -1 * angle / 180.0 * math.pi

    ox, oy = shape
    px, py = point

    ox /= 2
    oy /= 2

    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    new_x, new_y = newxy

    qx += ox - new_x
    qy += oy - new_y

    return int(qx + 0.5), int(qy + 0.5)

def pose_rotation(image,annos,mask):
    img_shape=np.shape(image)
    height  = img_shape[0]
    width   = img_shape[1]
    deg = random.uniform(-15.0, 15.0)

    img = image
    center = (img.shape[1] * 0.5, img.shape[0] * 0.5)       # x, y
    rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
    ret = cv2.warpAffine(img, rot_m, img.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    if img.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
    neww = min(neww, ret.shape[1])
    newh = min(newh, ret.shape[0])
    newx = int(center[0] - neww * 0.5)
    newy = int(center[1] - newh * 0.5)
    # print(ret.shape, deg, newx, newy, neww, newh)
    img = ret[newy:newy + newh, newx:newx + neww]

    # adjust meta data
    adjust_joint_list = []
    for joint in annos:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue

            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1, -1))
            #     continue
            x, y = _rotate_coord((width, height), (newx, newy), point, deg)
            # if x > neww or y > newh:
            #     adjust_joint.append((-1000, -1000))
            #     continue
            if x>neww-1 or y>newh-1:
                adjust_joint.append((-1000, -1000))
                continue
            if x < 0 or y < 0:
                adjust_joint.append((-1000, -1000))
                continue

            adjust_joint.append((x, y))
        adjust_joint_list.append(adjust_joint)
    joint_list = adjust_joint_list

    msk = mask
    center = (msk.shape[1] * 0.5, msk.shape[0] * 0.5)  # x, y
    rot_m = cv2.getRotationMatrix2D((int(center[0]), int(center[1])), deg, 1)
    ret = cv2.warpAffine(msk, rot_m, msk.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    if msk.ndim == 3 and msk.ndim == 2:
        ret = ret[:, :, np.newaxis]
    neww, newh = RotationAndCropValid.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
    neww = min(neww, ret.shape[1])
    newh = min(newh, ret.shape[0])
    newx = int(center[0] - neww * 0.5)
    newy = int(center[1] - newh * 0.5)
    # print(ret.shape, deg, newx, newy, neww, newh)
    msk = ret[newy:newy + newh, newx:newx + neww]
    return img, joint_list, msk
def random_flip(image,annos,mask_miss):
    flip_list=[0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16, 18]
    prob = random.uniform(0, 1.0)
    if prob > 0.5:
        return image,annos,mask_miss

    _, width, _ = np.shape(image)
    image = cv2.flip(image, 1)
    mask_miss=cv2.flip(mask_miss,1)
    new_joints = []
    for people in annos:
        new_keypoints = []
        for k in flip_list:
            point=people[k]
            if point[0] < 0 or point[1] < 0:
                new_keypoints.append((-1000, -1000))
                continue
            if point[0]>image.shape[1]-1 or point[1]>image.shape[0]-1:
                new_keypoints.append((-1000, -1000))
                continue
            if (width - point[0])>image.shape[1]-1:
                new_keypoints.append((-1000, -1000))
                continue
            new_keypoints.append((width - point[0], point[1]))
        new_joints.append(new_keypoints)
    annos=new_joints

    return image, annos, mask_miss

def pose_random_scale(image,annos,mask_miss):
    height=image.shape[0]
    width =image.shape[1]
    scalew = np.random.uniform(0.8, 1.2)
    scaleh = np.random.uniform(0.8, 1.2)
    # scalew =scaleh=np.random.uniform(0.5, 1.1)

    # scaleh=0.8934042054560039
    # scalew=1.0860957314059887
    neww = int(width * scalew)
    newh = int(height * scaleh)

    dst = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)
    mask_miss=cv2.resize(mask_miss, (neww, newh), interpolation=cv2.INTER_AREA)
    # adjust meta data
    adjust_joint_list = []
    for joint in annos:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0 or int(point[0] * scalew + 0.5) > neww or int(
            #                         point[1] * scaleh + 0.5) > newh:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((int(point[0] * scalew + 0.5), int(point[1] * scaleh + 0.5)))
        adjust_joint_list.append(adjust_joint)
    return dst,adjust_joint_list,mask_miss
def pose_resize_shortestedge_random(image,annos, mask):

    _target_height = 368
    _target_width = 368

    if len(np.shape(image))==2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    height,width,_=np.shape(image)


    ratio_w = _target_width / width
    ratio_h = _target_height / height
    ratio = min(ratio_w, ratio_h)
    target_size = int(min(width * ratio + 0.5, height * ratio + 0.5))
    random_target=random.uniform(0.95, 1.6)
    # random_target=1.1318003767113862
    target_size = int(target_size * random_target)
    # target_size = int(min(_network_w, _network_h) * random.uniform(0.7, 1.5))

    return pose_resize_shortestedge(image, annos, mask, target_size)


def pose_resize_shortestedge(image, annos,mask,target_size):
    _target_height = 368
    _target_width = 368
    img=image
    height, width, _ = np.shape(image)

    # adjust image
    scale = target_size / min(height, width)
    if height < width:
        newh, neww = target_size, int(scale * width + 0.5)
    else:
        newh, neww = int(scale * height + 0.5), target_size

    dst = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (neww, newh), interpolation=cv2.INTER_AREA)
    pw = ph = 0
    if neww < _target_width or newh < _target_height:
        pw = max(0, (_target_width - neww) // 2)
        ph = max(0, (_target_height - newh) // 2)
        mw = (_target_width - neww) % 2
        mh = (_target_height - newh) % 2
        color = np.random.uniform(0.0, 1.0)
        dst = cv2.copyMakeBorder(dst, ph, ph + mh, pw, pw + mw, cv2.BORDER_CONSTANT, value=(0,0 ,color))
        mask = cv2.copyMakeBorder(mask, ph, ph + mh, pw, pw + mw, cv2.BORDER_CONSTANT, value=1)
    # adjust meta data
    adjust_joint_list = []
    for joint in annos:
        adjust_joint = []
        for point in joint:
            if point[0] < -100 or point[1] < -100:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0 or int(point[0]*scale+0.5) > neww or int(point[1]*scale+0.5) > newh:
            #     adjust_joint.append((-1, -1))
            #     continue
            adjust_joint.append((int(point[0] * scale + 0.5) + pw, int(point[1] * scale + 0.5) + ph))
        adjust_joint_list.append(adjust_joint)
    return dst,adjust_joint_list,mask

def pose_crop_random(image,annos,mask):

    _target_height = 368
    _target_width = 368
    target_size = (_target_width, _target_height)

    if len(np.shape(image))==2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    height,width,_=np.shape(image)

    for _ in range(50):
        x = random.randrange(0, width - target_size[0]) if width > target_size[0] else 0
        y = random.randrange(0, height - target_size[1]) if height > target_size[1] else 0

        # check whether any face is inside the box to generate a reasonably-balanced datasets
        for joint in annos:
            if x <= joint[0][0] < x + target_size[0] and y <= joint[0][1] < y + target_size[1]:
                break

    return pose_crop(image,annos,mask, x, y, target_size[0], target_size[1])


def pose_crop(image,annos,mask, x, y, w, h):
    # adjust image
    target_size = (w, h)

    img = image
    resized = img[y:y+target_size[1], x:x+target_size[0], :]
    resized_mask = mask[y:y + target_size[1], x:x + target_size[0]]
    # adjust meta data
    adjust_joint_list = []
    for joint in annos:
        adjust_joint = []
        for point in joint:
            if point[0] < -10 or point[1] < -10:
                adjust_joint.append((-1000, -1000))
                continue
            # if point[0] <= 0 or point[1] <= 0:
            #     adjust_joint.append((-1000, -1000))
            #     continue
            new_x, new_y = point[0] - x, point[1] - y
            # if new_x <= 0 or new_y <= 0 or new_x > target_size[0] or new_y > target_size[1]:
            #     adjust_joint.append((-1, -1))
            #     continue
            if new_x > 367 or new_y > 367:
                adjust_joint.append((-1000, -1000))
                continue
            adjust_joint.append((new_x, new_y))
        adjust_joint_list.append(adjust_joint)

    return resized,adjust_joint_list,resized_mask
def drawing(image, annos):
    plt.imshow(image)
    for j in annos:
        for i in j:
            if i[0]>0 and i[1]>0:
                plt.scatter(i[0],i[1])
    plt.savefig('fig/'+str(i)+'.jpg', dpi=100)
    plt.show()

if __name__ == '__main__':
    data_dir = '/Users/Joel/Desktop/coco'
    data_type = 'val'
    anno_path = '{}/annotations/person_keypoints_{}2014.json'.format(data_dir, data_type)
    df_val = PoseInfo(data_dir, data_type, anno_path)

    for i in range (1050,1102):
        print('Img index',i)
        meta=df_val.metas[i]

        annos=meta.joint_list
        image=io.imread(meta.img_url)
        mask=meta.masks
        if len(image.shape)<3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h_mask = np.shape(image)[0]
        w_mask = np.shape(image)[1]

        mask_miss = np.ones((h_mask, w_mask), dtype=np.uint8)
        for seg in mask:
            bin_mask = maskUtils.decode(seg)
            bin_mask = np.logical_not(bin_mask)
            mask_miss = np.bitwise_and(mask_miss, bin_mask)

        image, annos, mask_miss = pose_random_scale(image, annos, mask_miss)
        image, annos, mask_miss = pose_rotation(image, annos, mask_miss)
        image, annos, mask_miss = random_flip(image, annos, mask_miss)
        image, annos, mask_miss = pose_resize_shortestedge_random(image, annos, mask_miss)
        image, annos, mask_miss = pose_crop_random(image, annos, mask_miss)
        for people in annos:
            for idx, jo in enumerate(people):
                if -1000 < jo[0] < 0 or -1000 < jo[1] < 0:
                    print('Err4 here')
        # plt.figure()
        # plt.imshow(image)
        # plt.imshow(mask_miss,alpha=0.3)

        # for people in annos:
        #     for idx,jo in enumerate(people):
        #
        #         if -100 < jo[0]  or -100 < jo[1]:
        #             plt.plot(jo[0],jo[1], '*')
        # plt.savefig('test_img/'+str(i)+".png")
        # plt.show()
        # plt.cla

        height, width, _ = np.shape(image)

        heatmap = get_heatmap2(annos, height, width)
        vectormap = get_vectormap2(annos, height, width)
        # show network output
        fig = plt.figure(figsize=(8, 8))
        a = fig.add_subplot(2, 3, 1)
        plt.imshow(image)
        for people in annos:
            for idx,jo in enumerate(people):
                if jo[0]>0 and jo[1]>0 :
                    plt.plot(jo[0],jo[1], '*')

        a = fig.add_subplot(2, 3, 2)
        a.set_title('Vectormap-1')
        tmp2 = vectormap.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        tmp2_odd = tmp2_odd * 255
        tmp2_odd = tmp2_odd.astype(np.int)
        plt.imshow(image)
        plt.imshow(tmp2_odd, alpha=0.3)

        tmp2_even = tmp2_even * 255
        tmp2_even = tmp2_even.astype(np.int)
        plt.imshow(tmp2_even, alpha=0.3)

        a = fig.add_subplot(2, 3, 3)
        tmp=heatmap
        tmp=np.amax(heatmap[:,:,:-1],axis=2)

        tmp = tmp * 255
        tmp = tmp.astype(np.int)
        plt.imshow(image)
        plt.imshow(tmp, alpha=0.3)

        a = fig.add_subplot(2, 3, 4)
        plt.imshow(mask_miss)
        plt.savefig('test_img/'+str(i)+".png")
        plt.show()
        plt.cla
