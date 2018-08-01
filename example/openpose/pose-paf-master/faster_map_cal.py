import math
import numpy as np
import cv2

def get_heatmap(annos,height,width):
    # 19 for coco, 15 for MPII
    num_joints = 19

    # the heatmap for every joints takes the maximum over all people
    joints_heatmap = np.zeros((num_joints, height, width), dtype=np.float32)

    # among all people
    for joint in annos:
        # generate heatmap for every keypoints
        # loop through all people and keep the maximum

        for i, points in enumerate(joint):
            if points[0] < 0 or points[1] < 0:
                continue
            joints_heatmap=put_heatmap(joints_heatmap, i, points, 8.0)

    # 0: joint index, 1:y, 2:x
    joints_heatmap = joints_heatmap.transpose((1, 2, 0))

    # background
    joints_heatmap[:, :, -1] = np.clip(1 - np.amax(joints_heatmap, axis=2), 0.0, 1.0)

    mapholder = []
    for i in range(0, 19):
        a = cv2.resize(np.array(joints_heatmap[:, :, i]), (46, 46))
        mapholder.append(a)
    mapholder = np.array(mapholder)
    joints_heatmap = mapholder.transpose(1, 2, 0)

    return joints_heatmap.astype(np.float16)

def cal_heatmap(heatmap, plane_idx, center, sigma):
    center_x, center_y = center
    _, height, width = heatmap.shape[:3]
    # exp(-th) ~0.01
    th = 4.6052
    delta = math.sqrt(th * 2)

    # give the area of gaussian heatmap
    x0 = int(max(0, center_x - delta * sigma))
    y0 = int(max(0, center_y - delta * sigma))

    x1 = int(min(width, center_x + delta * sigma))
    y1 = int(min(height, center_y + delta * sigma))

    # compute gaussian kernal
    for y in range(y0, y1):
        for x in range(x0, x1):
            d = (x - center_x) ** 2 + (y - center_y) ** 2
            exp = d / 2.0 / sigma / sigma
            # heat is so low
            if exp > th:
                continue
            # compare heat of index between different people and never exceed 1.0
            heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
            heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    return heatmap
def put_heatmap(heatmap, plane_idx, center, sigma):

    center_x, center_y = center
    _, height, width = heatmap.shape[:3]

    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))

    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

    exp_factor = 1 / 2.0 / sigma / sigma

    ## fast - vectorize
    arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y) ** 2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x) ** 2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    return heatmap
    ## slow - loops
    # for y in range(y0, y1 + 1):  # y0 to y1 include
    #     y_factor = (y - center_y) ** 2
    #     for x in range(x0, x1 + 1):
    #         # d = (x - center_x) ** 2 + (y - center_y) ** 2
    #         d = (x - center_x) ** 2 + y_factor
    #         # exp = d / 2.0 / sigma / sigma
    #         exp = d * exp_factor
    #         if exp > th:  # math.exp(-exp))
    #             continue
    #         val1 = math.exp(-exp)
    #         mat_val = heatmap[plane_idx, y, x]
    #         val2 = max(mat_val, val1)  # heatmap initilized to zero, cant be bigger then 1
    #         heatmap[plane_idx, y, x] = val2
    #         # heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
    #         # heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)
    # arr_heatmap2 = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]

def get_vectormap(annos,height,width):

    num_joints = 19

    limb = list(zip(
        [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
        [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
    ))

    vectormap = np.zeros((num_joints * 2, height, width), dtype=np.float32)
    counter = np.zeros((num_joints, height, width), dtype=np.int16)

    for joint in annos:
        if len(joint)!=19:
            print('THE LENGTH IS NOT 19 ERROR:',len(joint))
        for i, (a, b) in enumerate(limb):
            a -= 1
            b -= 1

            v_start = joint[a]
            v_end = joint[b]
            # exclude invisible or unmarked point
            if v_start[0] < -100 or v_start[1] < -100 or v_end[0] < -100 or v_end[1] < -100:
                continue
            vectormap = cal_vectormap(vectormap, counter, i, v_start, v_end)

    vectormap = vectormap.transpose((1, 2, 0))
    # normalize the PAF (otherwise longer limb gives stronger absolute strength)
    nonzero_vector = np.nonzero(counter)

    for i, y, x in zip(nonzero_vector[0], nonzero_vector[1], nonzero_vector[2]):

        if counter[i][y][x] <= 0:
            continue
        vectormap[y][x][i * 2 + 0] /= counter[i][y][x]
        vectormap[y][x][i * 2 + 1] /= counter[i][y][x]

    mapholder = []
    for i in range(0, 38):
        a = cv2.resize(np.array(vectormap[:, :, i]), (46, 46),interpolation=cv2.INTER_AREA)
        mapholder.append(a)
    mapholder = np.array(mapholder)
    vectormap = mapholder.transpose(1, 2, 0)

    return vectormap.astype(np.float16)

def cal_vectormap(vectormap, countmap, i, v_start, v_end):
    _, height, width = vectormap.shape[:3]
    # import copy
    # vectormap2=copy.deepcopy(vectormap)
    # vectormap3=copy.deepcopy(vectormap)
    # countmap2 = copy.deepcopy(countmap)
    threshold = 8
    # print(' v_start, v_end', v_start, v_end)
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]
    # print('Vec1', i)
    length = math.sqrt(vector_x ** 2 + vector_y ** 2)
    if length == 0:
        return vectormap


    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - v_start[0]
            bec_y = y - v_start[1]
            dist = abs(bec_x * norm_y - bec_y * norm_x)

            # orthogonal distance is < then threshold
            if dist > threshold:
                continue
            countmap[i][y][x] += 1
            vectormap[i * 2 + 0][y][x] = norm_x
            vectormap[i * 2 + 1][y][x] = norm_y

    return vectormap
def fast_vectormap(vectormap, countmap, i, v_start, v_end):
    _, height, width = vectormap.shape[:3]
    _, height, width = vectormap.shape[:3]
    # print(' v_start, v_end', v_start, v_end)
    threshold = 8
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]
    # print('Vec2', i)
    length = math.sqrt(vector_x ** 2 + vector_y ** 2)
    if length == 0:
        return vectormap

    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    x_vec = (np.arange(min_x, max_x) - v_start[0])*norm_y
    y_vec = (np.arange(min_y, max_y) - v_start[1])*norm_x



    xv, yv = np.meshgrid(x_vec, y_vec)

    dist_matrix=abs(xv-yv)
    filter_matrix=np.where(dist_matrix>threshold,0,1)

    # print('para',min_y,max_y, min_x, max_x,v_start,v_end)
    countmap[i, min_y: max_y, min_x: max_x]+=filter_matrix

    norm_x_map =filter_matrix*norm_x
    norm_y_map =filter_matrix*norm_y

    # padholder=np.zeros((height,width))
    # padholder[min_y: max_y, min_x: max_x]=filter_matrix
    # vectormap[i * 2 + 0, padholder.astype(bool)] = norm_x
    # vectormap[i * 2 + 1, padholder.astype(bool)] = norm_y

    for y in range(max_y-min_y):
        for x in range(max_x-min_x):
            if filter_matrix[y,x]!=0:
                vectormap[i * 2 + 0, min_y+y, min_x+x] = norm_x
                vectormap[i * 2 + 1, min_y+y, min_x+x] = norm_y
    return vectormap