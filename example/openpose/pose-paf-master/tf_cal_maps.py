import math
import numpy as np
import cv2


def get_heatmap(annos, height, width):
    # 19 for coco, 15 for MPII
    num_joints = 19

    # the heatmap for every joints takes the maximum over all people
    joints_heatmap = np.zeros((num_joints, height, width), dtype=np.float32)

    # among all people
    for joint in annos:
        # generate heatmap for every keypoints
        # loop through all people and keep the maximum
        for i, points in enumerate(joint):
            joints_heatmap = cal_heatmap(joints_heatmap, i, points, 8.0)

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
            d = (x - center_x)**2 + (y - center_y)**2
            exp = d / 2.0 / sigma / sigma
            # heat is so low
            if exp > th:
                continue
            # compare heat of index between different people and never exceed 1.0
            heatmap[plane_idx][y][x] = max(heatmap[plane_idx][y][x], math.exp(-exp))
            heatmap[plane_idx][y][x] = min(heatmap[plane_idx][y][x], 1.0)

    return heatmap


def get_vectormap(annos, height, width):

    num_joints = 19

    limb = list(
        zip(
            [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
            [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
        )
    )

    vectormap = np.zeros((num_joints * 2, height, width), dtype=np.float32)
    counter = np.zeros((num_joints, height, width), dtype=np.int16)

    for joint in annos:
        if len(joint) != 19:
            print('THE LENGTH IS NOT 19 ERROR:', len(joint))
        for i, (a, b) in enumerate(limb):
            a -= 1
            b -= 1

            v_start = joint[a]
            v_end = joint[b]
            # exclude invisible or unmarked point
            if v_start[0] < -100 or v_start[1] < -100 or v_end[0] < -100 or v_end[1] < -100:
                continue
            vectormap, counter = cal_vectormap(vectormap, counter, i, v_start, v_end)

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
        a = cv2.resize(np.array(vectormap[:, :, i]), (46, 46), interpolation=cv2.INTER_AREA)
        mapholder.append(a)
    mapholder = np.array(mapholder)
    vectormap = mapholder.transpose(1, 2, 0)

    return vectormap.astype(np.float16)


def cal_vectormap(vectormap, countmap, i, v_start, v_end):
    _, height, width = vectormap.shape[:3]
    threshold = 8
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]

    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap, countmap

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

    return vectormap, countmap
