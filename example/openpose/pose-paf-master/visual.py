import os
import numpy as np
import matplotlib.pyplot as plt

#moniter training
val_path = '/Users/Joel/Desktop/Log_1708/val/'
val_list = os.listdir(val_path)

idx=165000
heats_ground=[]
heats_result=[]
pafs_ground=[]
pafs_result=[]
masks=[]
images=[]
for val in val_list:
    if val.startswith('heat_ground'+str(idx)):
        heats_ground=np.load(val_path+val)
    elif val.startswith('heat_result' + str(idx)):
        heats_result = np.load(val_path + val)
    elif val.startswith('paf_result'+str(idx)):
        pafs_result=np.load(val_path+val)
    elif val.startswith('paf_ground' + str(idx)):
        pafs_ground = np.load(val_path + val)
    elif val.startswith('mask'+str(idx)):
        masks=np.load(val_path+val)
    elif val.startswith('image'+str(idx)):
        images=np.load(val_path+val)

interval=len(pafs_result)
for i in range(interval):
    heat_ground=heats_ground[i]
    heat_result=heats_result[i]
    paf_ground=pafs_ground[i]
    paf_result=pafs_result[i]

    mask=masks[i]

    mask = mask.reshape(46, 46, 1)
    mask1 = np.repeat(mask, 19, 2)
    mask2 = np.repeat(mask, 38, 2)

    image=images[i]

    fig = plt.figure(figsize=(8, 8))
    a = fig.add_subplot(2, 3, 1)
    plt.imshow(image )

    a = fig.add_subplot(2, 3, 2)
    a.set_title('Vectormap_ground')
    vectormap = paf_ground * mask2
    tmp2 = vectormap.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    # tmp2_odd = tmp2_odd * 255
    # tmp2_odd = tmp2_odd.astype(np.int)
    plt.imshow(tmp2_odd, alpha=0.3)

    # tmp2_even = tmp2_even * 255
    # tmp2_even = tmp2_even.astype(np.int)
    plt.colorbar()
    plt.imshow(tmp2_even, alpha=0.3)

    a = fig.add_subplot(2, 3, 3)
    a.set_title('Vectormap result')
    vectormap = paf_result * mask2
    tmp2 = vectormap.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
    plt.imshow(tmp2_odd, alpha=0.3)

    plt.colorbar()
    plt.imshow(tmp2_even, alpha=0.3)

    a = fig.add_subplot(2, 3, 4)
    a.set_title('Heatmap result')
    heatmap = heat_result * mask1
    tmp = heatmap
    tmp = np.amax(heatmap[:, :, :-1], axis=2)

    plt.colorbar()
    plt.imshow(tmp, alpha=0.3)

    a = fig.add_subplot(2, 3, 5)
    a.set_title('Heatmap ground truth')
    heatmap = heat_ground * mask1
    tmp = heatmap
    tmp = np.amax(heatmap[:, :, :-1], axis=2)

    plt.colorbar()
    plt.imshow(tmp, alpha=0.3)
    # plt.savefig(str(i)+'.png',dpi=300)
    plt.show()
