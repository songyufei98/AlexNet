import os
import time
import pickle
import torch
import read_data
import numpy as np
from torch.autograd import Variable
from model.AlexNet import AlexNet_LSM
from tqdm import tqdm
from utils import plot_save_lsm
from torch.utils.data import DataLoader, TensorDataset
import config

config = config.config


def save_LSM():
    plot_and_save = True
    print('*******************************************generate landslide susceptibility map*******************************************')
    model = AlexNet_LSM(13).to(config["device"])
    model.load_state_dict(torch.load(os.path.join('Result', 'AlexNet_LSM_0_1_FR', 'best.pth')))

    tensor_data = read_data.get_feature_data()
    print('整个预测区域大小：' + str(tensor_data.shape))
    creat = read_data.creat_dataset(tensor_data, config['size'])
    data = creat.creat_new_tensor()
    images_list = []
    probs = []
    model.eval()
    with torch.no_grad():   
        # 遍历扩展过边缘的数据图
        for i in range(creat.p, config["height"] + creat.p):
            for j in range(creat.p, config["width"] + creat.p):
                # 读取因子数据块
                images_list.append(data[:, i - creat.p:i + creat.p + 1, j - creat.p:j + creat.p + 1].astype(np.float32))
                if (i != creat.p and (i - creat.p) % config["Cutting_window"] == 0 and (j - creat.p)== config["width"] - 1) or ( 
                    (i - creat.p) == config["height"] - 1 and (j - creat.p) == config["width"] - 1):
                    start_time = time.time()
                    pred_data = np.stack(images_list)
                    print('i=' + str(i) + ' j=' + str(j))
                    images_list = []
                    pred_dataset = TensorDataset(torch.from_numpy(pred_data))
                    pred_loader = DataLoader(dataset=pred_dataset, batch_size=config["batch_size"], shuffle=False)
                    for images in tqdm(pred_loader):
                        images = torch.stack([image.cuda() for image in images])
                        images = Variable(images.squeeze(0)).to(config["device"])
                        probs.append(model(images).cpu()[:, 1])
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    minutes = int(elapsed_time // 60)
                    seconds = elapsed_time % 60
                    print(f"Processing time: {minutes} min {seconds} sec")
    probs = np.concatenate(probs)
    print('Probability list generation completed')
    if plot_and_save:
        plot_save_lsm(os.path.join('Result', 'AlexNet_LSM_0_1_FR'), probs)
    read_data.save_to_tif(probs, os.path.join('Result', 'AlexNet_LSM_0_1_FR', 'AlexNet_LSM.tif'))
            