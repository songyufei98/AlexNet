import read_data
import config
import generate_LSM
import train_cnn
from read_data import get_feature_data, creat_dataset
#Instantiate config
config = config.config

is_train = 0
is_saveLSM = 1

if __name__ == '__main__':
    if is_train:
        print('***************************************读取训练集 测试集***************************************')
        tensor_data = get_feature_data()  # reading training set feature*width*height
        creat = creat_dataset(tensor_data, config["size"])
        all_data_train, all_target_train = read_data.train_data(creat)
        all_data_val, all_target_val = read_data.test_data(creat)
        print('*******************************************开始训练*******************************************')
        train_cnn.train(all_data_train, all_target_train, all_data_val, all_target_val)
    if is_saveLSM:
        generate_LSM.save_LSM()