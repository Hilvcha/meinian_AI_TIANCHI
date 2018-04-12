# coding: utf-8
import os
import sys
import time

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# TERMINALNO, TIME, TRIP_ID, LONGITUDE, LATITUDE, DIRECTION, HEIGHT, SPEED, CALLSTATE, Y

from conf.configure import Configure
from utils.data_utils import save_features
from utils.feature_utils import time_this




@time_this
def save_all_features(train, test):
    funcs = {

    }
    for name in Configure.features:
        save_features(*funcs[name](train, test), name)


if __name__ == "__main__":
    print("****************** feature **********************")
    # 程序入口
    # save_all_features()
