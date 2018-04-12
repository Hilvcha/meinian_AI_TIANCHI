# coding: utf-8
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)


class Configure(object):
    base_path = os.path.abspath(os.path.join(__file__, '../..'))
    part1_path = os.path.join(base_path, "meinian_round1_data_part1_20180408.txt")
    part2_path = os.path.join(base_path, "meinian_round1_data_part2_20180408.txt")
    train_label_path=os.path.join(base_path, "meinian_round1_train_20180408.csv")
    test_vid_path=os.path.join(base_path, "[new] meinian_round1_test_a_20180409.csv")
    # 数据清洗后的路径
    cleaned_path = os.path.join(base_path, 'cleaned','tmp.csv')
    # 生成的特征的路径
    features_path = os.path.join(base_path, 'features')
    # 生成的模型可训练和预测的数据集
    datasets_path = os.path.join(base_path, 'datasets')
    # 最终结果csv存放处
    submit_result_path = os.path.abspath(os.path.join(base_path, 'model', 'submit.csv'))
    # 需要merge的特征
    features = [

    ]


if __name__ == '__main__':
    print('========== 当前项目目录 ==========')
    print(Configure.train_path)
    print(Configure.cleaned_path)
    print(Configure.features_path)
    print(Configure.datasets_path)
    print(Configure.submit_result_path)
