# coding : utf-8
import numpy as np
import pandas as pd
from utils.feature_utils import time_reform
import time
from conf.configure import Configure

from functools import wraps


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{}:{} seconds'.format(func.__module__, func.__name__, round(end - start , 2)))
        return r

    return wrapper


@timethis
def read_data(part1_path, part2_path):


    part_1 = pd.read_csv(part1_path, sep='$', encoding='utf8')
    part_2 = pd.read_csv(part2_path, sep='$', encoding='utf8')
    # train_data_id_label=pd.read_csv('meinian_round1_train_20180408.csv', encoding='ANSI')
    # test_data=pd.read_csv('meinian_round1_test_a_20180408.csv',encoding='ANSI',usecols=['vid'])


    part_1_2 = pd.concat([part_1, part_2])
    part_1_2 = pd.DataFrame(part_1_2).sort_values('vid').reset_index(drop=True)
    begin_time = time.time()
    print('begin')

    # 重复数据的拼接操作
    def merge_table(df):
        df['field_results'] = df['field_results'].astype(str)
        if df.shape[0] > 1:
            merge_df = " ".join(list(df['field_results']))
        else:
            merge_df = df['field_results'].values[0]
        return merge_df

    # 数据简单处理
    print('find_is_copy')
    print(part_1_2.shape)
    is_happen = part_1_2.groupby(['vid', 'table_id']).size().reset_index()
    # 重塑index用来去重
    is_happen['new_index'] = is_happen['vid'] + '_' + is_happen['table_id']
    is_happen_new = is_happen[is_happen[0] > 1]['new_index']

    part_1_2['new_index'] = part_1_2['vid'] + '_' + part_1_2['table_id']

    unique_part = part_1_2[part_1_2['new_index'].isin(list(is_happen_new))]
    unique_part = unique_part.sort_values(['vid', 'table_id'])
    no_unique_part = part_1_2[~part_1_2['new_index'].isin(list(is_happen_new))]
    print('begin')
    part_1_2_not_unique = unique_part.groupby(['vid', 'table_id']).apply(merge_table).reset_index()
    part_1_2_not_unique.rename(columns={0: 'field_results'}, inplace=True)
    print('xxx')
    tmp = pd.concat([part_1_2_not_unique, no_unique_part[['vid', 'table_id', 'field_results']]])
    # 行列转换
    print('finish')
    tmp = tmp.pivot(index='vid', values='field_results', columns='table_id')
    tmp.to_csv(Configure.cleaned_path,encoding="utf8")
    print(tmp.shape)
    print('totle time', time.time() - begin_time)

    return tmp


if __name__ == '__main__':
    read_data(Configure.part1_path,Configure.part2_path)
