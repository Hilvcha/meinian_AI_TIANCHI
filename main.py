# -*- coding:utf-8 -*-

import time
import numpy as np
import pandas as pd

from conf.configure import Configure
from functions.functions import save_all_features
from input.read_data import read_data
from train_model.xgboost_model import model_train

if __name__ == "__main__":
    now=time.localtime(time.time())
    print("******* start at:", time.strftime('%Y-%m-%d %H:%M:%S',now ), '*******')
    # 程序入口
    # read_data(Configure.part1_path, Configure.part2_path)

    # save_all_features(trainSet, testSet)

    # model_train(trainSet, testSet)
    test_id=pd.read_csv(Configure.test_vid_path,encoding='gbk')
    train_data=pd.read_csv(Configure.part1_path)
    data_history=pd.read_csv('history_first_data.csv')
    data_history_vid=pd.read_csv('history_first_data.csv',usecols=['vid'])
###########################################################################################
    def get_num_status(x):
        try:
            a=(x.count('甲状腺大小形态正常')+x.count('甲状腺形态正常')+x.count('甲状腺大小正常')+x.count('甲状腺彩超（双侧）未发现明显异常')+x.count('甲状腺彩超未发现明显异常')+x.count('甲状腺形态大小正常')+x.count('甲状腺彩超（双侧）未发现明显异常')+x.count('甲状腺彩超（含颈部淋巴细胞）未发现明显异常'))
            if(a==2):
                if(x.count('结节')>0):
                    return 0
                return 1
            return a
        except:
            return 1
    data_history['Thyroid_1']=data_history['0101'].map(lambda x:get_num_status(x))
    data_history['Thyroid_2']=data_history['0102'].map(lambda x:get_num_status(x))
    i=0
    s=[]
    for value in data_history['Thyroid_1']:
        if(((value==1) and (data_history['Thyroid_2'][i]==1)) or ((value==-1) and (data_history['Thyroid_2'][i]==1)) or ((value==1) and (data_history['Thyroid_2'][i]==-1))):
            s.append(1)
        if((value==0)or(data_history['Thyroid_2'][i]==0)):
            s.append(0)
        if((value==-1) and (data_history['Thyroid_2'][i]==-1)):
            s.append(-1)
        i=i+1
    s=pd.Series(s)
    data_history['Thyroid_status']=s
    del data_history['Thyroid_1']
    del data_history['Thyroid_2']
    ##################################################
    def get_num_status2(x):
        try:
            if(x.count('脂肪肝（轻度）')>0):
                return 1
            if(x.count('脂肪肝（中度）')>0):
                return 2
            if(x.count('脂肪肝（重度）')>0):
                return 3
            i=x.index('肝')
            x=x[i:]
            if((x.count('未发现明显异常')>0)or(x.count('大小、形态正常')>0)or(x.count('大小形态正常')>0)):
                return 0
            else:
                return -1
        except:
            return 1
    data_history['gan_1']=data_history['0102'].map(lambda x:get_num_status2(x))
    #(['vid', '0101', '0102', '0113', '0114', '0115', '0116', '0117', '0118',
    #     '0409', '0421', '0426', '10004', '1001', '1814', '1815', '1840', '1850',
    #    '190', '191', '2302', '2403', '2404', '2405', '300005', '3190', '3191',
        #   '3192', '3193', '3195', '3196', '3197', '3429', '3430', '3730',
        #  'Thyroid_status', 'gan_1'],
    #########################################################################################################
    def get_num_status3(x):
        try:
            if((x.count('胆囊大小、形态正常')>0) or (x.count('胆囊形态大小正常')>0) or (x.count('大小形态正常')>0) or (x.count('大小、形态正常')>0) or (x.count('胆囊大小、形态未见明显异常')>0) or (x.count('胆囊大小正常')>0) or (x.count('胆囊大小,形态正常')>0)):
                return 1
            if((x.count('胆囊区未显示胆囊回声')>0) or (x.count('自述')>0) or  (x.count('切除')>0)):
                return 2
            if(x.count('餐后')>0):
                return -1
            else:
                return 0
        except:
            return 1
    data_history['dan_1']=data_history['0114'].map(lambda x:get_num_status3(x))
    #data_history[data_history['dan_1']==0]['0114']
    #################################################################################################
    def get_num_status4(x):
        try:
            if((x.count('形态大小正常')>0) or (x.count('肝脏大小、形态正常')>0) or  (x.count('大小、形态正常')>0) or (x.count('肝脏大小、形态未见明显异常')>0) or (x.count('回声均匀')>0) or (x.count('大小形态正常')>0)) and (x.count('结节')==0):
                return 1
            if((x.count('呈点状密集弥漫性增强')>0)):
                return 2
            else:
                return 0
        except:
            return 1
    data_history['ganzang_1']=data_history['0113'].map(lambda x:get_num_status4(x))

    #data_history[data_history['ganzang_1']==0]['0113']
    ####################################################################################################
    def get_num_status5(x):
        try:
            if((x.count('形态大小正常')>0) or (x.count('胰腺头、体、尾大小测值正常，内回声均匀')>0) or  (x.count('大小、形态正常')>0) or (x.count('未见明显占位性病变')>0)):
                return 1
            if((x.count('受肠气影响')>0)):
                return 2
            else:
                return 0
        except:
            return 1
    data_history['yixian_1'] = data_history['0115'].map(lambda x: get_num_status5(x))
    #data_history[data_history['yixian_1']==0]['0115']
    ##################################################################################################
    def get_num_status6(x):
        try:
            if((x.count('形态大小正常')>0) or (x.count('回声均匀')>0) or  (x.count('大小、形态正常')>0) or (x.count('大小测值正常')>0)):
                return 1
            if((x.count('脾区未见脾脏回声')>0)):
                return 2
            else:
                return 0
        except:
            return 1
    data_history['pizang_1'] = data_history['0116'].map(lambda x: get_num_status6(x))
    #data_history[data_history['pizang_1']==0]['0116']
    ###########################################################################################################
    del data_history['0116']
    del data_history['0115']
    del data_history['0113']
    del data_history['0114']
    del data_history['0101']
    del data_history['0102']
    ##############################################################
    def get_num_status7(x):
        try:
            if((x.count('形态大小正常')>0) or (x.count('回声均匀')>0) or  (x.count('大小、形态正常')>0) or (x.count('左肾轮廓清，大小位置形态正常')>0) or (x.count('右肾轮廓清，大小位置形态正常，皮质部回声分布均匀，集合系统未见分离')>0)) and (x.count('强回声')==0):
                return 1
            if((x.count('强回声')>0)):
                return 2
            if(x.count('未见肾脏回声')>0):
                return 3
            else:
                return 0
        except:
            return 1
    data_history['left_shen_1'] = data_history['0117'].map(lambda x: get_num_status7(x))
    data_history['right_shen_1'] = data_history['0118'].map(lambda x: get_num_status7(x))
    #data_history[data_history['left_shen_1']==0]['0117']
    #data_history[data_history['right_shen_1']==0]['0118']
    del data_history['0117']
    del data_history['0118']
    ###################################################################################
    def get_num_status8(x):
        m=[]
        try:
            if((x.count('检查未发现明显异常')>0) or (x.count('未见明显异常')>0) or  (x.count('所检项目未见异常')>0) or (x.count('未发现明显异常')>0) or (x.count('内科检查未见异常')>0) or (x.count('未见异常')>0) or (x.count('正常')>0)) :
                return 0.5
            if((x.count('亢')>0)):
                m.append(2)
            if(x.count('高血')>0):
                m.append(3)
            if(x.count('糖尿病')>0):
                m.append(5)
            if(x.count('术后')>0):
                m.append(6)
            if (x.count('血糖偏高') > 0):
                m.append(7)
            if (x.count('心音弱') > 0):
                m.append(8)
            if (x.count('脂肪肝') > 0):
                m.append(8)
            if (x.count('血压偏高') > 0):
                m.append(9)
            if (x.count('血脂') > 0):
                m.append(10)
            if (x.count('甲状腺功能减退') > 0):
                m.append(12)
            if (x.count('结石') > 0):
                m.append(13)
            if (x.count('炎') > 0):
                m.append(14)
            if (x.count('心脏杂音') > 0):
                m.append(15)
            if (x.count('冠心') > 0):
                m.append(16)
            return sum(m)-1
        except:
            return 1
    data_history['health_profile']=data_history['0409'].map(lambda x: get_num_status8(x))
    #data_history[data_history['health_profile']==0]['0409']
    del data_history['0409']
    ######################################################################################################
    def get_num_status9(x):
        try:
            if((x.count('整齐')>0) or (len(x)==1) or  (x.count('正常')>0) or (x.count('齐 齐')>0) or (x.count('右肾轮廓清，大小位置形态正常，皮质部回声分布均匀，集合系统未见分离')>0)):
                return 1
            if((x.count('早搏')>0)):
                return 2
            if(x.count('房颤')>0):
                return 3
            if (x.count('不齐') > 0):
                return 4
            else:
                return 0
        except:
            return 1
    data_history['heart']=data_history['0421'].map(lambda x: get_num_status9(x))
    del data_history['0421']
    ####################################################################################################
    def fun(x):
        try:
            x=float(x)
            if(x<0):
                x=np.nan
            return x
        except:
            if(x.count(' ')==1):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            if(x.count('.3.70')==1):
                return 3.70
            if(x.count('4.14.')==1):
                return 4.14
            return np.nan
    data_history['shuzhi_feat1']=data_history['10004'].map(lambda x:fun(x))
    a=data_history['shuzhi_feat1']
    a=pd.DataFrame(a)
    data_history['shuzhi_feat1']=a.fillna(a.mean())
    del data_history['10004']
    ########################################################################################################
    def fun2(x):
        try:
            x=float(x)
            if(x>700 or x==0):
                x=np.nan
                return x
            return x
        except:
            if(x.count(' ')==1):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            #if(x.count('.3.70')==1):
            #    return 3.70
            #if(x.count('4.14.')==1):
            #    return 4.14
            return np.nan
    data_history['shuzhi_feat2'] = data_history['1814'].map(lambda x: fun2(x))
    b=data_history['shuzhi_feat2']
    b=pd.DataFrame(b)
    data_history['shuzhi_feat2']=b.fillna(b.mean())
    #data_history[data_history['shuzhi_feat2']>1000]
    del data_history['1814']
    ########################################################################################################
    def fun3(x):
        try:
            x=float(x)
            if(x>500 or x==0):
                x=np.nan
                return x
            return x
        except:
            if(x.count(' ')==1):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            if(x.count(' ')==2):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            #if(x.count('.3.70')==1):
            #    return 3.70
            #if(x.count('4.14.')==1):
            #    return 4.14
            return np.nan
    data_history['shuzhi_feat3'] = data_history['1815'].map(lambda x: fun3(x))
    c=data_history['shuzhi_feat3']
    c=pd.DataFrame(c)
    data_history['shuzhi_feat3']=c.fillna(c.mean())
    #data_history[data_history['shuzhi_feat3']>1000]['shuzhi_feat3']
    del data_history['1815']
    #############################################################################################
    def fun4(x):
        try:
            x=float(x)
        # if(x>500 or x==0):
        #     x=np.nan
        #     return x
            return x
        except:
            if(x.count(' ')==1):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            if(x.count(' ')==2):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            if(x.count('＜')>0 or x.count('<')>0 ):
                return 5.0
            if(x.count('阴')):
                return 4.8
            #if(x.count('.3.70')==1):
            #    return 3.70
            #if(x.count('4.14.')==1):
            #    return 4.14
            return np.nan
    data_history['shuzhi_feat4'] = data_history['1840'].map(lambda x: fun4(x))
    d=data_history['shuzhi_feat4']
    d=pd.DataFrame(d)
    data_history['shuzhi_feat4']=d.fillna(d.mean())
    #data_history[data_history['shuzhi_feat4']==0]['shuzhi_feat4']
    del data_history['1840']
    ####################################################################################################################
    def fun5(x):
        try:
            x=float(x)
        # if(x>500 or x==0):
        #     x=np.nan
        #     return x
            return x
        except:
            if(x.count(' ')==1):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            if(x.count(' ')==2):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            if(x.count('＜')>0 or x.count('<')>0 ):
                return 5.0
            if(x.count('阴')):
                return 4.8
            if(x.count('。')):
                return 3.89
            #if(x.count('.3.70')==1):
            #    return 3.70
            #if(x.count('4.14.')==1):
            #    return 4.14
            return np.nan
    data_history['shuzhi_feat5'] = data_history['1850'].map(lambda x: fun5(x))
    e=data_history['shuzhi_feat5']
    e=pd.DataFrame(e)
    data_history['shuzhi_feat5']=e.fillna(e.mean())
    #data_history[data_history['shuzhi_feat5']==0]['shuzhi_feat5']
    del data_history['1850']
    #####################################################################################################
    data_history['shuzhi_feat6'] = data_history['190'].map(lambda x: fun5(x))
    f=data_history['shuzhi_feat6']
    f=pd.DataFrame(f)
    data_history['shuzhi_feat6']=f.fillna(f.mean())
    del data_history['190']
    ################################################################################################
    def fun6(x):
        try:
            x=float(x)
            if(x<10):
                x=np.nan
                return x
            return x
        except:
            if(x.count(' ')==1):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            if(x.count(' ')==2):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            if(x.count('＜')>0 or x.count('<')>0 ):
                return 5.0
            if(x.count('阴')):
                return 4.8
            if(x.count('。')):
                return 3.89
            #if(x.count('.3.70')==1):
            #    return 3.70
            #if(x.count('4.14.')==1):
            #    return 4.14
            return np.nan
    data_history['shuzhi_feat7'] = data_history['191'].map(lambda x: fun6(x))
    g=data_history['shuzhi_feat7']
    g=pd.DataFrame(g)
    data_history['shuzhi_feat7']=g.fillna(g.mean())
    del data_history['191']
    ################################################################################################
    def get_num_status11(x):
        try:
            if((x.count('亚健康')>0) or (x.count('回声均匀')>0) or  (x.count('大小、形态正常')>0) or (x.count('左肾轮廓清，大小位置形态正常')>0) or (x.count('右肾轮廓清，大小位置形态正常，皮质部回声分布均匀，集合系统未见分离')>0)) and (x.count('强回声')==0):
                return 1
            if((x.count('健康')>0)):
                return 2
            if(x.count('疾病')>0):
                return 3
            else:
                return 0
        except:
            return 1
    data_history['health_status']=data_history['2302'].map(lambda x:get_num_status11(x))
    del data_history['2302']
    ####################################################################################################################
    #(['vid', '0101', '0102', '0113', '0114', '0115', '0116', '0117', '0118',
    #     '0409', '0421', '0426', '10004', '1001', '1814', '1815', '1840', '1850',
    #    '190', '191', '2302', '2403', '2404', '2405', '300005', '3190', '3191',
        #   '3192', '3193', '3195', '3196', '3197', '3429', '3430', '3730',
        #  'Thyroid_status', 'gan_1'],

    def fun6(x):
        try:
            x=float(x)
            if(x==0 or x>1000):
                x=np.nan
                return x
            return x
        except:
            if(x.count('未查') or x.count('弃查')):
                return np.nan
            if(x.count(' ')==1):
                i=x.index(' ')
                x=x[i+1:]
                x=float(x)
                return x
            if (x.count(' ') == 2):
                i = x.index(' ')
                x = x[:i]
                x = float(x)
                return x
        #    if(x.count('＜')>0 or x.count('<')>0 ):
    #           return 5.0
    #        if(x.count('阴')):
    #          return 4.8
    #       if(x.count('。')):
    #           return 3.89
            #if(x.count('.3.70')==1):
            #    return 3.70
            #if(x.count('4.14.')==1):
            #    return 4.14
            return 0
    data_history['shuzhi_feat8'] = data_history['2403'].map(lambda x: fun6(x))
    g=data_history['shuzhi_feat8']
    g=pd.DataFrame(g)
    data_history['shuzhi_feat8']=g.fillna(g.mean())
    del data_history['2403']
    ########################################################################################################################
    def fun6(x):
        try:
            x=float(x)
            if(x==0):
                x=np.nan
                return x
            return x
        except:
            if(x.count('未查') or x.count('弃查')):
                return np.nan
            if(x.count(' ')==1):
                i=x.index(' ')
                x=x[i+1:]
                x=float(x)
                return x
            if (x.count(' ') == 2):
                i = x.index(' ')
                x = x[:i]
                x = float(x)
                return x
        #    if(x.count('＜')>0 or x.count('<')>0 ):
    #           return 5.0
    #        if(x.count('阴')):
    #          return 4.8
    #       if(x.count('。')):
    #           return 3.89
            #if(x.count('.3.70')==1):
            #    return 3.70
            #if(x.count('4.14.')==1):
            #    return 4.14
            return np.nan
    data_history['shuzhi_feat9'] = data_history['2404'].map(lambda x: fun6(x))
    g=data_history['shuzhi_feat9']
    g=pd.DataFrame(g)
    data_history['shuzhi_feat9']=g.fillna(g.mean())
    del data_history['2404']
    #data_history[data_history['shuzhi_feat9']==0]['shuzhi_feat9']
    ####################################################################################################################
    def fun6(x):
        try:
            x=float(x)
            if(x==0 or x>100):
                x=np.nan
                return x
            return x
        except:
            if(x.count('未查') or x.count('弃查')):
                return np.nan
            if(x.count(' ')==1):
                i=x.index(' ')
                x=x[i+1:]
                x=float(x)
                return x
            if (x.count(' ') == 2):
                i = x.index(' ')
                x = x[:i]
                x = float(x)
                return x
        #    if(x.count('＜')>0 or x.count('<')>0 ):
    #           return 5.0
    #        if(x.count('阴')):
    #          return 4.8
    #       if(x.count('。')):
    #           return 3.89
            #if(x.count('.3.70')==1):
            #    return 3.70
            #if(x.count('4.14.')==1):
            #    return 4.14
            return 0
    data_history['shuzhi_feat10'] = data_history['2405'].map(lambda x: fun6(x))
    g=data_history['shuzhi_feat10']
    g=pd.DataFrame(g)
    data_history['shuzhi_feat10']=g.fillna(g.mean())
    del data_history['2405']
    #data_history[data_history['shuzhi_feat9']==0]['shuzhi_feat9']
    ######################################################################################################################
    def get_num_status21(x):
        try:
            if((x.count('未')) or  (x.count('正常')>0) or (x.count('无')>0) or (x.count('各瓣膜听诊区未闻及杂音')>0) or (x.count('未闻及')>0) or (x.count('未闻及病理性杂音')>0) or (x.count('未闻及器质性杂音')>0) or(x.count('无杂音')>0)):
                return 1
            if((x.count('Ⅱ级')>0) or x.count('2')>0):
                return 2
            if(x.count('Ⅰ级')>0 or x.count('1')>0):
                return 3
            if (x.count('Ⅲ级') > 0 or x.count('3') > 0):
                return 4
            if (x.count('Ⅳ级') > 0 ):
                return 5
            if (x.count('舒张期') > 0 ):
                return 6
            else:
                return 0
        except:
            return 1

    data_history['hh_status'] = data_history['0426'].map(lambda x: get_num_status21(x))
    del data_history['0426']
    ######################################################################################################################
    def fun21(x):
        try:
            x=float(x)
        #  if(x==0 or x>1000):
        #      x=np.nan
            #     return x
            return x
        except:
            if(x.count('未做') or x.count('阴性') or x.count('-')):
                return np.nan
            if(x.count('>=')==1):
                i=x.index('=')
                x=x[i+1:]
                x=float(x)
                return x
            if (x.count('\t') == 1):
                i = x.index('\t')
                x = x[i+1:]
                x = float(x)
                return x
        #    if(x.count('＜')>0 or x.count('<')>0 ):
    #           return 5.0
    #        if(x.count('阴')):
    #          return 4.8
    #       if(x.count('。')):
    #           return 3.89
            #if(x.count('.3.70')==1):
            #    return 3.70
            #if(x.count('4.14.')==1):
            #    return 4.14
            return np.nan
    data_history['shuzhi_feat21'] = data_history['3193'].map(lambda x: fun21(x))
    g=data_history['shuzhi_feat21']
    g=pd.DataFrame(g)
    data_history['shuzhi_feat21']=g.fillna(g.mean())
    del data_history['3193']
    #####################################################################################################################
    def get_num_status22(x):
        try:
            if((x.count('-')) or  (x.count('阴性')>0)):
                return 1
            if((x.count('阳')>0) or x.count('+')>0):
                return 2
            else:
                return 1
        except:
            return 1

    data_history['yin_yang_1'] = data_history['3191'].map(lambda x: get_num_status22(x))
    del data_history['3191']
    #####################################################################################################################
    def get_num_status23(x):
        try:
            if((x.count('-')) or  (x.count('阴性')>0)):
                return 1
            if((x.count('阳')>0) or x.count('+')>0):
                return 2
            else:
                return 1
        except:
            return 1

    data_history['yin_yang_2'] = data_history['3190'].map(lambda x: get_num_status23(x))
    del data_history['3190']
    ####################################################################################################################
    def get_num_status24(x):
        try:
            if((x.count('-')) or  (x.count('阴性')>0)):
                return 1
            if((x.count('阳')>0) or x.count('+')>0):
                return 2
            else:
                return 1
        except:
            return 1

    data_history['yin_yang_3'] = data_history['3192'].map(lambda x: get_num_status24(x))
    del data_history['3192']
    ####################################################################################################################
    def get_num_status25(x):
        try:
            if((x.count('-')) or  (x.count('阴性')>0)):
                return 1
            if((x.count('阳')>0) or x.count('+')>0):
                return 2
            else:
                return 1
        except:
            return 1

    data_history['yin_yang_4'] = data_history['3195'].map(lambda x: get_num_status25(x))
    del data_history['3195']
    #####################################################################################################################
    def get_num_status26(x):
        try:
            if((x.count('-')) or  (x.count('阴性')>0)):
                return 1
            if((x.count('阳')>0) or x.count('+')>0):
                return 2
            else:
                return 1
        except:
            return 1

    data_history['yin_yang_5'] = data_history['3196'].map(lambda x: get_num_status26(x))
    del data_history['3196']
    ######################################################################################################################
    def get_num_status27(x):
        try:
            if((x.count('-')) or  (x.count('阴性')>0)):
                return 1
            if((x.count('阳')>0) or x.count('+')>0):
                return 2
            else:
                return 1
        except:
            return 1

    data_history['yin_yang_6'] = data_history['3197'].map(lambda x: get_num_status27(x))
    del data_history['3197']
    ######################################################################################################################

    def fun30(x):
        try:
            x = float(x)
            if (x > 100):
                x = np.nan
                return x
            return x
        except:
            if (x.count('散在满视野') or x.count('脓白')):
                return np.nan
            if (x.count('+-') == 1):
                return 0
            if (x.count('+') == 1):
                return np.nan
            if(x.count('(2-4)/HP')==1):
                return 3
            if(x.count('(5-10)/HP')==1):
                return 7.5
            if(x.count('未见')==1 or x.count('阴') or x.count('未检出') or x.count('正常') or x.count('少许') or x.count('偶见')):
                return 0
            try:
                if (x.count('-') == 1):
                    i = x.index('-')
                    if (x.count('个') == 1):
                        i2 = x.index('个')
                        x1 = x[i + 1:i2]
                        x1 = float(x1)
                        x2 = x[:i]
                        x2 = float(x2)
                        x = (x1 + x2) / 2
                        return x
                    if (x.count('/') == 1):
                        i2 = x.index('/')
                        x1 = x[i + 1:i2]
                        x1 = float(x1)
                        x2 = x[:i]
                        x2 = float(x2)
                        x = (x1 + x2) / 2
                        return x
                    x1 = x[i +2:]
                    x2 = x[:i]
                    x1 = float(x1)
                    x2 = float(x2)
                    x = (x1 + x2) / 2
                    return x
            except:
                return np.nan
                #    if(x.count('＜')>0 or x.count('<')>0 ):
                #           return 5.0
            #        if(x.count('阴')):
            #          return 4.8
            #       if(x.count('。')):
            #           return 3.89
            # if(x.count('.3.70')==1):
            #    return 3.70
            # if(x.count('4.14.')==1):
            #    return 4.14
            return 0


    data_history['shuzhi_feat30'] = data_history['300005'].map(lambda x: fun30(x))
    g = data_history['shuzhi_feat30']
    g = pd.DataFrame(g)
    data_history['shuzhi_feat30'] = g.fillna(g.mean())
    del data_history['300005']

    #####################################################################################################################
    def get_num_status33(x):
        try:
            if((x.count('-')) or  (x.count('阴性')>0)):
                return 1
            if( x.count('++')>0):
                return 2
            if(x.count('+++')>0):
                return 3
            if(x.count('+')>0 or x.count('阳')>0):
                return 4
            if(x.count('/')>0):
                return 4
            else:
                return 1
        except:
            return 1

    data_history['yin_yang_7'] = data_history['3430'].map(lambda x: get_num_status33(x))
    del data_history['3430']
    # data_history[data_history['shuzhi_feat9']==0]['shuzhi_feat9']
    #####################################################################################################################
    #data_history['3429'].unique()


    def fun31(x):
        try:
            x = float(x)
            if (x > 100):
                x = np.nan
                return x
            return x
        except:
            if (x.count('散在满视野') or x.count('脓白')):
                return np.nan
            if (x.count('+-') == 1):
                return 0
            if (x.count('+') == 1):
                return np.nan
            if(x.count('(2-4)/HP')==1):
                return 3
            if(x.count('(5-10)/HP')==1):
                return 7.5
            if (x.count('未见') == 1 or x.count('阴') or x.count('未检出') or x.count('正常') or x.count('少许') or x.count('偶见')):
                return 0
            try:
                if (x.count('-') == 1):
                    i = x.index('-')
                    if (x.count('个') == 1):
                        i2 = x.index('个')
                        x1 = x[i + 1:i2]
                        x1 = float(x1)
                        x2 = x[:i]
                        x2 = float(x2)
                        x = (x1 + x2) / 2
                        return x
                    if (x.count('/') == 1):
                        i2 = x.index('/')
                        x1 = x[i + 1:i2]
                        x1 = float(x1)
                        x2 = x[:i]
                        x2 = float(x2)
                        x = (x1 + x2) / 2
                        return x
                    x1 = x[i +2:]
                    x2 = x[:i]
                    x1 = float(x1)
                    x2 = float(x2)
                    x = (x1 + x2) / 2
                    return x
            except:
                return np.nan
                #    if(x.count('＜')>0 or x.count('<')>0 ):
                #           return 5.0
            #        if(x.count('阴')):
            #          return 4.8
            #       if(x.count('。')):
            #           return 3.89
            # if(x.count('.3.70')==1):
            #    return 3.70
            # if(x.count('4.14.')==1):
            #    return 4.14
            return 0


    data_history['shuzhi_feat31'] = data_history['3429'].map(lambda x: fun31(x))
    g = data_history['shuzhi_feat31']
    g = pd.DataFrame(g)
    data_history['shuzhi_feat31'] = g.fillna(g.mean())
    del data_history['3429']

    #####################################################################################################################
    def get_heart_33(x):
        try:
            if((x.count('正常')>0) or (x.count('胆囊形态大小正常')>0) or (x.count('大小形态正常')>0) or (x.count('大小、形态正常')>0) or (x.count('胆囊大小、形态未见明显异常')>0) or (x.count('胆囊大小正常')>0) or (x.count('胆囊大小,形态正常')>0)):
                return 1
            if((x.count('过缓')>0) or (x.count('自述')>0) or  (x.count('切除')>0)):
                return 2
            if(x.count('早搏')>0):
                return -1
            if(x.count('不齐')>0):
                return 3
            if (x.count('异') > 0):
                return 4
            if (x.count('过速') > 0):
                return 5
            if (x.count('肥大') > 0):
                return 6
            else:
                return 4
        except:
            return 1
    data_history['heart_stat_33']=data_history['1001'].map(lambda x:get_num_status33(x))
    del data_history['1001']
    ###################################################################################################################
    def get_heart_35(x):
        try:
            if((x.count('未见')>0) or (x.count('未检出')>0) or (x.count('阴性')>0) or (x.count('0')>0 and len(x)==1)  or (x.count('-')>0) or (x.count('0个/LP')>0) or (x.count('胆囊大小,形态正常')>0)):
                return 1
            else:
                return 0
        except:
            return 1
    data_history['heart_stat_35']=data_history['3730'].map(lambda x:get_heart_35(x))
    del data_history['3730']
    ####################################################################################################################
    data_history_4_24=pd.read_csv('data_history_feat_clash3_4_24.csv')
    del data_history_4_24['0430']
    del data_history_4_24['0431']
    del data_history_4_24['0434']
    del data_history_4_24['0420']
    ########################################################################################################################
    def get_feat_0406(x):
        try:
            if((x.count('未')>0) or (x.count('正常')>0) or (x.count('不大')>0) ):
                return 1
            else:
                return 0
        except:
            return 1
    data_history['feat_0406']=data_history_4_24['0406'].map(lambda x:get_feat_0406(x))
    del data_history_4_24['0406']
    ###########################################################################################################################
    def get_feat_0407(x):
        try:
            if((x.count('未')>0) or (x.count('正常')>0) or (x.count('弃查')>0) or (x.count('不大')>0)):
                return 1
            else:
                return 0
        except:
            return 1
    data_history['feat_0407']=data_history_4_24['0407'].map(lambda x:get_feat_0407(x))
    del data_history_4_24['0407']
    ##########################################################################################################
    #['vid', '0413', '0423', '0424', '0432', '0433', '0435', '0901', '0911',
        #  '0912', '0929', '0947', '0949', '0954', '0974', '0975', '0976', '0985',
    #    '100010', '1117', '1308', '192', '193'],
    #   dtype='object')
    def get_feat_0433(x):
        try:
            if((x.count('未')>0) or (x.count('正常')>0) or (x.count('弃查')>0) or (x.count('不大')>0) or (x.count('无')>0)):
                return 1
            if(x.count('脾脏切除')):
                return 2
            else:
                return 0
        except:
            return 1
    data_history['feat_0433']=data_history_4_24['0433'].map(lambda x:get_feat_0433(x))
    del data_history_4_24['0433']
    ########################################################################################################################
    def get_feat_1117(x):
        try:
            x=float(x)
        #  if(x==0 or x>1000):
        #      x=np.nan
            #     return x
            if(x==0):
                x=np.nan
                return x
            return x
        except:
            if(x.count(' ')):
                i=x.index(' ')
                x=x[:i]
                x=float(x)
                return x
            return np.nan
    data_history['feat_1117']=data_history_4_24['1117'].map(lambda x:get_feat_1117(x))
    g = data_history['feat_1117']
    g = pd.DataFrame(g)
    data_history['feat_1117'] = g.fillna(g.mean())
    del data_history_4_24['1117']
    #######################################################################################################################
    def feat_combine():
        a=pd.get_dummies(data_history['Thyroid_status'],prefix='Thyroid_status')
        b=pd.get_dummies(data_history['gan_1'],prefix='gan_1')
        c=pd.get_dummies(data_history['dan_1'],prefix='dan_1')
        d=pd.get_dummies(data_history['ganzang_1'],prefix='ganzang_1')
        e=pd.get_dummies(data_history['yixian_1'],prefix='yixian_1')
        f=pd.get_dummies(data_history['pizang_1'],prefix='pizang_1')
        g=pd.get_dummies(data_history['left_shen_1'],prefix='left_shen_1')
        h = pd.get_dummies(data_history['right_shen_1'], prefix='right_shen_1')
        l=pd.get_dummies(data_history['health_profile'],prefix='profile_h')
        aa=pd.get_dummies(data_history['heart'],prefix='heart_')
        bb=pd.get_dummies(data_history['health_status'],prefix='hea_status')
        cc=pd.get_dummies(data_history['hh_status'],prefix='hh_status__')
        dd=pd.get_dummies(data_history['yin_yang_1'],prefix='yin_yang_')
        ee = pd.get_dummies(data_history['yin_yang_2'], prefix='yin_yang_2')
        ff = pd.get_dummies(data_history['yin_yang_3'], prefix='yin_yang_3')
        gg = pd.get_dummies(data_history['yin_yang_4'], prefix='yin_yang_4')
        hh = pd.get_dummies(data_history['yin_yang_5'], prefix='yin_yang_5')
        ii = pd.get_dummies(data_history['yin_yang_6'], prefix='yin_yang_6')
        jj=pd.get_dummies(data_history['yin_yang_7'], prefix='yin_yang_7')
        kk=pd.get_dummies(data_history['heart_stat_33'], prefix='heart_stat_33__')
        mm = pd.get_dummies(data_history['heart_stat_35'], prefix='heart_stat_35__')
        a_24=pd.get_dummies(data_history['feat_0407'], prefix='feat_0407_')
        b_24 = pd.get_dummies(data_history['feat_0406'], prefix='feat_0406_')
        c_24=pd.get_dummies(data_history['feat_0433'], prefix='feat_0433_')
        feat_now=pd.concat([data_history_vid,a,b,c,d,e,f,g,h,l,aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,mm,a_24,b_24,c_24,data_history['shuzhi_feat1'],data_history['shuzhi_feat2'],data_history['shuzhi_feat3'],data_history['shuzhi_feat4'],data_history['shuzhi_feat5'],data_history['shuzhi_feat6'],data_history['shuzhi_feat7'],data_history['shuzhi_feat8'],data_history['shuzhi_feat9'],data_history['shuzhi_feat10'],data_history['shuzhi_feat21'],data_history['shuzhi_feat30'],data_history['shuzhi_feat31'],data_history['feat_1117']],axis=1)
        return feat_now
    feat_now=feat_combine()
    feat_now.to_csv('feat_history_4_24_21_25.csv',index=False)

    print("******* start at:", time.strftime('%Y-%m-%d %H:%M:%S',now ), '*******')
