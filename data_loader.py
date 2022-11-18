# -*- encoding: utf-8 -*-
'''
@File    :   data_loader.py
@Time    :   2022/11/13 
@Author  :   gaoqiang 
@Version :   1.0
@Contact :   gaoqiang_mx@163.com
'''


""" 
本实验采用的数据集是多语言编码器的跨语言TRansfer评估（XTREME）基准的一个子集，称为PAN-X，该数据集由多种语言的维基百科文章组成，
包括瑞士最常用的四种语言，德语（62.9%）、法语（22.9%）、意大利语（8.4%）和英语（5.9%）。 每篇文章都用LOC（地点）、PER
（人物）和ORG（组织）标签以 "内-外-内"（IOB2）的格式进行了注释。 在这种格式中，B-前缀表示一
个实体的开始，而属于同一实体的连续标记被赋予I-前缀。 一个O标记表示该标记不属于任何实体。 
"""

# 为了制作一个真实的瑞士语料库，我们将根据口语比例对PAN-X的德语（de）、法语（fr）、意大利语（it）和英语（en）语料库进行采样
from datasets import get_dataset_config_names
from collections import defaultdict,Counter
from datasets import DatasetDict,load_dataset
import pandas as pd
def load_data():
    xtreme_subsets = get_dataset_config_names("xtreme")
    # panx_subsets = [s for s in xtreme_subsets if s.startswith("PAN")],PAN-X.

    langs = ['de','fr','it','en']#四种语言
    fracs = [0.629,0.229,0.084,0.059]#采样比例
    panx_ch = defaultdict(DatasetDict) #用来存储DatasetDict的语料

    for lang,frac in zip(langs,fracs):
        ds = load_dataset("xtreme",name=f"PAN-X.{lang}")
        """ 每种单语言数据的格式
        DatasetDict({
            train: Dataset({
                features: ['tokens', 'ner_tags', 'langs'],
                num_rows: 20000
            })
            validation: Dataset({
                features: ['tokens', 'ner_tags', 'langs'],
                num_rows: 10000
            })
            test: Dataset({
                features: ['tokens', 'ner_tags', 'langs'],
                num_rows: 10000
            })
        })
        """
        for split in ds:
            #根据比例对每种数据集的train,val,test数据进行选取，确保每种数据集的三类数据保持相同的比例
            panx_ch[lang][split] = (ds[split].shuffle(seed=0).select(range(int(frac*ds[split].num_rows))))
    
    return panx_ch
    # df = pd.DataFrame({lang:[panx_ch[lang]["train"].num_rows]for lang in langs},index=['number of train examples'])
    # print(df)

#以德语为起点，对法语、意大利语和英语进行Zeroshot跨语言转移
panx_ch = load_data()
tags = panx_ch['de']['train'].features['ner_tags'].feature #取得实体标签 ClassLabel(num_classes=7, names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'], id=None)



def de_data_process():
    panx_de = panx_ch['de'].map(create_tag_name,)
    de_example = panx_de['train'][0]
    # df = pd.DataFrame([de_example['tokens'],de_example['ner_tags_str']],index=['tokens','tags'])
    

    # df = pd.DataFrame.from_dict(cal_entity_frequency(panx_de),orient='index')
    # print(df)
    return panx_de

#为以id表示的ner_tags创建对应的names标签
def create_tag_name(batch):
    return {"ner_tags_str":[tags.int2str(idx)for idx in batch["ner_tags"]]}

def cal_entity_frequency(data):#计算每个实体出现的频率,其中data为某种语料的数据集
    split2freqs = defaultdict(Counter)
    for split,dataset in data.items():
        for row in dataset["ner_tags_str"]:
            for tag in row:
                if tag.startswith("B"):
                    tag_type = tag.split("-")[1]
                    split2freqs[split][tag_type]+=1
    return split2freqs

if __name__ =="__main__":
    de_data_process()