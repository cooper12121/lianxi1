import sys
sys.path.append("./")
from model import *
from transformers import AutoConfig,AutoTokenizer
from data_loader import *
from seqeval.metrics import classification_report,f1_score
from transformers import TrainingArguments,Trainer
from transformers import DataCollatorForTokenClassification
from datasets import concatenate_datasets
import torch
from torch.nn.functional import cross_entropy
import numpy as np
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
index2tag = {idx:tag for idx,tag in enumerate(tags.names)}
tag2index = {tag:idx for idx,tag in enumerate(tags.names)}
index2tag[-100]='IGN'
langs = ['de','fr','it','en']#四种语言

xlmr_model_name = "xlm-roberta-base"
xlmr_config = AutoConfig.from_pretrained(xlmr_model_name,num_labels = tags.num_classes,id2label=index2tag,label2id=tag2index)
xlmr_model = (XLMRobertaForTokenClassfication.from_pretrained(xlmr_model_name,config=xlmr_config).to(device))
xlmr_tokenizer= AutoTokenizer.from_pretrained(xlmr_model_name)
data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)#数据整理器，将输入序列填充到最大序列长度
panx_de_aligned = align_panx_dataset(panx_ch['de'])  
def train():
    
    num_epochs = 3
    batch_size=24
    logging_steps = len(panx_de_aligned['train'])//batch_size
    model_name = f"{xlmr_model_name}-finetuned-panx-de"
    training_args = TrainingArguments(output_dir=model_name,log_level="error",
    num_train_epochs=num_epochs,per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",save_steps=1e6,weight_decay=0.01,disable_tqdm=False,logging_steps=logging_steps,
    push_to_hub=False
    )
    trainer = Trainer(xlmr_model,args=training_args,data_collator=data_collator,compute_metrics=compute_metrics,
    train_dataset=panx_de_aligned['train'],eval_dataset=panx_de_aligned['validation'],tokenizer=xlmr_tokenizer
    )
    trainer.train()


    """ 对多种语料进行微调 """
    corpora_aligned,corpora = add_corpora()
    training_args.logging_steps = len(corpora_aligned['train'])//batch_size
    training_args.output_dir = "xlm-roberta-base-finetuned-panx-all"
    trainer_all = Trainer(xlmr_model,args=training_args,data_collator=data_collator,
    compute_metrics=compute_metrics,tokenizer=xlmr_tokenizer,train_dataset=corpora_aligned['train'],
    eval_dataset=corpora_aligned['validation'])
    trainer_all.train()
    return trainer,trainer_all


"""
    进行错误分析
        在验证集中查看损失 
    """
def forward_pass_with_label(batch):
    trainer = train()
    features = [dict(zip(batch,t))for t in zip(*batch.values())]
    batch = data_collator(features)

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        output = trainer.model(input_ids,attention_mask)
        predicted_label = torch.argmax(output.logits,axis=-1).cpu().numpy()
    loss = cross_entropy(output.logits.view(-1,7),labels.view(-1))
    loss = loss.view(len(input_ids),-1).cpu().numpy()
    return {'loss':loss,'precited_label':predicted_label}
def Analusis_loss():
    valid_set = panx_de_aligned['validation']
    valid_set = valid_set.map(forward_pass_with_label,batched=True,batch_size=32)

    #数据加载到dataframe中进一步分析
    df = valid_set.to_pandas()
    df['input_tokens']=df['input_ids'].apply(lambda x:xlmr_tokenizer.convert_ids_to_tokens(x))
    df['predicted_label'] = df['predicted_label'].apply(lambda x:[index2tag[i]for i in x])
    df['labels'] = df['labels'].apply(lambda x:[index2tag[i]for i in x])
    df['loss']=df.apply(lambda x:x['loss'][:len(x['input_ids'])],axis=1)
    df['predicted_label']=df.apply(lambda x:x['predicted_label'][:len(x['input_ids'])],axis=1)
    df.to_to_excel("all_examples_loss.xlsx") #所有样本及其损失

    #详细拆分每一个样本进行分析
    df_tokens = df.apply(pd.Series.explode) #为原始列表中的每个元素创建一个行
    df_tokens = df_tokens.query("labels!=IGN")#放弃对IGN元素的统计，因为其损失为0
    df_tokens['loss'] = df_tokens['loss'].astype(float).round(2)
    df_tokens.to_excel("token_loss.xlsx")#对每个token都进行了处理

    #统计每个tokens的损失：计数、平均值、总和
    df1 = (
    df_tokens.groupby('input_tokens')[['loss']]
    .agg(['count','mean','sum'])
    .droplevel(level=0,axis=1)
    .sort_values(by='mean',ascending=False)
    .reset_index()
    .round(2)
    .head(20)#取前20
    .T
    )
    df1.to_excel("token_loss_top20.xlsx")

    df2 = (
    df_tokens.groupby('labels')[['loss']]
    .agg(['count','mean','sum'])
    .droplevel(level=0,axis=1)
    .sort_values(by='mean',ascending=False)
    .reset_index()
    .round(2)
    .T
    )
    df2.t0_excell("entity_loss.xlsx")

#跨语言迁移,测试集的表现
def Cross_language_migration():
    f1_scores = defaultdict(defaultdict)
    trainer,trainer_all = train()
    f1_scores['de']['de'] = evaluate_lang_prefromance('de',trainer)#在德语微调后的模型在德语测试集的表现
    f1_scores['de']['fr'] = evaluate_lang_prefromance('fr',trainer)
    f1_scores['de']['it'] = evaluate_lang_prefromance('it',trainer)
    f1_scores['de']['en'] = evaluate_lang_prefromance('en',trainer)

    corpora_aligned,corpora = add_corpora()
    for idx,lang in enumerate(langs):
        f1_scores['all']['lang']=get_f1_score(trainer_all,corpora[idx]['test'])
    f1_json = json.dumps(f1_scores,sort_keys=False,indent=4,separators=(',',':'))
    f1_save = open('f1_scores_dict.json','w')
    f1_save.write(f1_json)
    f1_save.close()
def get_f1_score(trainer,dataset):
    return trainer.predict(dataset).metrics['test_f1']
def evaluate_lang_prefromance(lang,trainer):
    panx_ds = align_panx_dataset[panx_ch[lang]]
    return get_f1_score(trainer,panx_ds['test'])

#为了解决跨语言迁移引起的性能下降，我们采用一次性对多种语言进行微调
def concatenate_splits(corpora):#将不同的语料库连接
    multi_corpus=DatasetDict()
    for split in corpora[0].keys:
        multi_corpus[split]=concatenate_datasets([corpus[split] for corpus in corpora]).shuffle(seed=20)
    return multi_corpus
def add_corpora():
    corpora = [panx_de_aligned]
    for  lang in langs[1:]:
        ds_aligned = align_panx_dataset(panx_ch[lang])
        corpora.append(ds_aligned)
    corpora_aligned = concatenate_splits(corpora)
    return corpora_aligned,corpora
def tag_text(text,tags,model,tokenizer):#预测text的tags
    tokens = tokenizer(text).tokens()
    print(tokens)
    input_ids = tokenizer(text,return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids)[0]
    predictions = torch.argmax(outputs,dim=2)
    print(predictions)
    preds = [tags.names[p]for p in predictions[0].cpu().numpy()]
    return pd.DataFrame([tokens,preds],index=["tokens","tags"])

def tokenize_and_align_labels(examples):
    """ 标签对齐函数 
    对于一个实体块，只给其第一个token赋标签，其余和特殊字符一样赋"IGN",标签id=-100
    -100是因为torch.nn.CrossEntropyLoss对-100的值会忽略，不计入损失
    """
    tokenized_inputs = xlmr_tokenizer(examples['tokens'],truncation=True,is_split_into_words=True)
    lables=[]
    for idx,label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx=None
        label_ids=[]
        for word_idx in word_ids:
            if word_idx is None or word_idx==previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        lables.append(label_ids)
    tokenized_inputs["labels"]=lables
    return tokenized_inputs

def align_panx_dataset(corpus):
    return corpus.map(tokenize_and_align_labels,batched=True,remove_columns=['langs','ner_tags','tokens'])

def align_predictions(predictions,label_ids):
    """ 评估函数，告精度、召回率和F-score """
    preds = np.argmax(predictions,axis=2)
    batch_size,seq_len = preds.shape
    labels_list,preds_list = [],[]
    for batch_idx in range(batch_size):
        example_labels,example_preds=[],[]
        for seq_idx in range(seq_len):
            if label_ids[batch_idx,seq_idx]!=-100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])
        labels_list.append(example_labels)
        preds_list.append(example_preds)
    return preds_list,labels_list

def compute_metrics(eval_pred):#计算指标
    y_pred,y_true = align_predictions(eval_pred.predictions,eval_pred.label_ids)
    return {'f1':f1_score(y_true,y_pred)}

if __name__ =="__main__":
    train()
