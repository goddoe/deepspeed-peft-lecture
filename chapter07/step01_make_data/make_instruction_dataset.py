#!/usr/bin/env python
# coding: utf-8

# # Alpaca (KoAlpaca dataset)
# 
# - Reference
#   - Origin
#     - https://huggingface.co/datasets/tatsu-lab/alpaca
#   - 한국어 번역
#     - https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a

# In[1]:


from datasets import load_dataset


# In[ ]:


dataset_koalpaca = load_dataset("beomi/KoAlpaca-v1.1a")


# In[31]:


dataset_koalpaca['train'][0]


# In[ ]:


def proc_for_koalpaca(sample):
    sample['text'] = f"### User\n{sample['instruction']}\n\n### Bot\n{sample['output']}"
    return sample

dataset_koalpaca_cvted = dataset_koalpaca.map(proc_for_koalpaca,
                                              remove_columns=dataset_koalpaca['train'].features.keys())


# # ShareGPT data
# 
# 
# - Reference
#   - Origin
#     - https://sharegpt.com
#   - 한국어 번역 
#     - https://huggingface.co/datasets/junelee/sharegpt_deepl_ko

# In[3]:


# Reference: https://huggingface.co/datasets/junelee/sharegpt_deepl_ko

dataset_sharegpt_origin = load_dataset("junelee/sharegpt_deepl_ko", data_files="original_dataset.json")


# In[4]:


dataset_sharegpt_origin['train'][0]


# In[5]:


dataset_sharegpt_deepl_ko = load_dataset("junelee/sharegpt_deepl_ko", data_files="ko_dataset_2.json")


# In[6]:


dataset_sharegpt_deepl_ko['train'][0]


# In[18]:


# 번역된 ShareGPT 데이터를 대화형으로 CausalLM을 학습시키기 위해 변환

def proc_for_sharegpt_deepl_ko(sample):
    conv_list = []
    for turn in sample['conversations']:
        if turn['from'] == 'human':
            user_turn = f"### User\n{turn['value']}"
            conv_list.append(user_turn)
        elif turn['from'] == 'gpt' and len(conv_list) > 0:  # 사람이 먼저 말한 경우만 학습
            bot_turn = f"### Bot\n{turn['value']}"
            conv_list.append(bot_turn)
    sample['text'] = "\n\n".join(conv_list)
    return sample

dataset_sharegpt_deepl_ko_cvted = dataset_sharegpt_deepl_ko.map(proc_for_sharegpt_deepl_ko, remove_columns=["conversations", "id"])


# # Merge Dataset

# In[19]:


from datasets import concatenate_datasets


# In[20]:


dataset_merged = concatenate_datasets([dataset_sharegpt_deepl_ko_cvted['train'],
                                       dataset_koalpaca_cvted['train']], axis=0)

dataset_merged_splitted = dataset_merged.train_test_split(train_size=0.8, shuffle=True, seed=1234)

dataset_merged_splitted['valid'] = dataset_merged_splitted["test"]

del dataset_merged_splitted["test"]


# In[21]:


dataset_merged_splitted.save_to_disk("instruction_dataset")


# In[22]:


from datasets import load_from_disk

dataset = load_from_disk("instruction_dataset")


# In[23]:


dataset


# In[30]:


dataset['train'][6]


# In[ ]:




