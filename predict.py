# -*- coding: utf-8 -*-

import torch
from data_preprocess import load_json
from bert_multilabel_cls import BertMultiLabelCls
from transformers import BertTokenizer

hidden_size = 768
class_num = 3
label2idx_path = "data/label2idx.json"
save_model_path = "model/multi_label_cls.pth"
label2idx = load_json(label2idx_path)
idx2label = {idx: label for label, idx in label2idx.items()}
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
max_len = 128

model = BertMultiLabelCls(hidden_size=hidden_size, class_num=class_num)
model.load_state_dict(torch.load(save_model_path))
model.to(device)
model.eval()

def predict(texts):
    texts = split_text_by_length(texts, max_len)
    all_logits = []

    for text in texts:
        outputs = tokenizer(text, return_tensors="pt", max_length=max_len,
                            padding=True, truncation=True)

        logits = model(outputs["input_ids"].to(device),
                       outputs["attention_mask"].to(device),
                       outputs["token_type_ids"].to(device))
        logits = logits.cpu().tolist()
        all_logits.extend(logits)

    merged_logits = merge_lists(*all_logits)

    result = []
    result_pf = []
    for sample in merged_logits:
        pred_label = []
        for idx, logit in enumerate(sample):
            result_json_pf = {}
            result_json_pf["name"]= idx2label[idx]
            result_json_pf["score"]= logit
            result_pf.append(result_json_pf)
            if logit > 0.5:
                pred_label.append(idx2label[idx])
        result.append(pred_label)
    return result, result_pf

def split_text_by_length(text, length):
    # 去除文本中的换行符和空格
    text = ''.join(text).replace('\n', ' ').replace('，', '').replace('。', '').replace(' ', '')

    # 按照指定长度分割文本
    return [text[i:i + length] for i in range(0, len(text), length)]

def merge_lists(*args):
    merged_list = []
    for idx in range(len(args[0])):
        max_vals = max(sub_list[idx] for sub_list in args)
        merged_list.append(max_vals)
    return [merged_list]

if __name__ == '__main__':
    texts = ["中超-德尔加多扳平郭田雨绝杀 泰山2-1逆转亚泰"]
    result, result_pf_gs = predict(texts)
    print(result_pf_gs)
