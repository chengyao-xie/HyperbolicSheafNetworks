import torch
from pyG_CInd import BrainGraphCINDDataset
# 粘贴或导入 BrainGraphDataset 类定义

dataset = torch.load("brain_node_labeled_dataset_cind.pt")

print(f"dataset 类型: {type(dataset)}")
print(dir(dataset))
print(f"数据数量: {len(dataset)}")
print(f"第一个样本信息:")
print(dataset[0])
