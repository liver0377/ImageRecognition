import asyncio
import os
import cv2
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import models, transforms
from spider import get_baidu_baike_image_url
from PIL import ImageFont
from concurrent.futures import ThreadPoolExecutor


async def predict(img_pil, n, model, type):

    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
    plt.rcParams["axes.unicode_minus"] = False

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.eval()
    model = model.to(device)

    # 预处理
    input_img = test_transform(img_pil)
    input_img = input_img.unsqueeze(0).to(device)

    # 前向预测
    pred_logits = model(input_img)

    # 对分数进行softmax运算
    pred_softmax = F.softmax(pred_logits, dim=1)

    # 取置信度最高的n个类别
    top_n = torch.topk(pred_softmax, n)

    # 解析出类别
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()

    # 解析出置信度
    confs = top_n[0].cpu().detach().numpy().squeeze()

    results = []
    if type == "general":
        print("type == general")
        df = pd.read_csv("imagenet_class_index.csv")
        idx_to_labels = {}
        for _, row in df.iterrows():
            idx_to_labels[row["ID"]] = [row["wordnet"], row["Chinese"]]

        for i in range(n):
            class_name = idx_to_labels[pred_ids[i]][1]  # 获取类别名称
            confidence = confs[i]  # 获取置信度
            image_url = await get_baidu_baike_image_url(
                class_name.split(",")[0].strip()
            )
            results.append((class_name, confidence, image_url))
    elif type == "botany":
        print("type == botany")
        idx_to_labels = np.load("idx_to_labels.npy", allow_pickle=True).item()

        # Function to process a single item
        async def process_item(i, pred_id, conf):
            class_name = idx_to_labels[pred_id]  # 获取类别名称
            confidence = conf  # 获取置信度
            image_url = await get_baidu_baike_image_url(class_name.split(",")[0].strip())
            return (class_name, confidence, image_url)

        def process_item_sync(i, pred_id, conf):
            return asyncio.run(process_item(i, pred_id, conf))

        # Function to process all items using multithreading
        async def process_items_concurrently(n, pred_ids, confs):
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as executor:
                tasks = [
                    loop.run_in_executor(executor, process_item_sync, i, pred_ids[i], confs[i])
                    for i in range(n)
                ]
                return await asyncio.gather(*tasks)

        results = await process_items_concurrently(n, pred_ids, confs)

    return results


# img_pil: 待预测的图片
# n: n个置信度最高的返回结果
async def predict_general(img_pil, n):
    return await predict(img_pil, n, models.resnet18(pretrained=True), "general")


async def predict_botany(img_pil, n):
    model = torch.load("checkpoint/fruit30_pytorch_C1.pth")
    return await predict(img_pil, n, model, "botany")
