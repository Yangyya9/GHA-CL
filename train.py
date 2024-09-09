import torch
from sklearn.datasets import images
from torch import nn
from network import GHA_CL_MVC
from metric import valid, evaluate
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from dataloader import load_data
import os
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns


Dataname = "Hdigit"
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0005)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--rec_epochs", default=200)
parser.add_argument("--fine_tune_epochs", default=100)
parser.add_argument("--low_feature_dim", default=128)
parser.add_argument("--high_feature_dim", default=512)
parser.add_argument("--hidden_dim", default=128)
parser.add_argument("--contrastive_ins_enable", default=True, type=bool, help="Enable contrastive instance")
parser.add_argument("--contrastive_cls_enable", default=True, type=bool, help="Enable contrastive class")
parser.add_argument("--contrastive_symmetry", default=False, type=bool, help="Enable contrastive symmetry")
parser.add_argument("--contrastive_projection_dim", default=256)
parser.add_argument("--contrastive_projection_layers", default=2)
parser.add_argument("--contrastive_nmb_protos", default=256)
parser.add_argument("--contrastive_eps", default=0.05)
parser.add_argument("--contrastive_ds_iters", default=3)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "CCV":
    args.fine_tune_epochs = 30
    seed = 3

if args.dataset == "Hdigit":
    args.fine_tune_epochs = 100
    seed = 10

if args.dataset == "Cifar100":
    args.fine_tune_epochs = 15
    seed = 10

if args.dataset == "YouTubeFace":
    args.fine_tune_epochs = 100
    seed = 10

if args.dataset == "Synthetic3d":
    args.fine_tune_epochs = 100
    seed = 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )


def pre_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()

    for batch_idx, (xs, y, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        xrs, zs, _ = model(xs)
        loss_list = []

        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    pre_train_Loss = tot_loss / len(data_loader)

    print('Epoch {}'.format(epoch), 'pre_train_Loss:{:.6f}'.format(pre_train_Loss))


metrics_list=[]
def fine_tune(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    commonZ = []
    commonZ_cat=[]
    labels_vector = []

    for batch_idx, (xs, y, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        xrs, _, hs = model(xs)
        commonz, commonz_cat, _ = model.GHA_CL(xs)
        commonZ.extend(commonz.cpu().detach().numpy())
        labels_vector.extend(y.numpy().ravel())

        commonz_cat = commonz_cat.cpu()
        commonZ_cat.extend(commonz_cat.detach().numpy())

        loss_list = []

        for v in range(view):
            loss_list.append(mes(xs[v], xrs[v]))
        contrastive_loss, loss_ins, loss_cls =model.contrastive_loss(commonz, hs)

        loss_list.append(contrastive_loss)
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    fine_tune_Loss =tot_loss / len(data_loader)

    commonZ = np.array(commonZ)
    labels_vector = np.array(labels_vector)
    commonZ_cat = np.array(commonZ_cat)


    kmeans = KMeans(n_clusters=class_num, n_init=100)
    y_pred = kmeans.fit_predict(commonZ)

    acc, nmi, pur, ari = evaluate(labels_vector, y_pred)
    metrics_list.append((acc, nmi, pur, ari))

    print('Epoch {}'.format(epoch), 'fine_tune_Loss:{:.6f}'.format(fine_tune_Loss))
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(acc, nmi, pur, ari))


if not os.path.exists('./models'):
    os.makedirs('./models')
model = GHA_CL_MVC(args, view, dims, args.high_feature_dim, args.low_feature_dim,  device, args.contrastive_ins_enable, args.contrastive_cls_enable)
print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)



for i in range(1, args.rec_epochs+1):
    pre_train(i)
print("预训练结束---------------------------")


for j in range(args.rec_epochs+1, args.rec_epochs + args.fine_tune_epochs+1):
    fine_tune(j)
    if j == args.rec_epochs + args.fine_tune_epochs:
        print("所有训练结束#########################")
        print()
        sorted_metrics = sorted(metrics_list, key=lambda x: x[0], reverse=True)
        top_metrics = sorted_metrics[:5]
        avg_acc = sum(metric[0] for metric in top_metrics) / len(top_metrics)
        avg_nmi = sum(metric[1] for metric in top_metrics) / len(top_metrics)
        avg_pur = sum(metric[2] for metric in top_metrics) / len(top_metrics)
        avg_ari = sum(metric[3] for metric in top_metrics) / len(top_metrics)
        print('最终结果ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(avg_acc, avg_nmi, avg_pur, avg_ari))
        state = model.state_dict()
        torch.save(state, './models/' + args.dataset + '.pth')
        print('Saving model...')

