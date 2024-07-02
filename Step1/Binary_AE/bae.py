# ==============================================================================
#  Copyright (c) 2024. Imen Ben Amor
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
from tqdm import tqdm
from LR_framework.trials import load_trials
from preprocessing.preprocessing import *
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import shutil
import logging
from utils import *
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

def stringToList(string):
    listRes = list(string.split(" "))
    return listRes


def readVectors(filePath):
    vectors = []
    utt = []
    with open(filePath, "r") as f:
        line_idx = 0
        last_printed_percent = -1
        number_of_lines = 1021175
        for line in f:
            # line_idx += 1
            elems = line.split("  ")
            if not (elems[0].endswith("babble") or elems[0].endswith("reverb") or elems[0].endswith("noise") or
                    elems[0].endswith("music")):
                # logging.info(f"reading file: {elems[0]}..")
                vec = []
                utt.append(elems[0])
                for elem in stringToList(elems[1][2:-2].rstrip()):
                    vec.append(float(eval(elem)))
                vectors.append(vec)

            percent = round(line_idx / number_of_lines * 100, 0)
            if percent % 10 == 0 and percent != last_printed_percent:
                logging.info(f"{percent}%")
            last_printed_percent = percent

    return utt, np.array(vectors)


class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.Tanh()

        )
        self.binarization = StraightThroughEstimator()
        self.BN = nn.BatchNorm1d(512)
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256)

        )

    def forward(self, x):
        x = self.encoder(x)
        binary = self.binarization(x)
        recon = self.decoder(binary)
        return recon, binary, x

class CustomDatasetall(Dataset):
    def __init__(self, data):
        self.data = data
        self.classes = torch.unique(Y)
        self.num_classes = 27
        self.num_examples_per_class = 10
        # Calculate the number of complete batches
        self.num_complete_batches = len(self.classes) // self.num_classes

    def __getitem__(self, index):
        # Calculate the starting and ending indices for the selected classes
        start_class_index = index * self.num_classes
        end_class_index = (index + 1) * self.num_classes

        # Select the classes in the specified order
        selected_classes = self.classes[start_class_index:end_class_index]
        # selected_classes = random.sample(self.classes, self.num_classes)

        # Initialize lists for batches and labels
        batch_1 = []
        # batch_2 = []
        labels = []
        # nb_utt_spk={}
        for class_label in selected_classes:
            # print(class_label)
            # Select examples for the current class
            class_indices = torch.where(self.data['class'] == class_label)[0]
            # print(len(class_indices))
            # Randomly choose examples for the current class
            selected_indices1 = torch.randperm(len(class_indices))[:self.num_examples_per_class]
            # print(len(selected_indices1))
            # selected_indices2 = torch.randperm(len(class_indices))[:self.num_examples_per_class]
            # Append selected examples to the appropriate batch
            batch_1.extend(self.data['input'][class_indices[selected_indices1]])
            # batch_2.extend(self.data['input'][class_indices[selected_indices2]])

            # Append labels for the current class
            labels.extend([class_label] * self.num_examples_per_class)

        return torch.stack(batch_1), torch.tensor(labels, dtype=int)  # ,  torch.stack(batch_2),

    def __len__(self):
        num_batches = len(self.classes) // self.num_classes
        return num_batches
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
def sparse_loss(a,nb_classes,nb_samples):
    b=[]
    for c in range(nb_classes):
        b.append(a[c*nb_samples:nb_samples+c*nb_samples,:].sum(axis=0))
    b=torch.stack(b).float().cuda()
    #losses = [nn.MSELoss()(b[:, col], torch.zeros_like(b[:, col])) for col in range(b.shape[1])]
    #total_loss = sum(losses) / len(losses)
    #return total_loss
    loss=b.add_(torch.tensor(desired).cuda()).clamp(0,1).pow_(2).sum()#.pow_(2).sum()
    return loss
data_path = [Train (VOX2) XVEC PATH]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
utterances, vectors = readVectors(data_path)
xvectors = np.array(vectors).astype('float64')

utt_per_spk, loc_list = number_utterances(utterances)
data_path_test = [Test (VOX1) XVEC PATH]
utterances_test, vectors_test = readVectors(data_path_test)
xvectors_test = np.array(vectors_test).astype('float64')
non, tar = load_trials()
X_test = torch.tensor(xvectors_test, dtype=torch.float64)
utterances1=[]
for i in utterances:
    utterances1.append(i.replace("/","-"))
# encode class values as integers
utt_per_spk, loc_list = number_utterances(utterances1)

encoder = LabelEncoder()
encoder.fit(loc_list)
Y = encoder.transform(loc_list)

X = torch.tensor(xvectors, dtype=torch.float64)
#X = F.normalize(X)
Y = torch.tensor(Y, dtype=int).reshape(-1, 1)
data = {
    'input': X,
    'class': Y
}
custom_dataset = CustomDatasetall(data)
trainloader = DataLoader(custom_dataset, batch_size=1, shuffle=True)
BA=[f"BA{i}" for i in range(512)]
desired=-np.random.randint(0, 10,size=512)
shutil.rmtree(f'runs/sparse_binonly1', ignore_errors=True)
writer = SummaryWriter(f'runs/sparse_binonly1')


def scoring(target, non, df, c):
    if c == "balr":
        # jacctarget, jaccnon = jaccard_scoresb(target, non, df)
        costarget, cosnon = cosine_scoresb(target, non, df)
    else:
        # jacctarget, jaccnon = jaccard_scores(target, non, df)
        costarget, cosnon = cosine_scores(target, non, df)
    print("=============Cosine score================")
    eer, cllr_min, cllr_act = Cllr_min(costarget, cosnon)
    print(f"Cllr (min/act):({cllr_min}, {cllr_act}),eer= {eer} ")
    return eer


net = Autoencoder().to(device)
net.apply(init_weights)
optimizer = optim.Adam(net.parameters(), lr=0.001)
criterion_MSE = nn.MSELoss().to(device)
net.train()

MSE = []
tot = []
epochs = 1000
total_step = len(trainloader)
for e in range(epochs):
    total_step = len(trainloader)
    loss_batches = 0
    avg_loss = 0
    avg_mse = 0
    avg_sparse = 0
    logging.info(f'Starting epoch {e} of {epochs}')
    idx = 0
    for X1, y in tqdm(trainloader):
        optimizer.zero_grad()
        X1 = X1[0].to(device, dtype=torch.float)
        y = y[0].squeeze().to(device)
        features = X1
        recon, binary, x = net(features)
        mse = criterion_MSE(recon, features)
        sparsity = sparse_loss(x, 27, 10)
        loss = mse  +0.001*sparsity
        loss_batches += loss.item()
        avg_mse += mse.item()
        avg_sparse += sparsity.item()
        loss.backward()
        optimizer.step()

    avg_sparse /= len(trainloader)
    avg_mse /= len(trainloader)
    print(f"MSE={avg_mse}")
    print(f"sparsity={avg_sparse}")
    writer.add_scalar("total_loss/train", loss_batches / total_step, e)
    writer.add_scalar("sparsity/train", avg_sparse, e)
    writer.add_scalar("mse/train", avg_mse, e)
    torch.save(net.state_dict(), f"sparse_binonly1/binary_CE_{e}.pt")
    net.eval()
    if e % 2 == 0:
        dicti = {}
        for x, u in zip(X_test, utterances_test):
            x = x.reshape(1, -1).to(device, dtype=torch.float)
            recon, binary, x = net(x)
            dicti[u] = binary[0].cpu().detach().numpy()
        binary = np.array(list(dicti.values()))
        df = pd.DataFrame(binary)
        eer = scoring(tar, non, df, "balr")
        print(eer)
        writer.add_scalar("EER/Vox1", eer, e)
        net.train()

