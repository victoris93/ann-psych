import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import provider
from pointnet2_cls_msg import get_loss
from utils import GradLoader, load_data, test
import pandas as pd
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import optuna
import json

participants = pd.read_csv('participants.csv')
grads, participants = load_data(data_csv = participants, n_grad = 3)
labels = participants["diagnosis"].values
dataset = GradLoader(grads, labels)
train_dataset, test_dataset = GradLoader.stratified_split(dataset)
train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size

def define_model(trial, num_class, normal_channel=False, use_cpu = False):
    # Optuna suggestions for hyperparameters
    radius_list = trial.suggest_categorical(
        'radius_list',
        [[[0.015, 0.025, 0.05], [0.25, 0.5, 0.1]],[[0.1, 0.2, 0.4], [0.2, 0.4, 0.8]], [[0.2, 0.4, 0.8], [0.4, 0.8, 1.6]]]
        )
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

    class OptimPointNet(nn.Module):
        def __init__(self):
            super(OptimPointNet, self).__init__()
            in_channel = 3 if normal_channel else 0
            self.normal_channel = normal_channel
            self.sa1 = PointNetSetAbstractionMsg(512, radius_list[0], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
            self.sa2 = PointNetSetAbstractionMsg(128, radius_list[1], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
            self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
            self.fc1 = nn.Linear(1024, 512)
            self.bn1 = nn.BatchNorm1d(512)
            self.drop1 = nn.Dropout(0.4)
            self.fc2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.drop2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(256, num_class)

        def to_device(self):
            if not use_cpu and torch.cuda.is_available():
                self.cuda()

        def forward(self, xyz):
            B, _, _ = xyz.shape
            if self.normal_channel:
                norm = xyz[:, 3:, :]
                xyz = xyz[:, :3, :]
            else:
                norm = None
            l1_xyz, l1_points = self.sa1(xyz, norm)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
            x = l3_points.view(B, 1024)
            x = self.drop1(F.relu(self.bn1(self.fc1(x))))
            x = self.drop2(F.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            x = F.log_softmax(x, -1)

            return x,l3_points

    return OptimPointNet()

def objective(trial, train_dataset, test_dataset, num_class, use_cpu=False, num_epochs=200):
    model = define_model(trial, num_class, use_cpu=use_cpu)

    batch_size = trial.suggest_categorical('batch_size', [10, 20, 40, 80])
    criterion_name = trial.suggest_categorical('loss_function', ['nll_loss', 'cross_entropy'])
    if criterion_name == 'cross_entropy':
        loss_fn = F.cross_entropy
    # elif criterion_name == 'mse_loss':
    #     loss_fn = F.mse_loss
    else:
        loss_fn = F.nll_loss
    criterion = get_loss(loss_fn=loss_fn)

    if not use_cpu:
        model = DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()


    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=trial.suggest_float('adam_lr', 1e-5, 1e-2),
            betas=(trial.suggest_float('adam_beta1', 0.85, 0.95),
                   trial.suggest_float('adam_beta2', 0.99, 0.999))
        )
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=trial.suggest_float('sgd_lr', 1e-5, 1e-2),
            momentum=trial.suggest_float('sgd_momentum', 0.5, 0.99)
        )

    # best_val_loss = float('inf')
    best_instance_acc = 0.0
    best_val_acc = 0.0
    best_class_acc = 0.0
    epoch_instance_accs = []
    epoch_class_accs = []

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch {epoch + 1} of {num_epochs}")
        model.train()
        mean_correct = []

        for points, target in train_loader:

            optimizer.zero_grad()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = points.transpose(2, 1)

            if not use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = model(points)   
            loss = criterion(pred, target, trans_feat)
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        train_instance_acc = np.mean(mean_correct)
        epoch_instance_accs.append(train_instance_acc)

        # Validation loop - calculate instance and class accuracies
        print("Validation Loop...")
        mean_val_instance_acc = []
        with torch.no_grad():
            val_instance_acc, val_class_acc = test(model, val_loader, num_class=num_class, use_cpu=use_cpu)
            epoch_class_accs.append(val_class_acc)
            mean_val_instance_acc.append(val_instance_acc)

            # if val_instance_acc > best_instance_acc:
            #     best_instance_acc = val_instance_acc
            # if val_class_acc > best_class_acc:
            #     best_class_acc = val_class_acc

            # Save the model if it has the best validation loss so far
    mean_val_instance_acc = np.mean(mean_val_instance_acc)
    if mean_val_instance_acc > best_val_acc:
        best_val_acc = mean_val_instance_acc
        best_model_state = model.state_dict()
        best_hyperparams = {
            'batch_size': batch_size,
            'loss_function': criterion_name,
            'optimizer': optimizer_name,
            'learning_rate': trial.params['adam_lr'] if optimizer_name == 'Adam' else trial.params['sgd_lr'],
            'dropout_rate': trial.params['dropout_rate'],
            'radius_list': trial.params['radius_list'],
            'sgd_momentum': trial.params['sgd_momentum'] if optimizer_name == 'SGD' else 'NAN',
            'adam_beta1': trial.params['adam_beta1'] if optimizer_name == 'Adam' else 'NAN',
            'adam_beta2': trial.params['adam_beta2'] if optimizer_name == 'Adam' else 'NAN'
        }
    if best_model_state is not None:
        torch.save(best_model_state, 'best_model/best_optimized_model.pth')
        with open('best_model/best_hyperparameters.json', 'w') as f:
            json.dump(best_hyperparams, f)

    # Calculate mean instance and class accuracy across epochs
    mean_class_acc = np.mean(epoch_class_accs)
    
    # Print the mean accuracies
    print(f"Mean Instance Accuracy across epochs: {mean_val_instance_acc}")
    print(f"Mean Class Accuracy across epochs: {mean_class_acc}")

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_dataset, test_dataset, num_class = 2, use_cpu=False, num_epochs=200), n_trials=100)
    best_hyperparams = study.best_trial.params

    with open('best_model/best_hyperparameters.json', 'w') as f:
        json.dump(best_hyperparams, f)

    print("Optimization complete. The best hyperparameters are saved in 'best_hyperparameters.json'")

if __name__ == "__main__":
    main()
