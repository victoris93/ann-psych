import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import provider
from pointnet2_cls_msg import get_model, get_loss
from utils import GradLoader, load_data, test
import pandas as pd

# input size is (batch, num_points, dim)
# transposed input size is (batch, dim, num_points)

#load grads and labels

participants = pd.read_csv('participants.csv')
grads, participants = load_data(data_csv = participants, n_grad = 3)
labels = participants["diagnosis"].values
dataset = GradLoader(grads, labels)
train_dataset, test_dataset = GradLoader.stratified_split(dataset)
train_size = int(0.75 * len(dataset))
test_size = len(dataset) - train_size

num_class = 2
use_cpu = False
optimizer = 'Adam'
decay_rate = 1e-4
learning_rate = 0.001
n_epochs = 200
use_normals = False
batch_size = 20

classifier = get_model(num_class, normal_channel=use_normals)
criterion = get_loss()
if not use_cpu:
    classifier = classifier.cuda()
    criterion = criterion.cuda()

if optimizer == 'Adam':
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=decay_rate
    )
else:
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

start_epoch = 0
global_epoch = 0
global_step = 0
best_instance_acc = 0.0
best_class_acc = 0.0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(start_epoch, n_epochs):
    print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, n_epochs))
    mean_correct = []
    classifier.train()

    for points, target in tqdm(train_loader):
        optimizer.zero_grad()

        # Apply transformations to the points
        print(points.shape)
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = points.transpose(2, 1)

        if not use_cpu:
            points, target = points.cuda(), target.cuda()

        pred, trans_feat = classifier(points)
        loss = criterion(pred, target, trans_feat)
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
        loss.backward()
        optimizer.step()

    train_instance_acc = np.mean(mean_correct)
    print('Train Instance Accuracy: %f' % train_instance_acc)

    with torch.no_grad():
        instance_acc, class_acc = test(classifier, test_loader, num_class=num_class, use_cpu=use_cpu)

        if (instance_acc >= best_instance_acc):
            best_instance_acc = instance_acc
            best_epoch = epoch + 1

        if (class_acc >= best_class_acc):
            best_class_acc = class_acc
        print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        print('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

        if (instance_acc >= best_instance_acc):
            print('Save model...')
            # savepath = str(checkpoints_dir) + '/best_model.pth'
            savepath = './best_model/best_modep.pth'
            print('Saving at %s' % savepath)
            state = {
                'epoch': best_epoch,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        global_epoch += 1

print('End of training.')