import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import provider
import json
from utils import GradLoader, load_data, test
import pandas as pd
import csv

# input size is (batch, num_points, dim)
# transposed input size is (batch, dim, num_points)
def parse_args():
    parser = argparse.ArgumentParser(description='PointNet++ optimization')
    parser.add_argument('--radius1', type=float, required=True)
    parser.add_argument('--radius2', type=float, required=True)
    parser.add_argument('--radius3', type=float, required=True)
    parser.add_argument('--radius4', type=float, required=True)
    parser.add_argument('--radius5', type=float, required=True)
    parser.add_argument('--radius6', type=float, required=True)
    parser.add_argument('--dropout_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--loss_function', type=str, required=True)
    parser.add_argument('--optimizer', type=str, required=True)
    parser.add_argument('--adam_lr', type=float, default=None)
    parser.add_argument('--adam_beta1', type=float, default=None)
    parser.add_argument('--adam_beta2', type=float, default=None)
    parser.add_argument('--sgd_lr', type=float, default=None)
    parser.add_argument('--sgd_momentum', type=float, default=None)
    args = parser.parse_args()
    return args


class get_model(nn.Module):
    def __init__(self,num_class,radius_list, normal_channel=True):
        super(get_model, self).__init__()
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
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

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


class get_loss(nn.Module):
    def __init__(self, loss_fn=F.nll_loss):  # Default is F.nll_loss
        super(get_loss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, pred, target, trans_feat):
        total_loss = self.loss_fn(pred, target)
        return total_loss

#load grads and labels
def main(args, use_cpu=False):
    participants = pd.read_csv('participants.csv')
    grads, participants = load_data(data_csv = participants, n_grad = 3)
    labels = participants["diagnosis"].values
    dataset = GradLoader(grads, labels)
    train_dataset, test_dataset = GradLoader.stratified_split(dataset)
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size

    num_class = 2
    use_cpu = False
    n_epochs = 200
    use_normals = False
    radius_list = [[args.radius1, args.radius2, args.radius3], [args.radius4, args.radius5, args.radius6]]
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate  # Make sure to use this in your model if applicable

    classifier = get_model(num_class, radius_list = radius_list, normal_channel=use_normals)
    if args.loss_function == 'cross_entropy':
        loss_fn = F.cross_entropy
    else:
        loss_fn = F.nll_loss
    criterion = get_loss(loss_fn=loss_fn)
    if not use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # Setup optimizer with parsed arguments
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.adam_lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=1e-08,
            weight_decay=1e-4  # You can also make decay_rate a parsed argument
        )
    else:  # Assuming the only other option is 'SGD'
        optimizer = torch.optim.SGD(
            classifier.parameters(), 
            lr=args.sgd_lr, 
            momentum=args.sgd_momentum
        )


    start_epoch = 0
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    mean_test_instance_acc = []
    mean_test_class_acc = []
    for epoch in range(start_epoch, n_epochs):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, n_epochs))
        mean_correct = []
        classifier.train()

        for points, target in tqdm(train_loader):
            optimizer.zero_grad()

            # Apply transformations to the points
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

            # if (instance_acc >= best_instance_acc):
            #     print('Save model...')
            #     # savepath = str(checkpoints_dir) + '/best_model.pth'
            #     savepath = './best_model/best_modep.pth'
            #     print('Saving at %s' % savepath)
            #     state = {
            #         'epoch': best_epoch,
            #         'instance_acc': instance_acc,
            #         'class_acc': class_acc,
            #         'model_state_dict': classifier.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }
            #     torch.save(state, savepath)
            mean_test_instance_acc.append(instance_acc)
            mean_test_class_acc.append(class_acc)
            global_epoch += 1
    mean_test_instance_acc = np.mean(mean_test_instance_acc)
    mean_test_class_acc = np.mean(mean_test_class_acc)
    # open a csv file
    with open('results/optim_results.csv', 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        # if the csv file is empty, write the header
        if f.tell() == 0:
            writer.writerow(['radius1', 'radius2', 'radius3', 'radius4', 'radius5', 'radius6', 'dropout_rate', 'batch_size', 'loss_function', 'optimizer', 'adam_lr', 'adam_beta1', 'adam_beta2', 'sgd_lr', 'sgd_momentum', 'mean_test_instance_acc', 'mean_test_class_acc'])
        writer.writerow([args.radius1, args.radius2, args.radius3, args.radius4,args.radius5, args.radius6, args.dropout_rate, args.batch_size, args.loss_function, args.optimizer, args.adam_lr, args.adam_beta1, args.adam_beta2, args.sgd_lr, args.sgd_momentum, mean_test_instance_acc, mean_test_class_acc])
    print('End of training.')

if __name__ == '__main__':
    args = parse_args()
    main(args)
