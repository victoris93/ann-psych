import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

def load_data(data_csv, n_grad = None, aligned_grads = True):
    '''
    data_type: 'conn', 'disp', 'grad', 'nbs', 'eigen', 'disp_within_bth', 'disp_between_bth'
    '''
    data = []
    if aligned_grads:
        aligned = 'aligned'
            
    for subject in tqdm(data_csv['participant_id']):
        root_path = data_csv[data_csv['participant_id'] == subject]['path'].values[0]
        subj_path = f'{root_path}/sub-{subject}/func'
        try:
            features = [np.load(f'{subj_path}/{i}') for i in os.listdir(subj_path) if "gradients" in i and aligned in i and "labels" not in i][0]
            features = features[:,:, :n_grad]
            data.append(features)

        except FileNotFoundError as e:
            print(f"Data not found for subject {subject}: {e}.")
            data_csv = data_csv[data_csv['participant_id'] != subject]
    data = np.row_stack(data)
    return data, data_csv

class GradLoader(Dataset):
    def __init__(self, grads, labels):
        if isinstance(grads, str):
            self.grads = np.load(grads)[0, :, :3]
        elif isinstance(grads, np.ndarray):
            self.grads = grads
        if isinstance(labels, str):
            self.labels = np.load(labels)
        elif isinstance(labels, np.ndarray):
            self.labels = labels
        self.grads = torch.tensor(self.grads, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.grads)

    def __getitem__(self, idx):
        return self.grads[idx], self.labels[idx]
    
    @staticmethod
    def stratified_split(dataset, test_size=0.25):
        # Ensure labels are numpy array for compatibility with train_test_split
        labels = dataset.labels.numpy() if isinstance(dataset.labels, torch.Tensor) else dataset.labels

        # Get indices for training and test sets
        train_idx, test_idx = train_test_split(
            range(len(dataset)),
            test_size=test_size,
            stratify=labels
        )

        # Create training and testing subsets
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        return train_dataset, test_dataset

def test(model, test_loader, num_class=3, use_cpu=True):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for points, target in tqdm(test_loader):
        if not use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc
