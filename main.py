from models import *
from data_utils import *
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
import warnings
from train import *
warnings.filterwarnings("ignore")


def extract_and_save(dataset, n_classes, bs, new_data_dir, mode, network='ResNet50'):

    if network == 'ResNet50':
        model = FeResNet50(n_classes)

    counter = 0

    for vid_batch, labels, f_name in dataset:  # returns data in each batch
        counter += 1

        print('--  processing: %d / %d' % (counter, len(dataset)))

        # print('labels shape: ', labels.shape)
        # print('frame_arr shape: ', vid_batch.shape)
        # print('f_name shape: ', f_name.shape)

        for ind in range(bs):
            vid = vid_batch[ind, :, :, :, :]
            feats = model.get_features(vid)
            labs = labels[ind, :, :]
            fname = f_name[ind].split('/')[-1]

            new_feat_dir = os.path.join(new_data_dir, mode, 'features/')
            new_gt_dir = os.path.join(new_data_dir, mode, 'ground_truth/')

            if not os.path.exists(new_feat_dir):
                os.makedirs(new_feat_dir)

            if not os.path.exists(new_gt_dir):
                os.makedirs(new_gt_dir)

            np.save(new_feat_dir + fname + '.npy', feats.detach().numpy())
            np.save(new_gt_dir + fname + '.npy', labs.detach().numpy())


if __name__ == '__main__':

    data_path = '/media/saivt/DATA/Capstones/Emotions/Datasets/SEND'
    annot_path = ''
    network = 'ResNet50'  # for feature extraction
    new_data_dir = 'features_' + network + '/'

    do_extract_features = False
    do_train_model = True

    n_classes = 10
    sample_rate = 1
    image_size = 224
    batch_size = 1

    # ---- model ----

    num_stages = 4
    num_layers = 10
    num_f_maps = 300
    features_dim = 2048
    bz = 1
    lr = 0.0005
    num_epochs = 200

    model_dir = 'exp/' + network + '_features/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    transforms.Resize((image_size, image_size))
                                    ])

    # -- Save features -----------
    if do_extract_features:
        train_dataset = DataLoader(
            VideoDataset(data_path, 'Train', sample_rate, n_classes, transform),
            batch_size=batch_size,
            num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

        extract_and_save(train_dataset, n_classes, batch_size, new_data_dir, 'train')

        test_dataset = DataLoader(
            VideoDataset(data_path, 'Test', sample_rate, n_classes, transform),
            batch_size=batch_size,
            num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

        extract_and_save(test_dataset, n_classes, batch_size, new_data_dir, 'test')

        val_dataset = DataLoader(
            VideoDataset(data_path, 'Valid', sample_rate, n_classes, transform),
            batch_size=batch_size,
            num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

        extract_and_save(val_dataset, n_classes, batch_size, new_data_dir, 'val')

    # # -- model training ------
    if do_train_model:
        feat_train_dataset = DataLoader(
            FeatureDataset(new_data_dir, 'train', n_classes), batch_size=batch_size, num_workers=2,
            shuffle=True, pin_memory=True, drop_last=True)

        feat_val_dataset = DataLoader(
            FeatureDataset(new_data_dir, 'val', n_classes), batch_size=batch_size, num_workers=2,
            shuffle=True, pin_memory=True, drop_last=True)

        trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, n_classes)
        trainer.train(model_dir, feat_train_dataset, feat_val_dataset, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

