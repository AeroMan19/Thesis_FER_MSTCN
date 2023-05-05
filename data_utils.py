import numpy as np
import os
import warnings
import torch.utils.data as td
import torch
import cv2
from torchvision import transforms

warnings.filterwarnings("ignore")


def sample_data(vid_arr, sample_rate=2):
    vid_arr = vid_arr[::sample_rate, :, :, :]

    return vid_arr


class VideoDataset(td.Dataset):
    def __init__(self, data_path, mode, sample_rate, n_classes, transform=None):
        super(td.Dataset, self).__init__()
        self.mode = mode  # train | val | test
        self.data_path = os.path.join(data_path, self.mode)
        self.data_list = []
        self.folder_list = os.listdir(self.data_path)

        for folder in self.folder_list:

            files = os.listdir(os.path.join(self.data_path, folder))
            for file in files:
                self.data_list.append(os.path.join(folder, file))

        self.transform = transform
        self.sample_rate = sample_rate
        self.n_classes = n_classes

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        frame_arr = []
        frame_count = 1

        if torch.is_tensor(idx):
            idx = idx.tolist()

        f_name = self.data_list[idx]  # this outputs 'folder_name/file_name'
        video_path = os.path.join(self.data_path, f_name)

        print("Video path: ", video_path)

        # read video
        cap = cv2.VideoCapture(video_path)
        totframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if cap.isOpened() is False:
            print("Error opening video stream or file")

        while True:
            if frame_count > 200 or frame_count >= totframe:
                break
            ret, frame = cap.read()

            if self.transform:
                frame = self.transform(frame)

            frame = np.expand_dims(frame, axis=0)
            #print('frame:', frame.shape)
            if len(frame_arr) == 0:
                frame_arr = frame
            else:
                frame_arr = np.concatenate((frame_arr, frame), axis=0)
            #print('frame_arr:', frame_arr.shape)
            frame_count += 1

        frame_arr = torch.from_numpy(frame_arr)
        frame_arr = sample_data(frame_arr, self.sample_rate)

        label = torch.tensor(int(f_name.split('/')[0]))
        labels = label.repeat(frame_arr.shape[0], 1)  # repeates the same label for each frame

        cap.release()
        # print('labels shape: ', labels.shape)
        # print('frame_arr shape: ', frame_arr.shape)
        return frame_arr, labels, f_name


class FeatureDataset(td.Dataset):
    def __init__(self, feat_path, mode, n_classes):
        super(td.Dataset, self).__init__()
        self.mode = mode  # train | val | test
        self.feat_path = os.path.join(feat_path, self.mode, 'features/')
        self.gt_path = os.path.join(feat_path, self.mode, 'ground_truth/')

        self.feat_file_list = os.listdir(self.feat_path)
        self.n_classes = n_classes

    def __len__(self):
        return len(self.feat_file_list)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        f_name = self.feat_file_list[idx]
        vid_feat_path = os.path.join(self.feat_path, f_name)
        vid_gt_path = os.path.join(self.gt_path, f_name)

        features = np.load(vid_feat_path)
        g_truth = np.load(vid_gt_path)

        # vid_len = len(features)

        return features, g_truth, f_name


