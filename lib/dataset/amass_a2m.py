# This dataset is mostly a copy of amass.py in the same directory, except mofied
# to match the format of mdm dataloaders for action to motion (a2m). This is a 
# temp solution so we can plug it in easily. 

# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import joblib
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import tqdm

from torch.utils.data import Dataset

import os.path as osp
from mdm.utils import rotation_conversions
from lib.core.config import VIBE_DB_DIR
from lib.data_utils.img_utils import split_into_chunks

class AMASS(Dataset):
    def __init__(self, num_frames, split='train', restrict_subsets=None):
        """
        restrict_subsets: a list of strings of subsets to include in the final dataset
            If None then include all subdatasets. Valid subdataset names are:
            ['ACCAD', 'BioMotionLab', 'CMU', 'EKUT', 'Eyes', 'HumanEva', 'KIT',
             'MPI', 'SFU', 'SSM', 'TCD', 'TotalCapture', 'Transitions'],
        """
        self.dataname='amass'   # mdm training code uses this
        self.restrict_subsets=restrict_subsets
        self.seqlen = num_frames

        self.stride = self.seqlen

        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
        # del self.db['vid_name']

        if not restrict_subsets is None:
            # if True, this overwrites the `vid_indices` object with a restricted version
            self.create_subset(restrict_subsets)

        # transform the data to 6d
        self.create_db_6d_upfront()

        print(f'AMASS dataset number of videos: {len(self.vid_indices)}')

    def create_db_6d_upfront(self):

        print("   dataloader: doing 6d rotations")
        device='cuda'if torch.cuda.is_available() else 'cpu'
        
        thetas = torch.tensor(self.db['theta']).to(device)
        transes = torch.tensor(self.db['trans']).to(device)

        dset = TensorDataset(thetas, transes)
        loader = DataLoader(dset, batch_size=2048*2, shuffle=False, drop_last=False)
        all_data = []

        for (theta, trans) in tqdm.tqdm(loader):
            # like in amass dataset, concat a [1,0,0]: camera orientation (it will be)
            # removed, this is just for consistency
            cam = np.array([1., 0., 0.])[None, ...]
            cam = torch.Tensor(np.repeat(cam, theta.shape[0], axis=0)).to(device)
            theta = torch.cat([cam, theta], -1).to(device)
            # theta = torch.tensor(theta).to(device)

            ### now get the required pose vector
            pose = theta[...,3:75]     # (T,72)
            pose = pose.view(pose.shape[0], 24, 3) # (T,24,3)
            ## convert it to 6d representation that we need for the model 
            pose_6d = rotation_conversions.matrix_to_rotation_6d(
                rotation_conversions.axis_angle_to_matrix(
                    pose
            )) # (T,24,6)

            # get translation and add dummy values to match the 6d rep
            trans[...,[0,2]] -=  trans[0,[0,2]].unsqueeze(0)  # center the x and z coords.
            trans = torch.cat((trans, torch.zeros((trans.shape[0], 3), device=trans.device)), -1) # (N,T,6)
            trans = trans.unsqueeze(1)  # (T,1,6)
            
            # append the translation to the joint angle
            data = torch.cat((pose_6d, trans), 1) # (T,25,6)
            
            all_data.append(data.cpu().float())
        
        all_data = torch.cat(all_data)
        self.db['pose_6d'] = all_data
        return

    def create_subset(self, restrict_subsets):
        """  """
        # get the subdataset name for each video in the dataset
        start_idxs = np.array([d[0] for d in self.vid_indices])
        vid_subdataset = [s.split("_")[0] for s in self.db['vid_name'][start_idxs] ]
        valid_subsets = ['ACCAD', 'BioMotionLab', 'CMU', 'EKUT', 'Eyes', 'HumanEva', 'KIT',
                            'MPI', 'SFU', 'SSM', 'TCD', 'TotalCapture', 'Transitions']
        assert np.all(np.isin(restrict_subsets, valid_subsets)), f"invalid subsets list {restrict_subsets}"

        idxs_keep = np.where(np.isin(vid_subdataset, restrict_subsets))[0]
        self.vid_indices = [self.vid_indices[i] for i in idxs_keep]

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_file = osp.join(VIBE_DB_DIR, 'amass_db.pt')
        db = joblib.load(db_file)
        return db

    def get_single_item(self, index):
        # import ipdb; ipdb.set_trace()
        start_index, end_index = self.vid_indices[index]

        data = self.db['pose_6d'][start_index:end_index+1]
        data = data.permute(1,2,0)              # (25,6,T)
        # thetas = torch.tensor(self.db['theta'][start_index:end_index+1])
        # trans = torch.tensor(self.db['trans'][start_index:end_index+1])

        # # like in amass dataset, concat a [1,0,0]: camera orientation (it will be)
        # # removed, this is just for consistency
        # cam = np.array([1., 0., 0.])[None, ...]
        # cam = np.repeat(cam, thetas.shape[0], axis=0)
        # thetas = np.concatenate([cam, thetas], axis=-1)
        # thetas = torch.tensor(thetas)

        # ### now get the required pose vector
        # pose = thetas[...,3:75]     # (T,72)
        # pose = pose.view(pose.shape[0], 24, 3) # (T,24,3)
        # ## convert it to 6d representation that we need for the model 
        # pose_6d = rotation_conversions.matrix_to_rotation_6d(
        #     rotation_conversions.axis_angle_to_matrix(
        #         pose
        # )) # (T,24,6)

        # # get translation and add dummy values to match the 6d rep
        # trans[...,[0,2]] -=  trans[0,[0,2]].unsqueeze(0)  # center the x and z coords.
        # trans = torch.cat((trans, torch.zeros((trans.shape[0], 3))), -1) # (N,T,6)
        # trans = trans.unsqueeze(1)  # (T,1,6)
        
        # # append the translation to the joint angle
        # data = torch.cat((pose_6d, trans), 1) # (T,25,6)
        # data = data.permute(1,2,0)        # (25,6,T)
        
        # ## return value expected by mdm puts the motion data in `inp` key 
        # # and then for a2m task we also need `action` and `action_text` which
        # # we'll make as dummy vars. Then `collate_fn` in dataloader will turn 
        # # that into the final outputs. 
        ret = dict(
                inp=data.float(),
                action=0,
                action_text='',
                )

        return ret
