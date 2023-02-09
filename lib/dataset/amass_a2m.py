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
import os.path as osp
from torch.utils.data import Dataset

from mdm.utils import rotation_conversions
from lib.core.config import VIBE_DB_DIR
from lib.data_utils.img_utils import split_into_chunks

class AMASS(Dataset):
    def __init__(self, seqlen):
        self.seqlen = seqlen

        self.stride = seqlen

        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
        # del self.db['vid_name']
        print(f'AMASS dataset number of videos: {len(self.vid_indices)}')

    def create_db_6d_upfront():
        raise NotImplementedError()
        if hasattr(self, "db_6d"):
            return

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
        thetas = torch.tensor(self.db['theta'][start_index:end_index+1])
        trans = torch.tensor(self.db['trans'][start_index:end_index+1])

        # like in amass dataset, concat a [1,0,0]: camera orientation (it will be)
        # removed, this is just for consistency
        cam = np.array([1., 0., 0.])[None, ...]
        cam = np.repeat(cam, thetas.shape[0], axis=0)
        thetas = np.concatenate([cam, thetas], axis=-1)
        thetas = torch.tensor(thetas)

        ### now get the required pose vector
        pose = thetas[...,3:75]     # (T,72)
        pose = pose.view(pose.shape[0], 24, 3) # (T,24,3)
        ## convert it to 6d representation that we need for the model 
        pose_6d = rotation_conversions.matrix_to_rotation_6d(
            rotation_conversions.axis_angle_to_matrix(
                pose
        )) # (T,24,6)

        # get translation and add dummy values to match the 6d rep
        trans[...,[0,2]] -=  trans[0,[0,2]].unsqueeze(0)  # center the x and z coords.
        trans = torch.cat((trans, torch.zeros((trans.shape[0], 3))), -1) # (N,T,6)
        trans = trans.unsqueeze(1)  # (T,1,6)
        
        # append the translation to the joint angle
        data = torch.cat((pose_6d, trans), 1) # (T,25,6)
        data = data.permute(1,2,0)        # (25,6,T)
        
        ## now dummy values for the action category
        y = dict(
            y=dict(
                mask=torch.ones((1,1,data.shape[-1]), dtype=bool),
                lengths=[data.shape[-1]]*len(data),
                action= torch.zeros((data.shape[0], 1), dtype=torch.long),
                action_text=['']*len(data)
                )
            )

        return data.float(), y
