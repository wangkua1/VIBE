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
import roma
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

import os.path as osp
from mdm.utils import rotation_conversions
from lib.core.config import VIBE_DB_DIR
from lib.data_utils.img_utils import split_into_chunks


class VibeDataset(Dataset):
<<<<<<< HEAD

    def __init__(self,
                 num_frames,
                 split='train',
                 restrict_subsets=None,
                 dataset='amass',
                 normalize_translation=True):
=======
    def __init__(self, num_frames, split='train', restrict_subsets=None, 
        dataset='amass', normalize_translation=True, correct_frame_of_reference=False):
>>>>>>> bfbd3261148a3ad7a9b98137f73c6a25c0de0ad8
        """
        Args:
            dataset (str): one of ('amass','h36m')
            restrict_subsets: a list of strings of subsets to include in the final dataset
                If None then include all subdatasets. Valid subdataset names are:
                ['ACCAD', 'BioMotionLab', 'CMU', 'EKUT', 'Eyes', 'HumanEva', 'KIT',
                 'MPI', 'SFU', 'SSM', 'TCD', 'TotalCapture', 'Transitions'],
            correct_frame_of_reference: whether to switch x- and z- axis, which is needed sometimes
                If None, then it 
        """
        self.SUBSAMPLE = {
<<<<<<< HEAD
            'amass': 1,
            'h36m': 2,
=======
            'amass' : 1,
            'h36m' : 2,
            '3dpw' : 1,
>>>>>>> bfbd3261148a3ad7a9b98137f73c6a25c0de0ad8
        }
        self.dataset = dataset
        self.dataname = dataset  # mdm training code uses this
        self.split = split
        self.restrict_subsets = restrict_subsets
        self.seqlen = num_frames
        self.normalize_translation = normalize_translation

        self.stride = self.seqlen

        self.db = self.load_db(split=split,
                               subsample=self.SUBSAMPLE[self.dataset])

        self.vid_indices = split_into_chunks(np.array(self.db['vid_name']),
                                             self.seqlen, self.stride)
        # del self.db['vid_name']

        if not restrict_subsets is None:
            # if True, this overwrites the `vid_indices` object with a restricted version
            self.create_subset(restrict_subsets)

        # transform the data to 6d
        self.correct_frame_of_reference = True if self.dataset in ('amass','h36m','3dpw') else False
        self.correct_frame_of_reference = correct_frame_of_reference
        self.create_db_6d_upfront(self.correct_frame_of_reference)

        # filter some video types - e.g. treadmill
        if self.dataset == 'amass':
            self.filter_videos()

        print(f'  number of videos: {len(self.vid_indices)}')

    def create_db_6d_upfront(self, correct_frame_of_reference=False):
        """
        Convert the SMPL representation to the `pose_6d` representation. 
        Joint idx 0 is root orientation in 6d, joint idx 1-24 are the relative joint 
        orientations in 6d. Joint idx 25 has the root translation in its first 3 
        idxs and 

        `correct_frame_of_reference`=True should be used for Amass. It is equivalent 
        to switching the y- and z- positions and negating the z values (required for 
        getting amass in same frame as humanact12, 3dpw,) HumanML does this to amass data 
        by doing the switch in xyz coords. Here we do it by rotating -90deg
        about x-axis, switching the  rotation of the root joint (joint idx 0),
         and then flipping the.
        """
<<<<<<< HEAD
        print(
            "   Dataloader: doing 6d rotations and shifting the reference frame"
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        pose_key = 'theta' if self.dataset == 'amass' else 'pose'
        thetas = torch.tensor(self.db[pose_key]).to(device).float()
=======
        print("   Dataloader: doing 6d rotations")
        device='cuda'if torch.cuda.is_available() else 'cpu'
        
        pose_key = 'theta' if self.dataset=='amass' else 'pose'
        thetas = torch.tensor(self.db[pose_key]).to(device).float() # pose and beta
>>>>>>> bfbd3261148a3ad7a9b98137f73c6a25c0de0ad8
        transes = torch.tensor(self.db['trans']).to(device).float()

        dset = TensorDataset(thetas, transes)
        loader = DataLoader(dset,
                            batch_size=2048 * 2,
                            shuffle=False,
                            drop_last=False)
        all_data = []
<<<<<<< HEAD

        for theta, trans in tqdm.tqdm(loader):
            # like in amass dataset, concat a [1,0,0]: camera orientation (it will be)
            # removed, this is just for consistency
            cam = np.array([1., 0., 0.])[None, ...]
            cam = torch.Tensor(np.repeat(cam, theta.shape[0],
                                         axis=0)).to(device)
=======
        
        import ipdb; ipdb.set_trace
        for theta, trans in tqdm.tqdm(loader):
            # like in amass dataset, concat a [1,0,0]: camera orientation (it will be)
            # removed, this is just for consistency
            cam = np.array([1., 0., 0.], dtype=np.float32)[None, ...]
            cam = torch.Tensor(np.repeat(cam, theta.shape[0], axis=0)).to(device)
>>>>>>> bfbd3261148a3ad7a9b98137f73c6a25c0de0ad8
            theta = torch.cat([cam, theta], -1).to(device)
            # theta = torch.tensor(theta).to(device)

            ### now get the required pose vector
            pose = theta[..., 3:75]  # (T,72)
            pose = pose.view(pose.shape[0], 24, 3)  # (T,24,3)

            ## if flagged, rotate the root orientation -90deg about x
            if correct_frame_of_reference:
                root = pose.clone()[:, [0], :]
                root_rotated = apply_rotvec_to_aa2(
                    torch.Tensor(np.pi / 2 * np.array([-1, 0, 0])[None]).to(
                        pose.device).float(),
                    root.view(-1, 3),
                ).view(root.shape)
                pose[..., [0], :] = root_rotated

            ## convert it to 6d representation that we need for the model
            pose_6d = rotation_conversions.matrix_to_rotation_6d(
                rotation_conversions.axis_angle_to_matrix(pose))  # (T,24,6)

            # get translation and add dummy values to match the 6d rep
            trans[..., [0, 2]] -= trans[0, [0, 2]].unsqueeze(
                0)  # center the x and z coords.
            trans = torch.cat(
                (trans, torch.zeros(
                    (trans.shape[0], 3), device=trans.device)), -1)  # (N,T,6)
            trans = trans.unsqueeze(1)  # (T,1,6)

            ## if flagged, the translation also needs to change axes
            if correct_frame_of_reference:
                trans_copy = trans.clone()
                trans[..., 1] = trans_copy[..., 2].clone()
                trans[..., 2] = -trans_copy[..., 1].clone()

            # append the translation to the joint angle
            data = torch.cat((pose_6d, trans), 1)  # (T,25,6)
            all_data.append(data.cpu().float())

        all_data = torch.cat(all_data)

        self.db['pose_6d'] = all_data
        return

    def filter_videos(self):
        """
        Filter videos based where vidnames have the strings 
        from the list `FILTER_NAMES`
         """
        FILTER_NAMES = ['treadmill', 'normal_walk', 'TCD_handMocap']
        start_idxs = np.array([s[0] for s in self.vid_indices])
        vid_names = np.array([self.db['vid_name'][i] for i in start_idxs])

        mask_remove_vid = torch.zeros(len(vid_names))
        for filter_name in FILTER_NAMES:
            mask_remove_vid = mask_remove_vid + np.array([(filter_name in v)
                                                          for v in vid_names])
        idxs_keep_vid = np.where(~mask_remove_vid.bool())[0]
        self.vid_indices = [self.vid_indices[i] for i in idxs_keep_vid]

        return

    def correct_amass_frame_of_reference(self, data):
        """
        Expects data in shape ()
        """

    def create_subset(self, restrict_subsets):
        """  """
        # get the subdataset name for each video in the dataset
        start_idxs = np.array([d[0] for d in self.vid_indices])
        vid_subdataset = [
            s.split("_")[0] for s in self.db['vid_name'][start_idxs]
        ]
        valid_subsets = [
            'ACCAD', 'BioMotionLab', 'CMU', 'EKUT', 'Eyes', 'HumanEva', 'KIT',
            'MPI', 'SFU', 'SSM', 'TCD', 'TotalCapture', 'Transitions'
        ]
        assert np.all(
            np.isin(restrict_subsets,
                    valid_subsets)), f"invalid subsets list {restrict_subsets}"

        idxs_keep = np.where(np.isin(vid_subdataset, restrict_subsets))[0]
        self.vid_indices = [self.vid_indices[i] for i in idxs_keep]

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self, split='train', subsample=1):
        """ 
        Note that subsampling is implemented in each `load_db_{dataset}` call
        because the 3dpw uses it at a specific point to prevent RAM issues.
        """
        if self.dataset=='amass':
            db = self.load_db_amass(split)
            db = self.subsample(db)
        elif self.dataset=='h36m':
            db = self.load_db_h36m(split, subsample=subsample)
        elif self.dataset=='3dpw':
            db = self.load_db_3dpw(split)
        else:
            valid_datasets = ['amass','h36m','3dpw']
            raise ValueEror(f"Invalid dataset [{self.dataset}]. Must be one of {valid_datasets}")
        
        return db 

    def load_db_amass(self, split):
        db_file = osp.join(VIBE_DB_DIR, f'amass_db.pt')
        db = joblib.load(db_file)
        return db

    def load_db_h36m(self, split, subsample=2):
        if split == 'train':
            user_list = [1, 5, 6, 7, 8]
        elif split in ['val','test']: # JB added test for compatibility with mdm.sample.generate
            user_list = [9, 11]
        else:  
            user_list = [1]

        seq_db_list = []
        for user_i in user_list:
            print(f"  Loading Subject S{user_i}")
            db_subset = joblib.load(
                osp.join(VIBE_DB_DIR, f'h36m_{user_i}_db.pt'))
            seq_db_list.append(db_subset)

        dataset = defaultdict(list)

        for seq_db in seq_db_list:
            for k, v in seq_db.items():
                dataset[k] += list(v)

        # JW: temporary hack -- use slv data here
        dataset['pose'] = dataset['slv_mosh_theta']
        dataset['trans'] = dataset['slv_trans']

        dataset = self.subsample(dataset, subsample)

        # convert to array
        # for k in dataset.keys():
        #     dataset[k] = np.array(dataset[k])

        print(f'Loaded h36m split [{split}]')
        return dataset

    def load_db_3dpw(self, split):
        db_file = osp.join(VIBE_DB_DIR, f'3dpw_{split}_db.pt')
        db = joblib.load(db_file)
        return db

    def subsample(self, db, subsample=1):
        for k in db.keys():
            db[k] = db[k][::subsample]
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        data = self.db['pose_6d'][start_index:end_index + 1]
        data = data.permute(1, 2, 0)  # (25,6,T)
        vid_name = self.db['vid_name'][start_index]

        if self.normalize_translation:
            # has format (J,6,T). translation is the last joint (dim 0)
            data[-1, :, :] = data[-1, :, :] - data[-1, :, [0]]

        ret = dict(
            inp=data.float(),
            action_text='',
            vid_name=vid_name,
        )
        if 'features' in self.db.keys():
            ret['features'] = self.db['features'][start_index:end_index + 1]

        return ret


def apply_rotvec_to_aa2(rotvec, aa):
    N = aa.shape[0]
    rotvec = rotvec.repeat(N, 1)
    return roma.rotvec_composition([rotvec, aa])
