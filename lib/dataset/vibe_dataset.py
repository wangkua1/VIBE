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
from mdm.model.rotation2xyz import Rotation2xyz

import os.path as osp
from mdm.utils import rotation_conversions
from lib.core.config import VIBE_DB_DIR
from lib.data_utils.img_utils import split_into_chunks


class VibeDataset(Dataset):
    def __init__(self, num_frames, dataset='amass', split='train', restrict_subsets=None, 
        data_rep='rot6d', foot_vel_threshold=0.01, normalize_translation=True, 
        correct_frame_of_reference=False, no_motion=False):
        """
        Args:
            dataset (str): one of ('amass','h36m')
            restrict_subsets: a list of strings of subsets to include in the final dataset
                If None then include all subdatasets. Valid subdataset names are:
                ['ACCAD', 'BioMotionLab', 'CMU', 'EKUT', 'Eyes', 'HumanEva', 'KIT',
                 'MPI', 'SFU', 'SSM', 'TCD', 'TotalCapture', 'Transitions'],
            correct_frame_of_reference: (depracated) whether to switch y- and z- axis, which is needed sometimes
                If None, then it 
            data_rep (str): one of ('rot6d', 'rot6d_p_fc')
            no_motion (bool):
        """
        self.device='cuda'if torch.cuda.is_available() else 'cpu'
        self.SUBSAMPLE = {
            'amass' : 1,
            'h36m' : 2,
            '3dpw' : 1,
        }
        self.ROTATE_ABOUT_X = {
            'amass' : True,
            'h36m' : False,
            '3dpw' : False,
        }
        # self.FOOT_VEL_THRESHOLD = {
        #     'amass' : 0.03,
        #     'h36m' : 0.03,
        #     '3dpw' : 0.03,
        # }
        self.dataset=dataset 
        self.dataname=dataset   # mdm training code uses this
        self.num_frames=num_frames
        self.split=split
        self.data_rep=data_rep
        self.restrict_subsets=restrict_subsets
        self.seqlen=num_frames
        self.normalize_translation=normalize_translation
        self.foot_vel_threshold=foot_vel_threshold # for foot contact mask if it's used

        self.stride = self.seqlen

        self.db = self.load_db(split=split,
                subsample=self.SUBSAMPLE[self.dataset])

        self.vid_indices = split_into_chunks(np.array(self.db['vid_name']),
                                             self.seqlen, self.stride)
        
        self.no_motion=no_motion
        if self.no_motion: 
            # Preporcessing funcs take a few mins for the full dataset, reduce the dataset size. 
            # Should be as bigger than a typical batch size, otherwise calls to next() will
            N=100 
            print(f"   Dataset: only loading {N} samples")
            self.vid_indices = self.vid_indices[:N]

        if not restrict_subsets is None:
            # if True, this overwrites the `vid_indices` object with a restricted version
            self.create_subset(restrict_subsets)

        # Precompute the 6d articulation. Set a flags for axes to be changed.
        if correct_frame_of_reference: 
            # self.do_rotate_about_x=correct_frame_of_reference
            raise ValueError("arg `correct_frame_of_reference` no longer does anything")
        self.do_rotate_about_x = self.ROTATE_ABOUT_X[self.dataset]
        self.create_db_6d_upfront(do_rotate_about_x=self.do_rotate_about_x)
        # for data_rep with foot contact, prepare it 
        if self.data_rep in ('rot6d_fc'):
            self.do_fc_mask = True 
            # self.compute_fc_mask(self.FOOT_VEL_THRESHOLD[self.dataset])
            self.compute_fc_mask(foot_vel_threshold)
        else:
            self.do_fc_mask = False
        
        # filter some video types - e.g. treadmill, handmocap
        if self.dataset=='amass':
            self.filter_videos()

        # dataset feature parameters
        self.njoints, self.nfeats = self.__getitem__(0)['inp'].shape[:2]

        print(f'  number of videos: {len(self.vid_indices)}')

    def create_db_6d_upfront(self, do_rotate_about_x=False, do_fc_mask=False):
        """
        Convert the SMPL representation to the `pose_6d` representation. 
        Joint idx 0 is root orientation in 6d, joint idx 1-24 are the relative joint 
        orientations in 6d. Joint idx 25 has the root translation in its first 3 
        idxs and 

        `do_rotate_about_x`=True should be used for Amass. It is equivalent 
        to switching the y- and z- positions and negating the z values (required for 
        getting amass in same frame as humanact12, 3dpw,) HumanML does this to amass data 
        by doing the switch in xyz coords. Here we do it by rotating -90deg
        about x-axis, switching the  rotation of the root joint (joint idx 0),
         and then flipping the.
        """
        print("   Dataloader: doing 6d rotations")
        pose_key = 'theta' if self.dataset=='amass' else 'pose'
        thetas = torch.tensor(self.db[pose_key]).to(self.device).float() # pose and beta
        transes = torch.tensor(self.db['trans']).to(self.device).float()

        dset = TensorDataset(thetas, transes)
        loader = DataLoader(dset,
                            batch_size=2048 * 2,
                            shuffle=False,
                            drop_last=False)
        all_data = []
        all_foot_vels = []
        
        for i, (theta, trans) in enumerate(tqdm.tqdm(loader)):
            # first, break after 1 iteration if we have `no_motion` flag set
            if self.no_motion and i>1:  
                break
            # like in amass dataset, concat a [1,0,0]: camera orientation (it will be)
            # removed, this is just for consistency
            cam = np.array([1., 0., 0.], dtype=np.float32)[None, ...]
            cam = torch.Tensor(np.repeat(cam, theta.shape[0], axis=0)).to(self.device)
            theta = torch.cat([cam, theta], -1).to(self.device)

            ### now get the required pose vector
            pose = theta[..., 3:75]  # (T,72)
            pose = pose.view(pose.shape[0], 24, 3)  # (T,24,3)

            ## if flagged, rotate the root orientation -90deg about x
            if do_rotate_about_x:
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
            if do_rotate_about_x:
                trans_copy = trans.clone()
                trans[..., 1] = trans_copy[..., 2].clone()
                trans[..., 2] = -trans_copy[..., 1].clone()

            # append the translation to the joint angle
            data = torch.cat((pose_6d, trans), 1)  # (T,25,6)
            all_data.append(data.cpu().float())

        self.db['pose_6d'] = torch.cat(all_data)   


    def compute_fc_mask(self, foot_vel_threshold=0.03):
        
        assert hasattr(self,'db') and ('pose_6d' in self.db.keys()), "must run `create_db_6d_upfront` first"

        # this rot6d->xyz conversion code from the original 
        self.rot2xyz = Rotation2xyz(device=self.device, dataset=self.dataset)
        self.get_xyz = lambda sample: self.rot2xyz(
                        sample, mask=None, pose_rep="rot6d", translation=True,
                        glob=True, jointstype='smpl', vertstrans=False)
        
        self.foot_vel_threshold=foot_vel_threshold
        # create an array for putting foot velocities 
        T, J, D = self.db['pose_6d'].shape
        assert J==25 and D==6
        foot_vel = torch.zeros((T,4), dtype=torch.float32, device='cpu')
        # iterate over batches of vid indices
        batch_size = 128*60//self.num_frames # scale wrt vid length to prevent OOM error 
        loader = DataLoader(TensorDataset(torch.tensor(self.vid_indices)),
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False)
        
        print(f"  Dataloader: getting foot contact data with velocity threshold [{foot_vel_threshold}]")
        for batch_idx, (idxs,) in enumerate(tqdm.tqdm(loader)):
            # get the frames in a continuous sequnce
            start_idx, end_idx = idxs[:,0], idxs[:,1]
            slcs = [slice(start_idx[i].item(), end_idx[i].item()+1) for i in range(len(start_idx))]
            target = torch.stack([self.db['pose_6d'][s] for s in slcs]) # (N,T,25,6)
            
            # check that each video sequence is from the same video
            vid_names = np.stack([self.db['vid_name'][s] for s in slcs])
            assert np.all(np.all(np.char.equal(vid_names[:, 1:], vid_names[:, :-1]), axis=1)),\
                "Error: indexed frames in a sequence not from the same video "

            # do forward kinematics to get positions
            with torch.no_grad():
                target_xyz = self.get_xyz(target.to(self.device).permute(0,2,3,1))
            # get the foot and ankle joint positions and velocities
            l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
            relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
            gt_joint_xyz = target_xyz[:,relevant_joints,:,:]  # [BatchSize, 4, 3, Frames]
            gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:,:,:,1:] - gt_joint_xyz[:,:,:,:-1], axis=2)  # [BatchSize, 4, Frames]
            # velocity has shape (N,4,T-1) ... make it (N,4,T) by assuming the last value is the same 
            gt_joint_vel = torch.cat((
                gt_joint_vel,
                gt_joint_vel[...,[-1]]
                ), -1).permute(0,2,1).cpu() # (N,T,4)

            # Put these results back in the frame 
            for i, slc in enumerate(slcs):
                foot_vel[slcs[i]] = gt_joint_vel[i]

        self.db['foot_vel'] = foot_vel
        self.db['fc_mask'] = (foot_vel<=foot_vel_threshold)
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
        if split == 'db':
            db_file = osp.join(VIBE_DB_DIR, f'amass_db_db.pt')
        else:
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
        T,J,D = data.shape
        assert J==25 and D==6
        data = data.permute(1, 2, 0)  # (25,6,T)
        vid_name = self.db['vid_name'][start_index]

        if self.normalize_translation:
            # has format (J,6,T). translation is the last joint (dim 0)
            data[-1, :, :] = data[-1, :, :] - data[-1, :, [0]]

        if self.data_rep=="rot6d_fc":
            fc_mask = self.db['fc_mask'][start_index:end_index + 1] # (T,4)
            data = data.permute(2,0,1) # (T,25,6)
            data = torch.cat((
                data.view(T,J*D),
                fc_mask
                ), 1).unsqueeze(2).permute(1,2,0) # (154,1,T)

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
