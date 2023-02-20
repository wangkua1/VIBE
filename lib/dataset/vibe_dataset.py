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

import ipdb
import os
import torch
import joblib
import roma
import random
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
# from gthmr.lib.utils.geometry import apply_rotvec_to_aa2


class VibeDataset(Dataset):
    def __init__(self,
                 num_frames,
                 dataset='amass',
                 split='train',
                 restrict_subsets=None,
                 data_rep='rot6d',
                 foot_vel_threshold=0.01,
                 normalize_translation=True,
                 rotation_augmentation=False,
                 no_motion=False):
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.SUBSAMPLE = {
            'amass': 1,
            'amass_hml': 1,
            'h36m': 2,
            '3dpw': 1,
        }
        self.dataset = dataset
        self.dataname = dataset  # mdm training code uses this
        self.num_frames = num_frames
        self.split = split
        self.data_rep = data_rep
        self.restrict_subsets = restrict_subsets
        self.seqlen = num_frames
        self.normalize_translation = normalize_translation
        self.rotation_augmentation = rotation_augmentation
        self.foot_vel_threshold = foot_vel_threshold  # for foot contact mask if it's used

        self.stride = self.seqlen

        self.db = self.load_db(split=split,
                               subsample=self.SUBSAMPLE[self.dataset])

        if self.dataset != 'amass_hml':
            self.vid_indices = split_into_chunks(np.array(self.db['vid_name']),
                                                 self.seqlen, self.stride)
        else:
            self.vid_indices = self.get_amass_hml_vid_indices()

        self.no_motion = no_motion
        if self.no_motion:
            # Preporcessing funcs take a few mins for the full dataset, reduce the dataset size.
            # Should be as bigger than a typical batch size, otherwise calls to next() will
            N = 100
            print(f"   Dataset: only loading {N} samples")
            self.vid_indices = self.vid_indices[:N]

        if not restrict_subsets is None and dataset == 'amass':
            # if True, this overwrites the `vid_indices` object with a restricted version
            self.create_subset(restrict_subsets)

        # Precompute the 6d articulation.
        self.create_db_6d_upfront()
        
        # for data_rep with foot contact, prepare it 
        if self.data_rep=='rot6d_fc':
            self.do_fc_mask = True 
            # self.compute_fc_mask(self.FOOT_VEL_THRESHOLD[self.dataset])
            self.compute_fc_mask(foot_vel_threshold)
        else:
            self.do_fc_mask = False

        # filter some video types - e.g. treadmill, handmocap
        if self.dataset == 'amass':
            self.filter_videos()

        # dataset feature parameters
        self.njoints, self.nfeats = self.__getitem__(0)['inp'].shape[:2]

        print(f'  number of videos: {len(self.vid_indices)}')

    def _create_6d_from_theta(self, thetas, transes):

        """
        Convert the SMPL axis-angle representation to the `pose_6d` representation. 
        Joint idx 0 is root orientation in 6d, joint idx 1-24 are the relative joint 
        orientations in 6d. Joint idx 25 has the root translation in its first 3 
        idxs and zeros in the last 3.

        `do_rotate_about_x=True` should be used for Amass. It is equivalent 
        to switching the y- and z- positions and negating the z values (required for 
        getting amass in same frame as humanact12, 3dpw,) HumanML does this to amass data 
        by doing the switch in xyz coords. Here we do it by rotating the root orientation
        (axis 0) by -90deg about x-axis and then switching the y&z translation terms 
        (index -1) and multiplying z-translation by -1.

        Args: 
            thetas (np.ndarray): theta axis-angles from the smpl model
            transes (np.ndarray): translation vector. Same first dim size as `thetas`.
        """
        print("   Dataloader: doing 6d rotations")
        dset = TensorDataset(thetas, transes)
        loader = DataLoader(dset,
                            batch_size=2048 * 2,
                            shuffle=False,
                            drop_last=False)
        all_data = []
        all_foot_vels = []
        for i, (theta, trans) in enumerate(tqdm.tqdm(loader)):
            # first, break after 1 iteration if we have `no_motion` flag sets
            if self.no_motion and i > 1:
                break
            # like in amass dataset, concat a [1,0,0]: camera orientation (it will be)
            # removed, this is just for consistency
            cam = np.array([1., 0., 0.], dtype=np.float32)[None, ...]
            cam = torch.Tensor(np.repeat(cam, theta.shape[0],
                                         axis=0)).to(self.device)
            theta = torch.cat([cam, theta], -1).to(self.device)

            ### now get the required pose vector
            pose = theta[..., 3:75]  # (T,72)
            pose = pose.view(pose.shape[0], 24, 3)  # (T,24,3)
            
            ## convert axis-angle rep to 6d rep
            pose_6d = rotation_conversions.matrix_to_rotation_6d(
                rotation_conversions.axis_angle_to_matrix(pose))  # (T,24,6)

            # get translation and add dummy values to idexes 3-5 to match the 6d dimension
            trans = torch.cat(
                (trans, torch.zeros(
                    (trans.shape[0], 3), device=trans.device)), -1)  # (N,T,6)
            trans = trans.unsqueeze(1)  # (T,1,6)

            # append the translation to the joint angle
            data = torch.cat((pose_6d, trans), 1)  # (T,25,6)

            # rotate the motion depending on the dataset so that all datasets are in the same orientation
            data = self.correct_orientation(data, self.dataset)

            # save 
            all_data.append(data.cpu().float())

        return torch.cat(all_data)

    def correct_orientation(self, data, dataset):
        if dataset in ('amass','amass_hml'):
            return rotate_about_D(data.unsqueeze(-1), -np.pi/2, 0).squeeze(-1)
        elif dataset in('h36m','3dpw'):
            return rotate_about_D(data.unsqueeze(-1), np.pi, 0).squeeze(-1)
        else:
            raise ValueEror

    def create_db_6d_upfront(self):
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
        pose_key = 'theta' if self.dataset in ('amass',
                                               'amass_hml') else 'pose'
        thetas = torch.tensor(self.db[pose_key]).to(
            self.device).float()  # pose and beta
        transes = torch.tensor(self.db['trans']).to(self.device).float()

        self.db['pose_6d'] = self._create_6d_from_theta(
            thetas, transes)

        # For HMR datasets, do the same for camera-view data
        if self.dataset == 'amass':
            pass
        elif self.dataset == 'h36m':
            cv_thetas = torch.tensor(self.db['cam_view_pose']).to(
                self.device).float()  # pose and beta
            cv_transes = torch.tensor(self.db['cam_view_trans']).to(
                self.device).float()
            self.db['cv_pose_6d'] = self._create_6d_from_theta(
                cv_thetas, cv_transes)
        else:
            raise ValueError("Please implement this ... ")

    def compute_fc_mask(self, foot_vel_threshold=0.03):
        assert hasattr(self, 'db') and (
            'pose_6d'
            in self.db.keys()), "must run `create_db_6d_upfront` first"

        assert hasattr(self, 'db') and (
            'pose_6d'
            in self.db.keys()), "must run `create_db_6d_upfront` first"

        # this rot6d->xyz conversion code from the original
        self.rot2xyz = Rotation2xyz(device=self.device, dataset=self.dataset)
        self.get_xyz = lambda sample: self.rot2xyz(sample,
                                                   mask=None,
                                                   pose_rep="rot6d",
                                                   translation=True,
                                                   glob=True,
                                                   jointstype='smpl',
                                                   vertstrans=False)

        self.foot_vel_threshold = foot_vel_threshold
        # create an array for putting foot velocities
        T, J, D = self.db['pose_6d'].shape

        assert J == 25 and D == 6
        foot_vel = torch.zeros((T, 4), dtype=torch.float32, device='cpu')

        # iterate over batches of vid indices
        batch_size = 128 * 60 // self.num_frames  # scale wrt vid length to prevent OOM error
        print(
            f"  Dataloader: getting foot contact data with velocity threshold [{foot_vel_threshold}]"
        )
        target_xyz = torch.zeros((T, J - 1, 3),
                                 dtype=torch.float32)  # T is total frames
        batch_size = 2048 * 2
        n_iters = T // batch_size + 1
        for i in tqdm.tqdm(range(n_iters)):
            if i < n_iters - 1:
                slc = slice(i * batch_size, (i + 1) * batch_size)
            else:
                slc = slice(i * batch_size, T)
            target = self.db['pose_6d'][slc]
            with torch.no_grad():
                target_xyz[slc] = self.get_xyz(
                    target.to(self.device).unsqueeze(-1)).cpu().squeeze(-1)

        l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
        relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
        gt_joint_xyz = target_xyz.permute(
            1, 2, 0).unsqueeze(0)[:,
                                  relevant_joints, :, :]  # [N,4,3,T] set N=1
        gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] -
                                         gt_joint_xyz[:, :, :, :-1],
                                         axis=2)  # [BatchSize, 4, Frames]
        # velocity has shape (N,4,T-1) ... make it (N,4,T) by assuming the last value is the same
        gt_joint_vel = torch.cat((gt_joint_vel, gt_joint_vel[..., [-1]]),
                                 -1).permute(0, 2, 1)[0].cpu()  # (N,T,4)
        self.db['foot_vel'] = foot_vel
        self.db['fc_mask'] = (foot_vel <= foot_vel_threshold)

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
        if self.dataset in ('amass', 'amass_hml'):
            db = self.load_db_amass(split)
            db = self.subsample(db)
        elif self.dataset == 'h36m':
            db = self.load_db_h36m(split, subsample=subsample)
        elif self.dataset == '3dpw':
            db = self.load_db_3dpw(split)
        else:
            valid_datasets = ['amass', 'h36m', '3dpw']
            raise ValueEror(
                f"Invalid dataset [{self.dataset}]. Must be one of {valid_datasets}"
            )

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
        elif split in [
                'val', 'test'
        ]:  # JB added test for compatibility with mdm.sample.generate
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
                
        dataset['cam_view_pose'] = dataset['pose']
        dataset['cam_view_trans'] = dataset['trans']

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

    def get_amass_hml_vid_indices(self):
        """
        Build `self.amass_hml_vid_indices`. This is used when this dataset is `amass_humanml`.
        Read in a curated csv of amass data filenames with start and end frames from `amass_subsets_index.csv`
        which is originally from https://github.com/jmhb0/HumanML3D/blob/main/index.csv. 
        This is used in place of `self.vid_indices` in `get_single_item()`. 
        """
        print("    getting vid indicies for hml")
        path_amass_hml_vid_indices = os.path.join(os.path.dirname(__file__),
                                                  "amass_hml_indices.pt")

        if os.path.exists(path_amass_hml_vid_indices):
            return torch.load(path_amass_hml_vid_indices)

        else:
            import pandas as pd
            vid_names_uniq = np.unique(self.db['vid_name'])
            # the index of files and their ranges from humanml
            df_amass_hml = pd.read_csv(
                os.path.join(os.path.dirname(__file__),
                             "amass_subsets_index.csv"))
            cnt_missing_vids, cnt_found_videos = 0, 0
            amass_hml_vid_indices = []

            for i, row in tqdm.tqdm(df_amass_hml.iterrows(),
                                    total=len(df_amass_hml)):
                name = row['source_path'][12:].replace("/", "_")[:-4]
                if name not in vid_names_uniq:
                    cnt_missing_vids += 1
                    continue
                cnt_found_videos += 1
                idx_vid = np.where(self.db['vid_name'] == name)[0][0]
                start_index = idx_vid + row['start_frame']
                end_index = idx_vid + row['end_frame']
                assert (self.db['vid_name'][start_index]
                        == name) and (self.db['vid_name'][end_index - 1]
                                      == name)
                amass_hml_vid_indices.append([start_index, end_index])

            torch.save(amass_hml_vid_indices, path_amass_hml_vid_indices)

            return amass_hml_vid_indices

    def get_single_item(self, index):
        # get frame idxs
        ## indexing logic depends on the dataset
        # most datasets it's whatever is in self.vid_indices
        if self.dataset != 'amass_hml':
            start_index, end_index = self.vid_indices[index]
        # for amass_humanml we randomly sample within the video
        else:
            start_index, end_index_max = self.vid_indices[index]
            end_index_max = end_index_max - 1
            # handle the case that the motion is not long enough - we will extend it later
            if end_index_max - start_index < self.num_frames:
                end_index = end_index_max
            else:
                # this is the standard case where motion is long enough
                start_index=random.randint(start_index, end_index_max-self.num_frames)
                end_index=start_index+self.num_frames-1

        # get the 6d pose vector
        data = self.db['pose_6d'][start_index:end_index + 1]   # (T,25,6)
        # make a copy if we plan to do operations that change it in place 
        # (normalize_translation is fine for non amass_hml datasets since they always have the same start_index)
        if (self.normalize_translation and self.dataset=='amass_hml') or self.rotation_augmentation:
            data=torch.clone(data)

        T,J,D = data.shape
        assert J==25 and D==6

        # get vidname
        vid_name = self.db['vid_name'][start_index]

        if self.rotation_augmentation:
            rand_root_rot = random.uniform(0, 2 * np.pi) 
        
        def process_pose6d(pose6d):
            """
            This helper function was written because the same processing needs to be applied to both "pose" and "cv_pose".
            """
            pose6d = pose6d.permute(1, 2, 0)  # (25,6,T)
            if self.normalize_translation:
                # has format (J,6,T). translation is the last joint (dim 0)
                pose6d[-1, :, :] = pose6d[-1, :, :] - pose6d[-1, :, [0]]

            # rotation augmentation about y (vertical) axis
            if self.rotation_augmentation:
                pose6d = rotate_about_D(pose6d, rand_root_rot, D=1)

            if self.data_rep == "rot6d_fc":
                fc_mask = self.db['fc_mask'][start_index:end_index +
                                             1]  # (T,4)
                pose6d = pose6d.permute(2, 0, 1)  # (T,25,6)
                pose6d = torch.cat((pose6d.view(T, J * D), fc_mask),
                                   1).unsqueeze(2).permute(1, 2,
                                                           0)  # (154,1,T)
            # if the data is smaller than self.num_frames, pad it with the last frame value
            if pose6d.shape[-1] < self.num_frames:
                n_addframes = self.num_frames - pose6d.shape[-1]
                pose6d = torch.cat(
                    (pose6d, pose6d[..., [-1]].repeat(1, 1, n_addframes)), -1)
            return pose6d

        data = process_pose6d(data)

        # return dictionary
        ret = dict(
            inp=data.float(),
            action_text='',
            vid_name=vid_name,
        )

        # add features if the database has them
        if 'features' in self.db.keys():
            ret['features'] = self.db['features'][start_index:end_index + 1]

        if 'cv_pose_6d' in self.db.keys():
            cv_pose_6d = self.db['cv_pose_6d'][start_index:end_index + 1]
            ret['cv_pose_6d'] = process_pose6d(cv_pose_6d)

        if 'joints3D' in self.db.keys():
            ret['joints3D'] = self.db['joints3D'][start_index:end_index + 1]

        return ret

def apply_rotvec_to_aa1(rotvec, aa):
    """
    Version 1: rotate `aa` by `rotvec` in the *aa frame**.
    Args:
        rotvec: shape (1,3) 
        aa: shape (...,3) set of orientations in axis-angle orientation.
    """
    N = aa.shape[0]
    rotvec = rotvec.repeat(N, 1)
    return roma.rotvec_composition([rotvec, aa])

def apply_rotvec_to_aa2(rotvec, aa):
    """
    Version 2: rotate `aa` by `rotvec` in the *world frame*.

    Args:
        rotvec: shape (1,3) 
        aa: shape (...,3) set of orientations in axis-angle orientation.
    """
    N = aa.shape[0]
    rotvec = rotvec.repeat(N, 1)
    return roma.rotvec_composition([rotvec, aa])

def rotate_points(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin[0], origin[1]
    px, py = point[:, 0], point[:, 1]
    qx = np.cos(angle) * (px) - np.sin(angle) * (py)
    qy = np.sin(angle) * (px) + np.cos(angle) * (py)
    return torch.stack((qx, qy), -1)

def rotate_about_D(motions, theta, D=1):
    """ 
    For a batch of motions in rot6d format (N,25,6,T), rotate the whole
    motion about the X-axis (D=0), or Y-axis (D=1), or Z axis (D=2).
    The rotation is about the origin.

    Args:
        motions (torch.Tensor): shape (N,25,6,T). Note that each dimension in N*T
            is rotated independently, so it doesn't matter if this isn't a coherent 
            sequence. E.g. if you have a batch (N,25,6), then just unsqueeze(-1) to 
            add a dummy dim. 
        D: the dimension, XYZ, we are rotating about, e.g. D=1 is Y-axis.
    """
    assert motions.shape[1:3]==(25,6)
    motions_rotated = motions
    # motions_rotated = motions.unsqueeze(0)

    # rotate the root
    root = motions_rotated[:,[0]].permute(0,3,1,2)
    root = rotation_conversions.matrix_to_axis_angle(
                    rotation_conversions.rotation_6d_to_matrix(root))
    shape = root.shape
    # create the rotvec tensor and assign the rotation to axis D
    rotvec = torch.Tensor(np.array([0, 0, 0])[None]).to(root.device).float()
    rotvec[0,D] = theta
    root_rotated = apply_rotvec_to_aa2(rotvec, root.reshape(-1,3)).view(*shape)
    root_rotated = rotation_conversions.matrix_to_rotation_6d(
                    rotation_conversions.axis_angle_to_matrix(root_rotated))
    root_rotated = root_rotated.permute(0,2,3,1)    

    # rotate the translation points
    trans_axes = [0,1,2]
    trans_axes.remove(D)  # axis dimensinos for not-D that need to have translation rotated
    trans = motions_rotated[:,[-1],trans_axes].permute(0,2,1) # (N,T,2)
    shape = trans.shape
    # assert  trans[:,0].sum().item()==0, "rotation augmentation only allowed when also doing translation normalization"
    trans_rotated = rotate_points(np.array([0,0]), trans.reshape(-1,2), theta).reshape(shape)
    trans_rotated = trans_rotated.permute(0,2,1)

    # # assign the new values
    motions_rotated[:,[0]] = root_rotated
    motions_rotated[:,[-1],trans_axes] = trans_rotated

    return motions_rotated

def rotate_about_y(motions, theta):
    """ 
    For a motion in rot6d format (N,25,6,T), rotate the whole
    motion about the y (vertical) axes and about the origin.
    """
    assert motions.shape[:2]==(25,6)
    motions_rotated = motions.unsqueeze(0)
    # rotate the root
    root = motions_rotated[:,[0]].permute(0,3,1,2)
    root = rotation_conversions.matrix_to_axis_angle(
                    rotation_conversions.rotation_6d_to_matrix(root))
    shape = root.shape
    rotvec = torch.Tensor(theta*np.array([0, -1, 0])[None]).to(root.device).float()
    root_rotated = apply_rotvec_to_aa2(rotvec, root.reshape(-1,3)).view(*shape)
    root_rotated = rotation_conversions.matrix_to_rotation_6d(
                    rotation_conversions.axis_angle_to_matrix(root_rotated))
    root_rotated = root_rotated.permute(0,2,3,1)    

    # rotate the translation points
    trans = motions_rotated[:,[-1],[0,2]].permute(0,2,1) # (N,T,2)
    shape = trans.shape
    assert  trans[:,0].sum().item()==0, "rotation augmentation only allowed when also doing translation normalization"
    trans_rotated = rotate_points(np.array([0,0]), trans.reshape(-1,2), theta).reshape(shape)
    trans_rotated = trans_rotated.permute(0,2,1)

    # # assign the new values
    motions_rotated[:,[0]] = root_rotated
    motions_rotated[:,[-1],[0,2]] = trans_rotated

    return motions_rotated[0]
