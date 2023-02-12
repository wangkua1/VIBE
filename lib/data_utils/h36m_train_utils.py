import os
import os.path as osp
import sys
import cv2
import glob
import h5py
import pickle as pkl
import ipdb
import numpy as np
import roma
import argparse
from tqdm import tqdm
#from spacepy import pycdf
import cdflib
import joblib

# VIBE related
from VIBE.lib.core.config import VIBE_DB_DIR, VIBE_DATA_DIR, H36M_DIR
from VIBE.lib.models import spin
from VIBE.lib.data_utils.kp_utils import *
from lib.data_utils.feature_extractor import extract_features


# Viz
import torch
from utils.geometry import perspective_projection, perspective_projection_with_K
from nemo.utils.misc_utils import to_tensor, to_np

from nemo.utils.render_utils import add_keypoints_to_image, run_smpl_to_j3d
from hmr.renderer import Renderer
from hmr.smpl import SMPL
from hmr import hmr_config
from VIBE.lib.utils.renderer import Renderer as VIBERenderer
import roma 
from scipy.spatial.transform import Rotation as sR

ACTIONS = [
    "Directions 1", "Directions", "Discussion 1", "Discussion", "Eating 2",
    "Eating", "Greeting 1", "Greeting", "Phoning 1", "Phoning", "Posing 1",
    "Posing", "Purchases 1", "Purchases", "Sitting 1", "Sitting 2",
    "SittingDown 2", "SittingDown", "Smoking 1", "Smoking", "TakingPhoto 1",
    "TakingPhoto", "Waiting 1", "Waiting", "Walking 1", "Walking",
    "WalkingDog 1", "WalkingDog", "WalkTogether 1", "WalkTogether"
]

ACTIONS_WITHOUT_CHAIR = [
    "Directions 1", "Directions", "Discussion 1", "Discussion", "Greeting 1",
    "Greeting", "Posing 1", "Posing", "Purchases 1", "Purchases",
    "SittingDown 2", "SittingDown", "TakingPhoto 1", "TakingPhoto",
    "Waiting 1", "Waiting", "Walking 1", "Walking", "WalkingDog 1",
    "WalkingDog", "WalkTogether 1", "WalkTogether"
]

CAMERAS = ["54138969", "55011271", "58860488", "60457274"]
MOSH_CAMERAS = [ "58860488", "60457274","54138969", "55011271"]


def get_action_name_from_action_id(s):
    return s.split(' ')[0]


def action_id_without_chair(s):
    actions_with_chair = set([
        get_action_name_from_action_id(s) for s in ACTIONS
    ]).difference(
        set([get_action_name_from_action_id(s)
             for s in ACTIONS_WITHOUT_CHAIR]))

    return get_action_name_from_action_id(s) not in actions_with_chair


def action_id_without_chair0(s):
    raise
    # this is not good... use the one above...
    return get_action_name_from_action_id(s) in set(
        [get_action_name_from_action_id(s) for s in ACTIONS_WITHOUT_CHAIR])

def apply_rotvec_to_aa(rotvec, aa):
    N = aa.shape[0]
    rotvec = rotvec.repeat(N, 1)
    return roma.rotvec_composition([rotvec, aa])


def apply_rotvec_to_aa2(rotvec, aa):
    N = aa.shape[0]
    rotvec = rotvec.repeat(N, 1)
    return roma.rotvec_composition([aa, rotvec])


def read_db(split):
    debug=False
    extract_img=True
    IMG_D0=1002
    IMG_D1=1000
    FOCAL_LENGTH=1000
    smpl = SMPL(hmr_config.SMPL_MODEL_DIR, batch_size=1,
            create_transl=False).cuda()

    renderer = Renderer(focal_length=FOCAL_LENGTH,
                                 img_width=IMG_D1,
                                 img_height=IMG_D0,
                                 faces=smpl.faces)

    out_dir = '_h36m'
    os.makedirs(out_dir, exist_ok=True)

    dataset_path = H36M_DIR

    mosh_dir = osp.join(dataset_path, 'mosh/neutrMosh/neutrSMPL_H3.6/')
    # e.g. osp.join(mosh_dir, "S1", "SittingDown 2_cam0_aligned.pkl")

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # Output
    dataset = {
        'vid_name': [],
        'img_name': [],
        'joints3D': [],
        'joints2D': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'features': [],
    }

    if split == 'train':
        user_list = [1, 5, 6, 7, 8]
    else:
        user_list = [9, 11]

    model = spin.get_pretrained_hmr()

    # go over each user
    for user_i in user_list:
        print('User:', user_i)
        user_name = 'S%d' % user_i
        # path with GT bounding boxes
        bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat',
                                 'ground_truth_bb')
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                 'D3_Positions_mono')
        # path with GT 3D pose2
        pose2_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                  'D3_Positions')
        # path with GT 2D pose
        pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                   'D2_Positions')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()

        # # for debugging
        # seq_list_root = '/home/users/wangkua1/projects/bio-pose/VIBE/data/h36m/S1/MyPoseFeatures/D3_Positions_mono/'
        # seq_list = [osp.join(seq_list_root, f'WalkingDog.{CAMERAS[cam_id]}.cdf') for cam_id in range(4)]

        for cam_id, seq_i in enumerate(seq_list):
            print('\tSeq:', seq_i)
            sys.stdout.flush()
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action_w_space, camera, _ = seq_name.split('.')
            action = action_w_space.replace(' ', '_')

            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = cdflib.CDF(seq_i)['Pose'][0]

            # 2D pose file
            pose2d_file = os.path.join(pose2d_path, seq_name)
            poses_2d = cdflib.CDF(pose2d_file)['Pose'][0]

            # bbox file
            bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
            bbox_h5py = h5py.File(bbox_file)

            # Mosh
            mosh_cam_id = CAMERAS.index(camera)
            # mosh_cam_id = MOSH_CAMERAS.index(camera)
            mosh_path = osp.join(mosh_dir, user_name,
                                 f"{action_w_space}_cam{mosh_cam_id}_aligned.pkl")
            mosh = pkl.load(open(mosh_path, 'rb'), encoding="latin1")

            # Upsample frames by linear interpolation
            def interp(ar1, ar2, w):
                diff = ar2 - ar1
                return ar1 + w * diff

            N = mosh['new_poses'].shape[0]
            upsampled = np.zeros((N * 5, 72))
            for i in range(5):
                upsampled[np.arange(0, N * 5, 5)[:-1] + i] = interp(
                    mosh['new_poses'][:-1], mosh['new_poses'][1:], i / 5)

            # Upsample Mosh
            mosh_theta = to_tensor(upsampled)

            # Re-orient RF
            mosh_root_orient = mosh_theta[:, :3]
            # mosh_root_orient0 = apply_rotvec_to_aa(to_tensor(np.pi  * np.array([1, 0, 0]))[None], mosh_root_orient)
            mosh_root_orient = apply_rotvec_to_aa2(to_tensor(np.pi  * np.array([1, 0, 0]))[None], mosh_root_orient)
            mosh_theta[:, : 3] = mosh_root_orient


            # video file
            if extract_img:
                vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
                imgs_path = os.path.join(dataset_path, 'images')
                vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            # for frame_i in tqdm(range(200)):
            N = poses_3d.shape[0]
            assert N == poses_2d.shape[0]

            img_paths_array = []
            vid_name = []
            joints3D = []
            joints2D = []
            shape = []
            pose = []

            # for frame_i in tqdm(range(600)):
            for frame_i in tqdm(range(N - 10)):  # drop last few because of mosh interpolation
                # read video frame
                if extract_img:
                    success, image = vidcap.read()
                    if not success:
                        raise # can't read frame.

                protocol = 1
                if frame_i % 1 == 0 and (protocol == 1
                                         or camera == '60457274'):

                    vid_name_ = '%s_%s.%s' % (user_name, action, camera)
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera,
                                                     frame_i + 1)
                    img_path = osp.join(dataset_path, 'images', imgname)

                    # save image
                    if extract_img and not osp.exists(img_path):
                        cv2.imwrite(img_path, image)

                    
                    # read GT 2D pose
                    partall = np.reshape(poses_2d[frame_i, :], [-1, 2])
                    part17 = partall[h36m_idx]
                    part = np.zeros([24, 3])
                    part[global_idx, :2] = part17
                    part[global_idx, 2] = 1

                    # Below is almost the same, except it has 'Jaw (H36M)' instead of 'headtop'
                    # part2 = convert_kps(part17[None], src='h36m', dst='spin')[0, 25:]
                    part = np.vstack([np.zeros((25, 3)), part]) # SPIN format


                    # # read GT 3D pose
                    Sall = np.reshape(poses_3d[frame_i, :], [-1, 3]) / 1000.
                    S17 = Sall[h36m_idx]
                    # root_trans = S17[0]
                    # # S17 -= S17[0] # root-centered
                    # S24 = np.zeros([24, 4])
                    # S24[global_idx, :3] = S17
                    # S24[global_idx, 3] = 1
                    # S24 = np.vstack([np.zeros((25, 4)), S24]) # SPIN format

                    # Use Moshed 3D Joint XYZ instead
                    mosh_j3d, mosh_v3d = to_np(run_smpl_to_j3d(mosh_theta[frame_i], betas=to_tensor(mosh['betas'])))
                    mosh_j3d, mosh_v3d = to_np(mosh_j3d), to_np(mosh_v3d)
                    mosh_j3d = mosh_j3d + S17[7] # Mosh is relative to the "Spine" joint
                    S24 = mosh_j3d


                    vid_name.append(vid_name_)
                    img_paths_array.append(img_path)
                    joints3D.append(S24)
                    joints2D.append(part)
                    shape.append(mosh['betas'])
                    pose.append(mosh_theta[frame_i])

                    # # Viz
                    # im = cv2.imread(img_path)
                    # camera_rotation = torch.eye(3).unsqueeze(0).expand(
                    #     1, -1, -1)
                    # camera_translation = torch.zeros(1, 3)
                    # K = torch.load(
                    #     '/home/users/wangkua1/projects/bio-pose/camera_intrinsics.pt'
                    # )


                    # # projected_keypoints_2d = perspective_projection_with_K(
                    # #     torch.tensor(S24[:,:3])[None].float(),
                    # #     rotation=camera_rotation,
                    # #     translation=camera_translation,
                    # #     K=K).detach().numpy()[0]
                    # # projected_keypoints_2d = np.hstack([projected_keypoints_2d, S24[:, 3:]])
                    # # im1 = add_keypoints_to_image(np.copy(im),
                    # #                              projected_keypoints_2d)
                    # # cv2.imwrite(
                    # #     osp.join(out_dir, f'test_{cam_id}_{frame_i}.png'), im1)

                    # Viz SMPL
                    # mosh_j3d, mosh_v3d = to_np(run_smpl_to_j3d(mosh_theta[frame_i], betas=to_tensor(mosh['betas'])))
                    # mosh_j3d, mosh_v3d = to_np(mosh_j3d), to_np(mosh_v3d)
                    # mosh_j3d = mosh_j3d + S17[7]

                    # # Apply rotation
                    # # rot = sR.from_rotvec(np.pi * np.array([1, 0, 0]))
                    # # mosh_j3d = rot.apply(mosh_j3d)
                    # # mosh_v3d = rot.apply(mosh_v3d)
                    
                    # # mosh_j3d[:, 1] *= -1
                    # # mosh_v3d[:, 1] *= -1
                    # # mosh_j3d[:, 0] *= -1
                    # # mosh_v3d[:, 0] *= -1
                    # # mosh_j3d = to_np(run_smpl_to_j3d(mosh_theta[frame_i], betas=to_tensor(mosh['betas']))) * -1
                    # mosh_j3d = mosh_j3d + root_trans
                    # mosh_v3d = mosh_v3d + root_trans
                    
                    # nim = np.zeros((IMG_D0, IMG_D1, 3))
                    # nim[:im.shape[0], :im.shape[1]] = im
                    # im1 = renderer(
                    #     mosh_v3d,
                    #     camera_translation,
                    #     np.copy(nim),
                    #     return_camera=False)
                    

                    # projected_keypoints_2d_mosh = perspective_projection_with_K(
                    #     torch.tensor(mosh_j3d)[None].float(),
                    #     rotation=camera_rotation,
                    #     translation=camera_translation,
                    #     K=K).detach().numpy()[0]
                    # projected_keypoints_2d_mosh = projected_keypoints_2d_mosh[-24:, :]
                    # projected_keypoints_2d_mosh = np.hstack([projected_keypoints_2d_mosh, S24[:, 3:]])
                    # im2 = add_keypoints_to_image(np.copy(im1),
                    #                              projected_keypoints_2d_mosh)
                    # cv2.imwrite(
                    #     osp.join(out_dir, f'test_mosh_{cam_id}_{frame_i}.png'),
                    #     im2)

            vid_name = np.array(vid_name)
            img_paths_array = np.array(img_paths_array)
            joints3D = np.array(joints3D)
            joints2D = np.array(joints2D)
            shape = np.array(shape)
            pose = to_np(pose)
            N = joints2D.shape[0]

            # Compute BBOX based on J2D
            j2d = np.reshape(joints2D, [N, -1, 3])
            j2d = np.concatenate([j2d, np.ones((N, j2d.shape[1], 1))], -1)
            bbox, _, _ = generate_bbox_from_j2d(j2d)

            # Extract image (cropped) features from SPIN
            features = extract_features(model, img_paths_array, bbox, scale=1.2)

            
            # store data
            dataset['vid_name'].append(vid_name)
            dataset['img_name'].append(img_paths_array)
            dataset['joints3D'].append(joints3D)
            dataset['joints2D'].append(joints2D)
            dataset['shape'].append(shape)
            dataset['pose'].append(pose)
            dataset['bbox'].append(bbox)
            dataset['features'].append(features)

    final_dataset = {}
    for k, v in dataset.items():
        print(k)
        if len(v[0].shape) == 1:
            v = [vi[:,None] for vi in v]
        final_dataset[k] = np.vstack(v)

    return final_dataset

if __name__ == '__main__':
    """
    python -m VIBE.lib.data_utils.h36m_train_utils
    """
    final_dataset = read_db('train')
    joblib.dump(final_dataset, osp.join(VIBE_DB_DIR, 'h36m_train_db.pt'))
    final_dataset = read_db('val')
    joblib.dump(final_dataset, osp.join(VIBE_DB_DIR, 'h36m_val_db.pt'))

   