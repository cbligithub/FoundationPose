# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import trimesh
import numpy as np

from estimateModel import FoundationPose, ScorePredictor, PoseRefinePredictor
from utils import set_seed, depth2xyzmap, toOpen3dCloud, draw_posed_3d_box, draw_xyz_axis, dr
import argparse
import load_custom


def main():
    parser = argparse.ArgumentParser()
    tool_dir = os.path.dirname(os.path.realpath(__file__))
    code_dir = os.path.dirname(tool_dir)
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/charger/power_plug2.stl')
    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/charger')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    mesh.apply_scale(0.001)  # from mm to meter for charger data

    debug = args.debug
    debug_dir = args.debug_dir

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=scorer, refiner=refiner, glctx=glctx)

    reader = load_custom.CustomLoader(data_path=args.test_scene_dir)

    return

    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i == 0:
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, reader.K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

        # print('Initial pose:\n', pose)

        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            print('Refined pose:\n', center_pose)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K,
                                thickness=3, transparency=0, is_input_rgb=True)
        #  cv2.imshow('1', vis[...,::-1])
        #  cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)


if __name__ == '__main__':
    main()
