
from estimateModel import *
import argparse
import load_custom
import socket
import struct
import cv2
import numpy as np


class PoseEstimateServer:
    def __init__(self, mesh_file="meshes/charger/power_plug2.stl"):
        self.mesh = trimesh.load(mesh_file)
        self.mesh.apply_scale(0.001)  # from mm to meter for charger data
        self.is_first = True
        args = {
            'est_refine_iter': 5,
            'track_refine_iter': 2,
        }
        self.args = args
        self.downscale = 0.25

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals,
                                  mesh=self.mesh, scorer=scorer, refiner=refiner, glctx=glctx)

    def estimate_poses(self, color, depth, mask, K):
        H, W = color.shape[:2]
        H_ds = int(H * self.downscale)
        W_ds = int(W * self.downscale)
        color = cv2.resize(color, (W_ds, H_ds), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (W_ds, H_ds), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), (W_ds, H_ds), interpolation=cv2.INTER_NEAREST).astype(bool)
        K = K.copy()
        K[:2] *= self.downscale

        if self.is_first:
            pose = self.est.register(K=K, rgb=color, depth=depth,
                                     ob_mask=mask, iteration=self.args['est_refine_iter'])
            self.is_first = False
        else:
            pose = self.est.track_one(rgb=color, depth=depth,
                                      K=K, iteration=self.args['track_refine_iter'])

        print(pose)
        return pose


def server_program():
    # Create a TCP/IP socket
    host = "0.0.0.0"
    port = 1234

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"Server listening on {host}:{port}")

        while True:

            all_data = {'color': None, 'depth': None, 'mask': None, 'K': None}
            conn, address = server_socket.accept()
            print(f"Connection from {address} has been established.")
            estimator = PoseEstimateServer()
            try:

                while True:
                    header = conn.recv(16)
                    if header == b'':
                        print("No more data from client, closing connection.")
                        break

                    img_size, img_type, shape_h, shape_w = struct.unpack('>IIII', header)
                    print(f"Expecting image of size: {img_size}, type: {img_type}, shape: ({shape_h}, {shape_w})")

                    img_data = b''
                    while len(img_data) < img_size:
                        remaining_bytes = img_size - len(img_data)
                        data = conn.recv(remaining_bytes)
                        if not data:
                            break
                        img_data += data

                    if img_type == 1:  # Color image
                        print("Receiving color image...")
                        all_data['color'] = np.frombuffer(
                            img_data, dtype=np.uint8).reshape((shape_h, shape_w, 3))
                    elif img_type == 2:  # Depth image
                        print("Receiving depth image...")
                        all_data['depth'] = np.frombuffer(
                            img_data, dtype=np.float32).reshape((shape_h, shape_w))
                    elif img_type == 3:  # Mask image
                        print("Receiving mask image...")
                        all_data['mask'] = np.frombuffer(img_data, dtype=np.uint8).reshape(
                            (shape_h, shape_w)).astype(bool)
                    elif img_type == 4:  # Camera intrinsicsi
                        print("Receiving camera intrinsics...")
                        all_data['K'] = np.frombuffer(
                            img_data, dtype=np.float64).reshape((shape_h, shape_w))

                    if all(value is not None for value in all_data.values()):
                        print("Received all data, processing...")
                        pose = estimator.estimate_poses(
                            color=all_data['color'],
                            depth=all_data['depth'],
                            mask=all_data['mask'],
                            K=all_data['K'],
                        )
                        conn.sendall(np.array(pose).astype(np.float32).tobytes())
                        print('pose sent: ', pose)

            except Exception as e:
                print(f"An error occurred: {e}")

            finally:
                conn.close()


if __name__ == '__main__':
    server_program()
