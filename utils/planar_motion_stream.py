# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Camera Pose Generator:

This module allows to define a trajectory of camera poses
and generate continuous homographies by interpolating when
maximum optical flow is beyond a predefined threshold.
"""
# from __future__ import absolute_import

import numpy as np
import cv2

from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from numba import jit

DTYPE = np.float32


@jit
def interpolate_times_tvecs(tvecs, key_times, inter_tvecs, inter_times, nums):
    """
    Interpolates between key times times & translation vectors

    Args:
        tvecs (np.array): key translation vectors  (N, 3)
        key_times (np.array): key times (N, )
        inter_tvecs (np.array): interpolated translations (nums.sum(), 3)
        inter_times (np.array): interpolated times (nums.sum(),)
        nums (np.array): number of interpolation point between key points (N-1,)
                         nums[i] is the number of points between key_times[i] (included) and key_times[i+1] (excluded)
                         minimum is 1, which corresponds to key_times[i]
    """
    n = 0
    for i in range(nums.shape[0]):
        num = nums[i]
        tvec1 = tvecs[i]
        tvec2 = tvecs[i + 1]
        time1 = key_times[i]
        time2 = key_times[i + 1]
        for j in range(num):
            a = j / num
            ti = time1 * (1 - a) + a * time2
            inter_times[n] = ti
            inter_tvecs[n] = tvec1 * (1 - a) + a * tvec2
            n += 1


def generate_homography(rvec, tvec, nt, depth):
    """
    Generates a single homography

    Args:
        rvec (np.array): rotation vector
        tvec (np.array): translation vector
        nt (np.array): normal to camera
        depth (float): depth to camera
    """
    R = cv2.Rodrigues(rvec)[0].T
    H = R - np.dot(tvec.reshape(3, 1), nt) / depth
    return H


def generate_image_homography(rvec, tvec, nt, depth, K, Kinv):
    """
    Generates a single image homography

    Args:
        rvec (np.array): rotation vector
        tvec (np.array): translation vector
        nt (np.array): normal to camera
        depth (float): depth
        K (np.array): intrisic matrix
        Kinv (np.array): inverse intrinsic matrix
    """
    H = generate_homography(rvec, tvec, nt, depth)
    G = np.dot(K, np.dot(H, Kinv))
    G /= G[2, 2]
    return G


def generate_homographies_from_rotation_matrices(rot_mats, tvecs, nt, depth):
    """
    Generates multiple homographies from rotation matrices

    Args:
        rot_mats (np.array): N,3,3 rotation matrices
        tvecs (np.array): N,3 translation vectors
        nt (np.array): normal to camera
        depth (float): depth to camera
    """
    rot_mats = np.moveaxis(rot_mats, 2, 1)
    t = np.einsum('ik,jd->ikd', tvecs, nt)
    h = rot_mats - t / depth
    return h


def generate_homographies(rvecs, tvecs, nt, d):
    """
    Generates multiple homographies from rotation vectors

    Args:
        rvecs (np.array): N,3 rotation vectors
        tvecs (np.array): N,3 translation vectors
        nt (np.array): normal to camera
        d (float): depth
    """
    rot_mats = R.from_rotvec(rvecs).as_matrix()
    return generate_homographies_from_rotation_matrices(rot_mats=rot_mats, tvecs=tvecs, nt=nt, depth=d)


def generate_image_homographies_from_homographies(h, K, Kinv):
    """
    Multiplies homography left & right by intrinsic matrix

    Args:
        h (np.array): homographies N,3,3
        K (np.array): intrinsic
        Kinv (np.ndarray): inverse intrinsic
    """
    g = np.einsum('ikc,cd->ikd', h, Kinv)
    g = np.einsum('kc,jcd->jkd', K, g)
    g /= g[:, 2:3, 2:3]
    return g


def get_transform(rvec1, tvec1, rvec2, tvec2, nt, depth):
    """
    Get geometric Homography between 2 poses

    Args:
        rvec1 (np.array): rotation vector 1
        tvec1 (np.array): translation vector 1
        rvec2 (np.array): rotation vector 2
        tvec2 (np.array): translation vector 2
        nt (np.array): plane normal
        depth (float): depth from camera
    """
    H_0_1 = generate_homography(rvec1, tvec1, nt, depth)
    H_0_2 = generate_homography(rvec2, tvec2, nt, depth)
    H_1_2 = H_0_2.dot(np.linalg.inv(H_0_1))
    return H_1_2


def get_image_transform(rvec1, tvec1, rvec2, tvec2, nt, depth, K, Kinv):
    """
    Get image Homography between 2 poses (includes cam intrinsics)

    Args:
        rvec1 (np.array): rotation vector 1
        tvec1 (np.array): translation vector 1
        rvec2 (np.array): rotation vector 2
        tvec2 (np.array): translation vector 2
        nt (np.array): plane normal
        depth (float): depth from camera
        K (np.array): intrinsic
        Kinv (np.ndarray): inverse intrinsic
    """
    H_0_1 = generate_image_homography(rvec1, tvec1, nt, depth, K, Kinv)
    H_0_2 = generate_image_homography(rvec2, tvec2, nt, depth, K, Kinv)
    H_1_2 = H_0_2.dot(np.linalg.inv(H_0_1))
    return H_1_2


def interpolate_poses(rvecs, tvecs, nt, depth, K, Kinv, height, width, opt_flow_threshold=2, max_frames_per_bin=20):
    """
    Interpolate given poses

    Args:
        rvecs (np.array): N,3 rotation vectors
        tvecs (np.array): N,3 translation vectors
        nt (np.array): plane normal
        depth (float): depth to camera
        K (np.array): camera intrinsic
        Kinv (np.array): inverse camera intrinsic
        height (int): height of image
        width (int): width of image
        opt_flow_threshold (float): maximum flow threshold
        max_frames_per_bin (int): maximum number of pose interpolations between two consecutive poses
                                  of the original list of poses
    """
    max_frames = len(rvecs)
    key_times = np.linspace(0, max_frames - 1, max_frames, dtype=np.float32)  # (N,)

    rotations = R.from_rotvec(rvecs)

    # all homographies
    h_0_2 = generate_homographies_from_rotation_matrices(rotations.as_matrix(), tvecs, nt, depth)  # (N, 3, 3)
    hs = generate_image_homographies_from_homographies(h_0_2, K, Kinv)  # (N, 3, 3)

    h_0_1 = hs[:-1]  # (N-1, 3, 3)
    h_0_2 = hs[1:]  # (N-1, 3, 3)
    h_0_1 = np.einsum('ijk,ikc->ijc', h_0_2, np.linalg.inv(h_0_1))  # (N-1, 3, 3)

    # 4 corners
    uv1 = np.array([[0, 0, 1], [0, height - 1, 1], [width - 1, 0, 1], [width - 1, height - 1, 1]])  # (4, 3)

    # maximum flows / image
    xyz = np.einsum('jk,lck->ljc', uv1, h_0_1)  # equivalent to uv1.dot(h_0_1.T) for each 3x3 in h_0_1   (N-1, 4, 3)

    uv2 = xyz / xyz[..., 2:3]  # (N-1, 4, 3)
    flows = uv2[..., :2] - uv1[..., :2]  # (N-1, 4, 2)
    flow_mags = np.sqrt(flows[..., 0]**2 + flows[..., 1]**2)  # (N-1, 4)
    max_flows = flow_mags.max(axis=1)  # (N-1,)

    # interpolate
    nums = 1 + np.ceil(max_flows / opt_flow_threshold)
    if max_frames_per_bin > 0:
        nums = np.minimum(max_frames_per_bin, np.maximum(1, nums))
    nums = nums.astype(np.int32)
    total = nums.sum()

    interp_tvecs = np.zeros((total, 3), dtype=np.float32)
    times = np.zeros((total,), dtype=np.float32)
    interpolate_times_tvecs(tvecs, key_times, interp_tvecs, times, nums)

    slerp = Slerp(key_times, rotations)
    interp_rvecs = slerp(times).as_rotvec()  # (nums.sum(), 3)

    return interp_rvecs, interp_tvecs, times, max_flows.max()


def get_flow(rvec1, tvec1, rvec2, tvec2, nt, depth, K, Kinv, height, width):
    """
    Computes Optical Flow between 2 poses

    Args:
        rvec1 (np.array): rotation vector 1
        tvec1 (np.array): translation vector 1
        rvec2 (np.array): rotation vector 2
        tvec2 (np.array): translation vector 2
        nt (np.array): plane normal
        depth (float): depth from camera
        K (np.array): intrisic matrix
        Kinv (np.array): inverse intrisic matrix
        height (int): height of image
        width (int): width of image
        infinite (bool): plan is infinite or not
    """
    # 1. meshgrid of image 1
    uv1 = get_grid(height, width).reshape(height * width, 3)

    # adapt K with new height, width
    H_0_1 = generate_image_homography(rvec1, tvec1, nt, depth, K, Kinv)
    H_0_2 = generate_image_homography(rvec2, tvec2, nt, depth, K, Kinv)

    # 2. apply H_0_2.dot(H_1_0) directly
    H_1_0 = H_0_2.dot(np.linalg.inv(H_0_1))

    xyz = uv1.dot(H_1_0.T)
    uv2 = xyz / xyz[:, 2:3]

    flow = uv2[:, :2] - uv1[:, :2]
    flow = flow.reshape(height, width, 2)
    return flow


def get_grid(height, width):
    """
    Computes a 2d meshgrid

    Args:
        height (int): height of grid
        width (int): width of grid
    """
    x, y = np.linspace(0, width - 1, width, dtype=DTYPE), np.linspace(0, height - 1, height, dtype=np.float32)
    x, y = np.meshgrid(x, y)
    x, y = x[:, :, None], y[:, :, None]
    o = np.ones_like(x)
    xy = np.concatenate([x, y, o], axis=2)
    return xy


def generate_smooth_signal(num_signals, num_samples, min_speed=1e-4, max_speed=1e-1):
    """
    Generates a smooth signal

    Args:
        num_signals (int): number of signals to generate
        num_samples (int): length of multidimensional signal
        min_speed (float): minimum rate of change
        max_speed (float): maximum rate of change
    """
    t = np.linspace(0, num_samples - 1, num_samples)

    num_bases = 10
    samples = 0 * t
    samples = samples[None, :].repeat(num_signals, 0)
    for i in range(num_bases):
        speed = np.random.uniform(min_speed, max_speed, (num_signals,))[:, None]
        phase = np.random.uniform(np.pi, 10 * np.pi)
        test = np.sin((speed * t[None, :]) + phase)
        samples += test / num_bases

    # final
    speed = np.random.uniform(min_speed, max_speed, (num_signals,))[:, None]
    test = np.sin((speed * t[None, :]))
    test = (test + 1) / 2
    samples *= test
    return samples.astype(DTYPE)


def add_random_pause(signal, max_pos_size_ratio=0.1):
    """
    Adds a random pause in a multidimensional signal

    Args:
        signal (np.array): TxD signal
        max_pos_size_ratio (float): size of pause relative to signal Length
    """
    num_pos = len(signal)
    max_pause_size = int(max_pos_size_ratio * num_pos)
    min_pos = int(0.1 * num_pos)
    pause_size = max_pause_size
    pause_pos = np.random.randint(min_pos, num_pos - pause_size)
    value = signal[pause_pos:pause_pos + 1].repeat(pause_size, 0)
    cat = np.concatenate((signal[:pause_pos - 1], value, signal[pause_pos:]), axis=0)
    return cat


class CameraPoseGenerator(object):
    """
    CameraPoseGenerator generates a series of continuous homographies
    with interpolation.

    Args:
        height (int): height of image
        width (int): width of image
        max_frames (int): maximum number of poses
        pause_probability (float): probability that the sequence contains a pause
        max_optical_flow_threshold (float): maximum optical flow between two consecutive frames
        max_interp_consecutive_frames (int): maximum number of interpolated frames between two consecutive frames
    """

    def __init__(self, height, width, max_frames=300, pause_probability=0.5,
                 max_optical_flow_threshold=2., max_interp_consecutive_frames=20, min_speed=1e-4, max_speed=1e-1):
        self.K = np.array(
            [[width / 2, 0, width / 2], [0, height / 2, height / 2], [0, 0, 1]],
            dtype=DTYPE,
        )
        self.Kinv = np.linalg.inv(self.K)

        self.nt = np.array([0, 0, 1], dtype=DTYPE).reshape(1, 3)

        signal = generate_smooth_signal(6, max_frames, min_speed, max_speed).T
        if np.random.rand() < pause_probability:
            signal = add_random_pause(signal)
        rvecs = signal[:, :3]
        tvecs = signal[:, 3:]

        self.depth = np.random.uniform(1.0, 2.0)
        self.rvecs, self.tvecs, self.times, max_flow = interpolate_poses(
            rvecs, tvecs, self.nt, self.depth, self.K, self.Kinv, height, width,
            opt_flow_threshold=max_optical_flow_threshold,
            max_frames_per_bin=max_interp_consecutive_frames)

        self.time = 0
        self.max_frames = max_frames
        assert len(self.rvecs) >= max_frames
        self.rvecs, self.tvecs, self.times = self.rvecs[:max_frames], self.tvecs[:max_frames], self.times[:max_frames]

    def __len__(self):
        """
        Returns number of poses
        """
        return len(self.rvecs)

    def __call__(self):
        """
        Returns next homography
        """
        rvec2 = self.rvecs[self.time]
        tvec2 = self.tvecs[self.time]
        ts = self.times[self.time]
        H = generate_image_homography(rvec2, tvec2, self.nt, self.depth, self.K, self.Kinv)
        self.time += 1
        return H, ts

    def get_image_transform(self, rvec1, tvec1, rvec2, tvec2):
        """
        Get Homography between 2 poses

        Args:
            rvec1 (np.array): rotation vector 1
            tvec1 (np.array): translation vector 1
            rvec2 (np.array): rotation vector 2
            tvec2 (np.array): translation vector 2
        """
        return get_image_transform(rvec1, tvec1, rvec2, tvec2, self.nt, self.depth, self.K, self.Kinv)

    def get_flow(self, rvec1, tvec1, rvec2, tvec2, height, width):
        """
        Computes Optical flow between 2 poses

        Args:
            rvec1 (np.array): rotation vector 1
            tvec1 (np.array): translation vector 1
            rvec2 (np.array): rotation vector 2
            tvec2 (np.array): translation vector 2
            nt (np.array): plane normal
            depth (float): depth from camera
            height (int): height of image
            width (int): width of image
        """
        return get_flow(rvec1, tvec1, rvec2, tvec2, self.nt, self.depth, self.K, self.Kinv, height, width)


    def get_frame_homography(self, index):
        """
        Returns next homography
        """
        rvec2 = self.rvecs[index]
        tvec2 = self.tvecs[index]
        ts = self.times[index]
        H = generate_image_homography(rvec2, tvec2, self.nt, self.depth, self.K, self.Kinv)
        self.time += 1
        return H, ts
"""
6-DOF motion in front of image plane
All in numpy + OpenCV
Applies continuous homographies to your picture in time.
Also you can get the optical flow for this motion.
"""

class PlanarMotionStream(object):
    """
    Generates a planar motion in front of the image

    Args:
        image_filename (str): path to image
        height (int): desired height
        width (int): desired width
        max_frames (int): number of frames to stream
        rgb (bool): color images or gray
        infinite (bool): border is mirrored
        pause_probability (float): probability to add a pause during the stream
        max_optical_flow_threshold (float): maximum optical flow between two consecutive frames
        max_interp_consecutive_frames (int): maximum number of interpolated frames between two consecutive frames
        crop_image (bool): crop images or resize them
    """

    def __init__(self, image_filename, height, width, max_frames=1000, rgb=False, infinite=True,
                 pause_probability=0.5,
                 max_optical_flow_threshold=2., max_interp_consecutive_frames=20, crop_image=False,
                 dt=None, include_bounds=None):

        self.height = height
        self.width = width
        self.crop_image = crop_image
        self.max_frames = max_frames
        self.rgb = rgb
        self.filename = image_filename
        self.margins = []
        self.include_bounds = include_bounds
        
        if not self.rgb:
            frame = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
        else:
            frame = cv2.imread(image_filename)[..., ::-1]
        self.frame = frame
        self.frame_height, self.frame_width = self.frame.shape[:2]
        if self.height == -1 or self.width == -1:
            self.height, self.width = self.frame_height, self.frame_width
        self.camera = CameraPoseGenerator(self.frame_height, self.frame_width, self.max_frames, pause_probability,
                                          max_optical_flow_threshold=max_optical_flow_threshold,
                                          max_interp_consecutive_frames=max_interp_consecutive_frames)
        self.iter = 0
        # self.border_mode = cv2.BORDER_REFLECT101 if infinite else cv2.BORDER_CONSTANT
        self.border_mode = cv2.BORDER_REFLECT if infinite else cv2.BORDER_CONSTANT
        if dt is None:
            self.dt = np.random.randint(10000, 20000)
        else:
            self.dt = dt
        self.xy1 = None

    def get_size(self):
        return (self.height, self.width)

    def pos_frame(self):
        return self.iter

    def __len__(self):
        return self.max_frames

    def __next__(self):
        if self.iter >= len(self.camera):
            raise StopIteration

        G_0to2, ts = self.camera()

        out = cv2.warpPerspective(
            self.frame,
            G_0to2,
            dsize=(self.frame_width, self.frame_height),
            # dsize=(self.frame.shape[1]*3, self.frame.shape[0]*3),
            borderMode=self.border_mode,
        )
        
        self.iter += 1
        ts *= self.dt

        # if self.crop_image and out.shape[0] >= self.height and out.shape[1] >= self.width:
        #     if self.include_bounds is None:
        #         margin_height_top = int((out.shape[0] - self.height) // 2.0)
        #         margin_height_bottom = (out.shape[0] - self.height) - margin_height_top
        #         margin_width_left = int((out.shape[1] - self.width) // 2.0)
        #         margin_width_right = (out.shape[1] - self.width) - margin_width_left
        #         out = out[margin_height_top:-margin_height_bottom or None,
        #                 margin_width_left:-margin_width_right or None]
        #     else:
        #         bounds_corners = np.array([[self.include_bounds[2], self.include_bounds[0], 1], # top left
        #                                     [self.include_bounds[3], self.include_bounds[0], 1], # top right
        #                                     [self.include_bounds[2], self.include_bounds[1], 1], # bottom left
        #                                     [self.include_bounds[3], self.include_bounds[1], 1]]) # bottom right
        #         bound_corners_warped = np.matmul(G_0to2, bounds_corners.T).T
        #         bound_corners_warped = bound_corners_warped[:, :2]

        #         new_bounds = np.array([bound_corners_warped[:, 1].min(), bound_corners_warped[:, 1].max(),
        #                                 bound_corners_warped[:, 0].min(), bound_corners_warped[:, 0].max()])
                
        #         bounds_centre = np.array([(new_bounds[0] + new_bounds[1]) / 2,
        #                                 (new_bounds[2] + new_bounds[3]) / 2])
                
        #         margin_height_top = int(bounds_centre[0] - (self.height // 2.0))
        #         margin_height_bottom = out.shape[0] - (margin_height_top + self.height)
        #         margin_width_left = int(bounds_centre[1] - (self.width // 2.0))
        #         margin_width_right = out.shape[1] - (margin_width_left + self.width)
        #         out_copy = cv2.rectangle(out.copy(), (int(new_bounds[2]), int(new_bounds[0])),
        #                                 (int(new_bounds[3]), int(new_bounds[1])), (0, 255, 0), 2)
        #         cv2.imshow("frame", out_copy)
        #         cv2.waitKey(1)
        #         out = out[margin_height_top:-margin_height_bottom or None,
        #                 margin_width_left:-margin_width_right or None]
        #     margins = [margin_height_top, margin_height_bottom, margin_width_left, margin_width_right]
        #     self.margins.append(margins)
            
        # else:
        #     print("***Resizing image***")
        #     out = cv2.resize(out, (self.width, self.height), 0, 0, cv2.INTER_AREA)
        if self.rgb:
            # assert out.shape == (self.height, self.width, 3)
            assert (len(out.shape)==3) and (out.shape[-1] == 3)
        else:
            # assert out.shape == (self.height, self.width)
            assert (len(out.shape)==2)
        return out, ts

    def __iter__(self):
        return self

    def get_relative_homography(self, time_step):
        rvec1, tvec1 = self.camera.rvecs[time_step], self.camera.tvecs[time_step]
        rvec2, tvec2 = self.camera.rvecs[self.iter-1], self.camera.tvecs[self.iter-1]
        # H_2_1 = self.camera.get_transform(rvec2, tvec2, rvec1, tvec1, self.height, self.width)
        H_2_1 = self.camera.get_image_transform(rvec2, tvec2, rvec1, tvec1)
        return H_2_1


def get_harris_corners_from_image(img, return_mask=False):
    """
        takes an image as input and outputs harris corners

        Args:
            img: opencv image
            return_mask: returns a binary heatmap instead of corners positions
        Returns:
            harris corners in 3d with constant depth of one or a binary heatmap
    """
    harris_corners = cv2.cornerHarris(img, 5, 3, 0.06)
    filtered = ndimage.maximum_filter(harris_corners, 7)
    mask = (harris_corners == filtered)
    harris_corners *= mask
    mask = harris_corners > 0.001
    if mask.sum() == 0:
        mask = harris_corners >= 0.5 * harris_corners.max()
    mask = 1 * mask
    if return_mask:
        return mask
    y, x = np.nonzero(mask)
    harris_corners = np.ones((len(x), 3))
    harris_corners[:, 0] = x
    harris_corners[:, 1] = y
    return harris_corners


def project_points(points,
                   homography,
                   width,
                   height,
                   original_width,
                   original_height,
                   return_z=False,
                   return_mask=False,
                   filter_correct_corners=True):
    """
        projects 2d points given an homography and resize new points to new dimension.

        Args:
            points: 2d points in the form [x, y, 1] numpy array shape Nx3
            homography: 3*3 homography numpy array
            width: desired new dimension
            height: desired new dimension
            original_width: original dimension in which homography is given
            original_height: original dimension in which homography is given
            return_z: boolean to return points as 2d or 3d
            return_mask: boolean to return mask of out-of-bounds projected points
            filter_correct_corners: boolean whether to filter out-of-bounds projected points or not


        Returns:
            projected points: points projected in the new space and filtered by default to output only correct points
    """
    projected_points = np.matmul(homography, points.T).T
    projected_points /= np.expand_dims(projected_points[:, 2], 1)
    if return_z:
        projected_points = projected_points[:, :3]
    else:
        projected_points = projected_points[:, :2]
    projected_points[:, 0] *= width / original_width
    projected_points[:, 1] *= height / original_height
    if not filter_correct_corners and not return_mask:
        return projected_points
    mask = (projected_points[:, 0].round() >= 0) * (projected_points[:, 1].round() >= 0) * \
           (projected_points[:, 0].round() < width) * (projected_points[:, 1].round() < height)
    if not filter_correct_corners:
        return projected_points, mask
    else:
        if return_mask:
            return projected_points[mask], mask
        else:
            return projected_points[mask]


class CornerPlanarMotionStream(PlanarMotionStream):
    """
    Generates a planar motion in front of the image, returning both images and Harris' corners

    Args:
        image_filename: path to image
        height: desired height
        width: desired width
        max_frames: number of frames to stream
        rgb: color images or gray
        infinite: border is mirrored
        pause_probability: probability of stream to pause
        draw_corners_as_circle: if true corners will be 2 pixels circles
    """

    def __init__(self, image_filename, height, width, max_frames=1000, rgb=False, infinite=True,
                 pause_probability=0.5, draw_corners_as_circle=True):
        super().__init__(image_filename, height, width, max_frames=max_frames, rgb=rgb, infinite=infinite,
                         pause_probability=pause_probability)
        self.iter = 0
        self.corners = get_harris_corners_from_image(self.frame)
        self.draw_corners_as_circle = draw_corners_as_circle
        if self.draw_corners_as_circle:
            self.image_of_corners = np.zeros((self.frame_height, self.frame_width))
            rounded_corners = np.round(self.corners).astype(np.int16)
            if len(rounded_corners) > 0:
                for x, y, z in rounded_corners:
                    cv2.circle(self.image_of_corners, (x, y), 2, (255, 255, 255), -1)

    def __next__(self):
        if self.iter >= len(self.camera):
            raise StopIteration

        G_0to2, ts = self.camera()

        corners = np.zeros((self.height, self.width))
        if len(self.corners) != 0:
            if self.draw_corners_as_circle:
                corners = cv2.warpPerspective(
                    self.image_of_corners,
                    G_0to2,
                    dsize=(self.frame_width, self.frame_height),
                    borderMode=self.border_mode,
                )
                corners = cv2.resize(corners, (self.width, self.height), 0, 0, cv2.INTER_AREA)
            else:
                projected_corners = project_points(self.corners,
                                                   G_0to2,
                                                   self.width,
                                                   self.height,
                                                   self.frame_width,
                                                   self.frame_height)
                corners_rounded = np.round(projected_corners).astype(np.int16)
                corners[corners_rounded[:, 1], corners_rounded[:, 0]] = 1
        out = cv2.warpPerspective(
            self.frame,
            G_0to2,
            dsize=(self.frame_width, self.frame_height),
            borderMode=self.border_mode,
        )
        out = cv2.resize(out, (self.width, self.height), 0, 0, cv2.INTER_AREA)

        self.iter += 1
        ts *= self.dt

        return out, corners, ts
