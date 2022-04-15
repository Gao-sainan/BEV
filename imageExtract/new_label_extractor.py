import glob
import math
import os
from typing import List, Tuple, Union

import cv2 as cv
import habitat_sim
import habitat_sim.registry as registry
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from habitat_sim import registry as registry
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis, quat_rotate_vector, quat_from_two_vectors
from habitat_sim.utils.data import PoseExtractor
from numpy import bool_, float32, float64, ndarray
from quaternion import quaternion

from data_extractor2 import LabelExtractor


@registry.register_pose_extractor(name="td_pose_extractor")
class TopDownPoseExtractor(PoseExtractor):
    
    def extract_poses(
        self, view: ndarray, fp: str
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        
        poses = []
        
        # with open(f'sample_points.txt','r') as f:    #设置文件对象
        with open(f'BEV/data/sample_points/{fp[17:-26]}.txt','r') as f:
            data = f.readlines()
            for d in data:
                d = d.strip('\n')
                for i in range(len(d)):
                    if d[i] == '\t':
                        r = float(d[:i])
                        c = float(d[i+1:])
                        
                        poi = self.generate_poi((r, c))
                        for point_of_interest in poi:
                            pose = ((r, c), point_of_interest, fp)
                            poses.append(pose)

        # Returns poses in the coordinate system of the topdown view
        return poses
    
    def generate_poi(self, point):

        r, c = point
        neighbors = []
        # [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
        # step = np.linspace(0, 340, 18)
        step = [0]
        for s in step:
            neighbor = (r + s, c)
            neighbors.append(neighbor)

        return neighbors
    
    
    def _compute_quat(self, cam_angle) -> quaternion:
        """Rotations start from -z axis"""
        q1 = quat_from_two_vectors(habitat_sim.geo.FRONT, habitat_sim.geo.GRAVITY)
        axis = np.array([0., 0., 1.])
        q2 = quat_from_angle_axis(cam_angle, axis)
        return q1 * q2
    
    def _convert_to_scene_coordinate_system(
        self,
        poses: List[Tuple[Tuple[int, int], Tuple[int, int], str]],
        ref_point: Tuple[float32, float32, float32],
    ) -> List[Tuple[Tuple[int, int], quaternion, str]]:
        startw, starty, starth = ref_point
        for i, pose in enumerate(poses):
            pos, cpi, filepath = pose
            r1, c1 = pos
            r2, c2 = cpi
            cam_angle = math.radians(r2 - r1)
            
            new_pos = np.array(
                [
                    startw + c1 * self.meters_per_pixel,
                    # height 1 meters
                    starty + 10 * self.meters_per_pixel,
                    starth + r1 * self.meters_per_pixel,
                ]
            )
            
            new_rot = self._compute_quat(cam_angle)
            new_pos_t = tuple(new_pos)
            poses[i] = (new_pos_t, new_rot, filepath)
            
            data_dict = {'position':pos, 
                         'cpi': cpi, 
                         'angle': r2 - r1, 
                         'ref_pos':np.array(
                                            [
                                                startw + c1 * self.meters_per_pixel,
                                                starty,
                                                starth + r1 * self.meters_per_pixel,
                                            ]
                                        ),
                         'new_pos': new_pos_t}
            
            # with open(f'sample_points_label.txt','a') as f:
            with open(f'BEV/data/sample_points/{filepath[17:-26]}_label.txt','a') as f:
                f.write('{')
                for key in data_dict:
                    f.write('\n')
                    f.writelines('"' + str(key) + '":' + str(data_dict[key]))
                f.write('\n'+'}')

        return poses
    
def get_fov_lines(camera_pos, img, alpha=0.75):

    c, r = camera_pos
    
    left_l = (
        int(c - 240 * math.tan(math.radians(55))),
        int(r - 240)
    )
    
    left_r = (
        int(c + 240 * math.tan(math.radians(35))),
        int(r - 240)
    )
    
    right_l = (
        int(c - 240 * math.tan(math.radians(35))),
        int(r - 240)
    )
    
    right_r = (
        int(c + 240 * math.tan(math.radians(55))),
        int(r - 240)
    )
    overlay = img.copy()
    # draw left_view boundaries
    color_left = (172, 137, 59)
    tri_left = np.array([camera_pos, left_l, left_r])
    img = cv.drawContours(img, [tri_left], 0, color_left, thickness=-1)

    # draw right view boundaries
    color_right = (31, 135, 232)
    tri_right = np.array([camera_pos, right_l, right_r])
    img = cv.drawContours(img, [tri_right], 0, color_right, thickness=-1)

    # Perform weighted addition of the input image and the overlay
    result = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    result = cv.line(result, camera_pos, left_l, color_left, thickness=1)
    result = cv.line(result, camera_pos, left_r, color_left, thickness=1)
    result = cv.line(result, camera_pos, right_l, color_right, thickness=1)
    result = cv.line(result, camera_pos, right_r, color_right, thickness=1)
    
    # draw the camera position
    # color = (188, 158, 33)
    # img = cv.circle(img, camera_pos, radius=5, color=color, thickness=-1)
    
    f = 1 / ((2 / img.shape[1]) * math.tan(math.radians(45)))
    
    return result, f
    
habitat_test_scenes_dir = 'BEV/replica_data/'
scene_filepath = habitat_test_scenes_dir+"apartment_0/habitat/mesh_semantic.ply"
# scene_filepaths = glob.glob(habitat_test_scenes_dir+"/*/habitat/mesh_semantic.ply")
out_path = 'BEV/data/'

# for scene_id, scene_filepath in tqdm.tqdm(enumerate(scene_filepaths)):
label_extractor = LabelExtractor(
    scene_filepath,
    img_size=(480, 640),
    pose_extractor_name="td_pose_extractor",
    output=["semantic", 'rgba'],
    shuffle=False
)

label_extractor.set_mode('full')

semantic_classes = []

with open('BEV/imageExtract/replica_v1_all_semantic_class.txt') as f:
    semantic_classes  = [i.strip() for i in f]

make_path = os.path.join(out_path, f'apartment_0/label/')
if not os.path.isdir(make_path):
    os.makedirs(make_path)

for i, sample in tqdm.tqdm(enumerate(label_extractor)):
    
    img_rot_quat = label_extractor.poses[i][1]
    img_rot_quat /= quat_from_two_vectors(habitat_sim.geo.FRONT, habitat_sim.geo.GRAVITY)
    an, ax = quat_to_angle_axis(img_rot_quat)
    an = int(math.degrees(an))
    
    map_instance_to_semantic = [0]*(1+max(label_extractor.instance_id_to_name.keys()))
    for k,v in label_extractor.instance_id_to_name.items():
        map_instance_to_semantic[k] = semantic_classes.index(v)
    map_instance_to_semantic = np.array(map_instance_to_semantic,dtype=np.uint8)
    img = sample["rgba"]
    semantic_instance_id = sample["semantic"]
    semantic_label = map_instance_to_semantic[semantic_instance_id]
        
    #TODO:SET THR RULE TO GENERATE "NAVIGABLE" SEMANTIC LABEL
    semantic_label[semantic_label != semantic_classes.index('floor')] = 0
    semantic_label[semantic_label == semantic_classes.index('floor')] = 255
    

    camera_pos = (img.shape[1] // 2, img.shape[0] // 2)
    img, f = get_fov_lines(camera_pos, img)
    # semantic_label ,_ = get_fov_lines(camera_pos, semantic_label, alpha=0.5)

    w = semantic_label.shape[1] // 2
    h = semantic_label.shape[0] // 2
    crop_label = semantic_label[0:h, w//2 : 3*w//2]
    # resized = cv.resize(crop_label, semantic_label.shape, interpolation = cv.INTER_AREA)

    cv.imwrite(make_path + f'cam{i}_{an}.jpg',img)
    cv.imwrite(make_path + f'cam{i}__{an}_label.png',crop_label)
    
label_extractor.close()   
