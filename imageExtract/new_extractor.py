
import math
import os
from re import I
from typing import List, Tuple

import cv2 as cv
import habitat_sim.registry as registry
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from habitat_sim import registry as registry
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
from habitat_sim.utils.data import PoseExtractor
from numpy import bool_, float32, float64
from quaternion import quaternion
from data_extractor import ImageExtractor

@registry.register_pose_extractor(name="binocular_pose_extractor")
class BinocularPoseExtractor(PoseExtractor):
    
    # extract a path from A to B(save it to a configuration file)
    
    def extract_poses(self, view, fp):
        height, width = view.shape
        c = 45
        
        gridpoints = []
        for r in np.linspace(78, 150, 100):
            point = (r, c) 
            gridpoints.append(point)
                    
        poses = []
        for point in gridpoints:
            poi = self.generate_poi(point)
            pose = [(point, point_of_interest, fp) for point_of_interest in poi]
            poses.extend(pose)
            r, c = point
            
            with open(f'BEV/data/sample_points/{fp[17:-26]}.txt','a') as f:    #save to txt
                f.write(str(r))
                f.write('\t')
                f.write(str(c))
                f.write('\n')
                
        return poses
    
    
    def generate_poi(self, point):

        r, c = point
        neighbors = []
        # [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]
        # step = np.linspace(10, 350, 18)
        step = [10, 350]
        for s in step:
            neighbor = (r + s, c)
            neighbors.append(neighbor)

        return neighbors
    
    def _compute_quat(self, cam_angle) -> quaternion:
        """Rotations start from -z axis"""
        axis1 = np.array([0., 1., 0.])
        q1 = quat_from_angle_axis(cam_angle, axis1)
        # look down 20 degrees(modified)
        down_angle = math.radians(45)
        axis2 = np.array([-1., 0., 0.])
        q2 = quat_from_angle_axis(down_angle, axis2)
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
            new_pos = np.array(
                [
                    startw + c1 * self.meters_per_pixel,
                    starty,
                    starth + r1 * self.meters_per_pixel,
                ]
            )
            
            degree = r2 - r1
            angle = math.radians(degree)


            new_rot = self._compute_quat(angle)
            new_pos_t = tuple(new_pos)
            poses[i] = (new_pos_t, new_rot, filepath)
            
            data_dict = {'id': i,
                         'position':pos, 
                         'cpi': cpi, 
                         'angle': degree, 
                         'new_pos': new_pos_t}
            
            
            with open(f'BEV/data/sample_points/{filepath[17:-26]}_detial.txt','a') as f:
                f.write('{')
                for key in data_dict:
                    f.write('\n')
                    f.writelines('"' + str(key) + '":' + str(data_dict[key]))
                f.write('\n'+'}')

        return poses
    
def save_cam_cfg(filepath, extractor):
    
    agent = extractor.cfg.agents[0]
    # focal length
    f = 1 / ((2 / extractor.img_size[1]) * math.tan(math.radians(45)))
    cfg = {
        "default_agent": 0,
        "sensor_subtype": agent.sensor_specifications[0].sensor_subtype,
        "hfov": agent.sensor_specifications[0].hfov,
        "focal": f,
        "sensor_height": agent.height,  # Height of sensors in meters
    }
    
    with open (filepath, 'a') as f:
        f.write('colormap_sensor')
        f.write('{')
        for key in cfg:
            f.write('\n')
            f.writelines('"' + str(key) + '":' + str(cfg[key]))
        f.write('\n'+'}')
        


habitat_test_scenes_dir = 'BEV/replica_data/'
scene_filepath = habitat_test_scenes_dir+"apartment_0/habitat/mesh_semantic.ply"
# scene_filepaths = glob.glob(habitat_test_scenes_dir+"/*/habitat/mesh_semantic.ply")
out_path = 'BEV/data/'

# for scene_id, scene_filepath in tqdm.tqdm(enumerate(scene_filepaths)):
extractor = ImageExtractor(
    scene_filepath,
    img_size=(480, 640),
    pose_extractor_name="binocular_pose_extractor",
    output=['rgba', 'depth'],
    shuffle=False
)

extractor.set_mode('full')

cfg_path = 'BEV/data/config/cam_config.txt'
save_cfg = save_cam_cfg(cfg_path, extractor)

make_path = os.path.join(out_path, 'apartment_0/images/')
if not os.path.isdir(make_path):
    os.makedirs(make_path)

for i, sample in tqdm.tqdm(enumerate(extractor)):

    img = sample['rgba']
    depth = sample['depth']

    img_id = extractor.poses[i][0]
    img_rot = extractor.poses[i][1]
    img_rot /= quat_from_angle_axis(math.radians(45), np.array([-1., 0., 0.]))
    an, ax = quat_to_angle_axis(img_rot)
    
    an = int(math.degrees(an))
    
    if i % 2 == 0:
        id = i // 2
    else:
        id = (i - 1) // 2

    cv.imwrite(make_path + f'cam{id}_{an}.jpg',img)
    plt.imsave(make_path + f'cam{id}_{an}_depth.png', depth)
    # cv.imwrite(f'out1/{str(img_id)}_{an}_depth.png', depth)

extractor.close()   
    