"""
added by TSTM authors

robot_env segmentation utilities: generate precise masks based on depth maps and object positions
No external segmentation models (e.g., SAM) are needed; directly leverage MuJoCo's depth rendering and object states
"""

import numpy as np
import mujoco_py
from scipy.ndimage import binary_dilation, binary_erosion


def get_camera_params(sim, camera_name='third_person', width=84, height=84):
    """
    Get camera parameters for 3D-to-2D projection
    
    Returns:
        fovy: vertical field of view (radians)
        aspect: aspect ratio
        znear, zfar: near and far clipping planes
    """
    camera_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[camera_id] * np.pi / 180.0  # convert to radians
    aspect = width / height
    znear = 0.01
    zfar = 50.0
    return fovy, aspect, znear, zfar


def project_3d_to_2d(pos_3d, sim, camera_name='third_person', width=84, height=84):
    """
    Project 3D world coordinates to 2D image coordinates
    
    Args:
        pos_3d: (3,) 3D position in world coordinates
        sim: MuJoCo simulation instance
        camera_name: camera name
        width, height: image size
    
    Returns:
        (x, y): image pixel coordinates; return None if outside the view frustum
    """
    camera_id = sim.model.camera_name2id(camera_name)
    
    # Get camera position and rotation matrix
    cam_pos = sim.data.cam_xpos[camera_id]
    cam_mat = sim.data.cam_xmat[camera_id].reshape(3, 3)
    
    # Transform world coordinates into the camera coordinate system
    pos_cam = cam_mat.T @ (pos_3d - cam_pos)
    
    # Check if the point is in front of the camera
    if pos_cam[2] <= 0:
        return None
    
    # Project to Normalized Device Coordinates (NDC)
    fovy, aspect, znear, zfar = get_camera_params(sim, camera_name, width, height)
    
    # Perspective projection
    f = 1.0 / np.tan(fovy / 2.0)
    x_ndc = (f / aspect) * (pos_cam[0] / pos_cam[2])
    y_ndc = f * (pos_cam[1] / pos_cam[2])
    
    # Map NDC to pixel coordinates (NDC [-1, 1] -> [0, width/height])
    x_pixel = (x_ndc + 1.0) * width / 2.0
    y_pixel = (1.0 - y_ndc) * height / 2.0  # image y-axis points downward
    
    # Check if within image bounds
    if 0 <= x_pixel < width and 0 <= y_pixel < height:
        return int(x_pixel), int(y_pixel)
    return None


# Global dictionary storing a depth renderer per sim (avoid Cython object attribute issues)
_DEPTH_RENDERERS = {}


def render_depth_map(sim, camera_name='third_person', width=84, height=84):
    """
    Render a depth map
    
    Returns:
        depth: (H, W) depth map in meters; near is small, far is large
    """
    camera_id = sim.model.camera_name2id(camera_name)
    
    # Use the global dictionary to store the renderer
    sim_id = id(sim)
    if sim_id not in _DEPTH_RENDERERS:
        from mujoco_py import MjRenderContextOffscreen
        _DEPTH_RENDERERS[sim_id] = MjRenderContextOffscreen(sim, device_id=-1)
    
    renderer = _DEPTH_RENDERERS[sim_id]
    renderer.render(width, height, camera_id)
    depth = renderer.read_pixels(width, height, depth=True)[1]
    
    # Depth is in [0, 1]; convert to metric distances
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    depth_meters = near / (1.0 - depth * (1.0 - near / far))
    
    # Flip y-axis to match robot_env RGB rendering (see base.py: sim.render()[::-1])
    return depth_meters[::-1]


def generate_object_mask_from_depth(sim, object_site_name, camera_name='third_person', 
                                     width=84, height=84, depth_threshold=0.05,
                                     dilate_iters=2):
    """
    Generate an object mask based on depth and object position
    
    Core idea:
    1. Get the object's 3D position and project it to the image
    2. Render a depth map
    3. Around the projected position, keep pixels whose depth matches the object's depth
    
    Args:
        sim: MuJoCo simulation instance
        object_site_name: object site name (e.g., 'object0', 'grasp')
        camera_name: camera name
        width, height: image size
        depth_threshold: depth threshold in meters to determine object pixels
        dilate_iters: number of dilation iterations to fill mask holes
    
    Returns:
        mask: (H, W) boolean array; True indicates object pixels
    """
    # 1. Get object's 3D position
    obj_pos = sim.data.get_site_xpos(object_site_name).copy()
    
    # 2. Project to 2D
    pixel_coords = project_3d_to_2d(obj_pos, sim, camera_name, width, height)
    
    if pixel_coords is None:
        # Object is outside the camera frustum; return an all-zero mask
        return np.zeros((height, width), dtype=bool)
    
    center_x, center_y = pixel_coords
    
    # 3. Render depth map
    depth_map = render_depth_map(sim, camera_name, width, height)
    
    # 4. Get object depth in the camera coordinate system
    camera_id = sim.model.camera_name2id(camera_name)
    cam_pos = sim.data.cam_xpos[camera_id]
    cam_mat = sim.data.cam_xmat[camera_id].reshape(3, 3)
    pos_cam = cam_mat.T @ (obj_pos - cam_pos)
    object_depth = pos_cam[2]
    
    # 5. Create mask based on depth
    # Start from the center and keep pixels with similar depth
    mask = np.zeros((height, width), dtype=bool)
    
    # Keep pixels whose depth is within [object_depth - threshold, object_depth + threshold]
    depth_lower = object_depth - depth_threshold
    depth_upper = object_depth + depth_threshold
    
    # Initial mask: all pixels with depth close to the object
    rough_mask = (depth_map >= depth_lower) & (depth_map <= depth_upper)
    
    # Use connected components; keep the component that contains the center
    from scipy.ndimage import label
    labeled, num_features = label(rough_mask)
    
    if num_features > 0 and 0 <= center_y < height and 0 <= center_x < width:
        object_label = labeled[center_y, center_x]
        if object_label > 0:
            mask = (labeled == object_label)
        else:
            # If the center is not within any component, fall back to the rough mask
            mask = rough_mask
    else:
        mask = rough_mask
    
    # 6. Morphology: fill holes and smooth edges
    if dilate_iters > 0:
        mask = binary_dilation(mask, iterations=dilate_iters)
        mask = binary_erosion(mask, iterations=dilate_iters)
    
    return mask


def generate_agent_mask_from_segmentation(sim, camera_name='third_person', width=84, height=84):
    """
    Depth-based segmentation strategy (keep manipulator + near table, remove far background)
    
    Key idea:
    1. Manipulator and near table have similar depth (about 1.0-2.0m) → keep
    2. Far background (depth > 2.0m) → remove
    3. Gray walls at image borders → remove
    
    Args:
        sim: MuJoCo simulation instance
        camera_name: camera name
        width, height: image size
    
    Returns:
        mask: (H, W) boolean array; True indicates task-relevant pixels
    """
    # Render depth map
    depth_map = render_depth_map(sim, camera_name, width, height)
    
    # Create y-coordinate array
    # After flipping: y=0 is bottom (manipulator), y=83 is top (back wall)
    y_coords = np.arange(height).reshape(-1, 1)
    y_coords = np.broadcast_to(y_coords, (height, width))
    
    # Keep mid/near range (manipulator + near table), remove far background (back wall)
    # Layer 1: keep regions with depth < 1.15m (manipulator + near table + target)
    mask = depth_map < 1.15
    
    # Layer 2: remove far background (depth > 2.0m)
    is_far_background = depth_map > 2
    mask = mask & (~is_far_background)
    
    return mask


def generate_agent_mask(sim, camera_name='third_person', width=84, height=84):
    """Generate the overall agent (manipulator) mask by calling the main method"""
    return generate_agent_mask_from_segmentation(sim, camera_name, width, height)


def generate_full_scene_mask(sim, has_object=False, object_site='object0',
                             camera_name='third_person', width=84, height=84,
                             include_target=True, target_site='target0',
                             additional_sites=None):
    """
    Generate a full-scene segmentation mask (agent + object + target markers)
    
    Args:
        sim: MuJoCo simulation instance
        has_object: whether there is a manipulable object
        object_site: object site name
        camera_name: camera name
        width, height: image size
        include_target: whether to include target markers (default True)
        target_site: target marker site name (e.g., 'target0', 'nail_goal1')
        additional_sites: additional sites to include (e.g., ['box_hole'])
    
    Returns:
        mask: (H, W) boolean array; True indicates task-relevant pixels
    """
    # Get agent (manipulator) mask
    agent_mask = generate_agent_mask(sim, camera_name, width, height)
    
    full_mask = agent_mask
    
    # Add manipulable object mask
    if has_object:
        try:
            object_mask = generate_object_mask_from_depth(
                sim, object_site, camera_name, width, height
            )
            full_mask = full_mask | object_mask
        except Exception as e:
            print(f"Warning: Failed to extract mask for object '{object_site}': {e}")
    
    # Add target marker mask
    if include_target:
        try:
            target_mask = generate_object_mask_from_depth(
                sim, target_site, camera_name, width, height,
                depth_threshold=0.15,  # target is small; use a larger threshold
                dilate_iters=5  # increase dilation to ensure visibility
            )
            full_mask = full_mask | target_mask
        except Exception as e:
            print(f"Warning: Failed to extract mask for target '{target_site}': {e}")
        
        # Supplement: detect red target by color (useful when depth fails on small targets)
        try:
            rgb_image = sim.render(width=width, height=height, camera_name=camera_name, depth=False)[::-1, :, :]
            # Detect red: high R channel, low G and B channels
            red_mask = (rgb_image[:, :, 0] > 150) & \
                       (rgb_image[:, :, 1] < 100) & \
                       (rgb_image[:, :, 2] < 100)
            # Dilate to make red regions more prominent
            red_mask = binary_dilation(red_mask, iterations=2)
            full_mask = full_mask | red_mask
        except Exception:
            pass
    
    # Add masks for additional objects (e.g., box, nail board)
    if additional_sites is not None:
        for site_name in additional_sites:
            try:
                extra_mask = generate_object_mask_from_depth(
                    sim, site_name, camera_name, width, height,
                    depth_threshold=0.1,
                    dilate_iters=2
                )
                full_mask = full_mask | extra_mask
            except Exception as e:
                print(f"Warning: Failed to extract mask for extra site '{site_name}': {e}")
    
    return full_mask


# Helper to integrate with BaseEnv
def add_segmentation_to_env(env_instance):
    """
    Add segmentation capability to a BaseEnv instance
    
    Usage:
        After env initialization, call:
        add_segmentation_to_env(env)
        
        Then you can use:
        mask = env.render_segmentation()
        mask = env.render_segmentation(include_target=True)  # include target markers
    """
    def render_segmentation(width=None, height=None, include_target=True, 
                           target_site='target0', additional_sites=None):
        w = width or env_instance.image_size
        h = height or env_instance.image_size
        
        # Determine whether the current task has an object
        has_object = getattr(env_instance, 'has_object', False)
        
        # Generate mask
        mask = generate_full_scene_mask(
            env_instance.sim,
            has_object=has_object,
            object_site='object0',
            camera_name=env_instance.cameras[0] if env_instance.cameras else 'third_person',
            width=w,
            height=h,
            include_target=include_target,
            target_site=target_site,
            additional_sites=additional_sites
        )
        
        return mask.astype(np.uint8)
    
    env_instance.render_segmentation = render_segmentation
    return env_instance
