import numpy as np
import torch


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2

def calculate_alpha(end_points):
    
    # Get the spatial equilibrium weight by calculating alpha from the dimensions of the scene:
    # Get the dimensions of the scene from the point cloud
    pcd = end_points['point_clouds']
    pcd = torch.Tensor.cpu(pcd) if torch.is_tensor(pcd) else pcd
    pcd_flipped = flip_axis_to_camera(pcd)
    pcd_flipped = flip_axis_to_depth(pcd_flipped)
    # print("Shape of Point Cloud: ", pcd_flipped.shape)
    
    # Compute the minimum and maximum values along each axis
    min_coords = np.min(pcd_flipped, axis=1) if pcd_flipped.ndim == 3 else np.min(pcd_flipped, axis=0)
    max_coords = np.max(pcd_flipped, axis=1) if pcd_flipped.ndim == 3 else np.max(pcd_flipped, axis=0)
    
    # Calculate the dimensions of the scene (length, breadth, height)
    dimensions = max_coords - min_coords
    dimensions = dimensions[:,:3] if dimensions.ndim == 2 else dimensions[:3]
    # print("Shape of Dimensions: ", dimensions.shape)
    # print(dimensions)
    
    w, h, b = dimensions.T  # width, height, breadth
    # print("shape of w: ", w.shape)
    
    pred_centers = torch.Tensor.cpu(end_points["center"])
    # print("Shape of Pred Centers: ", pred_centers.shape)
    # Extract x, y, z values by splitting along the last dimension
    x = pred_centers[:, :, 0]  # Shape will be (8, 128)
    y = pred_centers[:, :, 1]  # Shape will be (8, 128)
    z = pred_centers[:, :, 2]  # Shape will be (8, 128)

    # Print shapes to verify
    # print("Shape of x: ", x.shape)  # Should output (8, 128)
    # print("Shape of y: ", y.shape)  # Should output (8, 128)
    # print("Shape of z: ", z.shape)  # Should output (8, 128)
    
    # Reshape w, h, b to (8, 1) so it can be broadcasted to (8, 128)
    if w.ndim == 1:
        w = w[:, np.newaxis]  # Shape becomes (8, 1)
        h = h[:, np.newaxis]  # Shape becomes (8, 1)
        b = b[:, np.newaxis]  # Shape becomes (8, 1)
    
    # Calculate the distance from the center of the image
    x_distance = np.abs(x.detach().numpy() - w / 2) * (1 / w)
    y_distance = np.abs(y.detach().numpy() - h / 2) * (1 / h)
    z_distance = np.abs(z.detach().numpy() - b / 2) * (1 / b)
    
    # First, find the maximum between x_distance and y_distance
    xy_max = np.maximum(x_distance, y_distance)

    # Then, find the maximum between the result and z_distance
    alpha = 3 * np.maximum(xy_max, z_distance)

    
    # print("Shape of Alpha: ", alpha.shape) # Should output (8, 128) or (batch_size, num_seeds)
    return alpha