import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../datasets")
from scannet_base import ScannetBaseDataset
from scannet_val import ScannetValDataset
import open3d as o3d
import os
import pandas as pd 

# Load datasets, classes, and box sizes:
train_dataset = ScannetBaseDataset(num_points=40000,
                                     augment=False)

val_dataset = ScannetValDataset(all_classes=True,
                                    num_novel_class=0,
                                    num_points=40000,
                                    augment=False)

classes = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'counter', 'curtain', 'desk', 'door', 'otherfurniture',
           'picture', 'refrigerator', 'showercurtain', 'sink', 'sofa', 'table', 'toilet', 'window']

box_sizes = {'cabinet': np.array([0.76966726, 0.81160211, 0.92573741]),
                       'bed': np.array([1.876858, 1.84255952, 1.19315654]),
                       'chair': np.array([0.61327999, 0.61486087, 0.71827014]),
                       'sofa': np.array([1.39550063, 1.51215451, 0.83443565]),
                       'table': np.array([0.97949596, 1.06751485, 0.63296875]),
                       'door': np.array([0.53166301, 0.59555772, 1.75001483]),
                       'window': np.array([0.96247056, 0.72462326, 1.14818682]),
                       'bookshelf': np.array([0.83221924, 1.04909355, 1.68756634]),
                       'picture': np.array([0.21132214, 0.4206159 , 0.53728459]),
                       'counter': np.array([1.44400728, 1.89708334, 0.26985747]),
                       'desk': np.array([1.02942616, 1.40407966, 0.87554322]),
                       'curtain': np.array([1.37664116, 0.65521793, 1.68131292]),
                       'refrigerator': np.array([0.66508189, 0.71111926, 1.29885307]),
                       'showercurtain': np.array([0.41999174, 0.37906947, 1.75139715]),
                       'toilet': np.array([0.59359559, 0.59124924, 0.73919014]),
                       'sink': np.array([0.50867595, 0.50656087, 0.30136236]),
                       'bathtub': np.array([1.15115265, 1.0546296 , 0.49706794]),
                       'otherfurniture': np.array([0.47535286, 0.49249493, 0.58021168])
                        }

color_name_to_rgb = {
    'red': [1.0, 0.0, 0.0],
    'green': [0.0, 1.0, 0.23529411764705882],
    'blue': [0.0, 0.4705882352941176, 1.0],
    'purple': [0.7058823529411765, 0.0, 1.0],
    'orange': [1.0, 0.23529411764705882, 0.0],
    'lime': [0.23529411764705882, 1.0, 0.0],
    'cyan': [0.0, 0.7058823529411765, 1.0],
    'magenta': [0.4705882352941176, 0.0, 1.0],
    'pink': [1.0, 0.0, 0.4705882352941176],
    'yellow': [0.4705882352941176, 1.0, 0.0],
    'skyblue': [0.0, 0.9411764705882353, 1.0],
    'indigo': [0.23529411764705882, 0.0, 1.0],
    'hotpink': [1.0, 0.0, 0.7058823529411765],
    'chartreuse': [0.7058823529411765, 1.0, 0.0],
    'aquamarine': [0.0, 1.0, 0.7058823529411765],
    'royalblue': [0.0, 0.23529411764705882, 1.0],
    'violet': [0.9411764705882353, 0.0, 1.0],
    'gold': [1.0, 0.4705882352941176, 0.0],
    'black': [0.0, 0.0, 0.0],
}

class_to_color_name = {
    'bathtub': 'black',
    'bed': 'green',
    'bookshelf': 'blue',
    'cabinet': 'purple',
    'chair': 'orange',
    'counter': 'lime',
    'curtain': 'cyan',
    'desk': 'magenta',
    'door': 'pink',
    'otherfurniture': 'yellow',
    'picture': 'skyblue',
    'refrigerator': 'indigo',
    'showercurtain': 'hotpink',
    'sink': 'chartreuse',
    'sofa': 'aquamarine',
    'table': 'royalblue',
    'toilet': 'violet',
    'window': 'gold',
}
    
# Helper functions for bbox manipulation
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

# Function to create a bounding box LineSet from center, heading, and size
def create_bbox(center, heading, size, color):
    corners = np.array([
        [-size[0]/2, -size[1]/2, -size[2]/2],
        [ size[0]/2, -size[1]/2, -size[2]/2],
        [ size[0]/2,  size[1]/2, -size[2]/2],
        [-size[0]/2,  size[1]/2, -size[2]/2],
        [-size[0]/2, -size[1]/2,  size[2]/2],
        [ size[0]/2, -size[1]/2,  size[2]/2],
        [ size[0]/2,  size[1]/2,  size[2]/2],
        [-size[0]/2,  size[1]/2,  size[2]/2],
    ])
    
    # Rotate the corners according to the heading
    cos_theta = np.cos(heading)
    sin_theta = np.sin(heading)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ])
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # Translate corners to the center
    corners =  center - rotated_corners
    
    # Create LineSet
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
    ]
    colors = [color for _ in range(len(lines))]  # Red color for all lines
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set


# Function to create a bounding box LineSet from corners
def create_bbox_from_corners(corners, color):
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
    ]
    colors = [color for _ in range(len(lines))]  # Same color for all lines
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

# Function that retrieves the ground truth bounding boxes' center coordinates and classnames in a given image
def get_gt_bboxes(img_id, dataset_type="val"):
    
    
    if dataset_type == "val":
        dataset = val_dataset
        
    elif dataset_type == "train":
        dataset = train_dataset
    
    data = dataset[img_id]
    
    scanname = dataset.scan_names[img_id]
    print(scanname)
    
    center_label = data["center_label"]
    heading_class_label = data["heading_class_label"]
    heading_residual_label = data["heading_residual_label"]
    size_residual_label = data["size_residual_label"]
    sem_cls_label = data["sem_cls_label"]
    pcd = data["point_clouds"]
    
    pcd_flipped = flip_axis_to_camera(pcd)
    pcd_flipped = flip_axis_to_depth(pcd_flipped)
    
    # print("pcd_flipped shape: ", pcd_flipped.shape)
    
    # print("No. of GT bboxes before removing duplicates: ", len(center_label))
    
    centers = []
    
    
    for i in range(len(center_label)):
        center = center_label[i]
        heading = heading_class_label[i] * (2 * np.pi / 12) + heading_residual_label[i]
        class_label = sem_cls_label[i]
        classname = classes[class_label]
        size = box_sizes[classname] + size_residual_label[i]
        
        # Remove duplicate filler boxes, which have center at [0, 0, 0]:
        if np.array_equal(center, np.array([0., 0., 0.])) and classname=="bathtub":
            continue
        
        centers.append((i, center, classname))
    
    # print("No. of GT bboxes after removing: ", len(centers))
    
    return centers

from sklearn.neighbors import NearestNeighbors

# Function to get the dimensions and center of a scene
def get_scene_dimensions_center(img_id, dataset_type="val"):
    
    if dataset_type == "val":
        dataset = val_dataset
        
    elif dataset_type == "train":
        dataset = train_dataset
    
    data = dataset[img_id]
    
    scanname = dataset.scan_names[img_id]
    
    pcd = data["point_clouds"]
    pcd_flipped = flip_axis_to_camera(pcd)
    pcd_flipped = flip_axis_to_depth(pcd_flipped)
    
    # Compute the minimum and maximum values along each axis
    min_coords = np.min(pcd_flipped, axis=0)
    max_coords = np.max(pcd_flipped, axis=0)
    
    # Calculate the dimensions of the scene (length, breadth, height)
    dimensions = max_coords - min_coords
    
    x, y, z = dimensions
    # print(f"x: {x}")
    # print(f"y: {y}")
    # print(f"z: {z}")
    
    # Calculate the center of the scene
    center = (max_coords + min_coords) / 2
    # print(f"Center: {center}")
    
    # Plot the point cloud and the center of the scene
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd_flipped)
    
    # Nearest Neighbors to map pcd_noRGB_flipped to points
    pcd_ply = o3d.io.read_point_cloud(f"../../../../../mnt/data/Datasets/ScanNet_v2/scans/{scanname}/{scanname}_vh_clean_2.ply")
    points = np.asarray(pcd_ply.points)
    colors = np.asarray(pcd_ply.colors)
    
    k = 10
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    distances, indices = nn.kneighbors(pcd_flipped)
    
    # Calculate the weighted average of the RGB values
    weights = 1 / (distances + 1e-8)  # Avoid division by zero
    weights /= weights.sum(axis=1, keepdims=True)  # Normalize weights
    
    # Compute the weighted average of RGB values
    rgb_vals = np.zeros((pcd_flipped.shape[0], 3))
    for i in range(k):
        rgb_vals += colors[indices[:, i]] * weights[:, i:i+1]
        
    point_cloud.colors = o3d.utility.Vector3dVector(rgb_vals)
        
    gt_entities = [point_cloud]
    
    # Create a sphere at the center of the scene
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    
    # Translate the sphere to the center of the scene
    sphere.translate(center)
    
    # Add the sphere to the list of entities
    gt_entities.append(sphere)
    
    # o3d.visualization.draw_plotly(gt_entities)
    
    return dimensions, center, gt_entities

# Function to divide the scene into n spatial zones
def divide_scene(n, dimensions, center):
    
    # Calculate the per-zone increment
    increment = dimensions / (2 * n)

    # Initialize the output array
    zones = np.zeros((n, 3, 2))

    # Calculate the bounds for each zone
    for i in range(n):
        for axis in range(3):
            lower_bound = center[axis] - (i + 1) * increment[axis]
            upper_bound = center[axis] + (i + 1) * increment[axis]
            zones[i, axis, 0] = lower_bound
            zones[i, axis, 1] = upper_bound

    # print("Zones:\n", zones)
    
    return zones

from plotly.colors import sample_colorscale

# Function to create a box from lower and upper bounds of each zone
def create_box_linetraces(zones):
    line_traces = []
    # Sample colors from a Plotly colorscale
    colors = sample_colorscale("Viridis", [i / len(zones) for i in range(len(zones))])
    
    for i in range(zones.shape[0]):
        lower_bound = zones[i, :, 0]
        upper_bound = zones[i, :, 1]
        points = [
            [lower_bound[0], lower_bound[1], lower_bound[2]],
            [upper_bound[0], lower_bound[1], lower_bound[2]],
            [upper_bound[0], upper_bound[1], lower_bound[2]],
            [lower_bound[0], upper_bound[1], lower_bound[2]],
            [lower_bound[0], lower_bound[1], upper_bound[2]],
            [upper_bound[0], lower_bound[1], upper_bound[2]],
            [upper_bound[0], upper_bound[1], upper_bound[2]],
            [lower_bound[0], upper_bound[1], upper_bound[2]],
        ]
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for line in lines:
            x = [points[line[0]][0], points[line[1]][0]]
            y = [points[line[0]][1], points[line[1]][1]]
            z = [points[line[0]][2], points[line[1]][2]]
            line_traces.append(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines',
                    line=dict(color=colors[i], width=5)  # Increase line width here
                )
            )
    return line_traces

import plotly.graph_objects as go

# Function to plot the scene with the bounding boxes and spatial zones
def plot_scene_go(zones, centers, gt_entities): 
    line_sets = create_box_linetraces(zones)

    # Create point cloud trace
    points = np.asarray(gt_entities[0].points)
    point_cloud_trace = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color='grey', opacity=0.4)  # Adjust point size and color
    )

    # Create trace for ground truth bounding box centers
    bbox_indexes = [center[0] for center in centers]
    bbox_centers = np.array([center[1] for center in centers])
    bbox_classes = [center[2] for center in centers]
    # Create hover text with index and class name
    hover_texts = [f"Index: {idx}<br>Class: {cls}" for idx, cls in zip(bbox_indexes, bbox_classes)]

    # Create trace for ground truth bounding box centers with class names and indexes
    bbox_centers_trace = go.Scatter3d(
        x=bbox_centers[:, 0], y=bbox_centers[:, 1], z=bbox_centers[:, 2],
        mode='markers',
        marker=dict(size=12, color='red', opacity=1),  # Adjust marker size, color, and opacity
        text=hover_texts,  # Add class names and indexes for hover text
        hoverinfo='text',  # Display the text on hover
        name='BBox Centers'
    )

    # Combine all traces
    fig = go.Figure(data=[point_cloud_trace] + line_sets + [bbox_centers_trace])

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='3D Point Cloud with Partitions'
    )

    # Show the plot
    fig.show()

# Function to create a box from lower and upper bounds of each zone
def create_box_linesets(zones):
    line_sets = []
    for i in range(zones.shape[0]):
        lower_bound = zones[i, :, 0]
        upper_bound = zones[i, :, 1]
        points = [
            [lower_bound[0], lower_bound[1], lower_bound[2]],
            [upper_bound[0], lower_bound[1], lower_bound[2]],
            [upper_bound[0], upper_bound[1], lower_bound[2]],
            [lower_bound[0], upper_bound[1], lower_bound[2]],
            [lower_bound[0], lower_bound[1], upper_bound[2]],
            [upper_bound[0], lower_bound[1], upper_bound[2]],
            [upper_bound[0], upper_bound[1], upper_bound[2]],
            [lower_bound[0], upper_bound[1], upper_bound[2]],
        ]
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for all lines
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)
    return line_sets

def plot_scene_o3d(zones, centers, gt_entities):
    line_sets = create_box_linesets(zones)

    for line_set in line_sets:
        gt_entities.append(line_set)
        
    # Add the GT bounding boxes' centers to the list of entities as spheres
    for i, center, classname in centers:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        sphere.translate(center)
        sphere.paint_uniform_color(color_name_to_rgb[class_to_color_name[classname]])
        gt_entities.append(sphere)
        
    o3d.visualization.draw_plotly(gt_entities)
    
    
def get_zone_for_bbox(bbox_center, zones):
    """
    Determines the zone index for a given bounding box center.

    Parameters:
    - bbox_center: A numpy array of shape (3,) representing the center coordinates [x, y, z] of the bounding box.
    - zones: A numpy array of shape (n, 3, 2) representing the zones. 
             n is the number of zones, 3 is the 3 axes (x, y, z), and 2 is the lower and upper bounds for each zone.

    Returns:
    - zone_index: An integer indicating the zone index for the bounding box center. 
                  Returns -1 if the bbox_center does not fall within any zone.
    """
    for i in range(zones.shape[0]):
        if (zones[i, 0, 0] <= bbox_center[0] <= zones[i, 0, 1] and
            zones[i, 1, 0] <= bbox_center[1] <= zones[i, 1, 1] and
            zones[i, 2, 0] <= bbox_center[2] <= zones[i, 2, 1]):
            return i
    return -1  # Return -1 if no zone matches the bounding box center

def get_zones_for_gt_bboxes(scene, n_zones, dataset_type="val", plot_viz=False):
    """
    Determines the zone index for each ground truth bounding box in the scene.

    Parameters:
    - scene: A dictionary containing the scene information.

    Returns:
    - zone_mapping: A dictionary mapping the bounding box index to the zone index.
    """
    bbox_centers = get_gt_bboxes(scene, dataset_type=dataset_type)     # List( (index, center, classname) )
    (scene_dimensions, scene_center, gt_entities) = get_scene_dimensions_center(scene, dataset_type=dataset_type)
    zones = divide_scene(n_zones, scene_dimensions, scene_center)
    
    if plot_viz:
        plot_scene_o3d(zones, bbox_centers, gt_entities)
        plot_scene_go(zones, bbox_centers, gt_entities)
    
    # Initialize the zone mapping
    zone_mapping = {}
    
    # Determine the zone for each bounding box
    for i, center, classname in bbox_centers:
        zone_index = get_zone_for_bbox(center, zones) + 1  # Add 1 to make the zone index 1-based
        zone_mapping[i] = classname, zone_index
        
    # print("Zone Mapping:\n", zone_mapping)
    
    return zone_mapping # Dictionary( bbox_index: (classname, zone_index) )


def get_all_zone_mappings(n=4, dataset_type="val"):
    if dataset_type == "val":
        dataset = val_dataset
        
    elif dataset_type == "train":
        dataset = train_dataset
        
    zone_mappings = {} # Dictionary( scanname: Dictionary( bbox_index: (classname, zone_index) ) )
    
    # print(dataset.scan_names)

    for i, scanname in enumerate(dataset.scan_names):
        # print("Index: ", i, " Scanname: ", scanname)
        
        zone_mapping = get_zones_for_gt_bboxes(i, n, dataset_type=dataset_type)
        
        zone_mappings[scanname] = zone_mapping
        
    return zone_mappings # Dictionary( scanname: Dictionary( bbox_index: (classname, zone_index) ) )

def analyze_spatial_bias_df(csv_file): 
    '''
    Input: csv_file with columns: img_id,scan_name,classname,gt_bbox_index,pred_bbox_index,spatial_zone
    
    Analyze the spatial bias of the predicted bounding boxes in a CSV file.
    '''
    df = pd.read_csv(csv_file)

    # Create a new column 'has_pred_bbox' that is True if the row has a non-NaN value in the 'pred_bbox_index' column
    df['has_pred_bbox'] = ~df['pred_bbox_index'].isna()

    print(df.columns)
    # Count the number of rows with a non-NaN value in the 'pred_bbox_index' column
    num_rows_with_pred_bbox = df['has_pred_bbox'].sum()
    print(f'Number of rows with a predicted bounding box: {num_rows_with_pred_bbox}')
    
    count_by_zone = df.groupby('spatial_zone')['has_pred_bbox'].value_counts().unstack()

    # Replace NaN values in count_by_zone[True] and count_by_zone[False] with 0
    count_by_zone[True] = count_by_zone[True].fillna(0)
    count_by_zone[False] = count_by_zone[False].fillna(0)

    count_by_zone['recall'] = count_by_zone[True] / (count_by_zone[True] + count_by_zone[False])
    display(count_by_zone)

    
    # Plot a bar chart of the recall by spatial zone
    count_by_zone['recall'].plot(kind='bar')
    plt.title('Recall by spatial zone for model with: ' + str(csv_file.split("_18classes_")[-1].split("_spatialzones.csv")[0]))
    plt.xlabel('Spatial zone')
    plt.ylabel('Recall')
    plt.xticks(rotation=0)
    
    
    # Display the recall value as text on top of the bars
    for i, recall in enumerate(count_by_zone['recall']):
        plt.text(i, recall, f'{recall:.2f}', ha='center', va='bottom')
    
    plt.show()
    
    # Count the number of True and False values for each class in each spatial zone
    count_by_class_and_zone = df.groupby(['spatial_zone', 'classname'])['has_pred_bbox'].value_counts().unstack()

    # Replace NaN values in count_by_zone[True] and count_by_zone[False] with 0
    count_by_class_and_zone[True] = count_by_class_and_zone[True].fillna(0)
    count_by_class_and_zone[False] = count_by_class_and_zone[False].fillna(0)

    # Calculate recall (T/(T+F)) for each class in each spatial zone
    count_by_class_and_zone['recall'] = count_by_class_and_zone[True] / (count_by_class_and_zone[True] + count_by_class_and_zone[False])

    '''
    # Iterate through each unique class
    for classname in count_by_class_and_zone.index.get_level_values('classname').unique():
        # Filter the data for the current class
        class_data = count_by_class_and_zone.loc[(slice(None), classname), :]
        
        # Ensure the spatial zones are in order (1, 2, 3, 4)
        class_data = class_data.reindex([1, 2, 3, 4], level='spatial_zone')
        display(class_data)
        
        # Calculate the overall recall as the average of the spatial zone recalls
        overall_recall = class_data['recall'].mean()
        
        # Create a bar chart
        plt.figure(figsize=(10, 6))
        
        # Plot recall for each spatial zone
        plt.bar(class_data.index.get_level_values('spatial_zone'), class_data['recall'], color='skyblue', label='Spatial Zone Recall')
        
        # Add the "Overall" recall as the fifth bar
        plt.bar(5, overall_recall, color='orange', label='Overall Recall')
        
        # Set the title and labels
        plt.title(f'Recall by Spatial Zone for Class: {classname}')
        plt.xlabel('Spatial Zone')
        plt.ylabel('Recall')
        
        # Adjust x-axis ticks to include the fifth "Overall" bar
        plt.xticks([1, 2, 3, 4, 5], ['1', '2', '3', '4', 'Overall'])
        
        # Display the legend
        plt.legend()
        
        # Display the plot
        plt.show()
    '''
    
    # Plotting the heatmap of number of rows with predicted bounding boxes by class and spatial zone:
    # Group by 'classname' and 'spatial_zone' and count where 'has_pred_bbox' is True
    grouped_data = df[df['has_pred_bbox']].groupby(['classname', 'spatial_zone']).size().unstack(fill_value=0)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(grouped_data, annot=True, fmt="d", cmap="YlGnBu", cbar=True)

    # Set the title and labels
    plt.title('Heatmap of Predicted Bounding Boxes by Class and Spatial Zone')
    plt.xlabel('Spatial Zone')
    plt.ylabel('Classname')

    # Display the plot
    plt.show()
    
    # Plotting the heatmap of recall by class and spatial zone:
    # Group by 'classname' and 'spatial_zone' and calculate recall
    recall_data = df.groupby(['classname', 'spatial_zone']).apply(
        lambda x: x['has_pred_bbox'].sum() / len(x)
    ).unstack(fill_value=0)

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(recall_data, annot=True, fmt=".2f", cmap="crest", cbar=True)

    # Set the title and labels
    plt.title('Heatmap of Recall by Class and Spatial Zone')
    plt.xlabel('Spatial Zone')
    plt.ylabel('Classname')

    # Display the plot
    plt.show()