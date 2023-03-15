import cv2
import numpy as np

def compute_corners_3d(dim, rotation_y):
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    l, w, h = dim[0], dim[1], dim[2]
    x_corners = [-l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2]
    y_corners = [w / 2, w / 2, w / 2, w / 2, -w / 2, -w / 2, -w / 2, -w / 2]
    z_corners = [h, h, 0, 0, h, h, 0, 0]
    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners).transpose(1, 0)
    return corners_3d

def compute_box_3d(dim, location, rotation_y):
    corners_3d = compute_corners_3d(dim, rotation_y)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(1, 3)
    return corners_3d

def get_cam_8_points(gt_boxes, r_velo2cam, t_velo2cam):
    camera_8_points_list = []
    for idx in range(gt_boxes.shape[0]):
        gt_box = gt_boxes[idx]
        lwh, loc, yaw_lidar = gt_box[3:6], gt_box[:3], gt_box[6]
        l, w, h = lwh
        x, y, z = loc
        z = z - h / 2
        bottom_center = [x, y, z]
        obj_size = [l, w, h]
        lidar_8_points = compute_box_3d(obj_size, bottom_center, yaw_lidar)
        camera_8_points = r_velo2cam * np.matrix(lidar_8_points).T
        camera_8_points = camera_8_points + t_velo2cam[:, np.newaxis]
        camera_8_points_list.append(camera_8_points.T)
    return camera_8_points_list

def points_cam2img(points_3d, calib_intrinsic, with_depth=False):
    points_num = list(points_3d.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0)
    points_4 = np.concatenate((points_3d, np.ones(points_shape)), axis=-1)
    point_2d = np.matmul(calib_intrinsic, points_4.T.swapaxes(1, 2).reshape(4, -1))
    point_2d = point_2d.T.reshape(-1, 8, 4)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    if with_depth:
        return np.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res

def plot_rect3d_on_img(img, num_rects, rect_corners, color=(0, 255, 0), thickness=1):
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for start, end in line_indices:
            radius = 5
            color = (0, 0, 250)
            thickness = 1
            cv2.circle(img, (corners[start, 0], corners[start, 1]), radius, color, thickness)
            cv2.circle(img, (corners[end, 0], corners[end, 1]), radius, color, thickness)
            color = (0, 255, 0)
            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return img.astype(np.uint8)

def vis_label_in_img(img, camera_8_points_list, calib_intrinsic):
    cam8points = np.array(camera_8_points_list)
    num_bbox = cam8points.shape[0]
    uv_origin = points_cam2img(cam8points, calib_intrinsic)
    uv_origin = (uv_origin - 1).round()
    plot_rect3d_on_img(img, num_bbox, uv_origin)
    return img

def demo(img_pth, gt_boxes, r_velo2cam, t_velo2cam, calib_intrinsic):
    P = np.eye(4)
    P[:3, :3] = calib_intrinsic
    img = cv2.imread(img_pth)
    camera_8_points_list = get_cam_8_points(gt_boxes, r_velo2cam, t_velo2cam)
    img = vis_label_in_img(img, camera_8_points_list, P)
    cv2.imwrite("debug/debug.jpg", img)
    return img