"""
THis module get ground truth annotation of LIDAR BEV from corresponding
camera-view annotation, which can be further used for training road detection
models.
"""
import os
import cv2

import numpy as np

def read_pcd(pcd_path):
    pcd = pypcd.PointCloud.from_path(pcd_path)
    x = np.transpose(pcd.pc_data["x"])
    y = np.transpose(pcd.pc_data["y"])
    z = np.transpose(pcd.pc_data["z"])
    return x, y, z

class PointCloudFilter(object):
    """
    Class for getting lidar-bev ground truth annotation from camera-view
    annotation.
    :param res: float. resolution in meters. Each output pixel will
                represent an square region res x res in size.
    :param side_range: tuple of two floats. (left-most, right_most)
    :param fwd_range: tuple of two floats. (back-most, forward-most)
    :param height_range: tuple of two floats. (min, max)
    :param calib: class instance for getting transform matrix from
                  calibration.
    """
    def __init__(self,
                 side_range=(-39.68, 39.68),
                 fwd_range=(0, 69.12),
                 height_range=(-2., -2.),
                 res=0.10
                ):
        self.res = res
        self.side_range = side_range
        self.fwd_range = fwd_range
        self.height_range = height_range
        self.calib = KittiCalibration()

    def set_range_patameters(self, side_range, fwd_range, height_range):
        self.side_range = side_range
        self.fwd_range = fwd_range
        self.height_range = height_range

    def read_bin(self, path):
        """
        Helper function to read one frame of lidar pointcloud in .bin format.
        :param path: where pointcloud is stored in .bin format.
        :return: (x, y, z, intensity) of pointcloud, N x 4.
        """
        points = np.fromfile(path, dtype=np.float32, count=-1).reshape([-1, 4])
        x_points, y_points, z_points, indices = self.get_pcl_range(points)
        filtered_points = np.concatenate((x_points[:,np.newaxis], y_points[:,np.newaxis], z_points[:,np.newaxis]), axis = 1)
        return filtered_points

    def scale_to_255(self, value, minimum, maximum, dtype=np.uint8):
        """
        Scales an array of values from specified min, max range to 0-255.
        Optionally specify the data type of the output (default is uint8).
        """
        if minimum!= maximum:
            return (((value - minimum) / float(maximum - minimum))
                    * 255).astype(dtype)
        else:
            return self.get_meshgrid()

    def get_pcl_range(self, points):
        """
        Get the pointcloud wihtin side_range and fwd_range.
        :param points: np.float, N x 4. each column is [x, y, z, intensity].
        :return: [x, y, z, intensity] of filtered points and corresponding
                 indices.
        """
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]
        indices = []
        for i in range(points.shape[0]):
            if points[i, 0] > self.fwd_range[0] and points[i, 0] < self.fwd_range[1]:
                if points[i, 1]  > self.side_range[0] and points[i, 1] < self.side_range[1]:
                    indices.append(i)

        indices = np.array(indices)
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        x_points = x_points[indices]
        y_points = y_points[indices]
        z_points = z_points[indices]
        return x_points, y_points, z_points, indices

    def clip_height(self, z_points):
        """
        Clip the height between (min, max).
        :param z_points: z_points from get_pcl_range
        :return: clipped height between (min,max).
        """
        height = np.clip(
            a=z_points, a_max=self.height_range[1], a_min=self.height_range[0]
        )
        return height

    def get_meshgrid(self):
        """
        Create mesh grids (size: res x res) in the x-y plane of the lidar
        :return: np.array: uint8, x-y plane mesh grids based on resolution.
        """
        x_max = 1 + int((self.side_range[1] - self.side_range[0]) / self.res)
        y_max = 1 + int((self.fwd_range[1] - self.fwd_range[0]) / self.res)
        img = np.zeros([y_max, x_max], dtype=np.uint8)
        return img

    def pcl2xy_plane(self, x_points, y_points):
        """
        Convert the lidar coordinate to x-y plane coordinate.
        :param x_points: x of points in lidar coordinate.
        :param y_points: y of points in lidar coordinate.
        :return: corresponding pixel position based on resolution.
        """
        x_img = (-y_points / self.res).astype(np.int32) # x axis is -y in lidar
        y_img = (-x_points / self.res).astype(np.int32) # y axis is -x in lidar
        # shift pixels to have minimum be (0,0)
        x_img -= int(np.floor(self.side_range[0] / self.res))
        y_img += int(np.ceil(self.fwd_range[1] / self.res))
        return x_img, y_img

    def pcl_2_bev(self, points):
        """
        Creates an 2D birds eye view representation of the pointcloud.
        :param points: np.float, N x 4. input pointcloud matrix,
                       each column is [x, y, z, intensity]
        :return: np.array, representing an image of the BEV.
        """
        # rescale the height values - to be between the range 0-255
        x_points, y_points, z_points, _ = self.get_pcl_range(points)
        x_img, y_img = self.pcl2xy_plane(x_points, y_points)
        height = self.clip_height(z_points)
        bev_img = self.get_meshgrid()
        pixel_values = self.scale_to_255(height,
                                         self.height_range[0],
                                         self.height_range[1])
        # fill pixel values in image array
        
        x_img = np.clip(x_img, 0, bev_img.shape[1] - 1)
        y_img = np.clip(y_img, 0, bev_img.shape[0] - 1)
        print(bev_img.shape)
        bev_img[y_img, x_img] = 255
        return bev_img

    def pcl2anno(self, points, calib_file, front_view_anno):
        """
        Project lidar pointcloud to corresponding front-view annotation image
        then determine which of its points belong to the free road.
        :param points: np.float, N x 4. input pointcloud matrix,which
                       you want to project to front-view annotation image.
        :param calib_file: pointcloud corresponding calibration file.
        :param front_view_anno: corresponding annotated front-view img,
                                violet region represent free road surface.
        :return: index of points on/out free road surface.
        """
        cabri_tr = self.calib.read_from_file(calib_file)
        img_anno = cv2.imread(front_view_anno, -1)
        height, width, _ = img_anno.shape
        # Project points_vel in Velodyne coordinates to the points points_img in
        # color image on the right coordinate.
        points_vel = np.hstack((points[:, :3], np.ones((len(points), 1))))
        points_img = np.dot(cabri_tr, points_vel.T)
        point_road_index = []
        point_out_road_index = []
        for i in range(len(points_vel)):
            # look forward
            if points_vel.T[0][i] > 1.5 and points_vel.T[0][i] < 300:
                arr = points_img[:, i]
                (u_pix, v_pix) = (arr[0] / arr[2], arr[1] / arr[2])
                if ((u_pix > 0 and u_pix < width) and
                    (v_pix > 0 and v_pix < height)):
                    pixel_value = img_anno[int(v_pix), int(u_pix)]
                    if np.all(pixel_value == [255, 0, 255]):
                        point_road_index.append(i)
                    if np.all(pixel_value == [0, 0, 255]):
                        point_out_road_index.append(i)
        return point_road_index, point_out_road_index

    def pcl_anno_bev(self, points, point_road_index, point_out_road_index):
        """
        :param points: np.float, N x 4. input pointcloud matrix.
        :param point_road_index: index of points on free road surface.
        :param point_out_road_index: index of points out free road surface.
        :return: BEV of lidar pointcloud containing free road annotation. 
        """
        x_points, y_points, _, _, indices = self.get_pcl_range(points)
        on_road = []
        out_road = []
        for i in range(len(indices)):
            if indices[i] in point_road_index:
                on_road.append(i)
            if indices[i] in point_out_road_index:
                out_road.append(i)
        pcl_anno_bev_img = self.get_meshgrid()
        x_points_road = x_points[on_road]
        y_points_road = y_points[on_road]
        x_road_img, y_road_img = self.pcl2xy_plane(x_points_road,
                                                   y_points_road)
        x_points_out_road = x_points[out_road]
        y_points_out_road = y_points[out_road]
        x_out_road_img, y_out_road_img = self.pcl2xy_plane(x_points_out_road,
                                                           y_points_out_road)

        pcl_anno_bev_img[y_road_img, x_road_img] = 255
        pcl_anno_bev_img[y_out_road_img, x_out_road_img] = 100
        return pcl_anno_bev_img

    @staticmethod
    def add_color_pcd(path1, path2, point_road_index):
        """
        Helper function to use different color to distinguish points on free 
        road or out free road.
        :param path1: where pointcloud is stored in .pcd format.
        :param path2: storage path after function add_color_pcd.
        :param point_road_index: index of points on free road in pointcloud 
                                 matrix.
        """
        with open(path1, 'rb') as fopen:
            lines = fopen.readlines()
        lines[2] = lines[2].split('\n')[0] + ' rgb\n'
        lines[3] = lines[3].split('\n')[0] + ' 4\n'
        lines[4] = lines[4].split('\n')[0] + ' I\n'
        lines[5] = lines[5].split('\n')[0] + ' 1\n'
        for i in range(len(lines)-11):
            if i in point_road_index:
                # 0xFFFF00: yellow, free road surface;
                # 0xFF0019: red, non-free road surface.
                lines[i + 11] = lines[i + 11].split('\n')[0] + ' ' + str(
                    0xFFFF00) + '\n'
            else:
                lines[i + 11] = lines[i + 11].split('\n')[0] + ' ' + str(
                    0xFF0019) + '\n'
        with open(path2, 'wb') as fwrite:
            fwrite.writelines(lines)

    @staticmethod
    def img_overlay_display(img1, img2):
        """
        Helper function to overlay two images to display.
        :param img1, img2: Images you want to overlay to display.
        :return: added image.
        """
        # img_1 has 3 channels, img2 has 1. 
        img_2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), 
                           interpolation=cv2.INTER_AREA)
        img_1 = np.array([[int(sum(img1[i][j])/3) for j in range(len(img1[i]))]
                          for i in range(len(img1))], dtype=np.uint8)
        alpha = 0.3
        beta = 1 - alpha
        gamma = 0
        img_add = cv2.addWeighted(img_1, alpha, img_2, beta, gamma)
        return img_add

    def get_line(self, original_point, box, x_0, y_0, pixel_x):
        """
        Assume that the obstacles and the areas behind the obstacles are all
        non-free road.
        Get a fit line from the lidar emission point to the obstacle boundary.       
        :param original_point: position of lidar emission point.
        :param box: representation of obstacle.
        :param x_0: boundary of box.
        :param y_0: boundary of box
        :param pixel_x: x range of area behind the obstacle.
        :return: a fit line, that is y range of area behind the obstacle.
        """
        y_line = int(original_point[1] -
                     (original_point[0] - pixel_x)*
                     (original_point[1] - box[1][y_0]) /
                     (original_point[0] - box[0][x_0])) + 1
        return y_line
    
    def anno_cor_pcl(self, anno_ipm, pcl_bev, r):
        '''
        annotation IPM correction by using point cloud projection.
        :param anno_ipm: ground truth annotation under bird's eye view.
        :param pcl_bev: BEV of lidar pointcloud.
        :param r: remove threshold of pointcloud noise.
        :return anno_ipm: corrected ground truth annotation.
        '''
        height, width, _ = anno_ipm.shape
        original_point = [height + np.floor(self.fwd_range[0] / self.res), width / 2]
        pcl_bev = cv2.resize(pcl_bev, (width, height), interpolation=cv2.INTER_AREA)
        mat = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                if (all(anno_ipm[i, j] == [255, 0, 255]) and
                        pcl_bev[i, j] == 100):
                    mat[i, j] = 1
        num_lst = []
        idx_lst = []
        for i in range(height):
            for j in range(width):
                mat_surr_r = mat[max(i - r, 0):min(i + r, mat.shape[0]) + 1,
                                 max(0, j - r):min(j + r, mat.shape[1]) + 1]
                surr_num = sum(sum(mat_surr_r))
                if surr_num > 3:
                    num_lst.append(surr_num)
                    idx_lst.append([i, j])
        if idx_lst == []:
            print('No pixels can be corrected!')
            return anno_ipm
        class_num = 2
        my = Hierarchical(class_num)
        my.fit(idx_lst)
        idx_lst = np.array(idx_lst)
        img = np.zeros((height, width), dtype=np.uint8)
        for num in range(class_num):
            one_class = idx_lst[np.where(np.array(my.labels) == num)]
            l = 1
            sum_num = 0
            x = [min(item[0] for item in one_class),
                 max(item[0] for item in one_class)]
            y = [min(item[1] for item in one_class),
                 max(item[1] for item in one_class)]
            while True:
                mat_surr_l = mat[max(x[0]-l, 0):min(x[1]+l, mat.shape[0]) + 1,
                                 max(0, y[0]-l):min(y[1]+l, mat.shape[1]) + 1]
                surr_num = sum(sum(mat_surr_l))
                if surr_num > sum_num:
                    sum_num = surr_num
                    l += 1
                else:
                    box = [[max(x[0]-(l-1), 0), min(x[1]+(l-1), mat.shape[0])],
                           [max(0, y[0]-(l-1)), min(y[1]+(l-1), mat.shape[1])]]
                    if max(0, y[0] - (l - 1)) > 99:
                        for i in range(min(box[0][1] + 1, height)):
                            y1 = self.get_line(original_point, box, 0, 0, i) + 1
                            y2 = self.get_line(original_point, box, 1, 1, i)
                            for j in range(max(y1, 0), min(y2, width)):
                                img[i, j] = 255
                    else:
                        if min(y[1] + (l - 1), mat.shape[1]) < 100:
                            for i in range(min(box[0][1] + 1, height)):
                                y1 = self.get_line(original_point, box, 1, 0, i) + 1
                                y2 = self.get_line(original_point, box, 0, 1, i)
                                for j in range(max(y1, 0), min(y2, width)):
                                    img[i, j] = 255
                        else:
                            for i in range(min(box[0][1] + 1, height)):
                                y1 = self.get_line(original_point, box, 1, 0, i) + 1
                                y2 = self.get_line(original_point, box, 1, 1, i)
                                for j in range(max(y1, 0), min(y2, width)):
                                    img[i, j] = 255
                    break
        anno_ipm[img == 255] = [0, 0, 255]
        return anno_ipm

    def get_bev_image(self, velodyne_path):
        if not os.path.exists(velodyne_path):
            raise ValueError(velodyne_path, "not Found")
        # filtered_points = self.read_bin(velodyne_path)
        x, y, z = read_pcd(velodyne_path)
        filtered_points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=-1)
        bev_image = self.pcl_2_bev(filtered_points)
        bev_image = cv2.merge([bev_image, bev_image, bev_image])
        return bev_image

class KittiCalibration(object):
    """
    Get transform matrix between different coordinate systems according to 
    Kitti calibration.
    :param p2: projection matrix after rectification.
    :param r0_rect: rectifying rotation matrix.
    :param tr_velo_to_cam: translation from velodyne to camera.
    :param tr: transform matrix
    """
    p2 = None
    r0_rect = None
    tr_velo_to_cam = None
    tr = None

    def __init__(self):
        """
        """
        pass

    def read_kitti_calib(self, filename, dtype='f8'):
        """
        Read data from calibration file.
        :param filename: KITTI calibration.
        :return: calibration content.
        """
        outdict = dict()
        output = open(filename, 'r')
        allcontent = output.readlines()
        output.close()
        for content_raw in allcontent:
            content = content_raw.strip()
            if content == '':
                continue
            if content[0] != '#':
                tmp = content.split(':')
                assert len(tmp) == 2, 'wrong file format'
                var = tmp[0].strip()
                values = np.array(tmp[-1].strip().split(' '), dtype)
                outdict[var] = values
        return outdict

    def setup(self, dict_kitti_stuff):
        """
        Get transform matrix from calibration content.
        :param dict_kitti_stuff: calibration content.  
        :return: transform matrix.
        """
        dtype_str = 'f8'
        self.p2 = np.matrix(dict_kitti_stuff['P2']).reshape((3, 4))
        r0_rect_raw = np.array(dict_kitti_stuff['R0_rect']).reshape((3, 3))
        # Rectification Matrix
        r0_rect_raw = np.vstack((r0_rect_raw, np.zeros((1, 3), dtype_str)))
        self.r0_rect = np.hstack((r0_rect_raw, np.zeros((4, 1), dtype_str)))
        self.r0_rect[3, 3] = 1.
        # intermediate result
        r2_1 = np.dot(self.p2, self.r0_rect)
        tr_velo_to_cam_raw = np.array(
            dict_kitti_stuff['Tr_velo_to_cam']).reshape(3, 4)
        self.tr_velo_to_cam = np.vstack((tr_velo_to_cam_raw,
                                         np.zeros((1, 4), dtype_str)))
        self.tr_velo_to_cam[3, 3] = 1.
        self.tr = np.dot(r2_1, self.tr_velo_to_cam)
        return self.tr

    def read_from_file(self, fn=None):
        """
        Get transform matrix from calibration file.
        :param fn: KITTI calibration.
        :return: transform matrix.
        """
        assert fn != None, 'Problem! filename must be != None'
        cur_calib_stuff_dict = self.read_kitti_calib(fn)
        self.tr = self.setup(cur_calib_stuff_dict)
        return self.tr

    @staticmethod
    def get_transform_matrix(lidar_calib_file, cam_calib_file):

        with open(cam_calib_file, 'r') as f:
            for line in f.readlines():
                if (line.split(' ')[0] == 'R_rect_00:'):
                    R0_rect = np.zeros((4, 4))
                    R0_rect[:3, :3] = np.array(line.split('\n')[0].split(' ')[1:]).astype(float).reshape(3,3)
                    R0_rect[3, 3] = 1
        with open(lidar_calib_file, 'r') as f:
            for line in f.readlines():
                if (line.split(' ')[0] == 'R:'):
                    R = np.array(line.split('\n')[0].split(' ')[1:]).astype(float).reshape(3,3)
                if (line.split(' ')[0] == 'T:'):
                    T = np.array(line.split('\n')[0].split(' ')[1:]).astype(float)
            Tr_velo_to_cam = np.zeros((4, 4))
            Tr_velo_to_cam[:3, :3] = R
            Tr_velo_to_cam[:3, 3] = T
            Tr_velo_to_cam[3, 3] = 1

        vel_to_cam = np.dot(R0_rect, Tr_velo_to_cam) 
        cam_to_vel = np.linalg.inv(vel_to_cam) 
        return vel_to_cam, cam_to_vel, R0_rect, Tr_velo_to_cam

    @staticmethod
    def get_transform_matrix_origin(calib_file):
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                if (line.split(' ')[0] == 'P2:'):
                    P2 = np.array(line.split('\n')[0].split(' ')[1:]).astype(float).reshape(3,4)
                if (line.split(' ')[0] == 'R0_rect:'):
                    R0_rect = np.zeros((4, 4))
                    R0_rect[:3, :3] = np.array(line.split('\n')[0].split(' ')[1:]).astype(float).reshape(3,3)
                    R0_rect[3, 3] = 1
                if (line.split(' ')[0] == 'Tr_velo_to_cam:'):
                    Tr_velo_to_cam = np.zeros((4, 4))
                    Tr_velo_to_cam[:3, :4] = np.array(line.split('\n')[0].split(' ')[1:]).astype(float).reshape(3,4)
                    Tr_velo_to_cam[3, 3] = 1

            vel_to_cam = np.dot(R0_rect,Tr_velo_to_cam)
            cam_to_vel = np.linalg.inv(vel_to_cam) 
            return vel_to_cam, cam_to_vel


def get_object_corners_in_lidar(cam_to_vel, dimensions, location, rotation_y):
    tr_matrix = np.zeros((2, 3))
    rotation_y = -rotation_y
    tr_matrix[:2, :2] = np.array([np.cos(rotation_y), -np.sin(rotation_y), np.sin(rotation_y), np.cos(rotation_y)]).astype(float).reshape(2,2)
    tr_matrix[:2, 2] = np.array([location[0], location[2]]).astype(float).reshape(1,2)
    dimensions = 0.5 * dimensions
    corner_points_2d = np.array([dimensions[2], dimensions[1], 1.0, dimensions[2], -dimensions[1], 1.0, -dimensions[2], dimensions[1], 1.0, -dimensions[2], -dimensions[1], 1.0]).astype(float).reshape(4,3).T
    corner_points_2d = np.dot(tr_matrix, corner_points_2d).T
    corner_points = np.ones((4, 4))
    corner_points[:, 0] = corner_points_2d[:, 0]
    corner_points[:, 2] = corner_points_2d[:, 1]
    corner_points = np.dot(corner_points, cam_to_vel.T)[:, :3]
    return corner_points

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

def _read_imageset_file(kitti_root, path):
    imagetxt = os.path.join(kitti_root, path)
    with open(imagetxt, 'r') as f:
        lines = f.readlines()
    total_img_ids = [int(line) for line in lines]
    img_ids = []
    for img_id in total_img_ids:
        if "test" in path:
            img_path = os.path.join(kitti_root, "testing/image_2", "{:06d}".format(img_id) + ".png")
        else:
            img_path = os.path.join(kitti_root, "training/image_2", "{:06d}".format(img_id) + ".png")
        if os.path.exists(img_path):
            img_ids.append(img_id)
    return img_ids
