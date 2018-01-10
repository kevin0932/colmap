# COLMAP - Structure-from-Motion and Multi-View Stereo.
# Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import collections
import numpy as np
import struct

import math

import numpy as np
import functools

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
ImagePairGT = collections.namedtuple(
    "ImagePairGT", ["id1", "id2", "qvec12", "tvec12", "camera_id1", "name1", "camera_id2", "name2", "rotmat12"])

def read_relative_poses_text(path):
    image_pair_gt = {}
    dummy_image_pair_id = 1
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id1 = int(elems[0])
                image_id2 = int(elems[1])
                qvec12 = np.array(tuple(map(float, elems[2:6])))
                tvec12 = np.array(tuple(map(float, elems[6:9])))
                camera_id1 = int(elems[9])
                image_name1 = elems[10]
                camera_id2 = int(elems[11])
                image_name2 = elems[12]
                rotmat_r1 = np.array(tuple(map(float, elems[13:16])))
                rotmat_r2 = np.array(tuple(map(float, elems[16:19])))
                rotmat_r3 = np.array(tuple(map(float, elems[19:22])))
                RelativeRotationMat = np.array([rotmat_r1, rotmat_r2, rotmat_r3])
                # print("RelativeRotationMat.shape = ", RelativeRotationMat.shape)
                image_pair_gt[dummy_image_pair_id] = ImagePairGT(id1=image_id1, id2=image_id2, qvec12=qvec12, tvec12=tvec12, camera_id1=camera_id1, name1=image_name1, camera_id2=camera_id2, name2=image_name2, rotmat12 = RelativeRotationMat)
                dummy_image_pair_id += 1
    return image_pair_gt


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=4, model_name="RADIAL", num_params=5),
    CameraModel(model_id=5, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=6, model_name="OPENCV", num_params=8),
    CameraModel(model_id=7, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=8, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=9, model_name="FOV", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
        relative_poses_pairGT = read_relative_poses_text(os.path.join(path, "relative_poses") + ext)
        return cameras, images, points3D, relative_poses_pairGT
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
        return cameras, images, points3D


# http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.184.3942&rep=rep1&type=pdf
def quaternion2RotMat(qw, qx, qy, qz):
    sqw = qw*qw
    sqx = qx*qx
    sqy = qy*qy
    sqz = qz*qz

    # invs (inverse square length) is only required if quaternion is not already normalised
    invs = 1 / (sqx + sqy + sqz + sqw)
    m00 = ( sqx - sqy - sqz + sqw)*invs     # since sqw + sqx + sqy + sqz =1/invs*invs
    m11 = (-sqx + sqy - sqz + sqw)*invs
    m22 = (-sqx - sqy + sqz + sqw)*invs

    tmp1 = qx*qy;
    tmp2 = qz*qw;
    m10 = 2.0 * (tmp1 + tmp2)*invs
    m01 = 2.0 * (tmp1 - tmp2)*invs

    tmp1 = qx*qz
    tmp2 = qy*qw
    m20 = 2.0 * (tmp1 - tmp2)*invs
    m02 = 2.0 * (tmp1 + tmp2)*invs
    tmp1 = qy*qz
    tmp2 = qx*qw
    m21 = 2.0 * (tmp1 + tmp2)*invs ;
    m12 = 2.0 * (tmp1 - tmp2)*invs

    rotation_matrix = np.array( [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]] )
    return rotation_matrix


def euler2quat(z=0, y=0, x=0):
    ''' Return quaternion corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    quat : array shape (4,)
       Quaternion in w, x, y z (real, then vector) format

    Notes
    -----
    We can derive this formula in Sympy using:

    1. Formula giving quaternion corresponding to rotation of theta radians
       about arbitrary axis:
       http://mathworld.wolfram.com/EulerParameters.html
    2. Generated formulae from 1.) for quaternions corresponding to
       theta radians rotations about ``x, y, z`` axes
    3. Apply quaternion multiplication formula -
       http://en.wikipedia.org/wiki/Quaternions#Hamilton_product - to
       formulae from 2.) to give formula for combined rotations.
    '''
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])



def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

def euler2angle_axis(z=0, y=0, x=0):
    ''' Return angle, axis corresponding to these Euler angles

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    theta : scalar
       angle of rotation
    vector : array shape (3,)
       axis around which rotation occurs

    Examples
    --------
    >>> theta, vec = euler2angle_axis(0, 1.5, 0)
    >>> print(theta)
    1.5
    >>> np.allclose(vec, [0, 1, 0])
    True
    '''
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return nq.quat2angle_axis(euler2quat(z, y, x))

def main():
    if len(sys.argv) != 3:
        # print("Usage: python read_model.py path/to/model/folder [.txt,.bin]")
        print("Usage: python read_model.py path/to/model/folder .txt")
        return

    cameras, images, points3D, relative_poses_pairGT = read_model(path=sys.argv[1], ext=sys.argv[2])

    outputPath = sys.argv[1]

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))
    print("num_relative_poses_pairGT:", len(relative_poses_pairGT))

    #Image = collections.namedtuple(
    #    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    print(images[1])
    print(images[1].qvec)
    print(images[1].tvec)
    print(images[1].tvec[0])
    print(type(images))

    for imgIdx, val in images.items():
        if val.name == "P1180142.JPG":
            print("the image id of name P1180141.JPG is ", imgIdx)


    outputGTfilepath = outputPath+'/southbuilding_RtAngleAxis_groundtruth_from_colmap.txt'

    GT_file = open(outputGTfilepath,'w')

    for i in range(len(images)):
        imgID = i+1

        tmpRotMat = quaternion2RotMat(images[imgID].qvec[0], images[imgID].qvec[1], images[imgID].qvec[2], images[imgID].qvec[3])
        #print(tmpRotMat.shape)
        eulerAnlges = mat2euler(tmpRotMat)
        recov_angle_axis_result = euler2angle_axis(eulerAnlges[0], eulerAnlges[1], eulerAnlges[2])
        R_angleaxis = recov_angle_axis_result[0]*(recov_angle_axis_result[1])
        R_angleaxis = np.array(R_angleaxis, dtype=np.float32)

        GT_file.write("%s %s %s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" % (images[imgID].id, images[imgID].camera_id, images[imgID].name, images[imgID].qvec[0], images[imgID].qvec[1], images[imgID].qvec[2], images[imgID].qvec[3], images[imgID].tvec[0], images[imgID].tvec[1], images[imgID].tvec[2], tmpRotMat[0,0], tmpRotMat[0,1], tmpRotMat[0,2], tmpRotMat[1,0], tmpRotMat[1,1], tmpRotMat[1,2], tmpRotMat[2,0], tmpRotMat[2,1], tmpRotMat[2,2], R_angleaxis[0], R_angleaxis[1], R_angleaxis[2]))
        # GT_file.write("%d %d %s %f %f %f %f %f %f %f\n" % (images[imgID].id, images[imgID].camera_id, images[imgID].name, images[imgID].qvec[0], images[imgID].qvec[1], images[imgID].qvec[2], images[imgID].qvec[3], images[imgID].tvec[0], images[imgID].tvec[1], images[imgID].tvec[2]))
        # GT_file.write("%d %d " % (images[imgID].id, images[imgID].camera_id) + images[imgID].name + "\n")

    GT_file.close()

    # test if theia and colmap have the same transformation systems
    outputGTfilepath_2 = '/home/kevin/JohannesCode/southbuilding_RtAngleAxis_groundtruth_from_colmap.txt'

    GT_file_2 = open(outputGTfilepath_2,'w')

    for i in range(len(images)):
        imgID = i+1

        tmpRotMat = quaternion2RotMat(images[imgID].qvec[0], images[imgID].qvec[1], images[imgID].qvec[2], images[imgID].qvec[3])
        #print(tmpRotMat.shape)
        eulerAnlges = mat2euler(tmpRotMat)
        recov_angle_axis_result = euler2angle_axis(eulerAnlges[0], eulerAnlges[1], eulerAnlges[2])
        R_angleaxis = recov_angle_axis_result[0]*(recov_angle_axis_result[1])
        R_angleaxis = np.array(R_angleaxis, dtype=np.float32)

        GT_file_2.write("%s %s %s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" % (images[imgID].id, images[imgID].camera_id, images[imgID].name, images[imgID].qvec[0], images[imgID].qvec[1], images[imgID].qvec[2], images[imgID].qvec[3], images[imgID].tvec[0], images[imgID].tvec[1], images[imgID].tvec[2], tmpRotMat[0,0], tmpRotMat[0,1], tmpRotMat[0,2], tmpRotMat[1,0], tmpRotMat[1,1], tmpRotMat[1,2], tmpRotMat[2,0], tmpRotMat[2,1], tmpRotMat[2,2], R_angleaxis[0], R_angleaxis[1], R_angleaxis[2]))
        # GT_file_2.write("%d %d %s %f %f %f %f %f %f %f\n" % (images[imgID].id, images[imgID].camera_id, images[imgID].name, images[imgID].qvec[0], images[imgID].qvec[1], images[imgID].qvec[2], images[imgID].qvec[3], images[imgID].tvec[0], images[imgID].tvec[1], images[imgID].tvec[2]))
        # GT_file_2.write("%d %d " % (images[imgID].id, images[imgID].camera_id) + images[imgID].name + "\n")

    GT_file_2.close()



if __name__ == "__main__":
    main()
