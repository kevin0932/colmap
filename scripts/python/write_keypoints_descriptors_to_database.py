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

# This script exports inlier matches from a COLMAP database to a text file.

import os
import argparse
import sqlite3
import numpy as np
import io
# import scipy
import scipy.io as spio
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", required=True)
    parser.add_argument("--matdata_path", required=True)
    # parser.add_argument("--min_num_matches", type=int, default=15)
    args = parser.parse_args()
    return args


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def main():
    args = parse_args()

    connection = sqlite3.connect(args.database_path)
    cursor = connection.cursor()
    cursorDescriptors = connection.cursor()
    cursorKeypoints = connection.cursor()

    images_id_to_name = {}
    images_name_to_id = {}
    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        image_name = row[2]
        images_id_to_name[image_id] = image_name
        images_name_to_id[image_name] = image_id
        print(image_id, ", ", image_name, "~~~~~~~~~~~~~~~~~~~~~")

        tmp = image_name.split('.')
        fdatapath = os.path.join(args.matdata_path, tmp[0]+'_f.mat')
        fdata = spio.loadmat(fdatapath, squeeze_me=True)
        kpdata = fdata['f']
        print("kpdata.shape = ", kpdata.shape)
        print("kpdata.dtype = ", kpdata.dtype)
        print("kpdata[0,:] = ", kpdata[0,:])
        kpdata = kpdata.astype(np.float32)
        kpdata = kpdata.T
        print("kpdata.dtype = ", kpdata.dtype)
        print("kpdata[0,:] = ", kpdata[0,:])
        scale_cos_orientation = np.multiply(kpdata[:,2], np.cos(kpdata[:,3]))
        scale_sin_orientation = np.multiply(kpdata[:,2], np.sin(kpdata[:,3]))
        colmap_keypoints = np.zeros((kpdata.shape[0], 6), dtype=np.float32)
        colmap_keypoints[:,0] = kpdata[:,0]
        colmap_keypoints[:,1] = kpdata[:,1]
        colmap_keypoints[:,2] = scale_cos_orientation
        colmap_keypoints[:,3] = -scale_sin_orientation
        colmap_keypoints[:,4] = scale_sin_orientation
        colmap_keypoints[:,5] = scale_cos_orientation
        print("colmap_keypoints.shape = ", colmap_keypoints.shape)

        ddatapath = os.path.join(args.matdata_path, tmp[0]+'_d.mat')
        ddata = spio.loadmat(ddatapath, squeeze_me=True)
        descriptordata = ddata['d']
        descriptordata = descriptordata.astype(np.uint8)
        descriptordata = descriptordata.T
        print("descriptordata.shape = ", descriptordata.shape)
        print("descriptordata.dtype = ", descriptordata.dtype)
        print("descriptordata[100,:] = ", descriptordata[0,:])
        # descriptordata = descriptordata.astype(np.uint8)
        # print("descriptordata.dtype = ", descriptordata.dtype)
        # print("descriptordata[0,:] = ", descriptordata[0,:])

        ## remove rows with only zeros descriptor
        colmap_keypoints = colmap_keypoints[~np.all(descriptordata == 0, axis=1)]
        descriptordata = descriptordata[~np.all(descriptordata == 0, axis=1)]
        print("descriptordata.shape = ", descriptordata.shape)
        print("colmap_keypoints.shape = ", colmap_keypoints.shape)

        # image_id = 1
        cursorKeypoints.execute("SELECT * FROM keypoints WHERE image_id=?", (image_id,))
        for rowKp in cursorKeypoints:
            image_id = rowKp[0]
            featureNum = rowKp[1]
            featureDim = rowKp[2]
            features = np.fromstring(rowKp[3], dtype=np.float32).reshape(-1, 6)
            image_name = images_id_to_name[image_id]
            print(image_id, ", ", image_name, ", ", featureNum, ", ",featureDim, ", ", features.shape)
        cursorKeypoints.execute('''UPDATE keypoints SET rows = ? WHERE image_id = ?''', (colmap_keypoints.shape[0], image_id))
        cursorKeypoints.execute('''UPDATE keypoints SET cols = ? WHERE image_id = ?''', (colmap_keypoints.shape[1], image_id))
        # dummydata = np.zeros((2,6), dtype=np.float32)
        cursorKeypoints.execute('''UPDATE keypoints SET data = ? WHERE image_id = ?''', (colmap_keypoints.tostring(), image_id))
        cursorKeypoints.execute("SELECT * FROM keypoints WHERE image_id=?", (image_id,))
        for rowKp in cursorKeypoints:
            image_id = rowKp[0]
            featureNum = rowKp[1]
            featureDim = rowKp[2]
            features = np.fromstring(rowKp[3], dtype=np.float32).reshape(-1, 6)
            image_name = images_id_to_name[image_id]
            print(image_id, ", ", image_name, ", ", featureNum, ", ",featureDim, ", ", features.shape)


        cursorDescriptors.execute("SELECT * FROM descriptors WHERE image_id=?;", (image_id,))
        for rowDes in cursorDescriptors:
            image_id = rowDes[0]
            descriptorNum = rowDes[1]
            descriptorDim = rowDes[2]
            descriptors = np.fromstring(rowDes[3],dtype=np.uint8).reshape(-1, 128)
            print(image_id, ", ", image_name, ", ", descriptorNum, ", ",descriptorDim, ", ", descriptors.shape)
        cursorDescriptors.execute('''UPDATE descriptors SET rows = ? WHERE image_id = ?''', (descriptordata.shape[0], image_id))
        cursorDescriptors.execute('''UPDATE descriptors SET cols = ? WHERE image_id = ?''', (descriptordata.shape[1], image_id))
        # dummydata = np.zeros((2,128), dtype=np.float32)
        cursorDescriptors.execute('''UPDATE descriptors SET data = ? WHERE image_id = ?''', (descriptordata.tostring(), image_id))
        cursorDescriptors.execute("SELECT * FROM descriptors WHERE image_id=?;", (image_id,))
        for rowDes in cursorDescriptors:
            image_id = rowDes[0]
            descriptorNum = rowDes[1]
            descriptorDim = rowDes[2]
            descriptors = np.fromstring(rowDes[3],dtype=np.uint8).reshape(-1, 128)
            print(image_id, ", ", image_name, ", ", descriptorNum, ", ",descriptorDim, ", ", descriptors.shape)


        connection.commit()
        # break

    cursor.close()
    cursorDescriptors.close()
    cursorKeypoints.close()
    connection.close()


if __name__ == "__main__":
    main()
