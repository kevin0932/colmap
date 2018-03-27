// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_BASE_RECONSTRUCTION_H_
#define COLMAP_SRC_BASE_RECONSTRUCTION_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "base/camera.h"
#include "base/database_cache.h"
#include "base/image.h"
#include "base/point2d.h"
#include "base/point3d.h"
#include "base/track.h"
#include "util/alignment.h"
#include "util/types.h"

namespace colmap {

// Reconstruction class holds all information about a single reconstructed
// model. It is used by the mapping and bundle adjustment classes and can be
// written to and read from disk.
class Reconstruction {
 public:
  Reconstruction();

  // Get number of objects.
  inline size_t NumCameras() const;
  inline size_t NumImages() const;
  inline size_t NumRegImages() const;
  inline size_t NumPoints3D() const;
  inline size_t NumImagePairs() const;

  // Get const objects.
  inline const class Camera& Camera(const camera_t camera_id) const;
  inline const class Image& Image(const image_t image_id) const;
  inline const class Point3D& Point3D(const point3D_t point3D_id) const;
  inline const std::pair<size_t, size_t>& ImagePair(
      const image_pair_t pair_id) const;
  inline std::pair<size_t, size_t>& ImagePair(const image_t image_id1,
                                              const image_t image_id2);

  // Get mutable objects.
  inline class Camera& Camera(const camera_t camera_id);
  inline class Image& Image(const image_t image_id);
  inline class Point3D& Point3D(const point3D_t point3D_id);
  inline std::pair<size_t, size_t>& ImagePair(const image_pair_t pair_id);
  inline const std::pair<size_t, size_t>& ImagePair(
      const image_t image_id1, const image_t image_id2) const;

  // Get reference to all objects.
  inline const EIGEN_STL_UMAP(camera_t, class Camera) & Cameras() const;
  inline const EIGEN_STL_UMAP(image_t, class Image) & Images() const;
  inline const std::vector<image_t>& RegImageIds() const;
  inline const EIGEN_STL_UMAP(point3D_t, class Point3D) & Points3D() const;
  inline const std::unordered_map<image_pair_t, std::pair<size_t, size_t>>&
  ImagePairs() const;

  // Identifiers of all 3D points.
  std::unordered_set<point3D_t> Point3DIds() const;

  // Check whether specific object exists.
  inline bool ExistsCamera(const camera_t camera_id) const;
  inline bool ExistsImage(const image_t image_id) const;
  inline bool ExistsPoint3D(const point3D_t point3D_id) const;
  inline bool ExistsImagePair(const image_pair_t pair_id) const;

  // Load data from given `DatabaseCache`.
  void Load(const DatabaseCache& database_cache);

  // Setup all relevant data structures before reconstruction. Note that the
  // scene graph object must live until the `TearDown` method is called.
  void SetUp(const SceneGraph* scene_graph);

  // Finalize the Reconstruction after the reconstruction has finished.
  //
  // Once a scene has been finalized, it cannot be used for reconstruction.
  //
  // This removes all not yet registered images and unused cameras, in order to
  // save memory.
  void TearDown();

  // Add new camera. There is only one camera per image, while multiple images
  // might be taken by the same camera.
  void AddCamera(const class Camera& camera);

  // Add new image.
  void AddImage(const class Image& image);

  // Add new 3D object, and return its unique ID.
  point3D_t AddPoint3D(const Eigen::Vector3d& xyz, const Track& track);

  // Add observation to existing 3D point.
  void AddObservation(const point3D_t point3D_id, const TrackElement& track_el);

  // Merge two 3D points and return new identifier of new 3D point.
  // The location of the merged 3D point is a weighted average of the two
  // original 3D point's locations according to their track lengths.
  point3D_t MergePoints3D(const point3D_t point3D_id1,
                          const point3D_t point3D_id2);

  // Delete a 3D point, and all its references in the observed images.
  void DeletePoint3D(const point3D_t point3D_id);

  // Delete one observation from an image and the corresponding 3D point.
  // Note that this deletes the entire 3D point, if the track has two elements
  // prior to calling this method.
  void DeleteObservation(const image_t image_id, const point2D_t point2D_idx);

  // Register an existing image.
  void RegisterImage(const image_t image_id);

  // De-register an existing image, and all its references.
  void DeRegisterImage(const image_t image_id);

  // Check if image is registered.
  inline bool IsImageRegistered(const image_t image_id) const;

  // Normalize scene by scaling and translation to avoid degenerate
  // visualization after bundle adjustment and to improve numerical
  // stability of algorithms.
  //
  // Translates scene such that the mean of the camera centers or point
  // locations are at the origin of the coordinate system.
  //
  // Scales scene such that the minimum and maximum camera centers are at the
  // given `extent`, whereas `p0` and `p1` determine the minimum and
  // maximum percentiles of the camera centers considered.
  void Normalize(const double extent = 10.0, const double p0 = 0.1,
                 const double p1 = 0.9, const bool use_images = true);

  // Apply the 3D similarity transformation to all images and points.
  void Transform(const double scale, const Eigen::Vector4d& qvec,
                 const Eigen::Vector3d& tvec);

  // Merge the given reconstruction into this reconstruction by registering the
  // images registered in the given but not in this reconstruction and by
  // merging the two clouds and their tracks. The coordinate frames of the two
  // reconstructions are aligned using the projection centers of common
  // registered images. Return true if the two reconstructions could be merged.
  bool Merge(const Reconstruction& reconstruction, const int min_common_images);

  // Align the given reconstruction with a set of pre-defined camera positions.
  // Assuming that locations[i] gives the 3D coordinates of the center
  // of projection of the image with name image_names[i].
  bool Align(const std::vector<std::string>& image_names,
             const std::vector<Eigen::Vector3d>& locations,
             const int min_common_images);

  // Robust alignment using RANSAC.
  bool AlignRobust(const std::vector<std::string>& image_names,
                   const std::vector<Eigen::Vector3d>& locations,
                   const int min_common_images,
                   const RANSACOptions& ransac_options);

  // Find image with name.
  //
  // @param name        Name of the image.
  //
  // @return            Nullptr if image was not found.
  const class Image* FindImageWithName(const std::string& name) const;

  // Filter 3D points with large reprojection error, negative depth, or
  // insufficient triangulation angle.
  //
  // @param max_reproj_error    The maximum reprojection error.
  // @param min_tri_angle       The minimum triangulation angle.
  // @param point3D_ids         The points to be filtered.
  //
  // @return                    The number of filtered observations.
  size_t FilterPoints3D(const double max_reproj_error,
                        const double min_tri_angle,
                        const std::unordered_set<point3D_t>& point3D_ids);
  size_t FilterPoints3DInImages(const double max_reproj_error,
                                const double min_tri_angle,
                                const std::unordered_set<image_t>& image_ids);
  size_t FilterAllPoints3D(const double max_reproj_error,
                           const double min_tri_angle);

  // Filter observations that have negative depth.
  //
  // @return    The number of filtered observations.
  size_t FilterObservationsWithNegativeDepth();

  // Filter images without observations or bogus camera parameters.
  //
  // @return    The identifiers of the filtered images.
  std::vector<image_t> FilterImages(const double min_focal_length_ratio,
                                    const double max_focal_length_ratio,
                                    const double max_extra_param);

  // Compute statistics for scene.
  size_t ComputeNumObservations() const;
  double ComputeMeanTrackLength() const;
  double ComputeMeanObservationsPerRegImage() const;
  double ComputeMeanReprojectionError() const;

  // Read data from text or binary file. Prefer binary data if it exists.
  void Read(const std::string& path);
  void Write(const std::string& path) const;

  // Read data from binary/text file.
  void ReadText(const std::string& path);
  void ReadBinary(const std::string& path);

  // Write data from binary/text file.
  void WriteText(const std::string& path) const;
  void WriteBinary(const std::string& path) const;

  // Import from other data formats. Note that these import functions are
  // only intended for visualization of data and usable for reconstruction.
  void ImportPLY(const std::string& path);

  // Export to other data formats.
  bool ExportNVM(const std::string& path) const;
  bool ExportBundler(const std::string& path,
                     const std::string& list_path) const;
  void ExportPLY(const std::string& path) const;
  void ExportVRML(const std::string& images_path,
                  const std::string& points3D_path, const double image_scale,
                  const Eigen::Vector3d& image_rgb) const;

  // Extract colors for 3D points of given image. Colors will be extracted
  // only for 3D points which are completely black.
  //
  // @param image_id      Identifier of the image for which to extract colors.
  // @param path          Absolute or relative path to root folder of image.
  //                      The image path is determined by concatenating the
  //                      root path and the name of the image.
  //
  // @return              True if image could be read at given path.
  bool ExtractColorsForImage(const image_t image_id, const std::string& path);

  // Extract colors for all 3D points by computing the mean color of all images.
  //
  // @param path          Absolute or relative path to root folder of image.
  //                      The image path is determined by concatenating the
  //                      root path and the name of the image.
  void ExtractColorsForAllImages(const std::string& path);

  // Create all image sub-directories in the given path.
  void CreateImageDirs(const std::string& path) const;

 private:
  size_t FilterPoints3DWithSmallTriangulationAngle(
      const double min_tri_angle,
      const std::unordered_set<point3D_t>& point3D_ids);
  size_t FilterPoints3DWithLargeReprojectionError(
      const double max_reproj_error,
      const std::unordered_set<point3D_t>& point3D_ids);

  void ReadCamerasText(const std::string& path);
  void ReadImagesText(const std::string& path);
  void ReadPoints3DText(const std::string& path);
  void ReadCamerasBinary(const std::string& path);
  void ReadImagesBinary(const std::string& path);
  void ReadPoints3DBinary(const std::string& path);

  void WriteCamerasText(const std::string& path) const;
  void WriteImagesText(const std::string& path) const;
  void WriteRelativePosesText(const std::string& path) const;
  void WriteGeoRegistrationText(const std::string& path) const;
  void WritePoints3DText(const std::string& path) const;
  void WriteCamerasBinary(const std::string& path) const;
  void WriteImagesBinary(const std::string& path) const;
  void WritePoints3DBinary(const std::string& path) const;

  void SetObservationAsTriangulated(const image_t image_id,
                                    const point2D_t point2D_idx,
                                    const bool is_continued_point3D);
  void ResetTriObservations(const image_t image_id, const point2D_t point2D_idx,
                            const bool is_deleted_point3D);

  const SceneGraph* scene_graph_;

  EIGEN_STL_UMAP(camera_t, class Camera) cameras_;
  EIGEN_STL_UMAP(image_t, class Image) images_;
  EIGEN_STL_UMAP(point3D_t, class Point3D) points3D_;
  std::unordered_map<image_pair_t, std::pair<size_t, size_t>> image_pairs_;

  // { image_id, ... } where `images_.at(image_id).registered == true`.
  std::vector<image_t> reg_image_ids_;

  // Total number of added 3D points, used to generate unique identifiers.
  point3D_t num_added_points3D_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t Reconstruction::NumCameras() const { return cameras_.size(); }

size_t Reconstruction::NumImages() const { return images_.size(); }

size_t Reconstruction::NumRegImages() const { return reg_image_ids_.size(); }

size_t Reconstruction::NumPoints3D() const { return points3D_.size(); }

size_t Reconstruction::NumImagePairs() const { return image_pairs_.size(); }

const class Camera& Reconstruction::Camera(const camera_t camera_id) const {
  return cameras_.at(camera_id);
}

const class Image& Reconstruction::Image(const image_t image_id) const {
  return images_.at(image_id);
}

const class Point3D& Reconstruction::Point3D(const point3D_t point3D_id) const {
  return points3D_.at(point3D_id);
}

const std::pair<size_t, size_t>& Reconstruction::ImagePair(
    const image_pair_t pair_id) const {
  return image_pairs_.at(pair_id);
}

const std::pair<size_t, size_t>& Reconstruction::ImagePair(
    const image_t image_id1, const image_t image_id2) const {
  const auto pair_id = Database::ImagePairToPairId(image_id1, image_id2);
  return image_pairs_.at(pair_id);
}

class Camera& Reconstruction::Camera(const camera_t camera_id) {
  return cameras_.at(camera_id);
}

class Image& Reconstruction::Image(const image_t image_id) {
  return images_.at(image_id);
}

class Point3D& Reconstruction::Point3D(const point3D_t point3D_id) {
  return points3D_.at(point3D_id);
}

std::pair<size_t, size_t>& Reconstruction::ImagePair(
    const image_pair_t pair_id) {
  return image_pairs_.at(pair_id);
}

std::pair<size_t, size_t>& Reconstruction::ImagePair(const image_t image_id1,
                                                     const image_t image_id2) {
  const auto pair_id = Database::ImagePairToPairId(image_id1, image_id2);
  return image_pairs_.at(pair_id);
}

const EIGEN_STL_UMAP(camera_t, Camera) & Reconstruction::Cameras() const {
  return cameras_;
}

const EIGEN_STL_UMAP(image_t, class Image) & Reconstruction::Images() const {
  return images_;
}

const std::vector<image_t>& Reconstruction::RegImageIds() const {
  return reg_image_ids_;
}

const EIGEN_STL_UMAP(point3D_t, Point3D) & Reconstruction::Points3D() const {
  return points3D_;
}

const std::unordered_map<image_pair_t, std::pair<size_t, size_t>>&
Reconstruction::ImagePairs() const {
  return image_pairs_;
}

bool Reconstruction::ExistsCamera(const camera_t camera_id) const {
  return cameras_.find(camera_id) != cameras_.end();
}

bool Reconstruction::ExistsImage(const image_t image_id) const {
  return images_.find(image_id) != images_.end();
}

bool Reconstruction::ExistsPoint3D(const point3D_t point3D_id) const {
  return points3D_.find(point3D_id) != points3D_.end();
}

bool Reconstruction::ExistsImagePair(const image_pair_t pair_id) const {
  return image_pairs_.find(pair_id) != image_pairs_.end();
}

bool Reconstruction::IsImageRegistered(const image_t image_id) const {
  return Image(image_id).IsRegistered();
}

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_RECONSTRUCTION_H_
