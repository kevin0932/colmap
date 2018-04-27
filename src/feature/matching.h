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

#ifndef COLMAP_SRC_FEATURE_MATCHING_H_
#define COLMAP_SRC_FEATURE_MATCHING_H_

#include <array>
#include <string>
#include <vector>

#include "base/database.h"
#include "feature/sift.h"
#include "util/alignment.h"
#include "util/cache.h"
#include "util/opengl_utils.h"
#include "util/threading.h"
#include "util/timer.h"

namespace colmap {

struct ExhaustiveMatchingOptions {
  // Block size, i.e. number of images to simultaneously load into memory.
  int block_size = 50;

  bool Check() const;
};

struct SequentialMatchingOptions {
  // Number of overlapping image pairs.
  int overlap = 5;

  // Whether to match images against their quadratic neighbors.
  bool quadratic_overlap = true;

  // Whether to enable vocabulary tree based loop detection.
  bool loop_detection = false;

  // Loop detection is invoked every `loop_detection_period` images.
  int loop_detection_period = 10;

  // The number of images to retrieve in loop detection. This number should
  // be significantly bigger than the sequential matching overlap.
  int loop_detection_num_images = 30;

  // Number of images to spatially verify for re-ranking.
  int loop_detection_num_verifications = 0;

  // The maximum number of features to use for indexing an image. If an
  // image has more features, only the largest-scale features will be indexed.
  int loop_detection_max_num_features = -1;

  // Path to the vocabulary tree.
  std::string vocab_tree_path = "";

  bool Check() const;
};

struct VocabTreeMatchingOptions {
  // Number of images to retrieve for each query image.
  int num_images = 100;

  // Number of images to spatially verify for re-ranking.
  int num_verifications = 0;

  // The maximum number of features to use for indexing an image. If an
  // image has more features, only the largest-scale features will be indexed.
  int max_num_features = -1;

  // Path to the vocabulary tree.
  std::string vocab_tree_path = "";

  // Optional path to file with specific image names to match.
  std::string match_list_path = "";

  bool Check() const;
};

struct SpatialMatchingOptions {
  // Whether the location priors in the database are GPS coordinates in
  // the form of longitude and latitude coordinates in degrees.
  bool is_gps = true;

  // Whether to ignore the Z-component of the location prior.
  bool ignore_z = true;

  // The maximum number of nearest neighbors to match.
  int max_num_neighbors = 50;

  // The maximum distance between the query and nearest neighbor. For GPS
  // coordinates the unit is Euclidean distance in meters.
  double max_distance = 100;

  bool Check() const;
};

struct TransitiveMatchingOptions {
  // The maximum number of image pairs to process in one batch.
  int batch_size = 1000;

  // The number of transitive closure iterations.
  int num_iterations = 3;

  bool Check() const;
};

struct ImagePairsMatchingOptions {
  // Number of image pairs to match in one batch.
  int block_size = 100;

  // Path to the file with the matches.
  std::string match_list_path = "";

  bool Check() const;
};

struct OFGuidedImagePairsMatchingOptions {
  // Number of image pairs to match in one batch.
  //int block_size = 100;
  int block_size = 1;

  // Path to the file with the pair matches and predicted optical flow quantization mapping.
  std::string match_list_path = "";
  std::string optical_flow_path = "";
  // size_t image_scale_factor = 24;
  // size_t OF_scale_factor = 1;
  int image_scale_factor = 84;
  int OF_scale_factor = 1;
  double uncertainty_radius = 63;
  // int image_scale_factor;
  // int OF_scale_factor;
  // double uncertainty_radius;
  // bool only_image_pairs_as_ref = false;
  bool Check() const;
};

struct FeaturePairsMatchingOptions {
  // Whether to geometrically verify the given matches.
  bool verify_matches = true;

  // Path to the file with the matches.
  std::string match_list_path = "";

  bool Check() const;
};

namespace internal {

struct FeatureMatcherData {
  image_t image_id1 = kInvalidImageId;
  image_t image_id2 = kInvalidImageId;
  FeatureMatches matches;
  TwoViewGeometry two_view_geometry;
  // added for matching quantization when it is guided by optical-flow
  // std::vector<FeatureMatches> quantization_maps;
  FeatureMatches quantization_map;
  Eigen::Matrix<float, 48, 64, Eigen::RowMajor> optical_flow_x;
  Eigen::Matrix<float, 48, 64, Eigen::RowMajor> optical_flow_y;
  Eigen::Matrix<float, 48, 64, Eigen::RowMajor> optical_flow_x_21;
  Eigen::Matrix<float, 48, 64, Eigen::RowMajor> optical_flow_y_21;
  Eigen::Matrix<float, 48, 64, Eigen::RowMajor> flowconf_x;
  Eigen::Matrix<float, 48, 64, Eigen::RowMajor> flowconf_y;
  Eigen::Matrix<float, 48, 64, Eigen::RowMajor> flowconf_x_21;
  Eigen::Matrix<float, 48, 64, Eigen::RowMajor> flowconf_y_21;
};

}  // namespace internal

// Cache for feature matching to minimize database access during matching.
class FeatureMatcherCache {
 public:
  FeatureMatcherCache(const size_t cache_size, const Database* database);

  void Setup();

  const Camera& GetCamera(const camera_t camera_id) const;
  const Image& GetImage(const image_t image_id) const;
  const FeatureKeypoints& GetKeypoints(const image_t image_id);
  const FeatureDescriptors& GetDescriptors(const image_t image_id);
  FeatureMatches GetMatches(const image_t image_id1, const image_t image_id2);
  std::vector<image_t> GetImageIds() const;

  bool ExistsMatches(const image_t image_id1, const image_t image_id2);
  bool ExistsInlierMatches(const image_t image_id1, const image_t image_id2);

  void WriteMatches(const image_t image_id1, const image_t image_id2,
                    const FeatureMatches& matches);
  void WriteInlierMatches(const image_t image_id1, const image_t image_id2,
                          const TwoViewGeometry& two_view_geometry);

  void DeleteMatches(const image_t image_id1, const image_t image_id2);
  void DeleteInlierMatches(const image_t image_id1, const image_t image_id2);

 private:
  const size_t cache_size_;
  const Database* database_;
  std::mutex database_mutex_;
  EIGEN_STL_UMAP(camera_t, Camera) cameras_cache_;
  EIGEN_STL_UMAP(image_t, Image) images_cache_;
  std::unique_ptr<LRUCache<image_t, FeatureKeypoints>> keypoints_cache_;
  std::unique_ptr<LRUCache<image_t, FeatureDescriptors>> descriptors_cache_;
};

class FeatureMatcherThread : public Thread {
 public:
  FeatureMatcherThread(const SiftMatchingOptions& options,
                       FeatureMatcherCache* cache);

  void SetMaxNumMatches(const int max_num_matches);

 protected:
  SiftMatchingOptions options_;
  FeatureMatcherCache* cache_;
};

class SiftCPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  SiftCPUFeatureMatcher(const SiftMatchingOptions& options,
                        FeatureMatcherCache* cache,
                        JobQueue<Input>* input_queue,
                        JobQueue<Output>* output_queue);

 protected:
  void Run() override;

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

class SiftGPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  SiftGPUFeatureMatcher(const SiftMatchingOptions& options,
                        FeatureMatcherCache* cache,
                        JobQueue<Input>* input_queue,
                        JobQueue<Output>* output_queue);

 protected:
  void Run() override;

  void GetDescriptorData(const int index, const image_t image_id,
                         const FeatureDescriptors** descriptors_ptr);

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;

  std::unique_ptr<OpenGLContextManager> opengl_context_;

  // The previously uploaded images to the GPU.
  std::array<image_t, 2> prev_uploaded_image_ids_;
  std::array<FeatureDescriptors, 2> prev_uploaded_descriptors_;
};

class GuidedSiftCPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  GuidedSiftCPUFeatureMatcher(const SiftMatchingOptions& options,
                              FeatureMatcherCache* cache,
                              JobQueue<Input>* input_queue,
                              JobQueue<Output>* output_queue);

 private:
  void Run() override;

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

class OFGuidedSiftCPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  OFGuidedSiftCPUFeatureMatcher(const SiftMatchingOptions& options,
                              FeatureMatcherCache* cache,
                              JobQueue<Input>* input_queue,
                              JobQueue<Output>* output_queue);

 private:
  void Run() override;

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

class NewOFGuidedSiftCPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  NewOFGuidedSiftCPUFeatureMatcher(const SiftMatchingOptions& options,
                              FeatureMatcherCache* cache,
                              JobQueue<Input>* input_queue,
                              JobQueue<Output>* output_queue);

 private:
  void Run() override;

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

class CrossCheckOFGuidedSiftCPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  CrossCheckOFGuidedSiftCPUFeatureMatcher(const SiftMatchingOptions& options,
                              FeatureMatcherCache* cache,
                              JobQueue<Input>* input_queue,
                              JobQueue<Output>* output_queue);

 private:
  void Run() override;

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};


class AdaptiveNewOFGuidedSiftCPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  AdaptiveNewOFGuidedSiftCPUFeatureMatcher(const SiftMatchingOptions& options,
                              FeatureMatcherCache* cache,
                              JobQueue<Input>* input_queue,
                              JobQueue<Output>* output_queue);

 private:
  void Run() override;

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

class AdaptiveCrossCheckOFGuidedSiftCPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  AdaptiveCrossCheckOFGuidedSiftCPUFeatureMatcher(const SiftMatchingOptions& options,
                              FeatureMatcherCache* cache,
                              JobQueue<Input>* input_queue,
                              JobQueue<Output>* output_queue);

 private:
  void Run() override;

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

class GuidedSiftGPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  GuidedSiftGPUFeatureMatcher(const SiftMatchingOptions& options,
                              FeatureMatcherCache* cache,
                              JobQueue<Input>* input_queue,
                              JobQueue<Output>* output_queue);

 private:
  void Run() override;

  void GetFeatureData(const int index, const image_t image_id,
                      const FeatureKeypoints** keypoints_ptr,
                      const FeatureDescriptors** descriptors_ptr);

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;

  std::unique_ptr<OpenGLContextManager> opengl_context_;

  // The previously uploaded images to the GPU.
  std::array<image_t, 2> prev_uploaded_image_ids_;
  std::array<FeatureKeypoints, 2> prev_uploaded_keypoints_;
  std::array<FeatureDescriptors, 2> prev_uploaded_descriptors_;
};

class OFGuidedSiftGPUFeatureMatcher : public FeatureMatcherThread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  OFGuidedSiftGPUFeatureMatcher(const SiftMatchingOptions& options,
                              FeatureMatcherCache* cache,
                              JobQueue<Input>* input_queue,
                              JobQueue<Output>* output_queue);

 private:
  void Run() override;

  void GetFeatureData(const int index, const image_t image_id,
                      const FeatureKeypoints** keypoints_ptr,
                      const FeatureDescriptors** descriptors_ptr);

  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;

  std::unique_ptr<OpenGLContextManager> opengl_context_;

  // The previously uploaded images to the GPU.
  std::array<image_t, 2> prev_uploaded_image_ids_;
  std::array<FeatureKeypoints, 2> prev_uploaded_keypoints_;
  std::array<FeatureDescriptors, 2> prev_uploaded_descriptors_;
};


class TwoViewGeometryVerifier : public Thread {
 public:
  typedef internal::FeatureMatcherData Input;
  typedef internal::FeatureMatcherData Output;

  TwoViewGeometryVerifier(const SiftMatchingOptions& options,
                          FeatureMatcherCache* cache,
                          JobQueue<Input>* input_queue,
                          JobQueue<Output>* output_queue);

 protected:
  void Run() override;

  const SiftMatchingOptions options_;
  TwoViewGeometry::Options two_view_geometry_options_;
  FeatureMatcherCache* cache_;
  JobQueue<Input>* input_queue_;
  JobQueue<Output>* output_queue_;
};

// Multi-threaded and multi-GPU SIFT feature matcher, which writes the computed
// results to the database and skips already matched image pairs. To improve
// performance of the matching by taking advantage of caching and database
// transactions, pass multiple images to the `Match` function. Note that the
// database should be in an active transaction while calling `Match`.
class SiftFeatureMatcher {
 public:
  SiftFeatureMatcher(const SiftMatchingOptions& options, Database* database,
                     FeatureMatcherCache* cache);

  ~SiftFeatureMatcher();

  // Setup the matchers and return if successful.
  bool Setup();

  // Match one batch of multiple image pairs.
  void Match(const std::vector<std::pair<image_t, image_t>>& image_pairs);

  // Optical-Flow-Guided Match one batch of multiple image pairs.
  void OFGuidedMatch(const std::vector<std::pair<image_t, image_t>>& image_pairs, const std::vector<FeatureMatches>& quantization_maps);
  void NewOFGuidedMatch(const std::vector<std::pair<image_t, image_t>>& image_pairs, const std::vector<FeatureMatches>& quantization_maps, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_x, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_y);
  void CrossCheckOFGuidedMatch(const std::vector<std::pair<image_t, image_t>>& image_pairs, const std::vector<FeatureMatches>& quantization_maps, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_x, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_y, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_x_21, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_y_21);

  void AdaptiveNewOFGuidedMatch(const std::vector<std::pair<image_t, image_t>>& image_pairs, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_x, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_y, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& flowconfs_x, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& flowconfs_y);
  void AdaptiveCrossCheckOFGuidedMatch(const std::vector<std::pair<image_t, image_t>>& image_pairs, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_x, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_y, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_x_21, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& optical_flows_y_21, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& flowconfs_x, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& flowconfs_y, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& flowconfs_x_21, const std::vector<Eigen::Matrix<float, 48, 64, Eigen::RowMajor>>& flowconfs_y_21);

 private:
  SiftMatchingOptions options_;
  Database* database_;
  FeatureMatcherCache* cache_;

  bool is_setup_;

  std::vector<std::unique_ptr<FeatureMatcherThread>> matchers_;
  std::vector<std::unique_ptr<FeatureMatcherThread>> guided_matchers_;
  std::vector<std::unique_ptr<Thread>> verifiers_;
  std::unique_ptr<ThreadPool> thread_pool_;

  JobQueue<internal::FeatureMatcherData> matcher_queue_;
  JobQueue<internal::FeatureMatcherData> verifier_queue_;
  JobQueue<internal::FeatureMatcherData> guided_matcher_queue_;
  JobQueue<internal::FeatureMatcherData> output_queue_;
};

// Exhaustively match images by processing each block in the exhaustive match
// matrix in one batch:
//
// +----+----+-----------------> images[i]
// |#000|0000|
// |1#00|1000| <- Above the main diagonal, the block diagonal is not matched
// |11#0|1100|                                                             ^
// |111#|1110|                                                             |
// +----+----+                                                             |
// |1000|#000|\                                                            |
// |1100|1#00| \ One block                                                 |
// |1110|11#0| / of image pairs                                            |
// |1111|111#|/                                                            |
// +----+----+                                                             |
// |  ^                                                                    |
// |  |                                                                    |
// | Below the main diagonal, the block diagonal is matched <--------------+
// |
// v
// images[i]
//
// Pairs will only be matched if 1, to avoid duplicate pairs. Pairs with #
// are on the main diagonal and denote pairs of the same image.
class ExhaustiveFeatureMatcher : public Thread {
 public:
  ExhaustiveFeatureMatcher(const ExhaustiveMatchingOptions& options,
                           const SiftMatchingOptions& match_options,
                           const std::string& database_path);

 private:
  void Run() override;

  const ExhaustiveMatchingOptions options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};

// Sequentially match images within neighborhood:
//
// +-------------------------------+-----------------------> images[i]
//                      ^          |           ^
//                      |   Current image[i]   |
//                      |          |           |
//                      +----------+-----------+
//                                 |
//                        Match image_i against
//
//                    image_[i - o, i + o]        with o = [1 .. overlap]
//                    image_[i - 2^o, i + 2^o]    (for quadratic overlap)
//
// Sequential order is determined based on the image names in ascending order.
//
// Invoke loop detection if `(i mod loop_detection_period) == 0`, retrieve
// most similar `loop_detection_num_images` images from vocabulary tree,
// and perform matching and verification.
class SequentialFeatureMatcher : public Thread {
 public:
  SequentialFeatureMatcher(const SequentialMatchingOptions& options,
                           const SiftMatchingOptions& match_options,
                           const std::string& database_path);

 private:
  void Run() override;

  std::vector<image_t> GetOrderedImageIds() const;
  void RunSequentialMatching(const std::vector<image_t>& image_ids);
  void RunLoopDetection(const std::vector<image_t>& image_ids);

  const SequentialMatchingOptions options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};

// Match each image against its nearest neighbors using a vocabulary tree.
class VocabTreeFeatureMatcher : public Thread {
 public:
  VocabTreeFeatureMatcher(const VocabTreeMatchingOptions& options,
                          const SiftMatchingOptions& match_options,
                          const std::string& database_path);

 private:
  void Run() override;

  const VocabTreeMatchingOptions options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};

// Match images against spatial nearest neighbors using prior location
// information, e.g. provided manually or extracted from EXIF.
class SpatialFeatureMatcher : public Thread {
 public:
  SpatialFeatureMatcher(const SpatialMatchingOptions& options,
                        const SiftMatchingOptions& match_options,
                        const std::string& database_path);

 private:
  void Run() override;

  const SpatialMatchingOptions options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};

// Match transitive image pairs in a database with existing feature matches.
// This matcher transitively closes loops. For example, if image pairs A-B and
// B-C match but A-C has not been matched, then this matcher attempts to match
// A-C. This procedure is performed for multiple iterations.
class TransitiveFeatureMatcher : public Thread {
 public:
  TransitiveFeatureMatcher(const TransitiveMatchingOptions& options,
                           const SiftMatchingOptions& match_options,
                           const std::string& database_path);

 private:
  void Run() override;

  const TransitiveMatchingOptions options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};

// Match images manually specified in a list of image pairs.
//
// Read matches file with the following format:
//
//    image_name1 image_name2
//    image_name1 image_name3
//    image_name2 image_name3
//    ...
//
class ImagePairsFeatureMatcher : public Thread {
 public:
  ImagePairsFeatureMatcher(const ImagePairsMatchingOptions& options,
                           const SiftMatchingOptions& match_options,
                           const std::string& database_path);

 private:
  void Run() override;

  const ImagePairsMatchingOptions options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};

// Import feature matches from a text file.
//
// Read matches file with the following format:
//
//      image_name1 image_name2
//      0 1
//      1 2
//      2 3
//      <empty line>
//      image_name1 image_name3
//      0 1
//      1 2
//      2 3
//      ...
//
class FeaturePairsFeatureMatcher : public Thread {
 public:
  FeaturePairsFeatureMatcher(const FeaturePairsMatchingOptions& options,
                             const SiftMatchingOptions& match_options,
                             const std::string& database_path);

 private:
  const static size_t kCacheSize = 100;

  void Run() override;

  const FeaturePairsMatchingOptions options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
};



//////////////////////////////////////////////////////////////////
// Match images manually specified in a list of image pairs.
//
// Read matches file with the following format:
//
//    image_name1 image_name2
//    image_name1 image_name3
//    image_name2 image_name3
//    ...
//
class OFGuidedImagePairsFeatureMatcher : public Thread {
 public:
  OFGuidedImagePairsFeatureMatcher(const OFGuidedImagePairsMatchingOptions& options,
                           const SiftMatchingOptions& match_options,
                           const std::string& database_path);

 private:
  void Run() override;

  const OFGuidedImagePairsMatchingOptions options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};


class NewOFGuidedImagePairsFeatureMatcher : public Thread {
 public:
  NewOFGuidedImagePairsFeatureMatcher(const OFGuidedImagePairsMatchingOptions& options,
                           const SiftMatchingOptions& match_options,
                           const std::string& database_path);

 private:
  void Run() override;

  const OFGuidedImagePairsMatchingOptions options_;
  const SiftMatchingOptions match_options_;
  Database database_;
  FeatureMatcherCache cache_;
  SiftFeatureMatcher matcher_;
};


}  // namespace colmap

#endif  // COLMAP_SRC_FEATURE_MATCHING_H_
