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

#include "feature/sift.h"

#include <array>
#include <fstream>
#include <memory>

#include "ext/SiftGPU/SiftGPU.h"
#include "ext/VLFeat/covdet.h"
#include "ext/VLFeat/sift.h"
#include "feature/utils.h"
#include "util/cuda.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/opengl_utils.h"

#include <math.h>

namespace colmap {
namespace {

// VLFeat uses a different convention to store its descriptors. This transforms
// the VLFeat format into the original SIFT format that is also used by SiftGPU.
FeatureDescriptors TransformVLFeatToUBCFeatureDescriptors(
    const FeatureDescriptors& vlfeat_descriptors) {
  FeatureDescriptors ubc_descriptors(vlfeat_descriptors.rows(),
                                     vlfeat_descriptors.cols());
  const std::array<int, 8> q{{0, 7, 6, 5, 4, 3, 2, 1}};
  for (FeatureDescriptors::Index n = 0; n < vlfeat_descriptors.rows(); ++n) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 8; ++k) {
          ubc_descriptors(n, 8 * (j + 4 * i) + q[k]) =
              vlfeat_descriptors(n, 8 * (j + 4 * i) + k);
        }
      }
    }
  }
  return ubc_descriptors;
}

Eigen::MatrixXi ComputeSiftDistanceMatrix(
    const FeatureKeypoints* keypoints1, const FeatureKeypoints* keypoints2,
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2,
    const std::function<bool(float, float, float, float)>& guided_filter) {
  if (guided_filter != nullptr) {
    CHECK_NOTNULL(keypoints1);
    CHECK_NOTNULL(keypoints2);
    CHECK_EQ(keypoints1->size(), descriptors1.rows());
    CHECK_EQ(keypoints2->size(), descriptors2.rows());
  }

  const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors1_int =
      descriptors1.cast<int>();
  const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors2_int =
      descriptors2.cast<int>();

  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(
      descriptors1.rows(), descriptors2.rows());

  for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
    for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
      if (guided_filter != nullptr &&
          guided_filter((*keypoints1)[i1].x, (*keypoints1)[i1].y,
                        (*keypoints2)[i2].x, (*keypoints2)[i2].y)) {
        dists(i1, i2) = 0;
      } else {
        dists(i1, i2) = descriptors1_int.row(i1).dot(descriptors2_int.row(i2));
      }
    }
  }

  return dists;
}

Eigen::MatrixXf ComputeSiftDistanceMatrix_Kevin(
    const FeatureDescriptors& descriptors1,
    const FeatureDescriptors& descriptors2) {

  const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors1_int =
      descriptors1.cast<int>();
  const Eigen::Matrix<int, Eigen::Dynamic, 128> descriptors2_int =
      descriptors2.cast<int>();

  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dists(
      descriptors1.rows(), descriptors2.rows());

  for (FeatureDescriptors::Index i1 = 0; i1 < descriptors1.rows(); ++i1) {
    for (FeatureDescriptors::Index i2 = 0; i2 < descriptors2.rows(); ++i2) {
        // dists(i1, i2) = descriptors1_int.row(i1).dot(descriptors2_int.row(i2));
        dists(i1, i2) = (descriptors1_int.row(i1)-descriptors2_int.row(i2)).squaredNorm();
    }
  }

  return dists;
}

size_t FindBestMatchesOneWay_One2Multi(const Eigen::MatrixXf& dists,
                             const float max_ratio, const float max_distance,
                             std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
  const float kDistNorm = 1.0f / (512.0f * 512.0f);

  size_t num_matches = 0;
  matches->resize(dists.rows(), -1);

  for (Eigen::MatrixXf::Index i1 = 0; i1 < dists.rows(); ++i1) {
    int best_i2 = -1;
    float best_dist = 32767000;
    float second_best_dist = 32767000;
    // std::cout << "##############################" << std::endl;
    for (Eigen::MatrixXf::Index i2 = 0; i2 < dists.cols(); ++i2) {
      const float dist = dists(i1, i2);
      if (dist <= best_dist) {
        best_i2 = i2;
        second_best_dist = best_dist;
        best_dist = dist;
      } else if (dist <= second_best_dist) {
        second_best_dist = dist;
      }
      // std::cout << dist * kDistNorm << " ";
    }
    // std::cout << "##############################" << std::endl;
    // Check if any match found.
    if (best_i2 == -1) {
      continue;
    }

    // const float best_dist_normed =
    //     std::acos(std::min(kDistNorm * best_dist, 1.0f));
    const float best_dist_normed = std::min(kDistNorm * best_dist, 1.0f);

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    // const float second_best_dist_normed =
    //     std::acos(std::min(kDistNorm * second_best_dist, 1.0f));
    const float second_best_dist_normed = std::min(kDistNorm * second_best_dist, 1.0f);

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    num_matches += 1;
    (*matches)[i1] = best_i2;
  }

  return num_matches;
}

size_t FindBestMatchesOneWay(const Eigen::MatrixXi& dists,
                             const float max_ratio, const float max_distance,
                             std::vector<int>* matches) {
  // SIFT descriptor vectors are normalized to length 512.
  const float kDistNorm = 1.0f / (512.0f * 512.0f);

  size_t num_matches = 0;
  matches->resize(dists.rows(), -1);

  for (Eigen::MatrixXi::Index i1 = 0; i1 < dists.rows(); ++i1) {
    int best_i2 = -1;
    int best_dist = 0;
    int second_best_dist = 0;
    for (Eigen::MatrixXi::Index i2 = 0; i2 < dists.cols(); ++i2) {
      const int dist = dists(i1, i2);
      if (dist > best_dist) {
        best_i2 = i2;
        second_best_dist = best_dist;
        best_dist = dist;
      } else if (dist > second_best_dist) {
        second_best_dist = dist;
      }
    }

    // Check if any match found.
    if (best_i2 == -1) {
      continue;
    }

    const float best_dist_normed =
        std::acos(std::min(kDistNorm * best_dist, 1.0f));

    // Check if match distance passes threshold.
    if (best_dist_normed > max_distance) {
      continue;
    }

    const float second_best_dist_normed =
        std::acos(std::min(kDistNorm * second_best_dist, 1.0f));

    // Check if match passes ratio test. Keep this comparison >= in order to
    // ensure that the case of best == second_best is detected.
    if (best_dist_normed >= max_ratio * second_best_dist_normed) {
      continue;
    }

    num_matches += 1;
    (*matches)[i1] = best_i2;
  }

  return num_matches;
}

void FindBestMatches(const Eigen::MatrixXi& dists, const float max_ratio,
                     const float max_distance, const bool cross_check,
                     FeatureMatches* matches) {
  matches->clear();

  std::vector<int> matches12;
  const size_t num_matches12 =
      FindBestMatchesOneWay(dists, max_ratio, max_distance, &matches12);

  if (cross_check) {
    std::vector<int> matches21;
    const size_t num_matches21 = FindBestMatchesOneWay(
        dists.transpose(), max_ratio, max_distance, &matches21);
    matches->reserve(std::min(num_matches12, num_matches21));
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
          matches21[matches12[i1]] == static_cast<int>(i1)) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  } else {
    matches->reserve(num_matches12);
    for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
      if (matches12[i1] != -1) {
        FeatureMatch match;
        match.point2D_idx1 = i1;
        match.point2D_idx2 = matches12[i1];
        matches->push_back(match);
      }
    }
  }
}

void WarnIfMaxNumMatchesReachedGPU(const SiftMatchGPU& sift_match_gpu,
                                   const FeatureDescriptors& descriptors) {
  if (sift_match_gpu.GetMaxSift() < descriptors.rows()) {
    std::cout << StringPrintf(
                     "WARNING: Clamping features from %d to %d - consider "
                     "increasing the maximum number of matches.",
                     descriptors.rows(), sift_match_gpu.GetMaxSift())
              << std::endl;
  }
}

void WarnDarknessAdaptivityNotAvailable() {
  std::cout << "WARNING: Darkness adaptivity only available for GLSL SiftGPU."
            << std::endl;
}

}  // namespace

bool SiftExtractionOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
  }
  CHECK_OPTION_GT(max_image_size, 0);
  CHECK_OPTION_GT(max_num_features, 0);
  CHECK_OPTION_GT(octave_resolution, 0);
  CHECK_OPTION_GT(peak_threshold, 0.0);
  CHECK_OPTION_GT(edge_threshold, 0.0);
  CHECK_OPTION_GT(max_num_orientations, 0);
  if (domain_size_pooling) {
    CHECK_OPTION_GT(dsp_min_scale, 0);
    CHECK_OPTION_GE(dsp_max_scale, dsp_min_scale);
    CHECK_OPTION_GT(dsp_num_scales, 0);
  }
  return true;
}

bool SiftMatchingOptions::Check() const {
  if (use_gpu) {
    CHECK_OPTION_GT(CSVToVector<int>(gpu_index).size(), 0);
  }
  CHECK_OPTION_GT(max_ratio, 0.0);
  CHECK_OPTION_GT(max_distance, 0.0);
  CHECK_OPTION_GT(max_error, 0.0);
  CHECK_OPTION_GT(max_num_trials, 0);
  CHECK_OPTION_GE(min_inlier_ratio, 0);
  CHECK_OPTION_LE(min_inlier_ratio, 1);
  CHECK_OPTION_GE(min_num_inliers, 0);
  return true;
}

bool ExtractSiftFeaturesCPU(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  CHECK(options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);

  CHECK(!options.estimate_affine_shape);
  CHECK(!options.domain_size_pooling);

  if (options.darkness_adaptivity) {
    WarnDarknessAdaptivityNotAvailable();
  }

  // Setup SIFT extractor.
  std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
      vl_sift_new(bitmap.Width(), bitmap.Height(), options.num_octaves,
                  options.octave_resolution, options.first_octave),
      &vl_sift_delete);
  if (!sift) {
    return false;
  }

  vl_sift_set_peak_thresh(sift.get(), options.peak_threshold);
  vl_sift_set_edge_thresh(sift.get(), options.edge_threshold);

  // Iterate through octaves.
  std::vector<size_t> level_num_features;
  std::vector<FeatureKeypoints> level_keypoints;
  std::vector<FeatureDescriptors> level_descriptors;
  bool first_octave = true;
  while (true) {
    if (first_octave) {
      const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
      std::vector<float> data_float(data_uint8.size());
      for (size_t i = 0; i < data_uint8.size(); ++i) {
        data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
      }
      if (vl_sift_process_first_octave(sift.get(), data_float.data())) {
        break;
      }
      first_octave = false;
    } else {
      if (vl_sift_process_next_octave(sift.get())) {
        break;
      }
    }

    // Detect keypoints.
    vl_sift_detect(sift.get());

    // Extract detected keypoints.
    const VlSiftKeypoint* vl_keypoints = vl_sift_get_keypoints(sift.get());
    const int num_keypoints = vl_sift_get_nkeypoints(sift.get());
    if (num_keypoints == 0) {
      continue;
    }

    // Extract features with different orientations per DOG level.
    size_t level_idx = 0;
    int prev_level = -1;
    for (int i = 0; i < num_keypoints; ++i) {
      if (vl_keypoints[i].is != prev_level) {
        if (i > 0) {
          // Resize containers of previous DOG level.
          level_keypoints.back().resize(level_idx);
          if (descriptors != nullptr) {
            level_descriptors.back().conservativeResize(level_idx, 128);
          }
        }

        // Add containers for new DOG level.
        level_idx = 0;
        level_num_features.push_back(0);
        level_keypoints.emplace_back(options.max_num_orientations *
                                     num_keypoints);
        if (descriptors != nullptr) {
          level_descriptors.emplace_back(
              options.max_num_orientations * num_keypoints, 128);
        }
      }

      level_num_features.back() += 1;
      prev_level = vl_keypoints[i].is;

      // Extract feature orientations.
      double angles[4];
      int num_orientations;
      if (options.upright) {
        num_orientations = 1;
        angles[0] = 0.0;
      } else {
        num_orientations = vl_sift_calc_keypoint_orientations(
            sift.get(), angles, &vl_keypoints[i]);
      }

      // Note that this is different from SiftGPU, which selects the top
      // global maxima as orientations while this selects the first two
      // local maxima. It is not clear which procedure is better.
      const int num_used_orientations =
          std::min(num_orientations, options.max_num_orientations);

      for (int o = 0; o < num_used_orientations; ++o) {
        level_keypoints.back()[level_idx] =
            FeatureKeypoint(vl_keypoints[i].x + 0.5f, vl_keypoints[i].y + 0.5f,
                            vl_keypoints[i].sigma, angles[o]);
        if (descriptors != nullptr) {
          Eigen::MatrixXf desc(1, 128);
          vl_sift_calc_keypoint_descriptor(sift.get(), desc.data(),
                                           &vl_keypoints[i], angles[o]);
          if (options.normalization ==
              SiftExtractionOptions::Normalization::L2) {
            desc = L2NormalizeFeatureDescriptors(desc);
          } else if (options.normalization ==
                     SiftExtractionOptions::Normalization::L1_ROOT) {
            desc = L1RootNormalizeFeatureDescriptors(desc);
          } else {
            LOG(FATAL) << "Normalization type not supported";
          }

          level_descriptors.back().row(level_idx) =
              FeatureDescriptorsToUnsignedByte(desc);
        }

        level_idx += 1;
      }
    }

    // Resize containers for last DOG level in octave.
    level_keypoints.back().resize(level_idx);
    if (descriptors != nullptr) {
      level_descriptors.back().conservativeResize(level_idx, 128);
    }
  }

  // Determine how many DOG levels to keep to satisfy max_num_features option.
  int first_level_to_keep = 0;
  int num_features = 0;
  int num_features_with_orientations = 0;
  for (int i = level_keypoints.size() - 1; i >= 0; --i) {
    num_features += level_num_features[i];
    num_features_with_orientations += level_keypoints[i].size();
    if (num_features > options.max_num_features) {
      first_level_to_keep = i;
      break;
    }
  }

  // Extract the features to be kept.
  {
    size_t k = 0;
    keypoints->resize(num_features_with_orientations);
    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
      for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
        (*keypoints)[k] = level_keypoints[i][j];
        k += 1;
      }
    }
  }

  // Compute the descriptors for the detected keypoints.
  if (descriptors != nullptr) {
    size_t k = 0;
    descriptors->resize(num_features_with_orientations, 128);
    for (size_t i = first_level_to_keep; i < level_keypoints.size(); ++i) {
      for (size_t j = 0; j < level_keypoints[i].size(); ++j) {
        descriptors->row(k) = level_descriptors[i].row(j);
        k += 1;
      }
    }
    *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
  }

  return true;
}

bool ExtractCovariantSiftFeaturesCPU(const SiftExtractionOptions& options,
                                     const Bitmap& bitmap,
                                     FeatureKeypoints* keypoints,
                                     FeatureDescriptors* descriptors) {
  CHECK(options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);

  if (options.darkness_adaptivity) {
    WarnDarknessAdaptivityNotAvailable();
  }

  // Setup covariant SIFT detector.
  std::unique_ptr<VlCovDet, void (*)(VlCovDet*)> covdet(
      vl_covdet_new(VL_COVDET_METHOD_DOG), &vl_covdet_delete);
  if (!covdet) {
    return false;
  }

  const int kMaxOctaveResolution = 1000;
  CHECK_LE(options.octave_resolution, kMaxOctaveResolution);

  vl_covdet_set_first_octave(covdet.get(), options.first_octave);
  vl_covdet_set_octave_resolution(covdet.get(), options.octave_resolution);
  vl_covdet_set_peak_threshold(covdet.get(), options.peak_threshold);
  vl_covdet_set_edge_threshold(covdet.get(), options.edge_threshold);

  {
    const std::vector<uint8_t> data_uint8 = bitmap.ConvertToRowMajorArray();
    std::vector<float> data_float(data_uint8.size());
    for (size_t i = 0; i < data_uint8.size(); ++i) {
      data_float[i] = static_cast<float>(data_uint8[i]) / 255.0f;
    }
    vl_covdet_put_image(covdet.get(), data_float.data(), bitmap.Width(),
                        bitmap.Height());
  }

  vl_covdet_detect(covdet.get(), options.max_num_features);

  if (!options.upright) {
    if (options.estimate_affine_shape) {
      vl_covdet_extract_affine_shape(covdet.get());
    } else {
      vl_covdet_extract_orientations(covdet.get());
    }
  }

  const int num_features = vl_covdet_get_num_features(covdet.get());
  VlCovDetFeature* features = vl_covdet_get_features(covdet.get());

  // Sort features according to detected octave and scale.
  std::sort(
      features, features + num_features,
      [](const VlCovDetFeature& feature1, const VlCovDetFeature& feature2) {
        if (feature1.o == feature2.o) {
          return feature1.s > feature2.s;
        } else {
          return feature1.o > feature2.o;
        }
      });

  const size_t max_num_features = static_cast<size_t>(options.max_num_features);

  // Copy detected keypoints and clamp when maximum number of features reached.
  int prev_octave_scale_idx = std::numeric_limits<int>::max();
  for (int i = 0; i < num_features; ++i) {
    FeatureKeypoint keypoint;
    keypoint.x = features[i].frame.x + 0.5;
    keypoint.y = features[i].frame.y + 0.5;
    keypoint.a11 = features[i].frame.a11;
    keypoint.a12 = features[i].frame.a12;
    keypoint.a21 = features[i].frame.a21;
    keypoint.a22 = features[i].frame.a22;
    keypoints->push_back(keypoint);

    const int octave_scale_idx =
        features[i].o * kMaxOctaveResolution + features[i].s;
    CHECK_LE(octave_scale_idx, prev_octave_scale_idx);

    if (octave_scale_idx != prev_octave_scale_idx &&
        keypoints->size() >= max_num_features) {
      break;
    }

    prev_octave_scale_idx = octave_scale_idx;
  }

  // Compute the descriptors for the detected keypoints.
  if (descriptors != nullptr) {
    descriptors->resize(keypoints->size(), 128);

    const size_t kPatchResolution = 15;
    const size_t kPatchSide = 2 * kPatchResolution + 1;
    const double kPatchRelativeExtent = 7.5;
    const double kPatchRelativeSmoothing = 1;
    const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
    const double kSigma =
        kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;

    std::vector<float> patch(kPatchSide * kPatchSide);
    std::vector<float> patchXY(2 * kPatchSide * kPatchSide);

    float dsp_min_scale = 1;
    float dsp_scale_step = 0;
    int dsp_num_scales = 1;
    if (options.domain_size_pooling) {
      dsp_min_scale = options.dsp_min_scale;
      dsp_scale_step = (options.dsp_max_scale - options.dsp_min_scale) /
                       options.dsp_num_scales;
      dsp_num_scales = options.dsp_num_scales;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor>
        scaled_descriptors(dsp_num_scales, 128);

    std::unique_ptr<VlSiftFilt, void (*)(VlSiftFilt*)> sift(
        vl_sift_new(16, 16, 1, 3, 0), &vl_sift_delete);
    if (!sift) {
      return false;
    }

    vl_sift_set_magnif(sift.get(), 3.0);

    for (size_t i = 0; i < keypoints->size(); ++i) {
      for (int s = 0; s < dsp_num_scales; ++s) {
        const double dsp_scale = dsp_min_scale + s * dsp_scale_step;

        VlFrameOrientedEllipse scaled_frame = features[i].frame;
        scaled_frame.a11 *= dsp_scale;
        scaled_frame.a12 *= dsp_scale;
        scaled_frame.a21 *= dsp_scale;
        scaled_frame.a22 *= dsp_scale;

        vl_covdet_extract_patch_for_frame(
            covdet.get(), patch.data(), kPatchResolution, kPatchRelativeExtent,
            kPatchRelativeSmoothing, scaled_frame);

        vl_imgradient_polar_f(patchXY.data(), patchXY.data() + 1, 2,
                              2 * kPatchSide, patch.data(), kPatchSide,
                              kPatchSide, kPatchSide);

        vl_sift_calc_raw_descriptor(sift.get(), patchXY.data(),
                                    scaled_descriptors.row(s).data(),
                                    kPatchSide, kPatchSide, kPatchResolution,
                                    kPatchResolution, kSigma, 0);
      }

      Eigen::Matrix<float, 1, 128> descriptor;
      if (options.domain_size_pooling) {
        descriptor = scaled_descriptors.colwise().mean();
      } else {
        descriptor = scaled_descriptors;
      }

      if (options.normalization == SiftExtractionOptions::Normalization::L2) {
        descriptor = L2NormalizeFeatureDescriptors(descriptor);
      } else if (options.normalization ==
                 SiftExtractionOptions::Normalization::L1_ROOT) {
        descriptor = L1RootNormalizeFeatureDescriptors(descriptor);
      } else {
        LOG(FATAL) << "Normalization type not supported";
      }

      descriptors->row(i) = FeatureDescriptorsToUnsignedByte(descriptor);
    }

    *descriptors = TransformVLFeatToUBCFeatureDescriptors(*descriptors);
  }

  return true;
}

bool CreateSiftGPUExtractor(const SiftExtractionOptions& options,
                            SiftGPU* sift_gpu) {
  CHECK(options.Check());
  CHECK_NOTNULL(sift_gpu);

  // SiftGPU uses many global static state variables and the initialization must
  // be thread-safe in order to work correctly. This is enforced here.
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);

  std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
  CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

  std::vector<std::string> sift_gpu_args;

  sift_gpu_args.push_back("./sift_gpu");

#ifdef CUDA_ENABLED
  // Use CUDA version by default if darkness adaptivity is disabled.
  if (!options.darkness_adaptivity && gpu_indices[0] < 0) {
    gpu_indices[0] = 0;
  }

  if (gpu_indices[0] >= 0) {
    sift_gpu_args.push_back("-cuda");
    sift_gpu_args.push_back(std::to_string(gpu_indices[0]));
  }
#endif  // CUDA_ENABLED

  // Darkness adaptivity (hidden feature). Significantly improves
  // distribution of features. Only available in GLSL version.
  if (options.darkness_adaptivity) {
    if (gpu_indices[0] >= 0) {
      WarnDarknessAdaptivityNotAvailable();
    }
    sift_gpu_args.push_back("-da");
  }

  // No verbose logging.
  sift_gpu_args.push_back("-v");
  sift_gpu_args.push_back("0");

  // Fixed maximum image dimension.
  sift_gpu_args.push_back("-maxd");
  sift_gpu_args.push_back(std::to_string(options.max_image_size));

  // Keep the highest level features.
  sift_gpu_args.push_back("-tc2");
  sift_gpu_args.push_back(std::to_string(options.max_num_features));

  // First octave level.
  sift_gpu_args.push_back("-fo");
  sift_gpu_args.push_back(std::to_string(options.first_octave));

  // Number of octave levels.
  sift_gpu_args.push_back("-d");
  sift_gpu_args.push_back(std::to_string(options.octave_resolution));

  // Peak threshold.
  sift_gpu_args.push_back("-t");
  sift_gpu_args.push_back(std::to_string(options.peak_threshold));

  // Edge threshold.
  sift_gpu_args.push_back("-e");
  sift_gpu_args.push_back(std::to_string(options.edge_threshold));

  if (options.upright) {
    // Fix the orientation to 0 for upright features.
    sift_gpu_args.push_back("-ofix");
    // Maximum number of orientations.
    sift_gpu_args.push_back("-mo");
    sift_gpu_args.push_back("1");
  } else {
    // Maximum number of orientations.
    sift_gpu_args.push_back("-mo");
    sift_gpu_args.push_back(std::to_string(options.max_num_orientations));
  }

  std::vector<const char*> sift_gpu_args_cstr;
  sift_gpu_args_cstr.reserve(sift_gpu_args.size());
  for (const auto& arg : sift_gpu_args) {
    sift_gpu_args_cstr.push_back(arg.c_str());
  }

  sift_gpu->ParseParam(sift_gpu_args_cstr.size(), sift_gpu_args_cstr.data());

  return sift_gpu->VerifyContextGL() == SiftGPU::SIFTGPU_FULL_SUPPORTED;
}

bool ExtractSiftFeaturesGPU(const SiftExtractionOptions& options,
                            const Bitmap& bitmap, SiftGPU* sift_gpu,
                            FeatureKeypoints* keypoints,
                            FeatureDescriptors* descriptors) {
  CHECK(options.Check());
  CHECK(bitmap.IsGrey());
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);
  CHECK_EQ(options.max_image_size, sift_gpu->GetMaxDimension());

  CHECK(!options.estimate_affine_shape);
  CHECK(!options.domain_size_pooling);

  // Note, that this produces slightly different results than using SiftGPU
  // directly for RGB->GRAY conversion, since it uses different weights.
  const std::vector<uint8_t> bitmap_raw_bits = bitmap.ConvertToRawBits();
  const int code =
      sift_gpu->RunSIFT(bitmap.ScanWidth(), bitmap.Height(),
                        bitmap_raw_bits.data(), GL_LUMINANCE, GL_UNSIGNED_BYTE);

  const int kSuccessCode = 1;
  if (code != kSuccessCode) {
    return false;
  }

  const size_t num_features = static_cast<size_t>(sift_gpu->GetFeatureNum());

  std::vector<SiftKeypoint> keypoints_data(num_features);

  // Eigen's default is ColMajor, but SiftGPU stores result as RowMajor.
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      descriptors_float(num_features, 128);

  // Download the extracted keypoints and descriptors.
  sift_gpu->GetFeatureVector(keypoints_data.data(), descriptors_float.data());

  keypoints->resize(num_features);
  for (size_t i = 0; i < num_features; ++i) {
    (*keypoints)[i] = FeatureKeypoint(keypoints_data[i].x, keypoints_data[i].y,
                                      keypoints_data[i].s, keypoints_data[i].o);
  }

  // Save and normalize the descriptors.
  if (options.normalization == SiftExtractionOptions::Normalization::L2) {
    descriptors_float = L2NormalizeFeatureDescriptors(descriptors_float);
  } else if (options.normalization ==
             SiftExtractionOptions::Normalization::L1_ROOT) {
    descriptors_float = L1RootNormalizeFeatureDescriptors(descriptors_float);
  } else {
    LOG(FATAL) << "Normalization type not supported";
  }

  *descriptors = FeatureDescriptorsToUnsignedByte(descriptors_float);

  return true;
}

void LoadSiftFeaturesFromTextFile(const std::string& path,
                                  FeatureKeypoints* keypoints,
                                  FeatureDescriptors* descriptors) {
  CHECK_NOTNULL(keypoints);
  CHECK_NOTNULL(descriptors);

  std::ifstream file(path.c_str());
  CHECK(file.is_open()) << path;

  std::string line;
  std::string item;

  std::getline(file, line);
  std::stringstream header_line_stream(line);

  std::getline(header_line_stream >> std::ws, item, ' ');
  const point2D_t num_features = std::stoul(item);

  std::getline(header_line_stream >> std::ws, item, ' ');
  const size_t dim = std::stoul(item);

  CHECK_EQ(dim, 128) << "SIFT features must have 128 dimensions";

  keypoints->resize(num_features);
  descriptors->resize(num_features, dim);

  for (size_t i = 0; i < num_features; ++i) {
    std::getline(file, line);
    std::stringstream feature_line_stream(line);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float x = std::stold(item);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float y = std::stold(item);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float scale = std::stold(item);

    std::getline(feature_line_stream >> std::ws, item, ' ');
    const float orientation = std::stold(item);

    (*keypoints)[i] = FeatureKeypoint(x, y, scale, orientation);

    // Descriptor
    for (size_t j = 0; j < dim; ++j) {
      std::getline(feature_line_stream >> std::ws, item, ' ');
      const float value = std::stod(item);
      CHECK_GE(value, 0);
      CHECK_LE(value, 255);
      (*descriptors)(i, j) = TruncateCast<float, uint8_t>(value);
    }
  }
}

// try to allow pixel imperfect case
void OFGuidedMatchSiftFeaturesCPU(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints& keypoints1,
                          const FeatureKeypoints& keypoints2,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          const FeatureMatches& quantization_map,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  // const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
  //     nullptr, nullptr, descriptors1, descriptors2, nullptr);
  //
  // FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
  //                 match_options.cross_check, matches);
  // std::cout << "#### Enter OFGuidedMatchSiftFeaturesCPU()" << std::endl;
  // std::cout << "keypoints1.size() = " << keypoints1.size() << std::endl;

  // point2D_t image_scale_factor = 24;
  // point2D_t image_scale_factor = 4;
  // point2D_t image_scale_factor = 2;
  // point2D_t image_scale_factor = 16;
  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;
  for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  {
      // std::cout << "quantization_map[cnt] = " << quantization_map[cnt].point2D_idx1 << "," << quantization_map[cnt].point2D_idx2 << std::endl;
      // FeatureDescriptors tmpDescriptors1;
      // std::cout << "tmpDescriptors1 is created!" << std::endl;
      // // std::vector<uint8_t> tmpDescriptors1Vector;
      //std::vector<size_t> tmpIndices1;
      //std::vector<int> tmpIndices1;
      std::vector<point2D_t> tmpIndices1;
      // std::cout << "tmpIndices1 is created!" << std::endl;

      for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
      // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
      {
          // std::cout << "loop kp1Idx ^" << std::endl;
          point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
          float quantizationCenter_y_1 = floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width) * image_scale_factor;
          float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width)*DeMoN_OF_Width);
          // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << std::endl;
          if((pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2))<=400)
          // if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
          {
              // std::cout << "tmpQuantizationIdx1 = quantization_map[cnt].point2D_idx1, they are " << tmpQuantizationIdx1 << std::endl;
              // // tmpDescriptors1Vector.push_back(descriptors1.block<1,128>(kp1Idx,0));
              // tmpDescriptors1 << descriptors1.block<1,128>(kp1Idx,0);
              // std::cout << "tmpDescriptors1 = " << tmpDescriptors1 << std::endl;
              tmpIndices1.push_back(kp1Idx);
          }
          // std::cout << "end of loop kp1Idx ^" << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;
      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
      for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
      {
          tmpDescriptors1.resize(kp1Idx+1, 128);
          tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
      }
      // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;


      // FeatureDescriptors tmpDescriptors2;
      //std::vector<size_t> tmpIndices2;
      //std::vector<int> tmpIndices2;
      std::vector<point2D_t> tmpIndices2;
      for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
      {
          point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
          float quantizationCenter_y_2 = floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width) * image_scale_factor;
          float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width)*DeMoN_OF_Width);
          if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=400)
          // if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
          {
              // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
              tmpIndices2.push_back(kp2Idx);
          }
      }
      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
      for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
      {
          tmpDescriptors2.resize(kp2Idx+1, 128);
          tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
      }
      // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;

      // remember to normalize the descriptors so that colmap threshold params can be used!
      Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
      Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
      desc1 = L1RootNormalizeFeatureDescriptors(desc1);
      desc2 = L1RootNormalizeFeatureDescriptors(desc2);
      // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
      // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);

      const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
          nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
      // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;

      FeatureMatches tmpQuantizationMatches;
      FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                      match_options.cross_check, &tmpQuantizationMatches);
      // std::cout << "FindBestMatches is done!" << std::endl;

      for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches.size(); resultCnt++)
      {
          FeatureMatch ConvertedMatch;
          ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches[resultCnt].point2D_idx1];
          ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches[resultCnt].point2D_idx2];
          matches->push_back(ConvertedMatch);
      }
      // std::cout << "index conversion is done!" << std::endl;
  }

}

void OFGuidedMatchSiftFeaturesCPU_PixelPerfectCase(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints& keypoints1,
                          const FeatureKeypoints& keypoints2,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          const FeatureMatches& quantization_map,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  double uncertainty_radius = match_options.uncertainty_radius;
  // const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
  //     nullptr, nullptr, descriptors1, descriptors2, nullptr);
  //
  // FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
  //                 match_options.cross_check, matches);
  // std::cout << "#### Enter OFGuidedMatchSiftFeaturesCPU()" << std::endl;
  // std::cout << "keypoints1.size() = " << keypoints1.size() << std::endl;

  // point2D_t image_scale_factor = 24;
  // point2D_t image_scale_factor = 32; // 24; // 12; // 48; // 16; //4;
  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t OF_scale_factor = match_options.OF_scale_factor;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;
  for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  {
      // std::cout << "quantization_map[cnt] = " << quantization_map[cnt].point2D_idx1 << "," << quantization_map[cnt].point2D_idx2 << std::endl;
      // FeatureDescriptors tmpDescriptors1;
      // std::cout << "tmpDescriptors1 is created!" << std::endl;
      // // std::vector<uint8_t> tmpDescriptors1Vector;
      //std::vector<size_t> tmpIndices1;
      //std::vector<int> tmpIndices1;
      std::vector<point2D_t> tmpIndices1;
      // std::cout << "tmpIndices1 is created!" << std::endl;

      for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
      // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
      {
          // // std::cout << "loop kp1Idx ^" << std::endl;
          // point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
          // // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << std::endl;
          point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
          float quantizationCenter_y_1 = floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
          if((pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2))<=uncertainty_radius*uncertainty_radius)
          // if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
          {
              // std::cout << "tmpQuantizationIdx1 = quantization_map[cnt].point2D_idx1, they are " << tmpQuantizationIdx1 << std::endl;
              // // tmpDescriptors1Vector.push_back(descriptors1.block<1,128>(kp1Idx,0));
              // tmpDescriptors1 << descriptors1.block<1,128>(kp1Idx,0);
              // std::cout << "tmpDescriptors1 = " << tmpDescriptors1 << std::endl;
              tmpIndices1.push_back(kp1Idx);
          }
          // std::cout << "end of loop kp1Idx ^" << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;
      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
      for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
      {
          tmpDescriptors1.resize(kp1Idx+1, 128);
          tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
      }
      // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;


      // FeatureDescriptors tmpDescriptors2;
      //std::vector<size_t> tmpIndices2;
      //std::vector<int> tmpIndices2;
      std::vector<point2D_t> tmpIndices2;
      for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
      {
          // point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
          point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
          float quantizationCenter_y_2 = floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
          if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
          // if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
          {
              // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
              tmpIndices2.push_back(kp2Idx);
          }
      }
      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
      for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
      {
          tmpDescriptors2.resize(kp2Idx+1, 128);
          tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
      }
      // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;

      // remember to normalize the descriptors so that colmap threshold params can be used!
      Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
      Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
      desc1 = L1RootNormalizeFeatureDescriptors(desc1);
      desc2 = L1RootNormalizeFeatureDescriptors(desc2);
      // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
      // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);

      const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
          nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
      // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;

      FeatureMatches tmpQuantizationMatches;
      FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                      match_options.cross_check, &tmpQuantizationMatches);
      // std::cout << "FindBestMatches is done!" << std::endl;

      for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches.size(); resultCnt++)
      {
          FeatureMatch ConvertedMatch;
          ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches[resultCnt].point2D_idx1];
          ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches[resultCnt].point2D_idx2];
          matches->push_back(ConvertedMatch);
      }
      // std::cout << "index conversion is done!" << std::endl;
  }

}

void OFGuidedMatchSiftFeaturesCPU_One2Multi_byPixel(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints& keypoints1,
                          const FeatureKeypoints& keypoints2,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          const FeatureMatches& quantization_map,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  double uncertainty_radius = match_options.uncertainty_radius;
  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t OF_scale_factor = match_options.OF_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;

  int numQuantizationMapping = quantization_map.size();
  std::unordered_map<point2D_t, point2D_t> mapping1to2;
  std::unordered_map<point2D_t, point2D_t> mapping2to1;
  for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  {
      mapping1to2[quantization_map[cnt].point2D_idx1] = quantization_map[cnt].point2D_idx2;
      mapping2to1[quantization_map[cnt].point2D_idx2] = quantization_map[cnt].point2D_idx1;
  }
  std::cout << "convert quantization map to unordered map successfully!" << std::endl;
  // auto key_selector = [](auto pair){return pair.first;};
  // std::vector<point2D_t> keys1to2(mapping1to2.size());
  // std::vector<point2D_t> keys2to1(mapping2to1.size());
  // std::transform(mapping1to2.begin(), mapping1to2.end(), keys1to2.begin(), key_selector);
  // std::transform(mapping2to1.begin(), mapping2to1.end(), keys2to1.begin(), key_selector);

  //for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  //{

      for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
      // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
      {
          // std::cout << "image_scale_factor = " << image_scale_factor << "; OF_scale_factor = " << OF_scale_factor << std::endl;
          // point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
          // float quantizationCenter_y_1 = floor(tmpQuantizationIdx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          // float quantizationCenter_x_1 = image_scale_factor * (tmpQuantizationIdx1-floor(tmpQuantizationIdx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          point2D_t tmpQuantizationIdx1 = 0;
          point2D_t mappedQuantizationIdx2;
          float retrieved_quantizationCenter_x_1;
          float retrieved_quantizationCenter_y_1;
          float tmpMinSquareDist = 10000.0;
          bool NNflag = false;
          // for(auto element : mapping1to2)
          // for(point2D_t key12 : keys1to2)
          for(std::unordered_map<point2D_t,point2D_t>::iterator it = mapping1to2.begin(); it != mapping1to2.end(); ++it)
          {
              point2D_t tmpIdx1 = it->first;
              float quantizationCenter_y_1 = floor(tmpIdx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
              float quantizationCenter_x_1 = image_scale_factor * (tmpIdx1-floor(tmpIdx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
              float tmpSquareDist = (pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2));
              if(tmpSquareDist<=tmpMinSquareDist)
              {
                  tmpQuantizationIdx1 = tmpIdx1;
                  tmpMinSquareDist = tmpSquareDist;
                  NNflag = true;
                  mappedQuantizationIdx2 = it->second;
                  retrieved_quantizationCenter_x_1 = quantizationCenter_x_1;
                  retrieved_quantizationCenter_y_1 = quantizationCenter_y_1;
              }
          }
          if(NNflag==false || tmpMinSquareDist>5*image_scale_factor*image_scale_factor)
          {
              // std::cout << "skip this kp1, no NN quantization center could be retrieved!" << std::endl;
              continue;
          }
          // // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << ", before conversion tmpQuantizationIdx1 = " << (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor) << "; quanCenter = (" << quantizationCenter_x_1 << ", " << quantizationCenter_y_1 << ")" << "; keypoints1[kp1Idx] = (" << keypoints1[kp1Idx].x << ", " << keypoints1[kp1Idx].y << ")" << std::endl;
          // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << "; quanCenter = (" << retrieved_quantizationCenter_x_1 << ", " << retrieved_quantizationCenter_y_1 << ")" << "; keypoints1[kp1Idx] = (" << keypoints1[kp1Idx].x << ", " << keypoints1[kp1Idx].y << ")" << std::endl;

          // //int shareIdKp1Cnt = 0;
          // // point2D_t mappedQuantizationIdx2 = mapping1to2[tmpQuantizationIdx1];
          // point2D_t mappedQuantizationIdx2;
          // if(mapping1to2.count(tmpQuantizationIdx1) > 0)
          // {
          //     mappedQuantizationIdx2 = mapping1to2[tmpQuantizationIdx1];
          // } else {
          //     continue;
          // }

          //if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
          //{
              //shareIdKp1Cnt++;
              std::vector<point2D_t> tmpIndices1;
              // std::cout << "tmpIndices1 is created!" << std::endl;

              tmpIndices1.push_back(kp1Idx);
              // std::cout << "tmpIndices1.size() = " << tmpIndices1.size() << ", tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << "; quanCenter = (" << retrieved_quantizationCenter_x_1 << ", " << retrieved_quantizationCenter_y_1 << ")" << "; keypoints1[kp1Idx] = (" << keypoints1[kp1Idx].x << ", " << keypoints1[kp1Idx].y << ")" << std::endl;

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
              for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
              {
                  tmpDescriptors1.resize(kp1Idx+1, 128);
                  tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
              }
              // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;

              // std::vector<point2D_t> tmpIndices2;
              // for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
              // {
              //     point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
              //     if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
              //     {
              //         // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
              //         tmpIndices2.push_back(kp2Idx);
              //     }
              // }

              std::vector<point2D_t> tmpIndices2;
              for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
              {
                  point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
                  float quantizationCenter_y_2 = floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
                  float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
                  // float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
                  if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
                  // if(tmpQuantizationIdx2==mappedQuantizationIdx2)
                  {
                      // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
                      tmpIndices2.push_back(kp2Idx);
                  }
                  // std::cout << "mappedQuantizationIdx2 = " << mappedQuantizationIdx2 << "; @@@ quantizationCenter 2 = (" << quantizationCenter_x_2 << ", " << quantizationCenter_y_2 << ") ? quantizationCenter 2 before saving to float = (" << image_scale_factor * (mappedQuantizationIdx2-floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor)) << ", " << floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor << "), keypoints2[kp2Idx] = (" << keypoints2[kp2Idx].x << ", " << keypoints2[kp2Idx].y << ")" << std::endl;
                  // std::cout << "## image_scale_factor = " << image_scale_factor << ", OF_scale_factor = "<< OF_scale_factor << ", tmpQuantizationIdx1 = "<< tmpQuantizationIdx1 << ", keypoints1[kp1Idx] = (" << keypoints1[kp1Idx].x << ", " << keypoints1[kp1Idx].y << "); mappedQuantizationIdx2 = " << mappedQuantizationIdx2 << "; @@@ quantizationCenter 2 = (" << quantizationCenter_x_2 << ", " << quantizationCenter_y_2 << "), keypoints2[kp2Idx] = (" << keypoints2[kp2Idx].x << ", " << keypoints2[kp2Idx].y << ")" << std::endl;
              }
              // std::cout << "~~ tmpIndices2.size() = " << tmpIndices2.size() << "; keypoints2.size() = " << keypoints2.size() << std::endl;


              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
              for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
              {
                  tmpDescriptors2.resize(kp2Idx+1, 128);
                  tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
              }
              // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;
              // std::cout << "!! tmpDescriptors1.rows() = " << tmpDescriptors1.rows()  << "; tmpDescriptor1.cols()  = " << tmpDescriptors1.cols() << "!! tmpDescriptors2.rows() = " << tmpDescriptors2.rows()  << "; tmpDescriptors2.cols()  = " << tmpDescriptors2.cols() << std::endl;
              // Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
              // std::cout << "!! tmpDescriptors1 = " << tmpDescriptors1.format(OctaveFmt)<< std::endl;
              // // remember to normalize the descriptors so that colmap threshold params can be used!
              // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
              // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
              // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
              // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
              // // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
              // // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);

              // const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
              //     nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
              const Eigen::MatrixXf dists = ComputeSiftDistanceMatrix_Kevin(tmpDescriptors1, tmpDescriptors2);
              // std::cout << "ComputeSiftDistanceMatrix is done! dists.rows() = " <<  dists.rows() << ";  dists.cols() = " <<  dists.cols() << std::endl;

              // FeatureMatches tmpQuantizationMatches;
              // FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
              //                 match_options.cross_check, &tmpQuantizationMatches);
              // const size_t numMatch12Tmp;
              std::vector<int> tmpQuantizationMatch;
              const size_t numMatch12Tmp = FindBestMatchesOneWay_One2Multi(dists, match_options.max_ratio, match_options.max_distance, &tmpQuantizationMatch);
              // std::cout << "FindBestMatches is done! numMatch12Tmp = " << numMatch12Tmp << "; tmpQuantizationMatch.size() = " << tmpQuantizationMatch.size()<< "; tmpQuantizationMatch = " << tmpQuantizationMatch[0] << std::endl;

              // for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches.size(); resultCnt++)
              // {
              //     FeatureMatch ConvertedMatch;
              //     ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches[resultCnt].point2D_idx1];
              //     ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches[resultCnt].point2D_idx2];
              //     matches->push_back(ConvertedMatch);
              // }
              if(numMatch12Tmp==1)
              {
                  FeatureMatch ConvertedMatch;
                  ConvertedMatch.point2D_idx1 = tmpIndices1[0];
                  ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatch[0]];
                  matches->push_back(ConvertedMatch);
              }
              // std::cout << "index conversion is done!" << std::endl;

          //}
          // // std::cout << "end of loop kp1Idx ^" << std::endl;
          // std::cout << "@@@ tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << ", @@@ mappedQuantizationIdx2 = " << mappedQuantizationIdx2 << std::endl;
          // std::cout << "@@@ tmpIndices2.size() = " << tmpIndices2.size() << ", @@@ tmpDescriptors2.rows() = " << tmpDescriptors2.rows() << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;

  //}
  std::cout << "@@@ Final raw match number => matches->size() = " << matches->size() << "; keypoints1.size() = " << keypoints1.size() << std::endl;
}


// void OFGuidedMatchSiftFeaturesCPU_One2Multi_byPixel_ColmapFormat_bak(const SiftMatchingOptions& match_options,
//                           const FeatureKeypoints& keypoints1,
//                           const FeatureKeypoints& keypoints2,
//                           const FeatureDescriptors& descriptors1,
//                           const FeatureDescriptors& descriptors2,
//                           const FeatureMatches& quantization_map,
//                           FeatureMatches* matches) {
//   CHECK(match_options.Check());
//   CHECK_NOTNULL(matches);
//
//   double uncertainty_radius = match_options.uncertainty_radius;
//   point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
//   point2D_t OF_scale_factor = match_options.OF_scale_factor; // 24; // 12; // 48; // 16; //4;
//   point2D_t DeMoN_OF_Height = 48;
//   point2D_t DeMoN_OF_Width = 64;
//
//   int numQuantizationMapping = quantization_map.size();
//   std::unordered_map<point2D_t, point2D_t> mapping1to2;
//   std::unordered_map<point2D_t, point2D_t> mapping2to1;
//   for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
//   {
//       mapping1to2[quantization_map[cnt].point2D_idx1] = quantization_map[cnt].point2D_idx2;
//       mapping2to1[quantization_map[cnt].point2D_idx2] = quantization_map[cnt].point2D_idx1;
//   }
//   std::cout << "convert quantization map to unordered map successfully!" << std::endl;
//   // auto key_selector = [](auto pair){return pair.first;};
//   // std::vector<point2D_t> keys1to2(mapping1to2.size());
//   // std::vector<point2D_t> keys2to1(mapping2to1.size());
//   // std::transform(mapping1to2.begin(), mapping1to2.end(), keys1to2.begin(), key_selector);
//   // std::transform(mapping2to1.begin(), mapping2to1.end(), keys2to1.begin(), key_selector);
//
//   //for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
//   //{
//
//       for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
//       // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
//       {
//           // std::cout << "image_scale_factor = " << image_scale_factor << "; OF_scale_factor = " << OF_scale_factor << std::endl;
//           // point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
//           // float quantizationCenter_y_1 = floor(tmpQuantizationIdx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
//           // float quantizationCenter_x_1 = image_scale_factor * (tmpQuantizationIdx1-floor(tmpQuantizationIdx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
//           point2D_t tmpQuantizationIdx1 = 0;
//           point2D_t mappedQuantizationIdx2;
//           float retrieved_quantizationCenter_x_1;
//           float retrieved_quantizationCenter_y_1;
//           float tmpMinSquareDist = 10000.0;
//           bool NNflag = false;
//           // for(auto element : mapping1to2)
//           // for(point2D_t key12 : keys1to2)
//           for(std::unordered_map<point2D_t,point2D_t>::iterator it = mapping1to2.begin(); it != mapping1to2.end(); ++it)
//           {
//               point2D_t tmpIdx1 = it->first;
//               float quantizationCenter_y_1 = floor(tmpIdx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
//               float quantizationCenter_x_1 = image_scale_factor * (tmpIdx1-floor(tmpIdx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
//               float tmpSquareDist = (pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2));
//               if(tmpSquareDist<=tmpMinSquareDist)
//               {
//                   tmpQuantizationIdx1 = tmpIdx1;
//                   tmpMinSquareDist = tmpSquareDist;
//                   NNflag = true;
//                   mappedQuantizationIdx2 = it->second;
//                   retrieved_quantizationCenter_x_1 = quantizationCenter_x_1;
//                   retrieved_quantizationCenter_y_1 = quantizationCenter_y_1;
//               }
//           }
//           if(NNflag==false || tmpMinSquareDist>5*image_scale_factor*image_scale_factor)
//           {
//               // std::cout << "skip this kp1, no NN quantization center could be retrieved!" << std::endl;
//               continue;
//           }
//           // // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << ", before conversion tmpQuantizationIdx1 = " << (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor) << "; quanCenter = (" << quantizationCenter_x_1 << ", " << quantizationCenter_y_1 << ")" << "; keypoints1[kp1Idx] = (" << keypoints1[kp1Idx].x << ", " << keypoints1[kp1Idx].y << ")" << std::endl;
//           // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << "; quanCenter = (" << retrieved_quantizationCenter_x_1 << ", " << retrieved_quantizationCenter_y_1 << ")" << "; keypoints1[kp1Idx] = (" << keypoints1[kp1Idx].x << ", " << keypoints1[kp1Idx].y << ")" << std::endl;
//
//           // //int shareIdKp1Cnt = 0;
//           // // point2D_t mappedQuantizationIdx2 = mapping1to2[tmpQuantizationIdx1];
//           // point2D_t mappedQuantizationIdx2;
//           // if(mapping1to2.count(tmpQuantizationIdx1) > 0)
//           // {
//           //     mappedQuantizationIdx2 = mapping1to2[tmpQuantizationIdx1];
//           // } else {
//           //     continue;
//           // }
//
//           //if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
//           //{
//               //shareIdKp1Cnt++;
//               std::vector<point2D_t> tmpIndices1;
//               // std::cout << "tmpIndices1 is created!" << std::endl;
//
//               tmpIndices1.push_back(kp1Idx);
//               // std::cout << "tmpIndices1.size() = " << tmpIndices1.size() << ", tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << "; quanCenter = (" << retrieved_quantizationCenter_x_1 << ", " << retrieved_quantizationCenter_y_1 << ")" << "; keypoints1[kp1Idx] = (" << keypoints1[kp1Idx].x << ", " << keypoints1[kp1Idx].y << ")" << std::endl;
//
//               Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
//               for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
//               {
//                   tmpDescriptors1.resize(kp1Idx+1, 128);
//                   tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
//               }
//               // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;
//
//               // std::vector<point2D_t> tmpIndices2;
//               // for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
//               // {
//               //     point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
//               //     if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
//               //     {
//               //         // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
//               //         tmpIndices2.push_back(kp2Idx);
//               //     }
//               // }
//
//               std::vector<point2D_t> tmpIndices2;
//               for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
//               {
//                   point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
//                   float quantizationCenter_y_2 = floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
//                   float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
//                   // float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
//                   if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
//                   // if(tmpQuantizationIdx2==mappedQuantizationIdx2)
//                   {
//                       // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
//                       tmpIndices2.push_back(kp2Idx);
//                   }
//                   // std::cout << "mappedQuantizationIdx2 = " << mappedQuantizationIdx2 << "; @@@ quantizationCenter 2 = (" << quantizationCenter_x_2 << ", " << quantizationCenter_y_2 << ") ? quantizationCenter 2 before saving to float = (" << image_scale_factor * (mappedQuantizationIdx2-floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor)) << ", " << floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor << "), keypoints2[kp2Idx] = (" << keypoints2[kp2Idx].x << ", " << keypoints2[kp2Idx].y << ")" << std::endl;
//                   // std::cout << "## image_scale_factor = " << image_scale_factor << ", OF_scale_factor = "<< OF_scale_factor << ", tmpQuantizationIdx1 = "<< tmpQuantizationIdx1 << ", keypoints1[kp1Idx] = (" << keypoints1[kp1Idx].x << ", " << keypoints1[kp1Idx].y << "); mappedQuantizationIdx2 = " << mappedQuantizationIdx2 << "; @@@ quantizationCenter 2 = (" << quantizationCenter_x_2 << ", " << quantizationCenter_y_2 << "), keypoints2[kp2Idx] = (" << keypoints2[kp2Idx].x << ", " << keypoints2[kp2Idx].y << ")" << std::endl;
//               }
//               // std::cout << "~~ tmpIndices2.size() = " << tmpIndices2.size() << "; keypoints2.size() = " << keypoints2.size() << std::endl;
//
//
//               Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
//               for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
//               {
//                   tmpDescriptors2.resize(kp2Idx+1, 128);
//                   tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
//               }
//               // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;
//               // std::cout << "!! tmpDescriptors1.rows() = " << tmpDescriptors1.rows()  << "; tmpDescriptor1.cols()  = " << tmpDescriptors1.cols() << "!! tmpDescriptors2.rows() = " << tmpDescriptors2.rows()  << "; tmpDescriptors2.cols()  = " << tmpDescriptors2.cols() << std::endl;
//               // Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
//               // std::cout << "!! tmpDescriptors1 = " << tmpDescriptors1.format(OctaveFmt)<< std::endl;
//               // // remember to normalize the descriptors so that colmap threshold params can be used!
//               // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
//               // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
//               // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
//               // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
//               // // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
//               // // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);
//
//               const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
//                   nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
//               // const Eigen::MatrixXf dists = ComputeSiftDistanceMatrix_Kevin(tmpDescriptors1, tmpDescriptors2);
//               // // std::cout << "ComputeSiftDistanceMatrix is done! dists.rows() = " <<  dists.rows() << ";  dists.cols() = " <<  dists.cols() << std::endl;
//
//               FeatureMatches tmpQuantizationMatches;
//               FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
//                               match_options.cross_check, &tmpQuantizationMatches);
//               // // const size_t numMatch12Tmp;
//               // std::vector<int> tmpQuantizationMatch;
//               // const size_t numMatch12Tmp = FindBestMatchesOneWay_One2Multi(dists, match_options.max_ratio, match_options.max_distance, &tmpQuantizationMatch);
//               // // std::cout << "FindBestMatches is done! numMatch12Tmp = " << numMatch12Tmp << "; tmpQuantizationMatch.size() = " << tmpQuantizationMatch.size()<< "; tmpQuantizationMatch = " << tmpQuantizationMatch[0] << std::endl;
//
//               for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches.size(); resultCnt++)
//               {
//                   FeatureMatch ConvertedMatch;
//                   ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches[resultCnt].point2D_idx1];
//                   ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches[resultCnt].point2D_idx2];
//                   matches->push_back(ConvertedMatch);
//               }
//               // if(numMatch12Tmp==1)
//               // {
//               //     FeatureMatch ConvertedMatch;
//               //     ConvertedMatch.point2D_idx1 = tmpIndices1[0];
//               //     ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatch[0]];
//               //     matches->push_back(ConvertedMatch);
//               // }
//               // // std::cout << "index conversion is done!" << std::endl;
//
//           //}
//           // // std::cout << "end of loop kp1Idx ^" << std::endl;
//           // std::cout << "@@@ tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << ", @@@ mappedQuantizationIdx2 = " << mappedQuantizationIdx2 << std::endl;
//           // std::cout << "@@@ tmpIndices2.size() = " << tmpIndices2.size() << ", @@@ tmpDescriptors2.rows() = " << tmpDescriptors2.rows() << std::endl;
//       }
//       // std::cout << "end of loop kp1Idx ^" << std::endl;
//
//   //}
//   std::cout << "@@@ Final raw match number => matches->size() = " << matches->size() << "; keypoints1.size() = " << keypoints1.size() << std::endl;
// }

// void OFGuidedMatchSiftFeaturesCPU_One2Multi_byPixel(const SiftMatchingOptions& match_options,
//                           const FeatureKeypoints& keypoints1,
//                           const FeatureKeypoints& keypoints2,
//                           const FeatureDescriptors& descriptors1,
//                           const FeatureDescriptors& descriptors2,
//                           const FeatureMatches& quantization_map,
//                           FeatureMatches* matches) {
//   CHECK(match_options.Check());
//   CHECK_NOTNULL(matches);
//
//   double uncertainty_radius = match_options.uncertainty_radius;
//   point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
//   point2D_t OF_scale_factor = match_options.OF_scale_factor; // 24; // 12; // 48; // 16; //4;
//   point2D_t DeMoN_OF_Height = 48;
//   point2D_t DeMoN_OF_Width = 64;
//   for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
//   {
//
//       for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
//       // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
//       {
//           // std::cout << "image_scale_factor = " << image_scale_factor << "; OF_scale_factor = " << OF_scale_factor << std::endl;
//           point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
//           // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << std::endl;
//           int shareIdKp1Cnt = 0;
//           if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
//           {
//               shareIdKp1Cnt++;
//               std::vector<point2D_t> tmpIndices1;
//               // std::cout << "tmpIndices1 is created!" << std::endl;
//
//               tmpIndices1.push_back(kp1Idx);
//
//               Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
//               for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
//               {
//                   tmpDescriptors1.resize(kp1Idx+1, 128);
//                   tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
//               }
//               // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;
//
//               // std::vector<point2D_t> tmpIndices2;
//               // for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
//               // {
//               //     point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
//               //     if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
//               //     {
//               //         // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
//               //         tmpIndices2.push_back(kp2Idx);
//               //     }
//               // }
//
//               std::vector<point2D_t> tmpIndices2;
//               // point2D_t tmpQuantizationIdx2 = (keypoints2[0].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[0].y / image_scale_factor);
//               // float quantizationCenter_y_2 = floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
//               // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
//               // std::cout << "~~ quantizationCenter_x_2 = " << quantizationCenter_x_2 << "~~ quantizationCenter_y_2 = " << quantizationCenter_y_2 << "~~ tmpQuantizationIdx2 = " << tmpQuantizationIdx2 << std::endl;
//               for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
//               {
//                   point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
//                   float quantizationCenter_y_2 = floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
//                   float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
//                   // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
//                   if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
//                   // if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
//                   {
//                       // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
//                       tmpIndices2.push_back(kp2Idx);
//                   }
//               }
//               // std::cout << "~~ tmpIndices2.size() = " << tmpIndices2.size() << std::endl;
//
//
//               Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
//               for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
//               {
//                   tmpDescriptors2.resize(kp2Idx+1, 128);
//                   tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
//               }
//               // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;
//
//               // // remember to normalize the descriptors so that colmap threshold params can be used!
//               // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
//               // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
//               // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
//               // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
//               // // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
//               // // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);
//
//               const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
//                   nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
//               // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;
//
//               FeatureMatches tmpQuantizationMatches;
//               FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
//                               match_options.cross_check, &tmpQuantizationMatches);
//               // std::cout << "FindBestMatches is done!" << std::endl;
//
//               for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches.size(); resultCnt++)
//               {
//                   FeatureMatch ConvertedMatch;
//                   ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches[resultCnt].point2D_idx1];
//                   ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches[resultCnt].point2D_idx2];
//                   matches->push_back(ConvertedMatch);
//               }
//               // std::cout << "index conversion is done!" << std::endl;
//
//           }
//           // std::cout << "end of loop kp1Idx ^" << std::endl;
//           // std::cout << "@@@ shareIdKp1Cnt = " << shareIdKp1Cnt << std::endl;
//
//       }
//       // std::cout << "end of loop kp1Idx ^" << std::endl;
//
//   }
//
// }

// void OFGuidedMatchSiftFeaturesCPU_One2Multi_byPixel_ManualCrossCheck(const SiftMatchingOptions& match_options,
//                           const FeatureKeypoints& keypoints1,
//                           const FeatureKeypoints& keypoints2,
//                           const FeatureDescriptors& descriptors1,
//                           const FeatureDescriptors& descriptors2,
//                           const FeatureMatches& quantization_map,
//                           FeatureMatches* matches) {
//   CHECK(match_options.Check());
//   CHECK_NOTNULL(matches);
//
//   double uncertainty_radius = match_options.uncertainty_radius;
//   point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
//   point2D_t OF_scale_factor = match_options.OF_scale_factor; // 24; // 12; // 48; // 16; //4;
//   point2D_t DeMoN_OF_Height = 48;
//   point2D_t DeMoN_OF_Width = 64;
//   for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
//   {
//       FeatureMatches matches1to2;
//       std::vector<int> matches12;
//       // const size_t num_matches12;
//       size_t num_matches12 = 0;
//       matches12.resize(keypoints1.size(), -1);
//
//       for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
//       // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
//       {
//           // std::cout << "image_scale_factor = " << image_scale_factor << "; OF_scale_factor = " << OF_scale_factor << std::endl;
//           point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
//           // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << std::endl;
//           int shareIdKp1Cnt = 0;
//
//           // float quantizationCenter_y_1 = floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
//           // float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
//           // // float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-quantizationCenter_y_1*(DeMoN_OF_Width * OF_scale_factor));
//           // if((pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2)) < image_scale_factor * 1)//uncertainty_radius*uncertainty_radius)
//           if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
//           {
//               std::cout << "direction 1 ---> 2" << std::endl;
//               shareIdKp1Cnt++;
//               std::vector<point2D_t> tmpIndices1;
//               // std::cout << "tmpIndices1 is created!" << std::endl;
//
//               tmpIndices1.push_back(kp1Idx);
//
//               Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
//               for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
//               {
//                   tmpDescriptors1.resize(kp1Idx+1, 128);
//                   tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
//               }
//               // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;
//
//               std::vector<point2D_t> tmpIndices2;
//               // point2D_t tmpQuantizationIdx2 = (keypoints2[0].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[0].y / image_scale_factor);
//               // float quantizationCenter_y_2 = floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
//               // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
//               // std::cout << "~~ quantizationCenter_x_2 = " << quantizationCenter_x_2 << "~~ quantizationCenter_y_2 = " << quantizationCenter_y_2 << "~~ tmpQuantizationIdx2 = " << tmpQuantizationIdx2 << std::endl;
//               for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
//               {
//                   point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
//                   float quantizationCenter_y_2 = floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
//                   float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
//                   // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
//                   if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
//                   // if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
//                   {
//                       // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
//                       tmpIndices2.push_back(kp2Idx);
//                   }
//               }
//               // std::cout << "~~ tmpIndices2.size() = " << tmpIndices2.size() << std::endl;
//
//               Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
//               for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
//               {
//                   tmpDescriptors2.resize(kp2Idx+1, 128);
//                   tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
//               }
//               // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;
//
//               // // remember to normalize the descriptors so that colmap threshold params can be used!
//               // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
//               // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
//               // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
//               // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
//               // // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
//               // // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);
//
//               const Eigen::MatrixXi dists12 = ComputeSiftDistanceMatrix(
//                   nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
//               // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;
//               // num_matches12 = FindBestMatchesOneWay(dists12, match_options.max_ratio, match_options.max_distance, &matches12);
//
//               FeatureMatches tmpQuantizationMatches12;
//               FindBestMatches(dists12, match_options.max_ratio, match_options.max_distance,
//                               match_options.cross_check, &tmpQuantizationMatches12);
//               // std::cout << "FindBestMatches is done!" << std::endl;
//
//               for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches12.size(); resultCnt++)
//               {
//                   FeatureMatch ConvertedMatch;
//                   ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches12[resultCnt].point2D_idx1];
//                   ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches12[resultCnt].point2D_idx2];
//                   // matches->push_back(ConvertedMatch);
//                   matches1to2.push_back(ConvertedMatch);
//                   matches12[ConvertedMatch.point2D_idx1] = ConvertedMatch.point2D_idx2;
//                   num_matches12++;
//               }
//               // std::cout << "index conversion is done!" << std::endl;
//           }
//           // std::cout << "end of loop kp1Idx ^" << std::endl;
//           // std::cout << "@@@ shareIdKp1Cnt = " << shareIdKp1Cnt << std::endl;
//       }
//       // std::cout << "end of loop kp1Idx ^" << std::endl;
//
//       ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//       FeatureMatches matches2to1;
//       std::vector<int> matches21;
//       // const size_t num_matches21;
//       size_t num_matches21 = 0;
//       matches21.resize(keypoints2.size(), -1);
//
//       for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
//       {
//           point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
//           int shareIdKp2Cnt = 0;
//
//           // float quantizationCenter_y_2 = floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
//           // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
//           // // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
//           // if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2)) < image_scale_factor * 1)//uncertainty_radius*uncertainty_radius)
//           if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
//           {
//               std::cout << "@ direction 2 ---> 1" << std::endl;
//               shareIdKp2Cnt++;
//               std::vector<point2D_t> tmpIndices2;
//               // std::cout << "tmpIndices2 is created!" << std::endl;
//
//               tmpIndices2.push_back(kp2Idx);
//
//               Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
//               for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
//               {
//                   tmpDescriptors2.resize(kp2Idx+1, 128);
//                   tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
//               }
//               // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;
//
//               std::vector<point2D_t> tmpIndices1;
//               for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
//               {
//                   point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
//                   float quantizationCenter_y_1 = floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
//                   float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
//                   // float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-quantizationCenter_y_1*(DeMoN_OF_Width * OF_scale_factor));
//                   if((pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2))<=uncertainty_radius*uncertainty_radius)
//                   // if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
//                   {
//                       // tmpDescriptors1 << descriptors1.block<1,128>(kp1Idx,0);
//                       tmpIndices1.push_back(kp1Idx);
//                   }
//               }
//               // std::cout << "~~ tmpIndices1.size() = " << tmpIndices1.size() << std::endl;
//
//               Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
//               for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
//               {
//                   tmpDescriptors1.resize(kp1Idx+1, 128);
//                   tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
//               }
//               // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;
//
//               const Eigen::MatrixXi dists21 = ComputeSiftDistanceMatrix(
//                   nullptr, nullptr, tmpDescriptors2, tmpDescriptors1, nullptr);
//               // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;
//
//               // num_matches21 = FindBestMatchesOneWay(dists21, match_options.max_ratio, match_options.max_distance, &matches21);
//
//               FeatureMatches tmpQuantizationMatches21;
//               FindBestMatches(dists21, match_options.max_ratio, match_options.max_distance,
//                               match_options.cross_check, &tmpQuantizationMatches21);
//               // std::cout << "FindBestMatches is done!" << std::endl;
//
//               for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches21.size(); resultCnt++)
//               {
//                   FeatureMatch ConvertedMatch;
//                   ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches21[resultCnt].point2D_idx2];
//                   ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches21[resultCnt].point2D_idx1];
//                   // matches->push_back(ConvertedMatch);
//                   matches2to1.push_back(ConvertedMatch);
//                   matches21[ConvertedMatch.point2D_idx2] = ConvertedMatch.point2D_idx1;
//                   num_matches21++;
//               }
//               // // std::cout << "index conversion is done!" << std::endl;
//           }
//           // std::cout << "end of loop kp1Idx ^" << std::endl;
//           // std::cout << "@@@ shareIdKp1Cnt = " << shareIdKp1Cnt << std::endl;
//       }
//       // std::cout << "end of loop kp1Idx ^" << std::endl;
//       /////////////////////////////////////////////////////////////////////
//       /******* Manually cross checking *******/
//       if (true) {
//         std::cout << "@@@ num_matches12 = " << num_matches12 << ", @@@ matches12.size() = " << matches12.size() << ", @@@ num_matches21 = " << num_matches21 << ", @@@ matches21.size() = " << matches21.size() << std::endl;
//         matches->reserve(std::min(num_matches12, num_matches21));
//         for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
//           if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
//               matches21[matches12[i1]] == static_cast<int>(i1)) {
//             FeatureMatch match;
//             match.point2D_idx1 = i1;
//             match.point2D_idx2 = matches12[i1];
//             matches->push_back(match);
//           }
//         }
//       }
//       /////////////////////////////////////////////////////////////////////
//   }
//
// }

void OFGuidedMatchSiftFeaturesCPU_One2Multi_byPixel_ColmapFormat(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints& keypoints1,
                          const FeatureKeypoints& keypoints2,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          const FeatureMatches& quantization_map,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  double uncertainty_radius = match_options.uncertainty_radius;
  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t OF_scale_factor = match_options.OF_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;

  int numQuantizationMapping = quantization_map.size();
  std::unordered_map<point2D_t, point2D_t> mapping1to2;
  std::unordered_map<point2D_t, point2D_t> mapping2to1;
  for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  {
      mapping1to2[quantization_map[cnt].point2D_idx1] = quantization_map[cnt].point2D_idx2;
      mapping2to1[quantization_map[cnt].point2D_idx2] = quantization_map[cnt].point2D_idx1;
  }
  std::cout << "convert quantization map to unordered map successfully!" << std::endl;

  for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
  // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
  {
      // std::cout << "image_scale_factor = " << image_scale_factor << "; OF_scale_factor = " << OF_scale_factor << std::endl;
      // point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
      // float quantizationCenter_y_1 = floor(tmpQuantizationIdx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
      // float quantizationCenter_x_1 = image_scale_factor * (tmpQuantizationIdx1-floor(tmpQuantizationIdx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
      point2D_t tmpQuantizationIdx1 = 0;
      point2D_t mappedQuantizationIdx2;
      float retrieved_quantizationCenter_x_1;
      float retrieved_quantizationCenter_y_1;
      float tmpMinSquareDist = 10000.0;
      bool NNflag = false;
      // for(auto element : mapping1to2)
      // for(point2D_t key12 : keys1to2)
      for(std::unordered_map<point2D_t,point2D_t>::iterator it = mapping1to2.begin(); it != mapping1to2.end(); ++it)
      {
          point2D_t tmpIdx1 = it->first;
          float quantizationCenter_y_1 = floor(tmpIdx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          float quantizationCenter_x_1 = image_scale_factor * (tmpIdx1-floor(tmpIdx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          float tmpSquareDist = (pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2));
          if(tmpSquareDist<=tmpMinSquareDist)
          {
              tmpQuantizationIdx1 = tmpIdx1;
              tmpMinSquareDist = tmpSquareDist;
              NNflag = true;
              mappedQuantizationIdx2 = it->second;
              retrieved_quantizationCenter_x_1 = quantizationCenter_x_1;
              retrieved_quantizationCenter_y_1 = quantizationCenter_y_1;
          }
      }
      if(NNflag==false || tmpMinSquareDist>5*image_scale_factor*image_scale_factor)
      {
          // std::cout << "skip this kp1, no NN quantization center could be retrieved!" << std::endl;
          continue;
      }

      //shareIdKp1Cnt++;
      std::vector<point2D_t> tmpIndices1;
      // std::cout << "tmpIndices1 is created!" << std::endl;

      tmpIndices1.push_back(kp1Idx);
      // std::cout << "tmpIndices1.size() = " << tmpIndices1.size() << ", tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << "; quanCenter = (" << retrieved_quantizationCenter_x_1 << ", " << retrieved_quantizationCenter_y_1 << ")" << "; keypoints1[kp1Idx] = (" << keypoints1[kp1Idx].x << ", " << keypoints1[kp1Idx].y << ")" << std::endl;

      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
      for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
      {
          tmpDescriptors1.resize(kp1Idx+1, 128);
          tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
      }
      // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;

      std::vector<point2D_t> tmpIndices2;
      for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
      {
          point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
          float quantizationCenter_y_2 = floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          // float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
          if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
          // if(tmpQuantizationIdx2==mappedQuantizationIdx2)
          {
              // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
              tmpIndices2.push_back(kp2Idx);
          }
      }
      // std::cout << "~~ tmpIndices2.size() = " << tmpIndices2.size() << "; keypoints2.size() = " << keypoints2.size() << std::endl;


      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
      for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
      {
          tmpDescriptors2.resize(kp2Idx+1, 128);
          tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
      }
      // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;
      // std::cout << "!! tmpDescriptors1.rows() = " << tmpDescriptors1.rows()  << "; tmpDescriptor1.cols()  = " << tmpDescriptors1.cols() << "!! tmpDescriptors2.rows() = " << tmpDescriptors2.rows()  << "; tmpDescriptors2.cols()  = " << tmpDescriptors2.cols() << std::endl;
      // Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
      // std::cout << "!! tmpDescriptors1 = " << tmpDescriptors1.format(OctaveFmt)<< std::endl;
      // // remember to normalize the descriptors so that colmap threshold params can be used!
      // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
      // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
      // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
      // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
      // // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
      // // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);

      const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
          nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
      // const Eigen::MatrixXf dists = ComputeSiftDistanceMatrix_Kevin(tmpDescriptors1, tmpDescriptors2);
      // // std::cout << "ComputeSiftDistanceMatrix is done! dists.rows() = " <<  dists.rows() << ";  dists.cols() = " <<  dists.cols() << std::endl;

      FeatureMatches tmpQuantizationMatches;
      FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                      match_options.cross_check, &tmpQuantizationMatches);
      // // const size_t numMatch12Tmp;
      // std::vector<int> tmpQuantizationMatch;
      // const size_t numMatch12Tmp = FindBestMatchesOneWay_One2Multi(dists, match_options.max_ratio, match_options.max_distance, &tmpQuantizationMatch);
      // // std::cout << "FindBestMatches is done! numMatch12Tmp = " << numMatch12Tmp << "; tmpQuantizationMatch.size() = " << tmpQuantizationMatch.size()<< "; tmpQuantizationMatch = " << tmpQuantizationMatch[0] << std::endl;

      for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches.size(); resultCnt++)
      {
          FeatureMatch ConvertedMatch;
          ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches[resultCnt].point2D_idx1];
          ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches[resultCnt].point2D_idx2];
          matches->push_back(ConvertedMatch);
      }
  }
  std::cout << "@@@ Final raw match number => matches->size() = " << matches->size() << "; keypoints1.size() = " << keypoints1.size() << std::endl;
}

float computeBilinearInterpolation(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y){
    float x2x1, y2y1, x2x, y2y, yy1, xx1;
    x2x1 = x2 - x1;
    y2y1 = y2 - y1;
    if(x2x1==0 && y2y1==0){
        return q11;
    }
    if(x2x1==0 && y2y1!=0){
        y2y = y2 - y;
        yy1 = y - y1;
        return q11 + (q12-q11)*(yy1/y2y1);
    }
    if(x2x1!=0 && y2y1==0){
        x2x = x2 - x;
        xx1 = x - x1;
        return q11 + (q21-q11)*(xx1/x2x1);
    }
    x2x = x2 - x;
    y2y = y2 - y;
    yy1 = y - y1;
    xx1 = x - x1;
    return 1.0 / (x2x1 * y2y1) * (
        q11 * x2x * y2y +
        q21 * xx1 * y2y +
        q12 * x2x * yy1 +
        q22 * xx1 * yy1
    );
}

void NewOFGuidedMatchSiftFeaturesCPU_One2Multi_byPixel_ColmapFormat(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints& keypoints1,
                          const FeatureKeypoints& keypoints2,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          const FeatureMatches& quantization_map,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_x,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_y,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  double uncertainty_radius = match_options.uncertainty_radius;
  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t OF_scale_factor = match_options.OF_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;

  // int numQuantizationMapping = quantization_map.size();
  // std::unordered_map<point2D_t, point2D_t> mapping1to2;
  // std::unordered_map<point2D_t, point2D_t> mapping2to1;
  // for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  // {
  //     mapping1to2[quantization_map[cnt].point2D_idx1] = quantization_map[cnt].point2D_idx2;
  //     mapping2to1[quantization_map[cnt].point2D_idx2] = quantization_map[cnt].point2D_idx1;
  // }
  // std::cout << "convert quantization map to unordered map successfully!" << std::endl;

  for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
  // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
  {
      // // std::cout << "image_scale_factor = " << image_scale_factor << "; OF_scale_factor = " << OF_scale_factor << std::endl;
      // // point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
      // // float quantizationCenter_y_1 = floor(tmpQuantizationIdx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
      // // float quantizationCenter_x_1 = image_scale_factor * (tmpQuantizationIdx1-floor(tmpQuantizationIdx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
      // point2D_t tmpQuantizationIdx1 = 0;
      // point2D_t mappedQuantizationIdx2;
      // float retrieved_quantizationCenter_x_1;
      // float retrieved_quantizationCenter_y_1;
      // float tmpMinSquareDist = 10000.0;
      // bool NNflag = false;
      // // for(auto element : mapping1to2)
      // // for(point2D_t key12 : keys1to2)
      // for(std::unordered_map<point2D_t,point2D_t>::iterator it = mapping1to2.begin(); it != mapping1to2.end(); ++it)
      // {
      //     point2D_t tmpIdx1 = it->first;
      //     float quantizationCenter_y_1 = floor(tmpIdx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
      //     float quantizationCenter_x_1 = image_scale_factor * (tmpIdx1-floor(tmpIdx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
      //     float tmpSquareDist = (pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2));
      //     if(tmpSquareDist<=tmpMinSquareDist)
      //     {
      //         tmpQuantizationIdx1 = tmpIdx1;
      //         tmpMinSquareDist = tmpSquareDist;
      //         NNflag = true;
      //         mappedQuantizationIdx2 = it->second;
      //         retrieved_quantizationCenter_x_1 = quantizationCenter_x_1;
      //         retrieved_quantizationCenter_y_1 = quantizationCenter_y_1;
      //     }
      // }
      // if(NNflag==false || tmpMinSquareDist>5*image_scale_factor*image_scale_factor)
      // {
      //     // std::cout << "skip this kp1, no NN quantization center could be retrieved!" << std::endl;
      //     continue;
      // }

      float kp1_lowReso_x = (keypoints1[kp1Idx].x / float(image_scale_factor));
      float kp1_lowReso_y = (keypoints1[kp1Idx].y / float(image_scale_factor));
      float kp1_lowReso_x1 = floor(kp1_lowReso_x);
      float kp1_lowReso_x2 = ceil(kp1_lowReso_x);
      float kp1_lowReso_y1 = floor(kp1_lowReso_y);
      float kp1_lowReso_y2 = ceil(kp1_lowReso_y);
      // float flow_kp1_x = 10;
      // float flow_kp1_y = 10;
      // DEBUG: Becareful with =!
      if(kp1_lowReso_y2>48 || kp1_lowReso_y1<0 || kp1_lowReso_x2>64 || kp1_lowReso_x1<0){
          std::cout << "keypoints1[kp1Idx].x = " << keypoints1[kp1Idx].x << "; keypoints1[kp1Idx].y = " << keypoints1[kp1Idx].y << "; kp1_lowReso_x = " << kp1_lowReso_x << "; kp1_lowReso_y = " << kp1_lowReso_y << "; kp1_lowReso_x1 = " << kp1_lowReso_x1 << "; kp1_lowReso_x2 = " << kp1_lowReso_x2 << "; kp1_lowReso_y1 = " << kp1_lowReso_y1 << "; kp1_lowReso_y2 = " << kp1_lowReso_y2 << std::endl;
          std::cout << "###### skip this kp1 since optical flow guidance is out of image border!" << std::endl;
          continue;
      }
      float flow_kp1_x = 64.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_x(kp1_lowReso_y1,kp1_lowReso_x1), optical_flow_x(kp1_lowReso_y2,kp1_lowReso_x1), optical_flow_x(kp1_lowReso_y1,kp1_lowReso_x2), optical_flow_x(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
      float flow_kp1_y = 48.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_y(kp1_lowReso_y1,kp1_lowReso_x1), optical_flow_y(kp1_lowReso_y2,kp1_lowReso_x1), optical_flow_y(kp1_lowReso_y1,kp1_lowReso_x2), optical_flow_y(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
      //float flow_kp1_x = 64.0 * image_scale_factor * computeBilinearInterpolation(0, 1, 0, 1, kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
      //float flow_kp1_y = 48.0 * image_scale_factor * computeBilinearInterpolation(0, 1, 0, 1, kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
      //std::cout << "kp1_lowReso_x1 = " << kp1_lowReso_x1 << "; kp1_lowReso_x2 = " << kp1_lowReso_x2 << "; kp1_lowReso_y1 = " << kp1_lowReso_y1 << "; kp1_lowReso_y2 = " << kp1_lowReso_y2 << std::endl;

      //std::cout << "kp1_lowReso_x = " << kp1_lowReso_x << "; kp1_lowReso_y = " << kp1_lowReso_y << "; flow_kp1_x = " << flow_kp1_x << "; flow_kp1_y = " << flow_kp1_y << std::endl;
      float quantizationCenter_y_2 = keypoints1[kp1Idx].y + flow_kp1_y;
      float quantizationCenter_x_2 = keypoints1[kp1Idx].x + flow_kp1_x;
      // DEBUG: Could be the reason that extremeCase is not expected!
      // if(quantizationCenter_y_2>48*image_scale_factor || quantizationCenter_y_2<0 || quantizationCenter_x_2>64*image_scale_factor || quantizationCenter_x_2<0){
      //     //std::cout << "###### skip this kp1 since optical flow guidance is out of image border!" << std::endl;
      //     continue;
      // }
      // //return;

      // //shareIdKp1Cnt++;
      // std::vector<point2D_t> tmpIndices1;
      // // std::cout << "tmpIndices1 is created!" << std::endl;
      //
      // tmpIndices1.push_back(kp1Idx);
      // // std::cout << "tmpIndices1.size() = " << tmpIndices1.size() << ", tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << "; quanCenter = (" << retrieved_quantizationCenter_x_1 << ", " << retrieved_quantizationCenter_y_1 << ")" << "; keypoints1[kp1Idx] = (" << keypoints1[kp1Idx].x << ", " << keypoints1[kp1Idx].y << ")" << std::endl;

      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1(1, 128);
      tmpDescriptors1.block<1,128>(0,0) = descriptors1.block<1,128>(kp1Idx,0);
      // for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
      // {
      //     tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
      // }
      // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;

      std::vector<point2D_t> tmpIndices2;
      for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
      {
          // point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
          // float quantizationCenter_y_2 = floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          // float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          // // float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));

          if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
          // if(tmpQuantizationIdx2==mappedQuantizationIdx2)
          {
              // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
              tmpIndices2.push_back(kp2Idx);
          }
      }
      // std::cout << "~~ tmpIndices2.size() = " << tmpIndices2.size() << "; keypoints2.size() = " << keypoints2.size() << std::endl;


      // Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
      // for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
      // {
      //     tmpDescriptors2.resize(kp2Idx+1, 128);
      //     //tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(kp2Idx,0);
      //     tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
      // }
      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2(tmpIndices2.size(), 128);
      for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
      {
          // tmpDescriptors2.resize(kp2Idx+1, 128);
          // //tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(kp2Idx,0);
          tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
      }
      // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;
      // std::cout << "!! tmpDescriptors1.rows() = " << tmpDescriptors1.rows()  << "; tmpDescriptor1.cols()  = " << tmpDescriptors1.cols() << "!! tmpDescriptors2.rows() = " << tmpDescriptors2.rows()  << "; tmpDescriptors2.cols()  = " << tmpDescriptors2.cols() << std::endl;
      // Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
      // std::cout << "!! tmpDescriptors1 = " << tmpDescriptors1.format(OctaveFmt)<< std::endl;
      // // remember to normalize the descriptors so that colmap threshold params can be used!
      // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
      // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
      // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
      // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
      // // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
      // // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);

      const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
          nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
      // const Eigen::MatrixXf dists = ComputeSiftDistanceMatrix_Kevin(tmpDescriptors1, tmpDescriptors2);
      // // std::cout << "ComputeSiftDistanceMatrix is done! dists.rows() = " <<  dists.rows() << ";  dists.cols() = " <<  dists.cols() << std::endl;

      FeatureMatches tmpQuantizationMatches;
      FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                      match_options.cross_check, &tmpQuantizationMatches);
      // // const size_t numMatch12Tmp;
      // std::vector<int> tmpQuantizationMatch;
      // const size_t numMatch12Tmp = FindBestMatchesOneWay_One2Multi(dists, match_options.max_ratio, match_options.max_distance, &tmpQuantizationMatch);
      // // std::cout << "FindBestMatches is done! numMatch12Tmp = " << numMatch12Tmp << "; tmpQuantizationMatch.size() = " << tmpQuantizationMatch.size()<< "; tmpQuantizationMatch = " << tmpQuantizationMatch[0] << std::endl;

      for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches.size(); resultCnt++)
      {
          FeatureMatch ConvertedMatch;
          // ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches[resultCnt].point2D_idx1];
          ConvertedMatch.point2D_idx1 = kp1Idx;
          ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches[resultCnt].point2D_idx2];
          matches->push_back(ConvertedMatch);
      }
  }
  std::cout << "@@@ Final raw match number => matches->size() = " << matches->size() << "; keypoints1.size() = " << keypoints1.size() << std::endl;
}

// should also be simplified and modifed!
void NewOFGuidedMatchSiftFeaturesCPU_One2Multi_byPixel_ManualCrossCheck(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints& keypoints1,
                          const FeatureKeypoints& keypoints2,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          const FeatureMatches& quantization_map,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_x,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_y,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_x_21,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_y_21,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  double uncertainty_radius = match_options.uncertainty_radius;
  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t OF_scale_factor = match_options.OF_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;

  // int numQuantizationMapping = quantization_map.size();
  // std::unordered_map<point2D_t, point2D_t> mapping1to2;
  // std::unordered_map<point2D_t, point2D_t> mapping2to1;
  // for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  // {
  //     mapping1to2[quantization_map[cnt].point2D_idx1] = quantization_map[cnt].point2D_idx2;
  //     mapping2to1[quantization_map[cnt].point2D_idx2] = quantization_map[cnt].point2D_idx1;
  // }
  // std::cout << "convert quantization map to unordered map successfully!" << std::endl;

  //for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  //{
      // FeatureMatches matches1to2;
      std::vector<int> matches12;
      // const size_t num_matches12;
      size_t num_matches12 = 0;
      matches12.resize(keypoints1.size(), -1);

      for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
      // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
      {
          float kp1_lowReso_x = (keypoints1[kp1Idx].x / float(image_scale_factor));
          float kp1_lowReso_y = (keypoints1[kp1Idx].y / float(image_scale_factor));
          float kp1_lowReso_x1 = floor(kp1_lowReso_x);
          float kp1_lowReso_x2 = ceil(kp1_lowReso_x);
          float kp1_lowReso_y1 = floor(kp1_lowReso_y);
          float kp1_lowReso_y2 = ceil(kp1_lowReso_y);
          // DEBUG: Becareful with =!
          if(kp1_lowReso_y2>48 || kp1_lowReso_y1<0 || kp1_lowReso_x2>64 || kp1_lowReso_x1<0){
              std::cout << "keypoints1[kp1Idx].x = " << keypoints1[kp1Idx].x << "; keypoints1[kp1Idx].y = " << keypoints1[kp1Idx].y << "; kp1_lowReso_x = " << kp1_lowReso_x << "; kp1_lowReso_y = " << kp1_lowReso_y << "; kp1_lowReso_x1 = " << kp1_lowReso_x1 << "; kp1_lowReso_x2 = " << kp1_lowReso_x2 << "; kp1_lowReso_y1 = " << kp1_lowReso_y1 << "; kp1_lowReso_y2 = " << kp1_lowReso_y2 << std::endl;
              std::cout << "###### skip this kp1 since optical flow guidance is out of image border!" << std::endl;
              continue;
          }
          float flow_kp1_x = 64.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_x(kp1_lowReso_y1,kp1_lowReso_x1), optical_flow_x(kp1_lowReso_y2,kp1_lowReso_x1), optical_flow_x(kp1_lowReso_y1,kp1_lowReso_x2), optical_flow_x(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
          float flow_kp1_y = 48.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_y(kp1_lowReso_y1,kp1_lowReso_x1), optical_flow_y(kp1_lowReso_y2,kp1_lowReso_x1), optical_flow_y(kp1_lowReso_y1,kp1_lowReso_x2), optical_flow_y(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);

          float quantizationCenter_y_2 = keypoints1[kp1Idx].y + flow_kp1_y;
          float quantizationCenter_x_2 = keypoints1[kp1Idx].x + flow_kp1_x;

          // float quantizationCenter_y_1 = floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          // float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          // // float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-quantizationCenter_y_1*(DeMoN_OF_Width * OF_scale_factor));
          // if((pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2)) < image_scale_factor * 1)//uncertainty_radius*uncertainty_radius)
          //if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
          //{
              //std::cout << "direction 1 ---> 2" << std::endl;
              //shareIdKp1Cnt++;
              std::vector<point2D_t> tmpIndices1;
              // std::cout << "tmpIndices1 is created!" << std::endl;

              tmpIndices1.push_back(kp1Idx);

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1(1, 128);
              tmpDescriptors1.block<1,128>(0,0) = descriptors1.block<1,128>(kp1Idx,0);

              std::vector<point2D_t> tmpIndices2;
              for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
              {
                  // point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
                  // float quantizationCenter_y_2 = floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
                  // float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
                  // // float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
                  if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
                  // if(tmpQuantizationIdx2==mappedQuantizationIdx2)
                  {
                      // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
                      tmpIndices2.push_back(kp2Idx);
                  }
              }
              // std::cout << "~~ tmpIndices2.size() = " << tmpIndices2.size() << std::endl;

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2(tmpIndices2.size(), 128);
              for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
              {
                  tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
              }

              // // remember to normalize the descriptors so that colmap threshold params can be used!
              // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
              // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
              // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
              // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
              // // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
              // // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);

              const Eigen::MatrixXi dists12 = ComputeSiftDistanceMatrix(
                  nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
              // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;
              // num_matches12 = FindBestMatchesOneWay(dists12, match_options.max_ratio, match_options.max_distance, &matches12);

              FeatureMatches tmpQuantizationMatches12;
              FindBestMatches(dists12, match_options.max_ratio, match_options.max_distance,
                              false, &tmpQuantizationMatches12);
              // std::cout << "FindBestMatches is done!" << std::endl;

              for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches12.size(); resultCnt++)
              {
                  FeatureMatch ConvertedMatch;
                  ConvertedMatch.point2D_idx1 = kp1Idx;
                  ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches12[resultCnt].point2D_idx2];
                  matches12[ConvertedMatch.point2D_idx1] = ConvertedMatch.point2D_idx2;
                  num_matches12++;
              }
              // std::cout << "index conversion is done!" << std::endl;
          //}
          // std::cout << "end of loop kp1Idx ^" << std::endl;
          // std::cout << "@@@ shareIdKp1Cnt = " << shareIdKp1Cnt << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // FeatureMatches matches2to1;
      std::vector<int> matches21;
      // const size_t num_matches21;
      size_t num_matches21 = 0;
      matches21.resize(keypoints2.size(), -1);

      for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
      {
          float kp2_lowReso_x = (keypoints2[kp2Idx].x / float(image_scale_factor));
          float kp2_lowReso_y = (keypoints2[kp2Idx].y / float(image_scale_factor));
          float kp2_lowReso_x1 = floor(kp2_lowReso_x);
          float kp2_lowReso_x2 = ceil(kp2_lowReso_x);
          float kp2_lowReso_y1 = floor(kp2_lowReso_y);
          float kp2_lowReso_y2 = ceil(kp2_lowReso_y);
          if(kp2_lowReso_y2>48 || kp2_lowReso_y1<0 || kp2_lowReso_x2>64 || kp2_lowReso_x1<0){
              std::cout << "###### skip this kp2 since optical flow guidance is out of image border!" << std::endl;
              continue;
          }
          float flow_kp2_x = 64.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_x_21(kp2_lowReso_y1,kp2_lowReso_x1), optical_flow_x_21(kp2_lowReso_y2,kp2_lowReso_x1), optical_flow_x_21(kp2_lowReso_y1,kp2_lowReso_x2), optical_flow_x_21(kp2_lowReso_y2,kp2_lowReso_x2), kp2_lowReso_x1, kp2_lowReso_x2, kp2_lowReso_y1, kp2_lowReso_y2, kp2_lowReso_x, kp2_lowReso_y);
          float flow_kp2_y = 48.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_y_21(kp2_lowReso_y1,kp2_lowReso_x1), optical_flow_y_21(kp2_lowReso_y2,kp2_lowReso_x1), optical_flow_y_21(kp2_lowReso_y1,kp2_lowReso_x2), optical_flow_y_21(kp2_lowReso_y2,kp2_lowReso_x2), kp2_lowReso_x1, kp2_lowReso_x2, kp2_lowReso_y1, kp2_lowReso_y2, kp2_lowReso_x, kp2_lowReso_y);
          //std::cout << "kp2_lowReso_x1 = " << kp2_lowReso_x1 << "; kp2_lowReso_x2 = " << kp2_lowReso_x2 << "; kp2_lowReso_y1 = " << kp2_lowReso_y1 << "; kp2_lowReso_y2 = " << kp2_lowReso_y2 << std::endl;

          //std::cout << "kp2_lowReso_x = " << kp2_lowReso_x << "; kp2_lowReso_y = " << kp2_lowReso_y << "; flow_kp2_x = " << flow_kp2_x << "; flow_kp2_y = " << flow_kp2_y << std::endl;
          float quantizationCenter_y_1 = keypoints2[kp2Idx].y + flow_kp2_y;
          float quantizationCenter_x_1 = keypoints2[kp2Idx].x + flow_kp2_x;

          // if(quantizationCenter_y_1>=48*image_scale_factor || quantizationCenter_y_1<0 || quantizationCenter_x_1>=64*image_scale_factor || quantizationCenter_x_1<0){
          //     //std::cout << "###### skip this kp2 since optical flow guidance is out of image border!" << std::endl;
          //     continue;
          // }
          // float quantizationCenter_y_2 = floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          // // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
          // if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2)) < image_scale_factor * 1)//uncertainty_radius*uncertainty_radius)
          //if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
          //{
              //std::cout << "@ direction 2 ---> 1" << std::endl;
              //shareIdKp2Cnt++;
              std::vector<point2D_t> tmpIndices2;
              // std::cout << "tmpIndices2 is created!" << std::endl;

              tmpIndices2.push_back(kp2Idx);

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2(1, 128);
              tmpDescriptors2.block<1,128>(0,0) = descriptors2.block<1,128>(kp2Idx,0);

              std::vector<point2D_t> tmpIndices1;
              for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
              {
                  if((pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2))<=uncertainty_radius*uncertainty_radius)
                  {
                      tmpIndices1.push_back(kp1Idx);
                  }
              }
              // std::cout << "~~ tmpIndices1.size() = " << tmpIndices1.size() << std::endl;

              // Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
              // for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
              // {
              //     tmpDescriptors1.resize(kp1Idx+1, 128);
              //     tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
              // }
              // // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;
              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1(tmpIndices1.size(), 128);
              for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
              {
                  tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
              }

              const Eigen::MatrixXi dists21 = ComputeSiftDistanceMatrix(
                  nullptr, nullptr, tmpDescriptors2, tmpDescriptors1, nullptr);
              // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;

              // num_matches21 = FindBestMatchesOneWay(dists21, match_options.max_ratio, match_options.max_distance, &matches21);

              FeatureMatches tmpQuantizationMatches21;
              FindBestMatches(dists21, match_options.max_ratio, match_options.max_distance,
                              false, &tmpQuantizationMatches21);
              // std::cout << "FindBestMatches is done!" << std::endl;

              for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches21.size(); resultCnt++)
              {
                  FeatureMatch ConvertedMatch;
                  ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches21[resultCnt].point2D_idx2];
                  // ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches21[resultCnt].point2D_idx1];
                  ConvertedMatch.point2D_idx2 = kp2Idx;
                  // // matches->push_back(ConvertedMatch);
                  // matches2to1.push_back(ConvertedMatch);
                  matches21[ConvertedMatch.point2D_idx2] = ConvertedMatch.point2D_idx1;
                  num_matches21++;
              }
              // // std::cout << "index conversion is done!" << std::endl;
          //}
          // std::cout << "end of loop kp1Idx ^" << std::endl;
          // std::cout << "@@@ shareIdKp1Cnt = " << shareIdKp1Cnt << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;
      /////////////////////////////////////////////////////////////////////
      /******* Manually cross checking *******/
      if (true) {
        // std::cout << "@@@ num_matches12 = " << num_matches12 << ", @@@ matches12.size() = " << matches12.size() << ", @@@ num_matches21 = " << num_matches21 << ", @@@ matches21.size() = " << matches21.size() << std::endl;
        matches->reserve(std::min(num_matches12, num_matches21));
        for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
          if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
              matches21[matches12[i1]] == static_cast<int>(i1)) {
            FeatureMatch match;
            match.point2D_idx1 = i1;
            match.point2D_idx2 = matches12[i1];
            matches->push_back(match);
          }
        }
        std::cout << "@@@ num_matches12 = " << num_matches12 << ", @@@ matches12.size() = " << matches12.size() << ", @@@ num_matches21 = " << num_matches21 << ", @@@ matches21.size() = " << matches21.size() << ", == ### cross-check survivors = " << matches->size() << std::endl;
      }
      /////////////////////////////////////////////////////////////////////
  //}

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void AdaptiveNewOFGuidedMatchSiftFeaturesCPU_One2Multi_byPixel_ColmapFormat(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints& keypoints1,
                          const FeatureKeypoints& keypoints2,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_x,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_y,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& flowconf_x,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& flowconf_y,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  double uncertainty_radius = match_options.uncertainty_radius;
  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t OF_scale_factor = match_options.OF_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;

  for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
  {
      float kp1_lowReso_x = (keypoints1[kp1Idx].x / float(image_scale_factor));
      float kp1_lowReso_y = (keypoints1[kp1Idx].y / float(image_scale_factor));
      float kp1_lowReso_x1 = floor(kp1_lowReso_x);
      float kp1_lowReso_x2 = ceil(kp1_lowReso_x);
      float kp1_lowReso_y1 = floor(kp1_lowReso_y);
      float kp1_lowReso_y2 = ceil(kp1_lowReso_y);

      // DEBUG: Becareful with =!
      if(kp1_lowReso_y2>48 || kp1_lowReso_y1<0 || kp1_lowReso_x2>64 || kp1_lowReso_x1<0){
          std::cout << "keypoints1[kp1Idx].x = " << keypoints1[kp1Idx].x << "; keypoints1[kp1Idx].y = " << keypoints1[kp1Idx].y << "; kp1_lowReso_x = " << kp1_lowReso_x << "; kp1_lowReso_y = " << kp1_lowReso_y << "; kp1_lowReso_x1 = " << kp1_lowReso_x1 << "; kp1_lowReso_x2 = " << kp1_lowReso_x2 << "; kp1_lowReso_y1 = " << kp1_lowReso_y1 << "; kp1_lowReso_y2 = " << kp1_lowReso_y2 << std::endl;
          std::cout << "###### skip this kp1 since optical flow guidance is out of image border!" << std::endl;
          continue;
      }
      float flow_kp1_x = 64.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_x(kp1_lowReso_y1,kp1_lowReso_x1), optical_flow_x(kp1_lowReso_y2,kp1_lowReso_x1), optical_flow_x(kp1_lowReso_y1,kp1_lowReso_x2), optical_flow_x(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
      float flow_kp1_y = 48.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_y(kp1_lowReso_y1,kp1_lowReso_x1), optical_flow_y(kp1_lowReso_y2,kp1_lowReso_x1), optical_flow_y(kp1_lowReso_y1,kp1_lowReso_x2), optical_flow_y(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
      //float flow_kp1_x = 64.0 * image_scale_factor * computeBilinearInterpolation(0, 1, 0, 1, kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
      //float flow_kp1_y = 48.0 * image_scale_factor * computeBilinearInterpolation(0, 1, 0, 1, kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
      //std::cout << "kp1_lowReso_x1 = " << kp1_lowReso_x1 << "; kp1_lowReso_x2 = " << kp1_lowReso_x2 << "; kp1_lowReso_y1 = " << kp1_lowReso_y1 << "; kp1_lowReso_y2 = " << kp1_lowReso_y2 << std::endl;

      float flowconf_kp1_x = computeBilinearInterpolation(flowconf_x(kp1_lowReso_y1,kp1_lowReso_x1), flowconf_x(kp1_lowReso_y2,kp1_lowReso_x1), flowconf_x(kp1_lowReso_y1,kp1_lowReso_x2), flowconf_x(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
      float flowconf_kp1_y = computeBilinearInterpolation(flowconf_y(kp1_lowReso_y1,kp1_lowReso_x1), flowconf_y(kp1_lowReso_y2,kp1_lowReso_x1), flowconf_y(kp1_lowReso_y1,kp1_lowReso_x2), flowconf_y(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
      float adaptiveFactor_x = 1;
      float adaptiveFactor_y = 1;
      float maxFactor = 3;
      if(flowconf_kp1_x<0.98){
          adaptiveFactor_x = maxFactor;
      } else {
          adaptiveFactor_x = (maxFactor-1)*(1-flowconf_kp1_x)/0.02+1;
      }
      if(flowconf_kp1_y<0.98){
          adaptiveFactor_y = maxFactor;
      } else {
          adaptiveFactor_y = (maxFactor-1)*(1-flowconf_kp1_y)/0.02+1;
      }

      //std::cout << "kp1_lowReso_x = " << kp1_lowReso_x << "; kp1_lowReso_y = " << kp1_lowReso_y << "; flow_kp1_x = " << flow_kp1_x << "; flow_kp1_y = " << flow_kp1_y << std::endl;
      float quantizationCenter_y_2 = keypoints1[kp1Idx].y + flow_kp1_y;
      float quantizationCenter_x_2 = keypoints1[kp1Idx].x + flow_kp1_x;

      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1(1, 128);
      tmpDescriptors1.block<1,128>(0,0) = descriptors1.block<1,128>(kp1Idx,0);

      std::vector<point2D_t> tmpIndices2;
      for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
      {
          //if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
          if(abs(keypoints2[kp2Idx].x-quantizationCenter_x_2)<=uncertainty_radius*adaptiveFactor_x && abs(keypoints2[kp2Idx].y-quantizationCenter_y_2)<=uncertainty_radius*adaptiveFactor_y)
          {
              tmpIndices2.push_back(kp2Idx);
          }
      }
      // std::cout << "~~ tmpIndices2.size() = " << tmpIndices2.size() << "; keypoints2.size() = " << keypoints2.size() << std::endl;

      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2(tmpIndices2.size(), 128);
      for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
      {
          tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
      }

      const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
          nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
      // const Eigen::MatrixXf dists = ComputeSiftDistanceMatrix_Kevin(tmpDescriptors1, tmpDescriptors2);
      // // std::cout << "ComputeSiftDistanceMatrix is done! dists.rows() = " <<  dists.rows() << ";  dists.cols() = " <<  dists.cols() << std::endl;

      FeatureMatches tmpQuantizationMatches;
      FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                      match_options.cross_check, &tmpQuantizationMatches);


      for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches.size(); resultCnt++)
      {
          FeatureMatch ConvertedMatch;
          ConvertedMatch.point2D_idx1 = kp1Idx;
          ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches[resultCnt].point2D_idx2];
          matches->push_back(ConvertedMatch);
      }
  }
  std::cout << "@@@ Final raw match number => matches->size() = " << matches->size() << "; keypoints1.size() = " << keypoints1.size() << std::endl;
}

// should also be simplified and modifed!
void AdaptiveNewOFGuidedMatchSiftFeaturesCPU_One2Multi_byPixel_ManualCrossCheck(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints& keypoints1,
                          const FeatureKeypoints& keypoints2,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_x,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_y,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_x_21,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& optical_flow_y_21,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& flowconf_x,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& flowconf_y,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& flowconf_x_21,
                          const Eigen::Matrix<float, 48, 64, Eigen::RowMajor>& flowconf_y_21,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  double uncertainty_radius = match_options.uncertainty_radius;
  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t OF_scale_factor = match_options.OF_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;

  //for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  //{
      // FeatureMatches matches1to2;
      std::vector<int> matches12;
      // const size_t num_matches12;
      size_t num_matches12 = 0;
      matches12.resize(keypoints1.size(), -1);

      for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
      // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
      {
          float kp1_lowReso_x = (keypoints1[kp1Idx].x / float(image_scale_factor));
          float kp1_lowReso_y = (keypoints1[kp1Idx].y / float(image_scale_factor));
          float kp1_lowReso_x1 = floor(kp1_lowReso_x);
          float kp1_lowReso_x2 = ceil(kp1_lowReso_x);
          float kp1_lowReso_y1 = floor(kp1_lowReso_y);
          float kp1_lowReso_y2 = ceil(kp1_lowReso_y);
          // DEBUG: Becareful with =!
          if(kp1_lowReso_y2>48 || kp1_lowReso_y1<0 || kp1_lowReso_x2>64 || kp1_lowReso_x1<0){
              std::cout << "keypoints1[kp1Idx].x = " << keypoints1[kp1Idx].x << "; keypoints1[kp1Idx].y = " << keypoints1[kp1Idx].y << "; kp1_lowReso_x = " << kp1_lowReso_x << "; kp1_lowReso_y = " << kp1_lowReso_y << "; kp1_lowReso_x1 = " << kp1_lowReso_x1 << "; kp1_lowReso_x2 = " << kp1_lowReso_x2 << "; kp1_lowReso_y1 = " << kp1_lowReso_y1 << "; kp1_lowReso_y2 = " << kp1_lowReso_y2 << std::endl;
              std::cout << "###### skip this kp1 since optical flow guidance is out of image border!" << std::endl;
              continue;
          }
          float flow_kp1_x = 64.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_x(kp1_lowReso_y1,kp1_lowReso_x1), optical_flow_x(kp1_lowReso_y2,kp1_lowReso_x1), optical_flow_x(kp1_lowReso_y1,kp1_lowReso_x2), optical_flow_x(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
          float flow_kp1_y = 48.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_y(kp1_lowReso_y1,kp1_lowReso_x1), optical_flow_y(kp1_lowReso_y2,kp1_lowReso_x1), optical_flow_y(kp1_lowReso_y1,kp1_lowReso_x2), optical_flow_y(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);


          float flowconf_kp1_x = computeBilinearInterpolation(flowconf_x(kp1_lowReso_y1,kp1_lowReso_x1), flowconf_x(kp1_lowReso_y2,kp1_lowReso_x1), flowconf_x(kp1_lowReso_y1,kp1_lowReso_x2), flowconf_x(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
          float flowconf_kp1_y = computeBilinearInterpolation(flowconf_y(kp1_lowReso_y1,kp1_lowReso_x1), flowconf_y(kp1_lowReso_y2,kp1_lowReso_x1), flowconf_y(kp1_lowReso_y1,kp1_lowReso_x2), flowconf_y(kp1_lowReso_y2,kp1_lowReso_x2), kp1_lowReso_x1, kp1_lowReso_x2, kp1_lowReso_y1, kp1_lowReso_y2, kp1_lowReso_x, kp1_lowReso_y);
          float adaptiveFactor_x = 1;
          float adaptiveFactor_y = 1;
          float maxFactor = 3;
          if(flowconf_kp1_x<0.98){
              adaptiveFactor_x = maxFactor;
          } else {
              adaptiveFactor_x = (maxFactor-1)*(1-flowconf_kp1_x)/0.02+1;
          }
          if(flowconf_kp1_y<0.98){
              adaptiveFactor_y = maxFactor;
          } else {
              adaptiveFactor_y = (maxFactor-1)*(1-flowconf_kp1_y)/0.02+1;
          }

          float quantizationCenter_y_2 = keypoints1[kp1Idx].y + flow_kp1_y;
          float quantizationCenter_x_2 = keypoints1[kp1Idx].x + flow_kp1_x;

          // float quantizationCenter_y_1 = floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          // float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          // // float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-quantizationCenter_y_1*(DeMoN_OF_Width * OF_scale_factor));
          // if((pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2)) < image_scale_factor * 1)//uncertainty_radius*uncertainty_radius)
          //if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
          //{
              //std::cout << "direction 1 ---> 2" << std::endl;
              //shareIdKp1Cnt++;
              std::vector<point2D_t> tmpIndices1;
              // std::cout << "tmpIndices1 is created!" << std::endl;

              tmpIndices1.push_back(kp1Idx);

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1(1, 128);
              tmpDescriptors1.block<1,128>(0,0) = descriptors1.block<1,128>(kp1Idx,0);

              std::vector<point2D_t> tmpIndices2;
              for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
              {
                  //if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
                  if(abs(keypoints2[kp2Idx].x-quantizationCenter_x_2)<=uncertainty_radius*adaptiveFactor_x && abs(keypoints2[kp2Idx].y-quantizationCenter_y_2)<=uncertainty_radius*adaptiveFactor_y)
                  {
                      // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
                      tmpIndices2.push_back(kp2Idx);
                  }
              }
              // std::cout << "~~ tmpIndices2.size() = " << tmpIndices2.size() << std::endl;

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2(tmpIndices2.size(), 128);
              for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
              {
                  tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
              }

              // // remember to normalize the descriptors so that colmap threshold params can be used!
              // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
              // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
              // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
              // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
              // // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
              // // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);

              const Eigen::MatrixXi dists12 = ComputeSiftDistanceMatrix(
                  nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
              // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;
              // num_matches12 = FindBestMatchesOneWay(dists12, match_options.max_ratio, match_options.max_distance, &matches12);

              FeatureMatches tmpQuantizationMatches12;
              FindBestMatches(dists12, match_options.max_ratio, match_options.max_distance,
                              false, &tmpQuantizationMatches12);
              // std::cout << "FindBestMatches is done!" << std::endl;

              for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches12.size(); resultCnt++)
              {
                  FeatureMatch ConvertedMatch;
                  ConvertedMatch.point2D_idx1 = kp1Idx;
                  ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches12[resultCnt].point2D_idx2];
                  matches12[ConvertedMatch.point2D_idx1] = ConvertedMatch.point2D_idx2;
                  num_matches12++;
              }
              // std::cout << "index conversion is done!" << std::endl;
          //}
          // std::cout << "end of loop kp1Idx ^" << std::endl;
          // std::cout << "@@@ shareIdKp1Cnt = " << shareIdKp1Cnt << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // FeatureMatches matches2to1;
      std::vector<int> matches21;
      // const size_t num_matches21;
      size_t num_matches21 = 0;
      matches21.resize(keypoints2.size(), -1);

      for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
      {
          float kp2_lowReso_x = (keypoints2[kp2Idx].x / float(image_scale_factor));
          float kp2_lowReso_y = (keypoints2[kp2Idx].y / float(image_scale_factor));
          float kp2_lowReso_x1 = floor(kp2_lowReso_x);
          float kp2_lowReso_x2 = ceil(kp2_lowReso_x);
          float kp2_lowReso_y1 = floor(kp2_lowReso_y);
          float kp2_lowReso_y2 = ceil(kp2_lowReso_y);
          if(kp2_lowReso_y2>48 || kp2_lowReso_y1<0 || kp2_lowReso_x2>64 || kp2_lowReso_x1<0){
              std::cout << "###### skip this kp2 since optical flow guidance is out of image border!" << std::endl;
              continue;
          }
          float flow_kp2_x = 64.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_x_21(kp2_lowReso_y1,kp2_lowReso_x1), optical_flow_x_21(kp2_lowReso_y2,kp2_lowReso_x1), optical_flow_x_21(kp2_lowReso_y1,kp2_lowReso_x2), optical_flow_x_21(kp2_lowReso_y2,kp2_lowReso_x2), kp2_lowReso_x1, kp2_lowReso_x2, kp2_lowReso_y1, kp2_lowReso_y2, kp2_lowReso_x, kp2_lowReso_y);
          float flow_kp2_y = 48.0 * image_scale_factor * computeBilinearInterpolation(optical_flow_y_21(kp2_lowReso_y1,kp2_lowReso_x1), optical_flow_y_21(kp2_lowReso_y2,kp2_lowReso_x1), optical_flow_y_21(kp2_lowReso_y1,kp2_lowReso_x2), optical_flow_y_21(kp2_lowReso_y2,kp2_lowReso_x2), kp2_lowReso_x1, kp2_lowReso_x2, kp2_lowReso_y1, kp2_lowReso_y2, kp2_lowReso_x, kp2_lowReso_y);
          //std::cout << "kp2_lowReso_x1 = " << kp2_lowReso_x1 << "; kp2_lowReso_x2 = " << kp2_lowReso_x2 << "; kp2_lowReso_y1 = " << kp2_lowReso_y1 << "; kp2_lowReso_y2 = " << kp2_lowReso_y2 << std::endl;

          float flowconf_kp2_x = computeBilinearInterpolation(flowconf_x(kp2_lowReso_y1,kp2_lowReso_x1), flowconf_x(kp2_lowReso_y2,kp2_lowReso_x1), flowconf_x(kp2_lowReso_y1,kp2_lowReso_x2), flowconf_x(kp2_lowReso_y2,kp2_lowReso_x2), kp2_lowReso_x1, kp2_lowReso_x2, kp2_lowReso_y1, kp2_lowReso_y2, kp2_lowReso_x, kp2_lowReso_y);
          float flowconf_kp2_y = computeBilinearInterpolation(flowconf_y(kp2_lowReso_y1,kp2_lowReso_x1), flowconf_y(kp2_lowReso_y2,kp2_lowReso_x1), flowconf_y(kp2_lowReso_y1,kp2_lowReso_x2), flowconf_y(kp2_lowReso_y2,kp2_lowReso_x2), kp2_lowReso_x1, kp2_lowReso_x2, kp2_lowReso_y1, kp2_lowReso_y2, kp2_lowReso_x, kp2_lowReso_y);
          float adaptiveFactor_x_2nd = 1;
          float adaptiveFactor_y_2nd = 1;
          float maxFactor_2nd = 3;
          if(flowconf_kp2_x<0.98){
              adaptiveFactor_x_2nd = maxFactor_2nd;
          } else {
              adaptiveFactor_x_2nd = (maxFactor_2nd-1)*(1-flowconf_kp2_x)/0.02+1;
          }
          if(flowconf_kp2_y<0.98){
              adaptiveFactor_y_2nd = maxFactor_2nd;
          } else {
              adaptiveFactor_y_2nd = (maxFactor_2nd-1)*(1-flowconf_kp2_y)/0.02+1;
          }

          //std::cout << "kp2_lowReso_x = " << kp2_lowReso_x << "; kp2_lowReso_y = " << kp2_lowReso_y << "; flow_kp2_x = " << flow_kp2_x << "; flow_kp2_y = " << flow_kp2_y << std::endl;
          float quantizationCenter_y_1 = keypoints2[kp2Idx].y + flow_kp2_y;
          float quantizationCenter_x_1 = keypoints2[kp2Idx].x + flow_kp2_x;

          // if(quantizationCenter_y_1>=48*image_scale_factor || quantizationCenter_y_1<0 || quantizationCenter_x_1>=64*image_scale_factor || quantizationCenter_x_1<0){
          //     //std::cout << "###### skip this kp2 since optical flow guidance is out of image border!" << std::endl;
          //     continue;
          // }
          // float quantizationCenter_y_2 = floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          // // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
          // if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2)) < image_scale_factor * 1)//uncertainty_radius*uncertainty_radius)
          //if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
          //{
              //std::cout << "@ direction 2 ---> 1" << std::endl;
              //shareIdKp2Cnt++;
              std::vector<point2D_t> tmpIndices2;
              // std::cout << "tmpIndices2 is created!" << std::endl;

              tmpIndices2.push_back(kp2Idx);

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2(1, 128);
              tmpDescriptors2.block<1,128>(0,0) = descriptors2.block<1,128>(kp2Idx,0);

              std::vector<point2D_t> tmpIndices1;
              for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
              {
                  //if((pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2))<=uncertainty_radius*uncertainty_radius)
                  if(abs(keypoints1[kp1Idx].x-quantizationCenter_x_1)<=uncertainty_radius*adaptiveFactor_x_2nd && abs(keypoints1[kp1Idx].y-quantizationCenter_y_1)<=uncertainty_radius*adaptiveFactor_y_2nd)
                  {
                      tmpIndices1.push_back(kp1Idx);
                  }
              }
              // std::cout << "~~ tmpIndices1.size() = " << tmpIndices1.size() << std::endl;

              // Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
              // for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
              // {
              //     tmpDescriptors1.resize(kp1Idx+1, 128);
              //     tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
              // }
              // // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;
              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1(tmpIndices1.size(), 128);
              for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
              {
                  tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
              }

              const Eigen::MatrixXi dists21 = ComputeSiftDistanceMatrix(
                  nullptr, nullptr, tmpDescriptors2, tmpDescriptors1, nullptr);
              // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;

              // num_matches21 = FindBestMatchesOneWay(dists21, match_options.max_ratio, match_options.max_distance, &matches21);

              FeatureMatches tmpQuantizationMatches21;
              FindBestMatches(dists21, match_options.max_ratio, match_options.max_distance,
                              false, &tmpQuantizationMatches21);
              // std::cout << "FindBestMatches is done!" << std::endl;

              for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches21.size(); resultCnt++)
              {
                  FeatureMatch ConvertedMatch;
                  ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches21[resultCnt].point2D_idx2];
                  // ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches21[resultCnt].point2D_idx1];
                  ConvertedMatch.point2D_idx2 = kp2Idx;
                  // // matches->push_back(ConvertedMatch);
                  // matches2to1.push_back(ConvertedMatch);
                  matches21[ConvertedMatch.point2D_idx2] = ConvertedMatch.point2D_idx1;
                  num_matches21++;
              }
              // // std::cout << "index conversion is done!" << std::endl;
          //}
          // std::cout << "end of loop kp1Idx ^" << std::endl;
          // std::cout << "@@@ shareIdKp1Cnt = " << shareIdKp1Cnt << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;
      /////////////////////////////////////////////////////////////////////
      /******* Manually cross checking *******/
      if (true) {
        // std::cout << "@@@ num_matches12 = " << num_matches12 << ", @@@ matches12.size() = " << matches12.size() << ", @@@ num_matches21 = " << num_matches21 << ", @@@ matches21.size() = " << matches21.size() << std::endl;
        matches->reserve(std::min(num_matches12, num_matches21));
        for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
          if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
              matches21[matches12[i1]] == static_cast<int>(i1)) {
            FeatureMatch match;
            match.point2D_idx1 = i1;
            match.point2D_idx2 = matches12[i1];
            matches->push_back(match);
          }
        }
        std::cout << "@@@ num_matches12 = " << num_matches12 << ", @@@ matches12.size() = " << matches12.size() << ", @@@ num_matches21 = " << num_matches21 << ", @@@ matches21.size() = " << matches21.size() << ", == ### cross-check survivors = " << matches->size() << std::endl;
      }
      /////////////////////////////////////////////////////////////////////
  //}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void OFGuidedMatchSiftFeaturesCPU_One2Multi_byPixel_ManualCrossCheck(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints& keypoints1,
                          const FeatureKeypoints& keypoints2,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          const FeatureMatches& quantization_map,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  double uncertainty_radius = match_options.uncertainty_radius;
  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t OF_scale_factor = match_options.OF_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;

  int numQuantizationMapping = quantization_map.size();
  std::unordered_map<point2D_t, point2D_t> mapping1to2;
  std::unordered_map<point2D_t, point2D_t> mapping2to1;
  for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  {
      mapping1to2[quantization_map[cnt].point2D_idx1] = quantization_map[cnt].point2D_idx2;
      mapping2to1[quantization_map[cnt].point2D_idx2] = quantization_map[cnt].point2D_idx1;
  }
  std::cout << "convert quantization map to unordered map successfully!" << std::endl;

  //for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  //{
      // FeatureMatches matches1to2;
      std::vector<int> matches12;
      // const size_t num_matches12;
      size_t num_matches12 = 0;
      matches12.resize(keypoints1.size(), -1);

      for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
      // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
      {
          // // std::cout << "image_scale_factor = " << image_scale_factor << "; OF_scale_factor = " << OF_scale_factor << std::endl;
          // point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
          // // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << std::endl;
          // //int shareIdKp1Cnt = 0;
          // // point2D_t mappedQuantizationIdx2 = mapping1to2[tmpQuantizationIdx1];
          // point2D_t mappedQuantizationIdx2;
          // if(mapping1to2.count(tmpQuantizationIdx1) > 0)
          // {
          //     mappedQuantizationIdx2 = mapping1to2[tmpQuantizationIdx1];
          // } else {
          //     continue;
          // }
          point2D_t tmpQuantizationIdx1 = 0;
          point2D_t mappedQuantizationIdx2;
          float retrieved_quantizationCenter_x_1;
          float retrieved_quantizationCenter_y_1;
          float tmpMinSquareDist = 10000.0;
          bool NNflag = false;
          // for(auto element : mapping1to2)
          // for(point2D_t key12 : keys1to2)
          for(std::unordered_map<point2D_t,point2D_t>::iterator it = mapping1to2.begin(); it != mapping1to2.end(); ++it)
          {
              point2D_t tmpIdx1 = it->first;
              float quantizationCenter_y_1 = floor(tmpIdx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
              float quantizationCenter_x_1 = image_scale_factor * (tmpIdx1-floor(tmpIdx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
              float tmpSquareDist = (pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2));
              if(tmpSquareDist<=tmpMinSquareDist)
              {
                  tmpQuantizationIdx1 = tmpIdx1;
                  tmpMinSquareDist = tmpSquareDist;
                  NNflag = true;
                  mappedQuantizationIdx2 = it->second;
                  retrieved_quantizationCenter_x_1 = quantizationCenter_x_1;
                  retrieved_quantizationCenter_y_1 = quantizationCenter_y_1;
              }
          }
          if(NNflag==false || tmpMinSquareDist>5*image_scale_factor*image_scale_factor)
          {
              // std::cout << "skip this kp1, no NN quantization center could be retrieved!" << std::endl;
              continue;
          }
          // float quantizationCenter_y_1 = floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          // float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-floor(quantization_map[cnt].point2D_idx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          // // float quantizationCenter_x_1 = image_scale_factor * (quantization_map[cnt].point2D_idx1-quantizationCenter_y_1*(DeMoN_OF_Width * OF_scale_factor));
          // if((pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2)) < image_scale_factor * 1)//uncertainty_radius*uncertainty_radius)
          //if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
          //{
              //std::cout << "direction 1 ---> 2" << std::endl;
              //shareIdKp1Cnt++;
              std::vector<point2D_t> tmpIndices1;
              // std::cout << "tmpIndices1 is created!" << std::endl;

              tmpIndices1.push_back(kp1Idx);

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
              for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
              {
                  tmpDescriptors1.resize(kp1Idx+1, 128);
                  tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
              }
              // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;

              std::vector<point2D_t> tmpIndices2;
              for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
              {
                  point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
                  float quantizationCenter_y_2 = floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
                  float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-floor(mappedQuantizationIdx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
                  // float quantizationCenter_x_2 = image_scale_factor * (mappedQuantizationIdx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
                  if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2))<=uncertainty_radius*uncertainty_radius)
                  // if(tmpQuantizationIdx2==mappedQuantizationIdx2)
                  {
                      // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
                      tmpIndices2.push_back(kp2Idx);
                  }
              }
              // std::cout << "~~ tmpIndices2.size() = " << tmpIndices2.size() << std::endl;

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
              for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
              {
                  tmpDescriptors2.resize(kp2Idx+1, 128);
                  tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
              }
              // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;

              // // remember to normalize the descriptors so that colmap threshold params can be used!
              // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
              // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
              // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
              // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
              // // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
              // // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);

              const Eigen::MatrixXi dists12 = ComputeSiftDistanceMatrix(
                  nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
              // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;
              // num_matches12 = FindBestMatchesOneWay(dists12, match_options.max_ratio, match_options.max_distance, &matches12);

              FeatureMatches tmpQuantizationMatches12;
              FindBestMatches(dists12, match_options.max_ratio, match_options.max_distance,
                              false, &tmpQuantizationMatches12);
              // std::cout << "FindBestMatches is done!" << std::endl;

              for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches12.size(); resultCnt++)
              {
                  FeatureMatch ConvertedMatch;
                  ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches12[resultCnt].point2D_idx1];
                  ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches12[resultCnt].point2D_idx2];
                  // // matches->push_back(ConvertedMatch);
                  // matches1to2.push_back(ConvertedMatch);
                  matches12[ConvertedMatch.point2D_idx1] = ConvertedMatch.point2D_idx2;
                  num_matches12++;
              }
              // std::cout << "index conversion is done!" << std::endl;
          //}
          // std::cout << "end of loop kp1Idx ^" << std::endl;
          // std::cout << "@@@ shareIdKp1Cnt = " << shareIdKp1Cnt << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;

      ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      // FeatureMatches matches2to1;
      std::vector<int> matches21;
      // const size_t num_matches21;
      size_t num_matches21 = 0;
      matches21.resize(keypoints2.size(), -1);

      for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
      {
          // point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
          // //int shareIdKp2Cnt = 0;
          // // point2D_t mappedQuantizationIdx1 = mapping2to1[tmpQuantizationIdx2];
          // point2D_t mappedQuantizationIdx1;
          // if(mapping2to1.count(tmpQuantizationIdx2) > 0)
          // {
          //     mappedQuantizationIdx1 = mapping2to1[tmpQuantizationIdx2];
          // } else {
          //     continue;
          // }
          point2D_t tmpQuantizationIdx2 = 0;
          point2D_t mappedQuantizationIdx1;
          float retrieved_quantizationCenter_x_2;
          float retrieved_quantizationCenter_y_2;
          float tmpMinSquareDist = 10000.0;
          bool NNflag = false;
          // for(auto element : mapping1to2)
          // for(point2D_t key12 : keys1to2)
          for(std::unordered_map<point2D_t,point2D_t>::iterator it = mapping2to1.begin(); it != mapping2to1.end(); ++it)
          {
              point2D_t tmpIdx2 = it->first;
              float quantizationCenter_y_2 = floor(tmpIdx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
              float quantizationCenter_x_2 = image_scale_factor * (tmpIdx2-floor(tmpIdx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
              float tmpSquareDist = (pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2));
              if(tmpSquareDist<=tmpMinSquareDist)
              {
                  tmpQuantizationIdx2 = tmpIdx2;
                  tmpMinSquareDist = tmpSquareDist;
                  NNflag = true;
                  mappedQuantizationIdx1 = it->second;
                  retrieved_quantizationCenter_x_2 = quantizationCenter_x_2;
                  retrieved_quantizationCenter_y_2 = quantizationCenter_y_2;
              }
          }
          if(NNflag==false || tmpMinSquareDist>5*image_scale_factor*image_scale_factor)
          {
              // std::cout << "skip this kp1, no NN quantization center could be retrieved!" << std::endl;
              continue;
          }
          // float quantizationCenter_y_2 = floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
          // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-floor(quantization_map[cnt].point2D_idx2 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
          // // float quantizationCenter_x_2 = image_scale_factor * (quantization_map[cnt].point2D_idx2-quantizationCenter_y_2*(DeMoN_OF_Width * OF_scale_factor));
          // if((pow(keypoints2[kp2Idx].x-quantizationCenter_x_2, 2)+pow(keypoints2[kp2Idx].y-quantizationCenter_y_2, 2)) < image_scale_factor * 1)//uncertainty_radius*uncertainty_radius)
          //if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
          //{
              //std::cout << "@ direction 2 ---> 1" << std::endl;
              //shareIdKp2Cnt++;
              std::vector<point2D_t> tmpIndices2;
              // std::cout << "tmpIndices2 is created!" << std::endl;

              tmpIndices2.push_back(kp2Idx);

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
              for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
              {
                  tmpDescriptors2.resize(kp2Idx+1, 128);
                  tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
              }
              // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;

              std::vector<point2D_t> tmpIndices1;
              for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
              {
                  point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + OF_scale_factor * DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
                  float quantizationCenter_y_1 = floor(mappedQuantizationIdx1 / DeMoN_OF_Width / OF_scale_factor) * image_scale_factor;
                  float quantizationCenter_x_1 = image_scale_factor * (mappedQuantizationIdx1-floor(mappedQuantizationIdx1 / DeMoN_OF_Width / OF_scale_factor)*(DeMoN_OF_Width * OF_scale_factor));
                  // float quantizationCenter_x_1 = image_scale_factor * (mappedQuantizationIdx1-quantizationCenter_y_1*(DeMoN_OF_Width * OF_scale_factor));
                  if((pow(keypoints1[kp1Idx].x-quantizationCenter_x_1, 2)+pow(keypoints1[kp1Idx].y-quantizationCenter_y_1, 2))<=uncertainty_radius*uncertainty_radius)
                  // if(tmpQuantizationIdx1==mappedQuantizationIdx1)
                  {
                      // tmpDescriptors1 << descriptors1.block<1,128>(kp1Idx,0);
                      tmpIndices1.push_back(kp1Idx);
                  }
              }
              // std::cout << "~~ tmpIndices1.size() = " << tmpIndices1.size() << std::endl;

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
              for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
              {
                  tmpDescriptors1.resize(kp1Idx+1, 128);
                  tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
              }
              // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;

              const Eigen::MatrixXi dists21 = ComputeSiftDistanceMatrix(
                  nullptr, nullptr, tmpDescriptors2, tmpDescriptors1, nullptr);
              // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;

              // num_matches21 = FindBestMatchesOneWay(dists21, match_options.max_ratio, match_options.max_distance, &matches21);

              FeatureMatches tmpQuantizationMatches21;
              FindBestMatches(dists21, match_options.max_ratio, match_options.max_distance,
                              false, &tmpQuantizationMatches21);
              // std::cout << "FindBestMatches is done!" << std::endl;

              for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches21.size(); resultCnt++)
              {
                  FeatureMatch ConvertedMatch;
                  ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches21[resultCnt].point2D_idx2];
                  ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches21[resultCnt].point2D_idx1];
                  // // matches->push_back(ConvertedMatch);
                  // matches2to1.push_back(ConvertedMatch);
                  matches21[ConvertedMatch.point2D_idx2] = ConvertedMatch.point2D_idx1;
                  num_matches21++;
              }
              // // std::cout << "index conversion is done!" << std::endl;
          //}
          // std::cout << "end of loop kp1Idx ^" << std::endl;
          // std::cout << "@@@ shareIdKp1Cnt = " << shareIdKp1Cnt << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;
      /////////////////////////////////////////////////////////////////////
      /******* Manually cross checking *******/
      if (true) {
        // std::cout << "@@@ num_matches12 = " << num_matches12 << ", @@@ matches12.size() = " << matches12.size() << ", @@@ num_matches21 = " << num_matches21 << ", @@@ matches21.size() = " << matches21.size() << std::endl;
        matches->reserve(std::min(num_matches12, num_matches21));
        for (size_t i1 = 0; i1 < matches12.size(); ++i1) {
          if (matches12[i1] != -1 && matches21[matches12[i1]] != -1 &&
              matches21[matches12[i1]] == static_cast<int>(i1)) {
            FeatureMatch match;
            match.point2D_idx1 = i1;
            match.point2D_idx2 = matches12[i1];
            matches->push_back(match);
          }
        }
        std::cout << "@@@ num_matches12 = " << num_matches12 << ", @@@ matches12.size() = " << matches12.size() << ", @@@ num_matches21 = " << num_matches21 << ", @@@ matches21.size() = " << matches21.size() << ", == ### cross-check survivors = " << matches->size() << std::endl;
      }
      /////////////////////////////////////////////////////////////////////
  //}

}

void OFGuidedMatchSiftFeaturesCPU_PixelPerfectCase_byPixel(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints& keypoints1,
                          const FeatureKeypoints& keypoints2,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          const FeatureMatches& quantization_map,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);

  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;
  for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  {

      for(point2D_t kp1Idx=0;kp1Idx<keypoints1.size(); kp1Idx++)
      // for(size_t kp1Idx=0;kp1Idx<1; kp1Idx++)
      {
          // std::cout << "loop kp1Idx ^" << std::endl;
          point2D_t tmpQuantizationIdx1 = (keypoints1[kp1Idx].x / image_scale_factor) + DeMoN_OF_Width * (keypoints1[kp1Idx].y / image_scale_factor);
          // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << std::endl;
          if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
          {
              std::vector<point2D_t> tmpIndices1;
              // std::cout << "tmpIndices1 is created!" << std::endl;

              tmpIndices1.push_back(kp1Idx);

              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
              for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
              {
                  tmpDescriptors1.resize(kp1Idx+1, 128);
                  tmpDescriptors1.block<1,128>(kp1Idx,0) = descriptors1.block<1,128>(tmpIndices1[kp1Idx],0);
              }
              // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;

              std::vector<point2D_t> tmpIndices2;
              for(point2D_t kp2Idx=0;kp2Idx<keypoints2.size(); kp2Idx++)
              {
                  point2D_t tmpQuantizationIdx2 = (keypoints2[kp2Idx].x / image_scale_factor) + DeMoN_OF_Width * (keypoints2[kp2Idx].y / image_scale_factor);
                  if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
                  {
                      // tmpDescriptors2 << descriptors2.block<1,128>(kp2Idx,0);
                      tmpIndices2.push_back(kp2Idx);
                  }
              }
              Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
              for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
              {
                  tmpDescriptors2.resize(kp2Idx+1, 128);
                  tmpDescriptors2.block<1,128>(kp2Idx,0) = descriptors2.block<1,128>(tmpIndices2[kp2Idx],0);
              }
              // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;

              // // remember to normalize the descriptors so that colmap threshold params can be used!
              // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
              // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
              // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
              // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
              // // tmpDescriptors1 = L2NormalizeFeatureDescriptors(tmpDescriptors1);
              // // tmpDescriptors2 = L2NormalizeFeatureDescriptors(tmpDescriptors2);

              const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
                  nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
              // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;

              FeatureMatches tmpQuantizationMatches;
              FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                              match_options.cross_check, &tmpQuantizationMatches);
              // std::cout << "FindBestMatches is done!" << std::endl;

              for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches.size(); resultCnt++)
              {
                  FeatureMatch ConvertedMatch;
                  ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches[resultCnt].point2D_idx1];
                  ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches[resultCnt].point2D_idx2];
                  matches->push_back(ConvertedMatch);
              }
              // std::cout << "index conversion is done!" << std::endl;

          }
          // std::cout << "end of loop kp1Idx ^" << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;

  }

}

void MatchSiftFeaturesCPU(const SiftMatchingOptions& match_options,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(matches);
  // std::cout << "#### Enter MatchSiftFeaturesCPU()" << std::endl;
  // // remember to normalize the descriptors so that colmap threshold params can be used!
  // Eigen::MatrixXf desc1 = tmpDescriptors1.cast <float> ();
  // Eigen::MatrixXf desc2 = tmpDescriptors2.cast <float> ();
  // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
  // desc2 = L1RootNormalizeFeatureDescriptors(desc2);

  const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
      nullptr, nullptr, descriptors1, descriptors2, nullptr);

  FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                  match_options.cross_check, matches);
}

void MatchSiftFeaturesCPU_Normalize(const SiftMatchingOptions& match_options,
                          const FeatureDescriptors& descriptors1,
                          const FeatureDescriptors& descriptors2,
                          FeatureMatches* matches) {
  // CHECK(match_options.Check());
  // CHECK_NOTNULL(matches);
  // std::cout << "#### Enter MatchSiftFeaturesCPU_Normalize()" << std::endl;
  // // remember to normalize the descriptors so that colmap threshold params can be used!
  // Eigen::MatrixXf desc1 = descriptors1.cast <float> ();
  // Eigen::MatrixXf desc2 = descriptors2.cast <float> ();
  // desc1 = L1RootNormalizeFeatureDescriptors(desc1);
  // desc2 = L1RootNormalizeFeatureDescriptors(desc2);
  //
  // const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
  //     nullptr, nullptr, desc1, desc2, nullptr);
  //
  // FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
  //                 match_options.cross_check, matches);
}

void MatchGuidedSiftFeaturesCPU(const SiftMatchingOptions& match_options,
                                const FeatureKeypoints& keypoints1,
                                const FeatureKeypoints& keypoints2,
                                const FeatureDescriptors& descriptors1,
                                const FeatureDescriptors& descriptors2,
                                TwoViewGeometry* two_view_geometry) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(two_view_geometry);

  const float max_residual = match_options.max_error * match_options.max_error;

  const Eigen::Matrix3f F = two_view_geometry->F.cast<float>();
  const Eigen::Matrix3f H = two_view_geometry->H.cast<float>();

  std::function<bool(float, float, float, float)> guided_filter;
  if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
      two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) {
    guided_filter = [&](const float x1, const float y1, const float x2,
                        const float y2) {
      const Eigen::Vector3f p1(x1, y1, 1.0f);
      const Eigen::Vector3f p2(x2, y2, 1.0f);
      const Eigen::Vector3f Fx1 = F * p1;
      const Eigen::Vector3f Ftx2 = F.transpose() * p2;
      const float x2tFx1 = p2.transpose() * Fx1;
      return x2tFx1 * x2tFx1 /
                 (Fx1(0) * Fx1(0) + Fx1(1) * Fx1(1) + Ftx2(0) * Ftx2(0) +
                  Ftx2(1) * Ftx2(1)) >
             max_residual;
    };
  } else if (two_view_geometry->config == TwoViewGeometry::PLANAR ||
             two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
             two_view_geometry->config ==
                 TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    guided_filter = [&](const float x1, const float y1, const float x2,
                        const float y2) {
      const Eigen::Vector3f p1(x1, y1, 1.0f);
      const Eigen::Vector2f p2(x2, y2);
      return ((H * p1).hnormalized() - p2).squaredNorm() > max_residual;
    };
  } else {
    return;
  }

  CHECK(guided_filter);

  const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
      &keypoints1, &keypoints2, descriptors1, descriptors2, guided_filter);

  FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
                  match_options.cross_check,
                  &two_view_geometry->inlier_matches);
}

bool CreateSiftGPUMatcher(const SiftMatchingOptions& match_options,
                          SiftMatchGPU* sift_match_gpu) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(sift_match_gpu);

  // SiftGPU uses many global static state variables and the initialization must
  // be thread-safe in order to work correctly. This is enforced here.
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);

  const std::vector<int> gpu_indices =
      CSVToVector<int>(match_options.gpu_index);
  CHECK_EQ(gpu_indices.size(), 1) << "SiftGPU can only run on one GPU";

  SiftGPU sift_gpu;
  sift_gpu.SetVerbose(0);

  *sift_match_gpu = SiftMatchGPU(match_options.max_num_matches);

#ifdef CUDA_ENABLED
  if (gpu_indices[0] >= 0) {
    sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA_DEVICE0 +
                                gpu_indices[0]);
  } else {
    sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);
  }
#else   // CUDA_ENABLED
  sift_match_gpu->SetLanguage(SiftMatchGPU::SIFTMATCH_GLSL);
#endif  // CUDA_ENABLED

  if (sift_match_gpu->VerifyContextGL() == 0) {
    return false;
  }

  if (!sift_match_gpu->Allocate(match_options.max_num_matches,
                                match_options.cross_check)) {
    std::cout << StringPrintf(
                     "ERROR: Not enough GPU memory to match %d features. "
                     "Reduce the maximum number of matches.",
                     match_options.max_num_matches)
              << std::endl;
    return false;
  }

#ifndef CUDA_ENABLED
  if (sift_match_gpu->GetMaxSift() < match_options.max_num_matches) {
    std::cout << StringPrintf(
                     "WARNING: OpenGL version of SiftGPU only supports a "
                     "maximum of %d matches - consider changing to CUDA-based "
                     "feature matching to avoid this limitation.",
                     sift_match_gpu->GetMaxSift())
              << std::endl;
  }
#endif  // CUDA_ENABLED

  return true;
}

void MatchSiftFeaturesGPU(const SiftMatchingOptions& match_options,
                          const FeatureDescriptors* descriptors1,
                          const FeatureDescriptors* descriptors2,
                          SiftMatchGPU* sift_match_gpu,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(sift_match_gpu);
  CHECK_NOTNULL(matches);

  if (descriptors1 != nullptr) {
    CHECK_EQ(descriptors1->cols(), 128);
    WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors1);
    sift_match_gpu->SetDescriptors(0, descriptors1->rows(),
                                   descriptors1->data());
  }

  if (descriptors2 != nullptr) {
    CHECK_EQ(descriptors2->cols(), 128);
    WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors2);
    sift_match_gpu->SetDescriptors(1, descriptors2->rows(),
                                   descriptors2->data());
  }

  matches->resize(static_cast<size_t>(match_options.max_num_matches));

  const int num_matches = sift_match_gpu->GetSiftMatch(
      match_options.max_num_matches,
      reinterpret_cast<uint32_t(*)[2]>(matches->data()),
      static_cast<float>(match_options.max_distance),
      static_cast<float>(match_options.max_ratio), match_options.cross_check);

  if (num_matches < 0) {
    std::cerr << "ERROR: Feature matching failed. This is probably caused by "
                 "insufficient GPU memory. Consider reducing the maximum "
                 "number of features and/or matches."
              << std::endl;
    matches->clear();
  } else {
    CHECK_LE(num_matches, matches->size());
    matches->resize(num_matches);
  }
}

void OFGuidedMatchSiftFeaturesGPU(const SiftMatchingOptions& match_options,
                          const FeatureKeypoints* keypoints1,
                          const FeatureKeypoints* keypoints2,
                          const FeatureDescriptors* descriptors1,
                          const FeatureDescriptors* descriptors2,
                          const FeatureMatches& quantization_map,
                          SiftMatchGPU* sift_match_gpu,
                          FeatureMatches* matches) {
  CHECK(match_options.Check());
  CHECK_NOTNULL(sift_match_gpu);
  CHECK_NOTNULL(matches);

  FeatureKeypoints Objectkeypoints1 = *keypoints1;
  FeatureKeypoints Objectkeypoints2 = *keypoints2;
  // std::cout << "Objectkeypoints1.size() = " << Objectkeypoints1.size() << std::endl;
  matches->resize(static_cast<size_t>(match_options.max_num_matches));

  point2D_t image_scale_factor = match_options.image_scale_factor; // 24; // 12; // 48; // 16; //4;
  point2D_t DeMoN_OF_Height = 48;
  point2D_t DeMoN_OF_Width = 64;
  for(point2D_t cnt=0;cnt<quantization_map.size(); cnt++)
  {
      std::vector<point2D_t> tmpIndices1;
      // std::cout << "tmpIndices1 is created!" << std::endl;

      for(point2D_t kp1Idx=0;kp1Idx<Objectkeypoints1.size(); kp1Idx++)
      {
          // std::cout << "loop kp1Idx ^" << std::endl;
          point2D_t tmpQuantizationIdx1 = (Objectkeypoints1[kp1Idx].x / image_scale_factor) + DeMoN_OF_Width * (Objectkeypoints1[kp1Idx].y / image_scale_factor);
          // std::cout << "tmpQuantizationIdx1 = " << tmpQuantizationIdx1 << std::endl;
          if(tmpQuantizationIdx1==quantization_map[cnt].point2D_idx1)
          {
              tmpIndices1.push_back(kp1Idx);
          }
          // std::cout << "end of loop kp1Idx ^" << std::endl;
      }
      // std::cout << "end of loop kp1Idx ^" << std::endl;
      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors1;
      for(point2D_t kp1Idx=0;kp1Idx<tmpIndices1.size(); kp1Idx++)
      {
          tmpDescriptors1.resize(kp1Idx+1, 128);
          tmpDescriptors1.block<1,128>(kp1Idx,0) = (*descriptors1).block<1,128>(kp1Idx,0);
      }
      // std::cout << "end of loop updating descriptor1 subblocks ^" << std::endl;

      std::vector<point2D_t> tmpIndices2;
      for(point2D_t kp2Idx=0;kp2Idx<Objectkeypoints2.size(); kp2Idx++)
      {
          point2D_t tmpQuantizationIdx2 = (Objectkeypoints1[kp2Idx].x / image_scale_factor) + DeMoN_OF_Width * (Objectkeypoints1[kp2Idx].y / image_scale_factor);
          if(tmpQuantizationIdx2==quantization_map[cnt].point2D_idx2)
          {
              tmpIndices2.push_back(kp2Idx);
          }
      }
      Eigen::Matrix<uint8_t, Eigen::Dynamic, 128, Eigen::RowMajor> tmpDescriptors2;
      for(point2D_t kp2Idx=0;kp2Idx<tmpIndices2.size(); kp2Idx++)
      {
          tmpDescriptors2.resize(kp2Idx+1, 128);
          tmpDescriptors2.block<1,128>(kp2Idx,0) = (*descriptors2).block<1,128>(kp2Idx,0);
      }
      // std::cout << "end of loop updating descriptor2 subblocks ^" << std::endl;

      // const Eigen::MatrixXi dists = ComputeSiftDistanceMatrix(
      //     nullptr, nullptr, tmpDescriptors1, tmpDescriptors2, nullptr);
      // // std::cout << "ComputeSiftDistanceMatrix is done!" << std::endl;
      //
      // FeatureMatches tmpQuantizationMatches;
      // FindBestMatches(dists, match_options.max_ratio, match_options.max_distance,
      //                 match_options.cross_check, &tmpQuantizationMatches);
      // // std::cout << "FindBestMatches is done!" << std::endl;

      // if (&tmpDescriptors1 != nullptr) {
      if (tmpDescriptors1.size()>0) {
        CHECK_EQ(tmpDescriptors1.cols(), 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, tmpDescriptors1);
        sift_match_gpu->SetDescriptors(0, tmpDescriptors1.rows(),
                                       tmpDescriptors1.data());
      } else {
          continue;
      }

      // if (&tmpDescriptors2 != nullptr) {
      if (tmpDescriptors2.size()>0) {
        CHECK_EQ(tmpDescriptors2.cols(), 128);
        WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, tmpDescriptors2);
        sift_match_gpu->SetDescriptors(1, tmpDescriptors2.rows(),
                                       tmpDescriptors2.data());
      } else {
          continue;
      }

      // std::cout << "+++++setting descriptors is done!" << std::endl;
      FeatureMatches tmpQuantizationMatches;
      tmpQuantizationMatches.resize(static_cast<size_t>(match_options.max_num_matches));
      std::cout << "///// resizing matches is done!" << std::endl;
      std::cout << "^^^ tmpQuantizationMatches.data() = " << tmpQuantizationMatches.data() << std::endl;

      // const int num_matches = 0;
      const int num_matches = sift_match_gpu->GetSiftMatch(
          match_options.max_num_matches,
          // reinterpret_cast<uint32_t(*)[2]>(tmpQuantizationMatches.data()),
          reinterpret_cast<uint32_t(*)[2]>(&(tmpQuantizationMatches[0])),
          // reinterpret_cast<uint32_t(*)[2]>(matches->data()),
          static_cast<float>(match_options.max_distance),
          static_cast<float>(match_options.max_ratio), match_options.cross_check);

      std::cout << "<<<<< GetSiftMatch is done!" << std::endl;
      // std::cout << "<<<<< GetSiftMatch is done! and num_matches = " << num_matches << std::endl;

      if (num_matches < 0) {
        std::cerr << "ERROR: Feature matching failed. This is probably caused by "
                     "insufficient GPU memory. Consider reducing the maximum "
                     "number of features and/or matches."
                  << std::endl;
        tmpQuantizationMatches.clear();
      } else {
        CHECK_LE(num_matches, tmpQuantizationMatches.size());
        tmpQuantizationMatches.resize(num_matches);
      }
      std::cout << ">>>>>>> resizing again to make the size fit is done!" << std::endl;

      for(point2D_t resultCnt=0;resultCnt<tmpQuantizationMatches.size(); resultCnt++)
      {
          FeatureMatch ConvertedMatch;
          ConvertedMatch.point2D_idx1 = tmpIndices1[tmpQuantizationMatches[resultCnt].point2D_idx1];
          ConvertedMatch.point2D_idx2 = tmpIndices2[tmpQuantizationMatches[resultCnt].point2D_idx2];
          matches->push_back(ConvertedMatch);
      }
      std::cout << "@@@@<<<<< conversion is done!" << std::endl;
      // std::cout << "index conversion is done!" << std::endl;
  }

}

void MatchGuidedSiftFeaturesGPU(const SiftMatchingOptions& match_options,
                                const FeatureKeypoints* keypoints1,
                                const FeatureKeypoints* keypoints2,
                                const FeatureDescriptors* descriptors1,
                                const FeatureDescriptors* descriptors2,
                                SiftMatchGPU* sift_match_gpu,
                                TwoViewGeometry* two_view_geometry) {
  static_assert(offsetof(FeatureKeypoint, x) == 0 * sizeof(float),
                "Invalid keypoint format");
  static_assert(offsetof(FeatureKeypoint, y) == 1 * sizeof(float),
                "Invalid keypoint format");
  static_assert(sizeof(FeatureKeypoint) == 6 * sizeof(float),
                "Invalid keypoint format");

  CHECK(match_options.Check());
  CHECK_NOTNULL(sift_match_gpu);
  CHECK_NOTNULL(two_view_geometry);

  const size_t kFeatureShapeNumElems = 4;

  if (descriptors1 != nullptr) {
    CHECK_NOTNULL(keypoints1);
    CHECK_EQ(descriptors1->rows(), keypoints1->size());
    CHECK_EQ(descriptors1->cols(), 128);
    WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors1);
    const size_t kIndex = 0;
    sift_match_gpu->SetDescriptors(kIndex, descriptors1->rows(),
                                   descriptors1->data());
    sift_match_gpu->SetFeautreLocation(
        kIndex, reinterpret_cast<const float*>(keypoints1->data()),
        kFeatureShapeNumElems);
  }

  if (descriptors2 != nullptr) {
    CHECK_NOTNULL(keypoints2);
    CHECK_EQ(descriptors2->rows(), keypoints2->size());
    CHECK_EQ(descriptors2->cols(), 128);
    WarnIfMaxNumMatchesReachedGPU(*sift_match_gpu, *descriptors2);
    const size_t kIndex = 1;
    sift_match_gpu->SetDescriptors(kIndex, descriptors2->rows(),
                                   descriptors2->data());
    sift_match_gpu->SetFeautreLocation(
        kIndex, reinterpret_cast<const float*>(keypoints2->data()),
        kFeatureShapeNumElems);
  }

  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> F;
  Eigen::Matrix<float, 3, 3, Eigen::RowMajor> H;
  float* F_ptr = nullptr;
  float* H_ptr = nullptr;
  if (two_view_geometry->config == TwoViewGeometry::CALIBRATED ||
      two_view_geometry->config == TwoViewGeometry::UNCALIBRATED) {
    F = two_view_geometry->F.cast<float>();
    F_ptr = F.data();
  } else if (two_view_geometry->config == TwoViewGeometry::PLANAR ||
             two_view_geometry->config == TwoViewGeometry::PANORAMIC ||
             two_view_geometry->config ==
                 TwoViewGeometry::PLANAR_OR_PANORAMIC) {
    H = two_view_geometry->H.cast<float>();
    H_ptr = H.data();
  } else {
    return;
  }

  CHECK(F_ptr != nullptr || H_ptr != nullptr);

  two_view_geometry->inlier_matches.resize(
      static_cast<size_t>(match_options.max_num_matches));

  const int num_matches = sift_match_gpu->GetGuidedSiftMatch(
      match_options.max_num_matches,
      reinterpret_cast<uint32_t(*)[2]>(
          two_view_geometry->inlier_matches.data()),
      H_ptr, F_ptr, static_cast<float>(match_options.max_distance),
      static_cast<float>(match_options.max_ratio),
      static_cast<float>(match_options.max_error * match_options.max_error),
      static_cast<float>(match_options.max_error * match_options.max_error),
      match_options.cross_check);

  if (num_matches < 0) {
    std::cerr << "ERROR: Feature matching failed. This is probably caused by "
                 "insufficient GPU memory. Consider reducing the maximum "
                 "number of features."
              << std::endl;
    two_view_geometry->inlier_matches.clear();
  } else {
    CHECK_LE(num_matches, two_view_geometry->inlier_matches.size());
    two_view_geometry->inlier_matches.resize(num_matches);
  }
}

}  //  namespace colmap
