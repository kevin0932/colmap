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

#include <QApplication>

#include "feature/matching.h"
#include "util/logging.h"
#include "util/option_manager.h"

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

#ifdef CUDA_ENABLED
  const bool kUseOpenGL = false;
#else
  const bool kUseOpenGL = true;
#endif

  // DEBUG
  std::cout << "test 1 ~~~~~~~~~~~~" << std::endl;

  OptionManager options;
  options.AddDatabaseOptions();

  // DEBUG
  std::cout << "argc = " << argc << std::endl;
  std::cout << "argv[0] = " << argv[0] << std::endl;
  std::cout << "argv[1] = " << argv[1] << std::endl;
  std::cout << "argv[2] = " << argv[2] << std::endl;
  std::cout << "argv[3] = " << argv[3] << std::endl;
  std::cout << "argv[4] = " << argv[4] << std::endl;
  std::cout << "argv[5] = " << argv[5] << std::endl;
  std::cout << "argv[6] = " << argv[6] << std::endl;
  // image_scale_factor
  std::cout << "argv[7] = " << argv[7] << std::endl;
  std::cout << "argv[8] = " << argv[8] << std::endl;
  // OF_scale_factor
  std::cout << "argv[9] = " << argv[9] << std::endl;
  std::cout << "argv[10] = " << argv[10] << std::endl;
  // uncertainty_radius
  std::cout << "argv[11] = " << argv[11] << std::endl;
  std::cout << "argv[12] = " << argv[12] << std::endl;
  // new_optical_flow_guided_matching
  std::cout << "argv[13] = " << argv[13] << std::endl;
  std::cout << "argv[14] = " << argv[14] << std::endl;
  // optical_flow_guided_matching
  std::cout << "argv[15] = " << argv[15] << std::endl;
  std::cout << "argv[16] = " << argv[16] << std::endl;
  // ManualCrossCheck
  std::cout << "argv[17] = " << argv[17] << std::endl;
  std::cout << "argv[18] = " << argv[18] << std::endl;
  // only_image_pairs_as_ref
  std::cout << "argv[19] = " << argv[19] << std::endl;
  std::cout << "argv[20] = " << argv[20] << std::endl;


  double uncertainty_radius = 42;
  std::stringstream ss;
  // std::string s = "3.1415";
  ss << argv[12];
  ss >> uncertainty_radius;

  options.sift_matching->use_gpu = false;
  options.OFGuided_matching->match_list_path = argv[6];
  options.OFGuided_matching->optical_flow_path = argv[4];
  // options.OFGuided_matching->image_scale_factor = 84;
  // options.OFGuided_matching->OF_scale_factor = 1;
  // options.OFGuided_matching->uncertainty_radius = 42;
  options.sift_matching->image_scale_factor = std::stoi(argv[8]);
  options.sift_matching->OF_scale_factor = std::stoi(argv[10]);
  options.sift_matching->uncertainty_radius = uncertainty_radius;
  // options.sift_matching->new_optical_flow_guided_matching = false;
  // options.sift_matching->optical_flow_guided_matching = true;
  // options.sift_matching->ManualCrossCheck = true;
  // options.sift_matching->only_image_pairs_as_ref = false;
  std::istringstream(argv[14]) >> options.sift_matching->new_optical_flow_guided_matching;
  std::istringstream(argv[16]) >> options.sift_matching->optical_flow_guided_matching;
  std::istringstream(argv[18]) >> options.sift_matching->ManualCrossCheck;
  std::istringstream(argv[20]) >> options.sift_matching->only_image_pairs_as_ref;

  // DEBUG
  std::cout << "test 2 ~~~~~~~~~~~~" << std::endl;

  if(options.sift_matching->only_image_pairs_as_ref == true){
      options.sift_matching->use_gpu = true;
  }

  options.AddDefaultOFGuidedMatchingOptions();
  // options.AddOFGuidedMatchingOptions(options.OFGuided_matching->image_scale_factor, options.OFGuided_matching->OF_scale_factor, options.OFGuided_matching->uncertainty_radius);


  char ** rArray = new char*[argc-18];
  for(int i=0; i < argc-18; i++) {
    rArray[i] = argv[i];
  }
  options.Parse(argc-18, rArray);
  delete [] rArray;

  // options.Parse(argc, argv);

  // DEBUG
  std::cout << "options.sift_matching->use_gpu = " << options.sift_matching->use_gpu << std::endl;
  std::cout << "options.OFGuided_matching->block_size = " << options.OFGuided_matching->block_size << std::endl;
  std::cout << "options.OFGuided_matching->match_list_path = " << options.OFGuided_matching->match_list_path << std::endl;
  std::cout << "options.OFGuided_matching->optical_flow_path = " << options.OFGuided_matching->optical_flow_path << std::endl;
  std::cout << "options.OFGuided_matching->image_scale_factor = " << options.OFGuided_matching->image_scale_factor << std::endl;
  std::cout << "options.OFGuided_matching->OF_scale_factor = " << options.OFGuided_matching->OF_scale_factor << std::endl;
  std::cout << "options.OFGuided_matching->uncertainty_radius = " << options.OFGuided_matching->uncertainty_radius << std::endl;
  std::cout << "options.sift_matching->image_scale_factor = " << options.sift_matching->image_scale_factor << std::endl;
  std::cout << "options.sift_matching->OF_scale_factor = " << options.sift_matching->OF_scale_factor << std::endl;
  std::cout << "options.sift_matching->uncertainty_radius = " << options.sift_matching->uncertainty_radius << std::endl;
  std::cout << "options.sift_matching->new_optical_flow_guided_matching = " << options.sift_matching->new_optical_flow_guided_matching << std::endl;
  std::cout << "options.sift_matching->optical_flow_guided_matching = " << options.sift_matching->optical_flow_guided_matching << std::endl;
  std::cout << "options.sift_matching->ManualCrossCheck = " << options.sift_matching->ManualCrossCheck << std::endl;
  std::cout << "options.sift_matching->only_image_pairs_as_ref = " << options.sift_matching->only_image_pairs_as_ref << std::endl;

  // std::cout << "options.sift_matching->use_gpu = " << options.sift_matching->use_gpu << std::endl;
  // std::cout << "(*options.OFGuided_matching).block_size = " << (*options.OFGuided_matching).block_size << std::endl;
  // std::cout << "(*options.OFGuided_matching).match_list_path = " << (*options.OFGuided_matching).match_list_path << std::endl;
  // std::cout << "(*options.OFGuided_matching).optical_flow_path = " << (*options.OFGuided_matching).optical_flow_path << std::endl;
  // std::cout << "(*options.OFGuided_matching).image_scale_factor = " << (*options.OFGuided_matching).image_scale_factor << std::endl;
  // std::cout << "(*options.OFGuided_matching).OF_scale_factor = " << (*options.OFGuided_matching).OF_scale_factor << std::endl;
  // std::cout << "(*options.OFGuided_matching).uncertainty_radius = " << (*options.OFGuided_matching).uncertainty_radius << std::endl;

  // std::unique_ptr<QApplication> app;
  // if (options.sift_matching->use_gpu && kUseOpenGL) {
  //   app.reset(new QApplication(argc, argv));
  // }


  OFGuidedImagePairsFeatureMatcher feature_matcher(*options.OFGuided_matching,
                                       *options.sift_matching,
                                       *options.database_path);
  // OFGuidedImagePairsFeatureMatcher * feature_matcher_ptr = new OFGuidedImagePairsFeatureMatcher(*options.OFGuided_matching,
  //                                      *options.sift_matching,
  //                                      *options.database_path);

  // feature_matcher.Start();
  // feature_matcher.Wait();


  bool EnabledFeatureMatcher = true;

  if(options.sift_matching->only_image_pairs_as_ref == true)
  {
      std::cout << "Global Standard Matching is used as reference!" << std::endl;
      // delete feature_matcher_ptr;
      // feature_matcher.~OFGuidedImagePairsFeatureMatcher();
      // options.sift_matching->use_gpu = true;
      std::cout << "options.sift_matching->use_gpu = " << options.sift_matching->use_gpu << std::endl;
      NewOFGuidedImagePairsFeatureMatcher feature_matcher(*options.OFGuided_matching,
                                           *options.sift_matching,
                                           *options.database_path);
      // NewOFGuidedImagePairsFeatureMatcher * feature_matcher_ptr = new NewOFGuidedImagePairsFeatureMatcher(*options.OFGuided_matching,
      //                                      *options.sift_matching,
      //                                      *options.database_path);
  } else if (options.sift_matching->optical_flow_guided_matching == true)
  {
      // delete feature_matcher_ptr;
      // feature_matcher.~OFGuidedImagePairsFeatureMatcher();
      std::cout << "Adaptive-Radius O.F.Guided Matching is activated!" << std::endl;
      OFGuidedImagePairsFeatureMatcher feature_matcher(*options.OFGuided_matching,
                                           *options.sift_matching,
                                           *options.database_path);
      // OFGuidedImagePairsFeatureMatcher * feature_matcher_ptr = new OFGuidedImagePairsFeatureMatcher(*options.OFGuided_matching,
      //                                      *options.sift_matching,
      //                                      *options.database_path);
  } else if (options.sift_matching->new_optical_flow_guided_matching == true)
  {
      // delete feature_matcher_ptr;
      // feature_matcher.~OFGuidedImagePairsFeatureMatcher();
      std::cout << "Fixed-Radius O.F.Guided Matching is activated!" << std::endl;
      NewOFGuidedImagePairsFeatureMatcher feature_matcher(*options.OFGuided_matching,
                                           *options.sift_matching,
                                           *options.database_path);
      // NewOFGuidedImagePairsFeatureMatcher * feature_matcher_ptr = new NewOFGuidedImagePairsFeatureMatcher(*options.OFGuided_matching,
      //                                      *options.sift_matching,
      //                                      *options.database_path);
  } else
  {
      EnabledFeatureMatcher = false;
      std::cout<< "Warning: choose one of optical flow guided matcher!" << std::endl;
  }


  // if (options.sift_matching->use_gpu && kUseOpenGL) {
  //   RunThreadWithOpenGLContext(&feature_matcher);
  // } else {
  if(EnabledFeatureMatcher)
  {
      feature_matcher.Start();
      feature_matcher.Wait();
      // feature_matcher_ptr->Start();
      // feature_matcher_ptr->Wait();
  }
  // }

  return EXIT_SUCCESS;
}
