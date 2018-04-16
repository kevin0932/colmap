function [f_, d_] = compute_denseSIFT_VLFeat(filepath)
    imgRGB = imread(filepath);
    imgGRAY = rgb2gray(imgRGB);
    imgGRAY = im2single(imgGRAY);

    %% Extract dense SIFT feature keypoints
    binSize = 8 ;
    magnif = 3 ;
%     binSize = 4 ;
%     magnif = 1 ;
%     pixStep = 1;
%     pixStep = 2;
%     pixStep = 3;
%     pixStep = 4;
%     pixStep = 5;
%     pixStep = 7;
%     pixStep = 8;
%     pixStep = 10;
    pixStep = 12;
%     pixStep = 16;
%     pixStep = 20;
%     pixStep = 24;
%     pixStep = 32;
    imgGRAY_smoothed = vl_imsmooth(imgGRAY, sqrt((binSize/magnif)^2 - .25)) ;

    [f, d] = vl_dsift(imgGRAY_smoothed, 'Step', pixStep, 'size', binSize) ;
    f(3,:) = binSize/magnif ;
    f(4,:) = 0 ;
    
    [f_, d_] = vl_sift(imgGRAY, 'frames', f) ;
%     [f, d] = vl_sift(imgGRAY, 'PeakThresh', 0.025, 'EdgeThresh', 10.0) ;
%     [f, d] = vl_sift(imgGRAY, 'PeakThresh', 0.001, 'EdgeThresh', 10.0) ;
end