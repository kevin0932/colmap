function [f, d] = compute_denseSIFT_VLFeat(filepath)
    imgRGB = imread(filepath);
    imgGRAY = rgb2gray(imgRGB);
    imgGRAY = im2single(imgGRAY);

    %% Extract dense SIFT feature keypoints
    binSize = 8 ;
    magnif = 3 ;
%     binSize = 4 ;
%     magnif = 1 ;
    pixStep = 3;
%     pixStep = 6;
%     pixStep = 10;
%     pixStep = 16;
%     pixStep = 24;
%     pixStep = 32;
    imgGRAY_smoothed = vl_imsmooth(imgGRAY, sqrt((binSize/magnif)^2 - .25)) ;

    [f, d] = vl_dsift(imgGRAY_smoothed, 'Step', pixStep, 'size', binSize) ;
    f(3,:) = binSize/magnif ;
    f(4,:) = 0 ;
    [f_, d_] = vl_sift(imgGRAY, 'frames', f) ;
end