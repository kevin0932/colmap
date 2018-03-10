%% load all images in a give directory
% directory = '/media/kevin/SamsungT5_F/ThesisDATA/southbuilding/other_sized_colmap_undistorted_images/images_demon_6/';
% directory = '/media/kevin/MYDATA/images_demon_2/';
% directory = '/media/kevin/MYDATA/images_demon_1/';
% directory = '/media/kevin/MYDATA/images_demon_96_128/';
% img_dir = fullfile(directory, 'images');
% mat_out_dir = fullfile(directory, 'DenseSIFT', 'mat_denseSIFT');
% directory = '/media/kevin/MYDATA/textureless_labwall_10032018/DenseSIFT';
% img_dir = fullfile(directory, 'images_demon_96_128');
directory = '/media/kevin/MYDATA/textureless_desk_10032018/DenseSIFT';
img_dir = fullfile(directory, 'images_demon_96_128');
mat_out_dir = fullfile(directory, 'mat_denseSIFT');
image_rgb_info  = dir( fullfile(img_dir, '*.JPG'));
image_rgb_filenames = fullfile(img_dir, {image_rgb_info.name} );
addpath('/home/kevin/devel_lib/SfM/colmap/scripts/matlab');
%% loop over each image to compute dense SIFT and save corresponding .mat file
for i=1:length(image_rgb_filenames)   
% for i=1:20  
% for i=20:length(image_rgb_filenames)   
    tic;
    img_filepath=image_rgb_filenames{i};
    img_filename = image_rgb_info(i).name;
    tmp = strsplit(img_filename,'.');
    imgname = tmp{1};
    formatSpec = "processing item %d, %s";
    str = sprintf(formatSpec,i,img_filename)
    
%     [f, d] = compute_denseSIFT_VLFeat(string(img_filepath));
    [f, d] = compute_denseSIFT_VLFeat(img_filepath);
%     f = zeros(8);
%     d = zeros(8);
    
    out_filepath_f = fullfile(mat_out_dir, [imgname, '_f.mat']);
    out_filepath_d = fullfile(mat_out_dir, [imgname, '_d.mat']);
    save(out_filepath_f, 'f');
    save(out_filepath_d, 'd');
    toc
end