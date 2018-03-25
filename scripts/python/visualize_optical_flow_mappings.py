#%matplotlib inline
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import PIL

def read_quantization_map(filepath='/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/test_quantization_map_OFscale_8_err_1000.txt', image_pair_to_retrieved = None):
    if image_pair_to_retrieved is None:
        quantization_maps = dict()
        fh = open(filepath)
        image_pair_num = 25
        while True:
            line = fh.readline()
            line = line.strip()
            #print(line)
            image_names = line.split(' ')
            print(image_names)
            if len(image_names)<2:
                print("end of quantization map file!")
                break
            image_pair = image_names[0]+'---'+image_names[1]

            id_map = []
            while True:
                line = fh.readline()
                if line=="\n":
                    break
                # # check if line is not empty
                # if not line or line=="\n":
                #     break
                if line[0]=="#":
                    continue

                if line[0]=="I":
                    line = line.strip()
                    image_names = line.split(' ')
                    print(image_names)
                    image_pair = image_names[0]+'---'+image_names[1]
                    continue
                line = line.strip()
                #print(line)
                index_pairs = line.split(' ')

                #print(line, " ", line[0])
                id_map.append([index_pairs[0], index_pairs[1]])
                #print(np.array(id_map).shape)
                #return
            id_map_npArr = np.array(id_map)
            #print(type(id_map_npArr))
            #print(type(id_map_npArr.dtype))
            quantization_maps[image_pair] = id_map_npArr
            if len(quantization_maps)  == image_pair_num:
                break

        fh.close()
        print('reading done!')
        return quantization_maps
    else:
        print("image_pair_to_retrieved = ", image_pair_to_retrieved)
        quantization_maps = dict()
        fh = open(filepath)
        image_pair_num = 1000000
        while True:
            line = fh.readline()
            line = line.strip()
            #print(line)
            image_names = line.split(' ')
            print(image_names)
            if len(image_names)<2:
                print("end of quantization map file!")
                break
            image_pair = image_names[0]+'---'+image_names[1]

            id_map = []
            while True:
                line = fh.readline()
                if line=="\n":
                    break
                # # check if line is not empty
                # if not line or line=="\n":
                #     break
                if line[0]=="#":
                    continue

                if line[0]=="I":
                    line = line.strip()
                    image_names = line.split(' ')
                    print(image_names)
                    image_pair = image_names[0]+'---'+image_names[1]
                    continue
                line = line.strip()
                #print(line)
                index_pairs = line.split(' ')

                #print(line, " ", line[0])
                id_map.append([index_pairs[0], index_pairs[1]])
                #print(np.array(id_map).shape)
                #return
            id_map_npArr = np.array(id_map)
            #print(type(id_map_npArr))
            #print(type(id_map_npArr.dtype))
            quantization_maps[image_pair] = id_map_npArr
            if len(quantization_maps)  == image_pair_num or image_pair_to_retrieved == image_pair:
                break

        fh.close()
        print('reading done!')
        quantization_map_retrieved = dict()
        if image_pair_to_retrieved in quantization_maps.keys():
            quantization_map_retrieved[image_pair_to_retrieved] = quantization_maps[image_pair_to_retrieved]
            return quantization_map_retrieved
        else:
            print("The image_pair ", image_pair, " doesn't exist in the file!")

def show_rgb_img(img):
    """Convenience function to display a typical color image"""
    return plt.imshow(cv2.cvtColor(img, cv2.CV_32S))

#show_rgb_img(img1)

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray


def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    print("show_sift_features len(kp) = ", len(kp))

    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy(),(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img


def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid

#image_height = 48 * 8
#image_width = 64 * 8
image_scale_factor = 1
OF_scale_factor = 8
image_height = 48 * OF_scale_factor
image_width = 64 * OF_scale_factor
if __name__ == '__main__':

    print ('OpenCV Version (should be 3.1.0, with nonfree packages installed, for this tutorial):')
    print (cv2.__version__)
    #input_images_dir = "/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/images_demon_2304_3072"
    #quantization_maps = read_quantization_map('/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/test_quantization_map_OFscale_1_err_1000.txt')
    #quantization_maps = read_quantization_map('/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/test_quantization_map_OFscale_16_err_1000.txt')
    #quantization_maps = read_quantization_map('/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/test_quantization_map_OFscale_4_err_1000.txt')
    #quantization_maps = read_quantization_map('/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/test_quantization_map_OFscale_1_err_8000.txt')
    #quantization_maps = read_quantization_map('/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/test_quantization_map_OFscale_2_err_1000.txt')
    #quantization_maps = read_quantization_map('/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/test_quantization_map_OFscale_8_err_8000.txt')
    #quantization_maps = read_quantization_map('/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/test_quantization_map_OFscale_4_err_8000.txt')

    # input_images_dir = "/media/kevin/MYDATA/cab_front/DenseSIFT/images_demon_1536_2048"
    # # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/DenseSIFT/test_quantization_map_OFscale_1_err_4000.txt')
    # # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/DenseSIFT/test_quantization_map_OFscale_1_err_1000.txt')
    # # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/DenseSIFT/test_quantization_map_OFscale_1_err_500.txt')
    # # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/DenseSIFT/test_quantization_map_OFscale_1_err_100.txt')
    # # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/DenseSIFT/test_quantization_map_OFscale_4_err_2000.txt')
    # # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/DenseSIFT/full_quantization_map_OFscale_1.txt')
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_1000.txt')
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/demon_prediction_exhaustive_pairs/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_1000_survivorRatio_500_validPairNum_23.txt')
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/demon_prediction_exhaustive_pairs/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_4000_survivorRatio_500_validPairNum_37.txt')
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/demon_prediction_exhaustive_pairs/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_8000_survivorRatio_500_validPairNum_40.txt')
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/demon_prediction_exhaustive_pairs/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_16000_survivorRatio_500_validPairNum_52.txt')
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/demon_prediction_exhaustive_pairs/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_32000_survivorRatio_500_validPairNum_55.txt')
    # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/demon_prediction_exhaustive_pairs/CrossCheckSurvivor_full_quantization_map_OFscale_4_err_8000_survivorRatio_500_validPairNum_29.txt')

    # input_images_dir = "/media/kevin/MYDATA/textureless_desk_10032018/DenseSIFT/images_demon_1536_2048"
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/textureless_desk_10032018/demon_prediction_exhaustive_pairs/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_2000_survivorRatio_500_validPairNum_218.txt')
    # quantization_maps = read_quantization_map('/media/kevin/MYDATA/textureless_desk_10032018/demon_prediction_exhaustive_pairs/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_1000_survivorRatio_500_validPairNum_73.txt')

    # quantization_maps = read_quantization_map('/media/kevin/MYDATA/cab_front/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_8_err_2000.txt')
    # input_images_dir = "/media/kevin/MYDATA/southbuilding_2304_3072/DenseSIFT/images_demon_2304_3072"
    # # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/southbuilding_2304_3072/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_4_err_2000.txt')
    # quantization_maps = read_quantization_map('/media/kevin/MYDATA/southbuilding_2304_3072/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_1000.txt')
    # quantization_maps = read_quantization_map('/media/kevin/MYDATA/southbuilding_2304_3072/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_4_err_4000.txt')

    # input_images_dir = "/media/kevin/MYDATA/southbuilding_768_1024/DenseSIFT/images_demon_768_1024"
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/southbuilding_768_1024/DenseSIFT/test.txt')
    # quantization_maps = read_quantization_map('/media/kevin/MYDATA/southbuilding_768_1024/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_16_err_16000.txt')

    input_images_dir = "/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/images_demon_384_512"
    # quantization_maps = read_quantization_map('/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_8_err_8000_survivorRatio_500_validPairNum_254.txt')
    quantization_maps = read_quantization_map('/media/kevin/MYDATA/Datasets_14032018/CalibBoard/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_8_err_8000_survivorRatio_500_validPairNum_254.txt', 'IMG_0928.JPG---IMG_0981.JPG')

    # input_images_dir = "/media/kevin/MYDATA/southbuilding_2304_3072/DenseSIFT/images_demon_2304_3072"
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/southbuilding_2304_3072/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_1000_survivorRatio_500_validPairNum_171.txt')
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/southbuilding_2304_3072/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_2_err_2000_survivorRatio_500_validPairNum_183.txt')
    # # quantization_maps = read_quantization_map('/media/kevin/MYDATA/southbuilding_2304_3072/DenseSIFT/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_1000_survivorRatio_500_validPairNum_171.txt')
    # quantization_maps = read_quantization_map('/media/kevin/MYDATA/southbuilding_10032018/demon_prediction_exhaustive_pairs/CrossCheckSurvivor_full_quantization_map_OFscale_1_err_1000_survivorRatio_500_validPairNum_274.txt')

    for image_pair in quantization_maps.keys():
        print("image_pair to be retrieved is ", image_pair)
        image_name1, image_name2 = image_pair.split('---')
        # I cropped out each stereo image into its own file.
        # You'll have to download the images to run this for yourself
        imagepath1 = os.path.join(input_images_dir,image_name1)
        print(imagepath1)
        imagepath2 = os.path.join(input_images_dir,image_name2)
        img1 = cv2.imread(imagepath1)
        img2 = cv2.imread(imagepath2)

        img1_gray = to_gray(img1)
        img2_gray = to_gray(img2)

        # plt.imshow(img1_gray, cmap='gray');
        #
        #
        #
        # plt.figure()
        # plt.imshow(cv2.cvtColor(img1, cv2.CV_32S))
        # plt.show()
        #
        # plt.figure(figsize=(12,6))#
        # plt.imshow(img1_gray);
        # plt.show()
        # # generate SIFT keypoints and descriptors
        #img1_kp, img1_desc = gen_sift_features(img1_gray)
        #img2_kp, img2_desc = gen_sift_features(img2_gray)
        # print("len(img1_kp) = ", len(img1_kp))
        # print(img1_kp[0])
        # print(img1_kp[0].pt)
        # #print("img1_kp.shape = ", img1_kp.shape)
        # #print("img1_desc.shape = ", img1_desc.shape)
        quantization_map = quantization_maps[image_pair]
        #print(quantization_map)
        print("quantization_map[0,:] = ", quantization_map[0,:])
        quantization_map = quantization_map.astype(np.uint32)
        kp1 = []
        for cnt in range(quantization_map.shape[0]):
            quan_id1 = quantization_map[cnt,0]
            # print(type(quan_id1))
            # print(quan_id1)
            #y1 = quan_id1 % image_width
            #x1 = quan_id1 - image_width*y1
            x1 = quan_id1 % image_width
            y1 = int((quan_id1-x1) / image_width)
            #print(quan_id1, " ", x1, " ", y1)
            kp = cv2.KeyPoint(x1*image_scale_factor, y1*image_scale_factor, 20, 1, 10)
            kp1.append(kp)
        #print(len(kp1))
        kp2 = []
        for cnt in range(quantization_map.shape[0]):
            quan_id2 = quantization_map[cnt,1]
            # print(type(quan_id2))
            # print(quan_id2)
            #y2 = quan_id2 % image_width
            #x2 = quan_id2 - image_width*y2
            x2 = quan_id2 % image_width
            y2 = int((quan_id2-x2) / image_width)
            #print(quan_id2, " ", x2, " ", y2)
            kp = cv2.KeyPoint(x2*image_scale_factor, y2*image_scale_factor, 20, 1, 10)
            kp2.append(kp)
        #print(len(kp2))
        print ('Here are what our optical flow quantization mapping looks like (matches are subsampled!)')
        #show_sift_features(img1_gray, img1, img1_kp);
        #show_sift_features(img1_gray, img1, kp1);
        #plt.show()

        ## create a BFMatcher object which will match up the SIFT features
        #bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        #matches = bf.match(img1_desc, img2_desc)

        ## Sort the matches in the order of their distance.
        #matches = sorted(matches, key = lambda x:x.distance)

        ## draw the top N matches
        N_MATCHES = 100

        # match_img = cv2.drawMatches(
        #     img1, img1_kp,
        #     img2, img2_kp,
        #     matches[:N_MATCHES], img2.copy(), flags=0)
        # print(type(matches))
        # print(len(matches))
        # print(matches[0].distance, ' ', matches[0].queryIdx, ' ', matches[0].trainIdx)
        dummyMatches = []
        if len(kp2)==len(kp1):
            subsamplingrate = int(len(kp1)/1000)
            for match_i in range(len(kp2)):
                match = cv2.DMatch(match_i, match_i, 0.0)
                dummyMatches.append(match)
            print(len(dummyMatches))
            if subsamplingrate<=0:
                subsamplingrate=1

            match_img = cv2.drawMatches(
                img1, kp1,
                img2, kp2,
                #dummyMatches[:N_MATCHES], img2.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                dummyMatches[::subsamplingrate], img2.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            kp_img = cv2.drawMatches(
                img1, kp1,
                img2, kp2,
                #dummyMatches[:N_MATCHES], img2.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                [], img2.copy(), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            plt.figure(figsize=(12,12))
            plt.subplot(211)
            plt.imshow(match_img);
            plt.subplot(212)
            plt.imshow(kp_img);
            # plt.figure(figsize=(12,6))
            # plt.imshow(concat_images(img1, img2));

            # images = map(Image.open, [imagepath1, imagepath2])
            # widths, heights = zip(*(i.size for i in images))
            #
            # total_width = sum(widths)
            # max_height = max(heights)
            #
            # new_im = Image.new('RGB', (total_width, max_height))
            #
            # x_offset = 0
            # for im in images:
            #   new_im.paste(im, (x_offset,0))
            #   x_offset += im.size[0]
            # list_im = [imagepath1, imagepath2]
            # imgs    = [ PIL.Image.open(i) for i in list_im ]
            # # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
            # min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
            # imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
            #
            # # save that beautiful picture
            # imgs_comb = PIL.Image.fromarray( imgs_comb)
            # imgs_comb.save( 'Trifecta.jpg' )
            #
            # # for a vertical stacking it is simple: use vstack
            # imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
            # imgs_comb = PIL.Image.fromarray( imgs_comb)
            # # imgs_comb.save( 'Trifecta_vertical.jpg' )

            # imgs_comb = pil_grid([img1,img2], 1)
            # plt.figure(figsize=(12,6))
            # plt.imshow(imgs_comb);

            plt.show()
