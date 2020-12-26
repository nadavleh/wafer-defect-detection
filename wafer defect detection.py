import copy
import numpy as np
import cv2
from matplotlib import pyplot as plt

class defectInspector:
    
    def __init__(self, img_ref):
        rows, cols = img_ref.shape 
        self.img_ref = img_ref
        self.rows = rows
        self.cols = cols
        self.D_ref = set() #pixels to ignore which we got from the inspected image. updated in separate_pipeline_inspection()
        self.D_ins = set() #pixels to ignore from the reference image. updated in separate_pipeline_inspection()
        self.D_ignore = set() # pixels to ignore due to image alignment
        
    def inspect_image(self, img_ins):
    # =============================================================================
    #   this method is the main driver of the defectInspector class. It recieves 
    #   the inspected image as an input, and outputs the binary mask.
    # =============================================================================
        
        ## get the afine transformation which relates the referance image to he inspected image
        Affine_mat = self.get_transform(img_ins)
        ## align the ref image according to the Affine_mat we've found. 
        img_ref_transformed, img_ins, self.D_ignore = self.alignImgs(img_ins, Affine_mat)
        
        ## get the mask depicting conducting channels defects, via the separate_pipeline_inspection method
        binary_mask1 = self.separate_pipeline_inspection(img_ref_transformed, img_ins)
        ## uncomment to plot results
        # plt.figure(0)
        # plt.imshow(binary_mask1, cmap='gray')
        # plt.title('separate pipeline mask')
        # plt.axis([0, self.cols, self.rows, 0])
        
        ## get the mask depicting non-channels defects, via the joint_pipeline_inspection method
        binary_mask2 = self.joint_pipeline_inspection(img_ref_transformed, img_ins)
        ## uncomment to plot results
        # plt.figure(1)
        # plt.imshow(binary_mask2, cmap='gray')
        # plt.title('joint pipeline mask')
        # plt.axis([0, self.cols, self.rows, 0])

        ## join he two masks via the join_masks mehod, and output the result
        output = self.join_masks(binary_mask1, binary_mask2)
        
        return output
    
    def separate_pipeline_inspection(self, img_ref, img_ins):
    # =============================================================================
    # this method implements the seperate pipeline processing, which essentially send each
    # img seperatly through the same image-processing pipeline which de-noise the image and threshold it
    # input: the reference and inspected images
    # output: the binary mask of te first pipeline
    # =============================================================================
    
        ## pass each image through the imgProcessPipeline_seperate method
        img_ins2 = self.imgProcessPipeline_seperate(img_ins, False, False) # no reason to preform opening on the two TH images, we do that at the end of this function on the diff image
        img_ref2 = self.imgProcessPipeline_seperate(img_ref, False, True)
        ## uncomment to plot results
        plt.figure(3)
        plt.imshow(img_ins2, cmap='gray')
        plt.title('inspected image')
        plt.axis([0, self.cols, self.rows, 0])
        plt.figure(4)
        plt.title('reference image')
        plt.imshow(img_ref2, cmap='gray')
        plt.axis([0, self.cols, self.rows, 0])
        
        ## here we yield all the pixels which are to be ignored (blackend\zero'ed) in the joint_pipeline_inspection.
        ## these include the conducting-channels and a margin around them               
        ## this method takes care of getting the conducting channels pixels + a margin around them, and renders the loop above as redundant
        self.get_ignored_pixels(img_ref2)
        
        ## get the absolute value of the difference of the two thresholded images and apply another threshold (inverted this time to get the defects as white)
        diff = cv2.absdiff(img_ins2, img_ref2)
        thresh, diff = cv2.threshold(diff,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        

        ## preform once more the closing morphological opperation to emphesize the defects 
        ## and reduce the wafer's conductor channels boarder effect
        kenel_size = (3,3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kenel_size)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
        thresh, diff = cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV)

        return diff
    
    def get_ignored_pixels(self, img):
    # =============================================================================
    # this method receives a binary image, copies it so that the original image is unchanged
    # and then inflate the conducting channels by morphological dilation, and records all the pixels
    # of the dilated channels, for later use.
    # input: the thresholded reference image
    # output: None
    # =============================================================================
        ## img is a binary mask of the referance image
        mask = copy.deepcopy(img)
        _, mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV)
        ## uncomment to plot. for debugging puposses
        # plt.figure(5)
        # plt.imshow(mask, cmap = 'gray')
        
        # dilate the channels to get a margin around them
        kenel_size = (20,20)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kenel_size)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        
        ## uncomment to plot. for debugging puposses
        # plt.figure(6)
        # plt.imshow(mask, cmap = 'gray')
        
        # go through all the pixels and record which are part of the channels (the white ones)
        for i in range(self.rows):
            for j in range(self.cols):
                if mask[i,j] == 255:
                    self.D_ref.add((i,j))
        
    def joint_pipeline_inspection(self, img_ref, img_ins):
    # =============================================================================
    # this method implements the joint pipeline processing, which subtracts the two given images,
    # de-noise the result and threshold it. finally we zero all the pixels that are to be ignored (which
    # we got through the separate_pipeline_inspection method)
    #
    # input: the reference and inspected images
    # output: the binary mask of the second pipeline
    # =============================================================================
        diff = cv2.absdiff(img_ins, img_ref)
        plt.figure(8)
        plt.imshow(diff, cmap = 'gray')
        diff = self.imgProcessPipeline_joint(diff, preform_closing = True)
        return diff

    def get_transform(self, img_ins):
    # =============================================================================
    # this method finds features in both the inspected image and the reference image,
    # and then matches this features and estimates the affine transformation between them
    #
    # input: the reference 
    # output: the affine transformation between ref and ins images, as a 2 by 3 matrix.
    # =============================================================================
        ## Create ORB object to find features with 
        orb = cv2.ORB_create()
        ## Find the keypoints and descriptors with ORB
        kpts1, descs1 = orb.detectAndCompute(self.img_ref,None)
        kpts2, descs2 = orb.detectAndCompute(img_ins,None)
        ## match descriptors and sort them by distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descs1, descs2)
        dmatches = sorted(matches, key = lambda x:x.distance)
        ## extract the matched keypoints from "kpts" objects and convert them to type np.float32
        src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
        dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
        ## estimate the afine transform
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        mask = mask.ravel() 
        ## Select inlier data points
        src_pts=src_pts[mask == 1]
        dst_pts=dst_pts[mask == 1]

        return M
        
    def alignImgs(self, img_ins, M):
    # =============================================================================
    # this method simply alligns the ref image to ins_img's perspective, and past the alligned ref image ontop of a copy of ins_image
    # to reduce the chance of detecting a defect in the non-common section of the two images
    # input: the reference and afine transform matrix
    # output: ref_img aligned to ins_img, ins_img, pixels of non-intersecting regions of two images (which will later be set to zero)
    # =============================================================================
        
        
        ## using  cv2.warpAffine() to warp img_ref ontop of img_ins, also changes img_ins for some reason, hence 
        ## we create a copy of img_ins that can be changed without demaging img_ins. We also create a copy of img_ref
        ## so that the original ref image is not changed according to some arbitrary inspected image.
        img_ins_cpy = copy.deepcopy(img_ins)
        img_ref_cpy = copy.deepcopy(self.img_ref)

        
        ## warp the reference image (or a copy of it to be precise) and paste it ontop of the inspected image. use cubic interpolation 
        img_ref_cpy = cv2.warpAffine(img_ref_cpy, M, (self.cols, self.rows), img_ins_cpy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        
        # =============================================================================
        ## get pixels which are to be overllooked
        black_img = np.zeros((self.rows, self.cols))
        white_img = np.ones((self.rows, self.cols))
        black_zone_img = cv2.warpAffine(white_img, M, (self.cols, self.rows), black_img, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT) 
        overlooked_pixels = set()
        for i in range(self.rows):
            for j in range(self.cols):
                if black_zone_img[i,j] == 0:
                    overlooked_pixels.add((i,j))

        # =============================================================================

        return (img_ref_cpy, img_ins, overlooked_pixels)
    
    def imgProcessPipeline_seperate(self, img, preform_opening = False, ref_img = False):
    # =============================================================================
    # this method receives either the ref or ins images, and runs it through 
    # some processing which include de-noising and thresholding
    #
    # input: either ref_img or ins_img, a boolean flag to indicate if morphological openning is to take place
    #        and a ref_img boolean flag to indicate if the passed image is the ref_image, so we can prefom differnt de-noising on it
    #        if we want to.
    # output: the thresholded input image
    # =============================================================================
        ## Pass the image through a low-pass filter to reduce noise.
        ## the Gassian blur works fine, but we may tru diferent options as seen here.
        ## simply uncomment one of the options to try it (and comment the previous filter implemented)
        blured_img = cv2.GaussianBlur(img,(3,3),0)
        # blured_img = cv2.bilateralFilter(img,11,11,11)
        # if ref_img:
        #     blured_img = cv2.fastNlMeansDenoising(img,7,7,7,21)
        # else:
        #     blured_img = cv2.bilateralFilter(img,11,11,11)
        
        
        ## preform adaptive thresholding via the cv2.threshold() method and Otsu's algorithm: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
        ## this will produce a binary image
        thresh, blured_img = cv2.threshold(blured_img,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY) # the thresh var is to be discarded, its the threshold that was yielded via Otsu's addaptive thresholding
    
        if preform_opening:
            kenel_size = (5,5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kenel_size) # changing the kernel size really effects the detection, try replacing (5,5) with (10,10)
            blured_img = cv2.morphologyEx(blured_img, cv2.MORPH_OPEN, kernel)
              
        return blured_img
    
    
    def imgProcessPipeline_joint(self, img, preform_closing = False):
    # =============================================================================
    # this method receives the absolute difference image from the joint_pipeline method
    # and runs it through some processing to get the final mask.
    #
    # input: either ref_img or ins_img, a boolean flag to indicate if morphological openning is to take place
    #        and a ref_img boolean flag to indicate if the passed image is the ref_image, so we can prefom differnt de-noising on it
    #        if we want to.
    # output: the thresholded input image
    # =============================================================================
        ## Pass the image through a low-pass filter to reduce noise
        ## simply uncomment one of the options to try it (and comment the previous filter implemented)
        # blured_img = cv2.GaussianBlur(img,(3,3),0)
        blured_img = cv2.fastNlMeansDenoising(img,7,7,7,21)
        # blured_img = cv2.medianBlur(img,3)
        # blured_img = cv2.bilateralFilter(img,13,15,15)
        

        
        
        
        ## preform adaptive thresholding via the cv2.threshold() method and Otsu's algorithm: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
        ## this will produce a binary image
        thresh, blured_img = cv2.threshold(blured_img,0,255,cv2.THRESH_OTSU | cv2.THRESH_BINARY) # the thresh var is to be discarded, its the threshold that was yielded via Otsu's addaptive thresholding
        ## uncomment to show result. for debugging purposses
        # plt.figure(7)
        # plt.imshow(blured_img, cmap='gray')
        
        ## now we finally have the binary mask of the non-channel defects, however it contains the boarders of the channels, 
        ## and thus we remove them (zero them out). luckily we kept track of them previously.
        for i in range(self.rows):
            for j in range(self.cols):
                if (i,j) in self.D_ref or (i,j) in self.D_ins or (i,j) in self.D_ignore:
                    blured_img[i,j] = 0      
        ## uncomment to show result. for debugging purposses            
        # plt.figure(8)
        # plt.imshow(blured_img, cmap='gray')
        
        ## finally we preform mophological closing opperation, to get rid of the noise
        if preform_closing:               
            kenel_size = (2,2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kenel_size) # changing the kernel size really effects the detection, try replacing (5,5) with (10,10)
            mask1 = cv2.morphologyEx(blured_img, cv2.MORPH_CLOSE, kernel)
       
        ## uncomment to show result. for debugging purposses 
        # plt.figure(9)
        # plt.imshow(mask1, cmap='gray')
        
        return mask1

    def join_masks(self, img1, img2):
        output = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                if img1[i,j] == 255 or img2[i,j] == 255:
                    output[i,j] = 255
                    
        kenel_size = (20,20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kenel_size) # changing the kernel size really effects the detection, try replacing (5,5) with (10,10)
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
        
        ## blacken\ignore the top 5 pixels. This is a temporary solution to deal with some noise issues with the case3 images
        for i in range(self.rows):
            for j in range(self.cols):
                if i < 5 :
                    output[i,j] = 0    
            
        return output


def main():
    ## load images
    path = r'insert path here'
    img_ref = cv2.imread(path, 0) # the '0' is to read image as grayscale (in rgb all the channels are the same because the image is grayscale already)

    path = r'insert path here'
    img_ins = cv2.imread(path, 0)
    

    
    DI = defectInspector(img_ref)
    
    
    res = DI.inspect_image(img_ins)
    
    plt.figure(2)
    plt.imshow(res, cmap='gray')
    plt.title('final result')
    plt.axis([0, DI.cols, DI.rows, 0])

    

if __name__ == "__main__":
    main()
