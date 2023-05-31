import glob
import os
import pandas as pd
import csv
import numpy as np
import math
from PIL import Image
from scipy import interpolate
import time
import cv2
from tqdm import tqdm

def read_csv(csvpath,imgs_paths_list):

    
    img_names = [os.path.basename(os.path.normpath(imgpath)).split('.')[0] for imgpath in imgs_paths_list ] # list of img names instead of img paths
    
    with open(csvpath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        imgdata = [row for row in reader if row['file_name'] in img_names]
    
    return imgdata 

# For BEV warp ----------------------------------------------------------------------------------------------------------------------------------------------

    
def FoV(CCDH,CCDW,Psx,Psy,f_RC):
    sensor_h = CCDH*Psy # sensor height converted to mm
    sensor_w = CCDW*Psx # sensor width converted to mm
    FoV_v = 2*math.degrees(math.atan(sensor_h/(2*f_RC)))
    FoV_h = 2*math.degrees(math.atan(sensor_w/(2*f_RC)))
    # https://www.omnicalculator.com/other/camera-field-of-view
    return FoV_v, FoV_h

def camera_intrinsics(focal_length_mm,Xpp,Ypp,Psx,Psy,CCDW,CCDH):
    
    # mm, focal length (assume same for x and y)
    cx = Xpp/Psx # optical centre x coord in pixels
    cy = Ypp/Psy # optical centre y coord in pixels
    
    fx = focal_length_mm/Psx # focal length in pixels x dirn
    fy = focal_length_mm/Psy # focal length in pixels y dirn

        
    K = np.array([[fx, 0, cx+(CCDW/2)],
                [0, fy, cy+(CCDH/2)],
                [0, 0, 1]])
    
    return K

def RC_extrinsics(theta, h_RC, FoV_v):
    # Calculate h_VC (height of VC) and L_VC (horiz distance from WC origin to VC) and L1, L2, L3 (RC horiz distances)
    # https://ksimek.github.io/2012/08/22/extrinsic/ for getting the signs are coord systems right
    L1 = h_RC * math.tan(math.radians(90-theta-(FoV_v/2)))
    L2 = h_RC * math.tan(math.radians(90-theta))
    L3 = h_RC * math.tan(math.radians(90-theta+(FoV_v/2)))
    L_VC = (L1+L3)/2
    h_VC = (L_VC-L1)/(math.tan(math.radians(FoV_v/2)))

    # translation vector
    t_z = h_RC * math.cos(math.radians(90-theta))
    t_y = h_RC * math.sin(math.radians(90-theta))
    t = np.array([0,t_y, t_z])[:,np.newaxis] # new axis converts it to 2D single column vector

    # rotation matrix - note: only rotating around x axis
    alpha = (90 + theta) # RC to WC in RC coord system?
    c_a = math.cos(math.radians(alpha))
    s_a = math.sin(math.radians(alpha))
    R = np.array([[1, 0, 0],
                [0, c_a, -s_a],
                [0, s_a, c_a]])

    M_RC = np.vstack((np.hstack((R,t)),[0,0,0,1])) # combine R, t and make it square matrix
    return M_RC

def VC_extrinsics(theta, h_RC, FoV_v): # assumes end goal is bird's eye view
    # Calculate h_VC (height of VC) and L_VC (horiz distance from WC origin to VC) and L1, L2, L3 (RC horiz distances)
    L1 = h_RC * math.tan(math.radians(90-theta-(FoV_v/2)))
    L2 = h_RC * math.tan(math.radians(90-theta))
    L3 = h_RC * math.tan(math.radians(90-theta+(FoV_v/2)))
    L_VC = (L1+L3)/2
    h_VC = (L_VC-L1)/(math.tan(math.radians(FoV_v/2)))

    # rotation matrix - only rotates around x axis
    alpha = 180 # degrees
    c_a = math.cos(math.radians(alpha))
    s_a = math.sin(math.radians(alpha))
    # print(s_a)
    # print(np.sin(np.pi))
    R = np.array([[1, 0, 0],
                [0, c_a, -s_a],
                [0, s_a, c_a]])
    
    # translation vector
    t_y = L_VC
    t_z = h_VC
    t = np.array([0,t_y, t_z])[:,np.newaxis] # new axis converts it to 2D single column vector
    
    M_RC = np.vstack((np.hstack((R,t)),[0,0,0,1])) # combine R, t and make it square matrix

    return M_RC

def homography(K, M): # computes homography matrix using planar projective transform assumption (homography), z=0 
    M = np.delete(M,2,1) # deletes third column, axis=1 is column 
    K = np.hstack((K,[[0],[0],[0]])) # adds a 0 column to K
    H = np.matmul(K,M)
    return H



def calc_BEV_params(sample_img_path,theta,h_RC,FoV_v,f_RC,f_VC,Xpp,Ypp,Psx,Psy,CCDW,CCDH):
    
    # get matrices
    K_RC = camera_intrinsics(f_RC,Xpp,Ypp,Psx,Psy,CCDW,CCDH)
    K_VC = camera_intrinsics(f_VC,Xpp,Ypp,Psx,Psy,CCDW,CCDH)
    M_RC = RC_extrinsics(theta, h_RC, FoV_v)
    M_VC = VC_extrinsics(theta, h_RC, FoV_v)
    H_RC = homography(K_RC, M_RC)
    H_VC = homography(K_VC, M_VC)
    H = np.matmul(H_VC, np.linalg.inv(H_RC))
    
    img = cv2.imread(sample_img_path)
    h_img = img.shape[0]
    w_img = img.shape[1]
    
    # Warping corner coords
    img_corners = np.array([[[0,0],[0,h_img],[w_img,h_img],[w_img,0]]],dtype=np.float32)
    warped_corners = cv2.perspectiveTransform(img_corners,H)
    x_warp = warped_corners[0][:,0]
    y_warp = warped_corners[0][:,1]

    # Shift the warped image viewing window so it is not cut off:
    # https://stackoverflow.com/questions/22220253/cvwarpperspective-only-shows-part-of-warped-image
    xmin = int(min(x_warp))
    xmax = int(max(x_warp))
    ymin = int(min(y_warp))
    ymax = int(max(y_warp))
    
    T_offset = np.array([[1, 0, -xmin],
                        [0, 1, -ymin],
                        [0, 0, 1]]) # again, always want to do negative of xmin or ymin, so if ymin is -ve, we want to add it, if positive we want to subtract

    H_offset = np.matmul(T_offset,H)
    
    return H_offset, xmax, xmin, ymax, ymin
    
    
def BEV(imgpath,BEV_folder,H_offset, xmax, xmin, ymax, ymin):
    
    # convert to png, files made here will get overwritten by the warped result
    im = Image.open(imgpath)
    img_name = os.path.basename(os.path.normpath(imgpath)).split('.')[0] # name of img e.g. A11redlodge0006200010Lane1
    img_type = os.path.splitext(imgpath)[1]

    if img_type == '.jpg' or '.jpeg': 
        im.save(os.path.join(BEV_folder,img_name+'.png'))
        
    elif img_type == '.png':
        im.save(BEV_folder,img_name+img_type)
    
    else:
        print("error in file type, check whats happening")
        exit()

    img = cv2.imread(os.path.join(BEV_folder,img_name+'.png'))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) # alpha channel for transparent

    result = cv2.warpPerspective(img,H_offset,(xmax + (-xmin), ymax + (-ymin)), borderMode=cv2.BORDER_CONSTANT,borderValue = [0, 0, 0, 0]) # if xmin is negative, we want to add it to xmax, if its positive, we wanna subtract
    # print("End img dimensions, hxw",result.shape)
    end_file_path = os.path.join(BEV_folder,img_name +'.png')

    cv2.imwrite(end_file_path,result)

    
# For overlap calculation --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calcPhi(img1_coords, img2_coords):
    
    # phi is the angle that the trajectory makes between North (y axis) with positive CLOCKWISE- but depends on which direction car is travelling
    # always assume img1 has happened first, img2 second. i.e. in the direction of travel, img 2 is always AHEAD becuase it has happened more rceently
    
    x1, y1, z1 = img1_coords
    x2, y2, z2 = img2_coords    
    
    # Scenario A - car travels South West
    if x1 > x2 and y1 > y2 :     
        phi = math.degrees(math.atan((x1-x2)/(y1-y2)))
        direction = "SW"
    
    # Scenario B - car travels North East 
    elif x1 < x2 and y1 < y2 :
        phi = -(math.degrees(math.atan((y2-y1)/(x2-x1)))+ 90) # phi should be obtuse and -ve cus we are going anticlockwise
        direction = 'NE'
        
    # Scenario C - car travels North West
    elif x1 > x2 and y2 > y1 :    
        phi = math.degrees(math.atan((y2-y1)/(x1-x2))) + 90
        direction = 'NW'
        
    # Scenario D - car travels South East
    elif x2 > x1 and y1 > y2 :
        phi = -(math.degrees(math.atan((x2-x1)/(y1 - y2))))
        direction = 'SE'
        
    # Scenario E - car hasn't moved, return same image
    elif x1 == x2 and y1 == y2:
        phi = 0
        direction = 'hasnt moved'
        
    # Scenario F - car travels North
    elif x1 == x2 and y2 > y1 :
        phi = 180 # we travel North, and because camera is backward facing, it means camera is swung 180˚ from north
        direction = 'N'
        
    # Scenario G - car travels South
    elif x1 == x2 and y1 > y2 :
        phi = 0
        direction = 'S'
        
    # Scenario H - car travels East
    elif y1 == y2 and x1 < x2 :
        phi = -90
        direction = 'E'
    
    # Scenario I - car travels West
    elif y1 == y2 and x1 > x2 :
        phi = 90
        direction = 'W'
        
    else:
        print("Error in image coordinates")
        exit()
    
    # print("phi:", phi)
    # print("Direction car travels:", direction)
    return phi

def calcOverlap(theta,h_RC,im2_height,FoV_v,img1_coords,img2_coords, heading_1, heading_2):

    img_height_pix = im2_height
    
    # Vertical offset b/w images just based on distance between centre of camera coordinates (projected x,y,z in csv files)
    v_offset_m = math.sqrt((img1_coords[0]-img2_coords[0])**2 + (img1_coords[1]-img2_coords[1])**2+(img1_coords[2]-img2_coords[2])**2) # vertical offset in m - just the physical dist between the two camera centre points
    L1 = h_RC * math.tan(math.radians(90-theta-(FoV_v/2)))  # to get v offset in pixels, need to do scaling laws
    L2 = h_RC * math.tan(math.radians(90-theta))
    L3 = h_RC * math.tan(math.radians(90-theta+(FoV_v/2)))
    L_VC = (L1+L3)/2
    
    img_height_m = L3-L1 # image height meteres
    v_offset_pix = img_height_pix*(v_offset_m/img_height_m) # vertical offset in pix world obtained by scaling laws

    # img_width_m = img_width_pix*(img_height_m/img_height_pix) # using scaling between pixel world and meters world

    phi = calcPhi(img1_coords, img2_coords) # degrees
    epsilon = (L1+L3)/2 


    if phi > 90 or phi<0:
        print("warning, havent tested for this, maybe convert -ve angles to positive, i think will then work, also need to ")
        print("think about where image will be in y direction now if not in this quad")
        
    change_x = 0 # +ve then im1 is RIGHT of im2    
    
    # CASE 1
    if heading_1 > phi and heading_2 > phi:
        x_t = v_offset_m * math.sin(math.radians(heading_2-phi))
        x_r = epsilon * math.sin(math.radians(heading_2-heading_1)) # +ve if heading1<heading 2, -ve if head2<head1
        delx = x_t + x_r # if delx +ve here, im1 is LEFT of im2, hence:
        change_x = -delx # meters
        
        y_t = v_offset_m * math.cos(math.radians(heading_2-phi))
        y_r = epsilon*(1 - math.cos(math.radians(heading_2-heading_1))) # always positive
        dely = y_t - y_r
        change_y = dely # meters
    
    # CASE 2
    elif phi > heading_1 and phi > heading_2:
        x_t = v_offset_m * math.sin(math.radians(phi-heading_2))
        x_r = epsilon * math.sin(math.radians(heading_1-heading_2)) # +ve if heading1<heading 2, -ve if head2<head1
        delx = x_t + x_r # if delx +ve, im1 is RIGHT of im2, hence
        change_x = delx # meters
        
        y_t = v_offset_m * math.cos(math.radians(phi-heading_2))
        y_r = epsilon*(1 - math.cos(math.radians(heading_1-heading_2))) # always positive
        dely = y_t - y_r
        change_y = dely # meters
    
    # CASE 3
    elif heading_2 < phi < heading_1:
        x_r = epsilon * math.sin(math.radians(heading_1-heading_2))
        x_t = v_offset_m * math.sin(math.radians(phi-heading_2))
        delx = x_t + x_r # delx always +ve, im1 is RIGHT of im2, hence
        change_x = delx # meters   
        
        y_t = v_offset_m * math.cos(math.radians(phi-heading_2))
        y_r = epsilon*(1 - math.cos(math.radians(heading_1-heading_2))) # always positive
        dely = y_t - y_r
        change_y = dely # meters         
    
    # CASE 4
    elif heading_1 < phi < heading_2:
        x_t = v_offset_m * math.sin(math.radians(heading_2-phi))
        x_r = epsilon * math.sin(math.radians(heading_2-heading_1))
        delx = x_t + x_r # delx always +ve here, im1 is LEFT of im2, hence:
        change_x = -delx  # meters     

        y_t = v_offset_m * math.cos(math.radians(heading_2-phi))
        y_r = epsilon*(1 - math.cos(math.radians(heading_2-heading_1))) # always positive
        dely = y_t - y_r
        change_y = dely # meters

    delta_x_pix = round(change_x * (img_height_pix/img_height_m))
    delta_y_pix = round(change_y * (img_height_pix/img_height_m))
    

    
    return delta_x_pix, delta_y_pix, L1, L3, L_VC,L2   

def create_2_same_imgs(img1,img2,delta_x, delta_y):
    
    # Note: this code assumes car is always travellign downwards, i.e. img 2 is below img 1. Need to adjust code if otherwise
    assert img2.shape == img1.shape 
    
    img_height = img2.shape[0]
    img_width = img2.shape[1] 
    
    if delta_y<0:
        print(" We have a problem - dely is negtaive meaning car hasnt moved forward - please check maths")
        exit()
    
    if delta_x > 0:
        overlap_img1 = img1[delta_y:img_height, 0:img_width-delta_x]
        overlap_img2 = img2[0:img_height-delta_y, delta_x: img_width] 
 
    elif delta_x < 0: 
        overlap_img1 = img1[delta_y:img_height, abs(delta_x): img_width]
        overlap_img2 = img2[0:img_height-delta_y, 0:img_width-abs(delta_x)]        
        
    elif delta_x == 0: 
        overlap_img1 = img1[delta_y:img_height, :]
        overlap_img2 = img2[0:img_height-delta_y, :]  
    
    return overlap_img1, overlap_img2

def make_same_area_transparent(img1_overlap, img2_overlap):
    
    im1 = img1_overlap
    im2 = img2_overlap
    
    y_zeros_im1 = np.argwhere((im1 ==0).any(axis=2))[:,0] # x coords of img pix plane where we have data pts from projected las
    x_zeros_im1 = np.argwhere((im1 ==0).any(axis=2))[:,1] # y coords of img pix plane where we have data pts from projected las
    im2[y_zeros_im1,x_zeros_im1] = [0,0,0,0]# gives all the array values [x_wc, y_wc, z_wc] that have non 0 value, i.e. the x_wc_16,y_wc_16,z_wc_16 values of the projected points, i.e. all the poitns stored at each xlas, ylas
    
    y_zeros_im2 = np.argwhere((im2 ==0).any(axis=2))[:,0]
    x_zeros_im2 = np.argwhere((im2 ==0).any(axis=2))[:,1]
    im1[y_zeros_im2,x_zeros_im2] = [0,0,0,0]
    
    return im1, im2

def crop_transparent_4_shadow_removal(im1_same_area_transparent, im2_same_area_transparent):
    
    im1 = im1_same_area_transparent
    im2 = im2_same_area_transparent
    
    x_clr_pixs_im1 = np.argwhere((im1[:,:,3] !=0))[:,1] # x coords where we have coloured pixels
    x_clr_pixs_im2 = np.argwhere((im2[:,:,3] !=0))[:,1]
    
    assert x_clr_pixs_im1.all() == x_clr_pixs_im2.all()
    
    og_width = im1.shape[1]
    
    first_clr_pix = min(x_clr_pixs_im1)
    last_clr_pix = max(x_clr_pixs_im1)
    im1_cropped = im1[:,first_clr_pix:last_clr_pix+1]
    im2_cropped = im2[:,first_clr_pix:last_clr_pix+1]
    
    left_chopped_to = first_clr_pix # this pixel was included in clr image
    right_chopped_from = last_clr_pix # this pixel was included in clr image
    

    return im1_cropped, im2_cropped, left_chopped_to, right_chopped_from, og_width

# Shadow remove --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def balance(target_img, ref_img, channels_2_operate_on ,cvtColorStart=None, cvtColorEnd=None, kernel_target = 7, kernel_ref = 3):
    
    if cvtColorStart == None and cvtColorEnd==None:
        target = target_img
        ref = ref_img
    else:
        # convert original imgs from bgr2 hsv
        target = cv2.cvtColor(target_img,cvtColorStart)
        ref = cv2.cvtColor(ref_img,cvtColorStart)

        
    
    # compute LOCAL avergae of pixels by applying box blur, with kernel sizes in arguments
    target_blurred = cv2.blur(target, (kernel_target,kernel_target))
    ref_blurred = cv2.blur(ref, (kernel_ref,kernel_ref))
    
    
    # if cvtColorStart != None and cvtColorEnd!=None:
    #     # convert blurred images from bgr to hsv
    #     target_blurred = cv2.cvtColor(target_blurred,cvtColorStart)
    #     ref_blurred = cv2.cvtColor(ref_blurred,cvtColorStart)
    
    
    # Split channels into hsv and tuple so cant chnage order
    (t_H, t_S, t_V) = cv2.split(target)
    t = (t_H, t_S, t_V)
    (tm_H,tm_S, tm_V) = cv2.split(target_blurred) # m for mean
    tm = (tm_H, tm_S, tm_V)
    (rm_H, rm_S, rm_V) = cv2.split(ref_blurred)
    rm = (rm_H, rm_S, rm_V)
    
    # result channels starts with the orginal targets HSV channels, as we may only operate on one specific channel so wangt to retain the other original ones
    result_channels = [t[0],t[1],t[2]]
    
    # Iterate thru channel(s) and perform SCALING operation on each channel, to then merge at end
    for channel in channels_2_operate_on: 
        
        target_sf = t[channel]/tm[channel]
        r = np.round(target_sf*rm[channel]) # make sure its an inetger
        r = np.uint8(np.clip(r,0,255)) # clip all values to be between 0 and 255 and make sure it is uint8
        result_channels[channel] = r
        
    # result channels know contains 3 arrays, each is H, S, V. Each of those is a 2D array with the values
    result_img = cv2.merge(result_channels)
    if cvtColorStart != None and cvtColorEnd!=None:
        result_img = cv2.cvtColor(result_img,cvtColorEnd)
    
    return result_img
    
def removeShadow(shadow_im1, clean_im2):
    
    shadow_im1 = cv2.cvtColor(shadow_im1,cv2.COLOR_BGRA2BGR)
    clean_im2 = cv2.cvtColor(clean_im2,cv2.COLOR_BGRA2BGR)
    
    # Only balance the V channel of HSV, it essentially reduces brightness of shadow area
    # because value channel 
    brightness_balanced = balance(shadow_im1,clean_im2,channels_2_operate_on=[2],cvtColorStart=cv2.COLOR_BGR2HSV, cvtColorEnd=cv2.COLOR_HSV2BGR) 
    
    # now to get rid of the blue, rebalnace all rgb channels
    deshadowed_result = balance(brightness_balanced,clean_im2,channels_2_operate_on=[0,1,2], cvtColorStart=None, cvtColorEnd=None)
    
    # h_balance = balance(deshadowed_result,clean_im2,channels_2_operate_on=[0],cvtColorStart=cv2.COLOR_BGR2HSV, cvtColorEnd=cv2.COLOR_HSV2BGR) 
    
    deshadowed_result2 = balance(deshadowed_result,clean_im2,channels_2_operate_on=[0,1,2], cvtColorStart=None, cvtColorEnd=None)

    
    return deshadowed_result2
    

def main():
    
    # SET WHAT OUPUT YOU WANT ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    FINAL_RESULT_SINGLE_IMGS = True 
    COMPARISON_MODE = False
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    start_time = time.time()
    print(" ")
    print("Starting algorithm --->")
    
    # SET THESE PARAMS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    theta = 40# RED LODGE, CHAN GE IF NEEDED - pitch angle rel to horizon
    h_RC = 1.9680 # height of camera centre above ground
    f_RC = 8.5 #mm focal length of RC
    f_VC = 8.5
    # Camera parameters - mvBlueCOUGAR-X-105bC
    CCDW = 2464 # pixels, CCD (image) width 
    CCDH = 2056 # pixels, CCD (image) height
    Xpp = -0.00558 # mm, principle point x coord
    Ypp = 0.14785  # mm, principle point y coord
    Psx = 0.00345 # mm, width of pixel
    Psy = 0.00345 # mm, height of pixel
    FoV_v, FoV_h = FoV(CCDH,CCDW,Psx,Psy,f_RC)
    
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # paths to folders and data
    rootdir_path = os.path.dirname(__file__) # gives path to registration full code folder (i.e folder where this file is stored)
    csv_path = (glob.glob(os.path.join(rootdir_path,"Data","csv","*.csv")))[0]
    imgs_path_list = sorted(glob.glob(os.path.join(rootdir_path,"Data","rgb images","*"))) # returns lsit of images, dont constrain it to be a specific filetype
    BEV_warp_folder = os.path.join(rootdir_path,'Result','BEV warp')
    comparison_folder = os.path.join(rootdir_path,'Result','comparison result')
    final_result_folder = os.path.join(rootdir_path,'Result','final result imgs')

    # fetched all csv rows for the imgs present in data folder
    csvdata = read_csv(csv_path,imgs_path_list)
    csv_img_data = csvdata.copy()

    # STEP 1: BEV warp img (and removing DS store files)
    
    # remove dsstore separately first
    for img in imgs_path_list:
        if img == os.path.join(rootdir_path,'Data','rgb images','.DS_Store'): # sometime hidden DS_Store files, come up, thsi ignores the,
            imgs_path_list.remove(img)
     
     
    # Calculate BEV parameters: all images are the same size. This has to be the condition for this code to work, hence calc BEV params based on first image, use these params to BEV warp  all images
    warp_matrix, xmax, xmin, ymax, ymin = calc_BEV_params(imgs_path_list[0],theta,h_RC,FoV_v,f_RC,f_VC,Xpp,Ypp,Psx,Psy,CCDW,CCDH)


    BEV_start = time.time()
    BEV_count = 0
    print('Stage 1: BEV warping')
    no_BEV_imgs = len(imgs_path_list)
    for img in imgs_path_list:
        
        BEV(img,BEV_warp_folder,warp_matrix, xmax, xmin, ymax, ymin) # this warps img and saves the result in BEV folder with the same name as original filename
        BEV_count+=1
        
        if BEV_count%50 == 0:
            BEV_progress = int((BEV_count)/(no_BEV_imgs)*100) #minus 1 because we cant get shadow of last im
            print(f"{BEV_progress}% done ------- {BEV_count}/{no_BEV_imgs} images warped")

    BEV_end = time.time()
    print(f"Time taken to warp {no_BEV_imgs} images:{BEV_end-BEV_start}s")
    
    
    
    
    
    BEV_imgs_path_list = sorted(glob.glob(os.path.join(rootdir_path,'Result','BEV warp','*')))
    
    for img in BEV_imgs_path_list:
        if img == os.path.join(rootdir_path,'Result','BEV warp','.DS_Store'): # sometime hidden DS_Store files, come up, thsi ignores the,
            imgs_path_list.remove(img)

    no_imgs = len(BEV_imgs_path_list) # we will assume there are no incorrect images
    print(' ')
    print('Stage 2: Stitch and Remove Shadows')
    overlap_and_shadow_start_time = time.time()
    # Now working on WARPED IMAGES! PNGS not JPG, so filename is diff, but imgname is same abc.png not abc.jpg
    for i in range(len(BEV_imgs_path_list)):
        
        if i == len((BEV_imgs_path_list))-1:
            print("note: Last image cannot have its shadow removed")
            continue
        
        path2og_im1 = imgs_path_list[i] # unwarped
        og_im1 = cv2.imread(path2og_im1)
        path2im1 = BEV_imgs_path_list[i] # warped img
        path2im2 = BEV_imgs_path_list[i+1]
        
        im1 = cv2.imread(path2im1,cv2.IMREAD_UNCHANGED)


        im1_name = os.path.basename(os.path.normpath(path2im1)).split('.')[0] # name of img e.g. A11redlodge0006200010Lane1

        im1_type = os.path.splitext(path2im1)[1] # jpg
        filename1 = os.path.basename(os.path.normpath(path2im1))
        
        im2 = cv2.imread(path2im2,cv2.IMREAD_UNCHANGED)
        im2_name = os.path.basename(os.path.normpath(path2im2)).split('.')[0] # name of img e.g. A11redlodge0006200010Lane1
        im2_type = os.path.splitext(path2im2)[1] # jpg
        filename2 = os.path.basename(os.path.normpath(path2im2))
        
        im1_height = im1.shape[0]
        im1_width = im1.shape[1]
        
        im2_height = im2.shape[0]
        im2_width = im2.shape[1]

        assert im2_height == im1_height
        assert im2_width == im1_width

        
        data1 = None
        data2 = None

        for row in csv_img_data: #extract the correct row for correspondign img
            
            if row['file_name'] == im1_name:
                data1 = row
            elif row['file_name'] == im2_name:
                data2 = row
            
            if data1!=None and data2!= None:
                break # we have found the rows
        
        if data1 == None or data2 == None: 
            print("Couldn't find the csv row for image, please check images are correct")
            exit()
        
        heading1 = float(data1['heading[deg]'])
        heading2 = float(data2['heading[deg]'])
        
        x1 = float(data1['projectedX[m]'])
        y1 = float(data1['projectedY[m]'])
        z1 = float(data1['projectedZ[m]'])
        pitch1 = float(data1['pitch[deg]'])
        xyz1 = [x1,y1,z1]
        
        x2 = float(data2['projectedX[m]'])
        y2 = float(data2['projectedY[m]'])
        z2 = float(data2['projectedZ[m]'])
        pitch2 = float(data2['pitch[deg]'])      
        xyz2 = [x2,y2,z2]
        
        
        # Step 2 - Calc overlap and get two overlap images
        delta_x, delta_y = calcOverlap(theta, h_RC, im2_height,FoV_v, xyz1,xyz2,heading1,heading2)[:2] # delx is positive if im1 is RIGHT of im2, dely positive if im1 if above im2
        im1_overlap, im2_overlap = create_2_same_imgs(im1,im2,delta_x,delta_y)
        
        # Make same areas transparent
        im1_SAT, im2_SAT = make_same_area_transparent(im1_overlap, im2_overlap)

        # Crop transparent so img is smaller for shadow removal
        im1_crop, im2_crop, left_chopped_to, right_chopped_from,og_width = crop_transparent_4_shadow_removal(im1_SAT,im2_SAT)
        # cv2.imwrite(os.path.join(overlap_folder,f'pair{i+1}_im1_{filename1}'),im1_crop)
        # cv2.imwrite(os.path.join(overlap_folder,f'pair{i+1}_im2_{filename2}'),im2_crop)
        
        # STEP 3: remove the shadows
        shadow_removed = removeShadow(im1_crop,im2_crop)

        
        # STEP 4: Reverse the steps to insert shadow removed into original image
        reverse_CT4SR = cv2.cvtColor(im1_SAT,cv2.COLOR_BGRA2BGR) # convert all back to BGR now, lose alpha channel
        
        # Step 4a: get back to same shape as im1_SAT, so reverse crop_transparent_4_shadow_removal
        reverse_CT4SR[:,left_chopped_to:right_chopped_from+1]=shadow_removed
        assert im1_SAT.shape[1] == reverse_CT4SR.shape[1]

        # Step 4b: Reverse make same area transparent func to get im1 overlap
        # im1_overlap and im1_SAT are the same shape, they just have different areas of pixels that are transparent
        x_clr_pixs = np.argwhere((reverse_CT4SR[:,:] !=[0,0,0]))[:,1] # x coords where we have coloured pixels
        y_clr_pixs = np.argwhere((reverse_CT4SR[:,:] !=[0,0,0]))[:,0] 
        reverse_MSAT = cv2.cvtColor(im1_overlap,cv2.COLOR_BGRA2BGR)
        reverse_MSAT[y_clr_pixs,x_clr_pixs] = reverse_CT4SR[y_clr_pixs,x_clr_pixs]
        
        
        # Step 4c: Reverse overlap area calc
        result_im1 = cv2.cvtColor(im1,cv2.COLOR_BGRA2BGR)
        
        if delta_x > 0:
            result_im1[delta_y:im1_height, 0:im1_width-delta_x] = reverse_MSAT
        elif delta_x < 0: 
            result_im1[delta_y:im1_height, abs(delta_x): im1_width] = reverse_MSAT
        elif delta_x == 0: 
            result_im1[delta_y:im1_height,:] = reverse_MSAT

        if COMPARISON_MODE == True:
            og_shadow_img = cv2.cvtColor(im1,cv2.COLOR_BGRA2BGR)
            comparison_img = cv2.hconcat([og_shadow_img,result_im1])
            cv2.imwrite(os.path.join(comparison_folder,f'{im1_name}_comparison_shadow_removed.jpg'),comparison_img)
        
        
        inv_warp_mat = np.linalg.inv(warp_matrix)
        final_result = cv2.warpPerspective(result_im1,inv_warp_mat,(og_im1.shape[1], og_im1.shape[0]))
        
        if FINAL_RESULT_SINGLE_IMGS == True:
            cv2.imwrite(os.path.join(final_result_folder,f'{im1_name}_shadow_removed.jpg'),final_result)
        if i%50 == 0:
            progress = int((i+1)/(no_imgs-1)*100) #minus 1 because we cant get shadow of last image
            print(f"{progress}% done ------- {i+1}/{no_imgs-1} images completed")
        
        # HOW TO SAVE COMPTATION:
        # 1. get rid of alpha stuff, so that you dont have to save pngs or any BEV images, only the last image
        # 2. Everythign in BEV doesnt need to be calculated N times, we are doing the smae warp to all the images, just calc everything once
    
    end_time = time.time()
    shadow_andstitch_time = end_time-overlap_and_shadow_start_time
    print(f"Time taken to stitch and remove shadows from {no_imgs-1} images:{shadow_andstitch_time}s")
    print(' ')
    tot_runtime = end_time-start_time
    print(f"Total runtime:{tot_runtime}")
    print("Operation complete")
        
    

if __name__ == '__main__':
    main()
    
