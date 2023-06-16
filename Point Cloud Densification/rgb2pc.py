import glob
import os
import pandas as pd
import csv
import laspy
import numpy as np
import math
from PIL import Image
from scipy import interpolate
import time
import cv2

def read_img_uint32(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    rgb_png = np.array(img_file, dtype='uint32')  # LAS stores its coordinates in uint32
    img_file.close()
    return rgb_png

# fetches CSV data of the images present in rgb images folder
def read_csv(csvpath,imgs_paths_list):
    
    img_names = [os.path.splitext(imgpath.split('/')[-1])[0] for imgpath in imgs_paths_list ] # list of img names instead of img paths
    
    with open(csvpath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        imgdata = [row for row in reader if row['file_name'] in img_names]

    # first = imgdata[0]
    # print(first['file_name'])
    
    return imgdata

def convertLAS2numpy(path2las):
    
    las = laspy.read(path2las)
   
    x_scale, y_scale, z_scale = las.header.scales
    x_offset, y_offset, z_offset = las.header.offsets
    x_coords = (las.X * x_scale) + x_offset
    y_coords = (las.Y * y_scale) + y_offset
    z_coords = (las.Z * z_scale) + z_offset
    ones = np.ones(len(x_coords))
    
    reds  = (las.red)/256 #LAS stores RGB as 16-bit, but usually we want 8-bit so convert
    greens = (las.green)/256
    blues = (las.blue)/256 
    greyscale = (np.add(reds,(np.add(blues,greens))))/3
    homogeneous_data_matrix = np.vstack((x_coords,y_coords,z_coords,ones,blues,greens,reds,greyscale)) # returns [x,y,z,1,B,G,R,greyscale]

    return homogeneous_data_matrix

def rotMat(roll, pitch, heading, mode='norm'):
    
    alpha = math.radians(heading)
    cosa = math.cos(alpha)
    sina = math.sin(alpha)
    
    beta  = math.radians(pitch)
    cosb = math.cos(beta)
    sinb = math.sin(beta)
    
    gamma = math.radians(roll)
    cosg = math.cos(gamma)
    sing = math.sin(gamma)
    
    yaw_mat = np.array([[cosa , -sina , 0],
                        [sina, cosa, 0],
                        [0, 0, 1]])
    
    pitch_mat = np.array([[cosb, 0, sinb],
                          [0, 1, 0],
                          [-sinb, 0, cosb]])
    
    roll_mat = np.array([[1, 0, 0],
                         [0, cosg, -sing],
                         [0, sing, cosg]])
    if mode =='norm':
        rotmat = yaw_mat @ pitch_mat @ roll_mat
    elif mode == 'rev':
        rotmat = roll_mat @ pitch_mat @ yaw_mat
    else:
        print("error in mode")
    
    return rotmat

def extrinsicsMat(rph_list, xyz_list, error_correction): # orientation and position matrix of CCS rel to WCS

    wcs2ccs = np.row_stack((np.column_stack((rotMat(180,0,0) ,np.array([[0],[0],[0]]))), np.array([0,0,0,1])))  # world coord system to camera coordinate system
    convert2projangles = [90+rph_list[1],rph_list[0],-rph_list[2]] # according to the orinetation of the LAS file - very odd but found by trial and error
    t = np.swapaxes(np.array([xyz_list]),0,1) # in [[x],[y],[z]] rather than [x,y,z]
    A, B, C = np.add(convert2projangles,error_correction)
    R = rotMat(A,B,C)
    extrinsics_mat = (np.row_stack((np.column_stack((R,t)), np.array([0,0,0,1])))) @ wcs2ccs

    return extrinsics_mat
        
def intrinsicsMat(focal_length_mm):
    
    CCDW = 2464 # pixels, CCD (image) width 
    CCDH = 2056 # pixels, CCD (image) height
    Xpp = -0.00558 # mm, principle point x coord
    Ypp = 0.14785  # mm, principle point y coord
    Psx = 0.00345 # mm, width of pixel
    Psy = 0.00345 # mm, height of pixel
    cx = Xpp/Psx # optical centre x coord in pixels
    cy = Ypp/Psy # optical centre y coord in pixels
    fx = focal_length_mm/Psx # focal length in pixels x dirn
    fy = focal_length_mm/Psy # focal length in pixels y dirn
    
    K = np.array([[fx, 0, cx+(CCDW/2)],
                [0, fy, cy+(CCDH/2)],
                [0, 0, 1]])
    
    return K
    
def projection_WCS2PCS(intrinsicsMat:np.ndarray, extrinsicsMat:np.ndarray, points: np.ndarray,
                   map_width, map_height, path2las):
    
    las = laspy.read(path2las)
    x_offset, y_offset, z_offset = las.header.offsets
    x_scale, y_scale, z_scale = las.header.scales
    
    "depth_map = np.zeros(shape=(map_height, map_width), dtype=np.uint16)"
    
    rgbd_map = np.zeros(shape=(map_height, map_width,4), dtype=np.uint8) # each element at coord x,y has array [r,g,b,depth from cam cneter] with rgb in uint8 (normal), is rgb image wtih depth value for each coloured pixel
    LAS_data_map = np.zeros(shape=(map_height, map_width,6), dtype='uint32') # stores [x_wc,y_wc,z_wc,r_wc_uint32,g_wc_uint32,b_wc_uint32] so all xyz and rgb of real world points but in LAS format that are captured in img
    LAS_new = []
    points_ccs_hg = np.linalg.inv(extrinsicsMat) @ points[0:4, :]
    points_pix_cs_hg = intrinsicsMat @  points_ccs_hg # converts homogeneous world points into homogeneous pixel coord points, inv exmat because need wcs rel2 ccs, but exmat is ccs rel2 wcs
    
    uv_rows = points_pix_cs_hg[0:2, :].reshape((2, points.shape[1])) # just takes the first two rows of projected_pts_homo (x/s, y/s) and makes new matrix
    w_row = points_pix_cs_hg[2, :].reshape((1, points.shape[1])) # creates 1D row vector of the s values (homogeneous last row of pix cord sustem points)
    w_points = uv_rows / w_row # all LAS points in pixel coord system (non homo) (2 x N array)

    for pt_pix, pt_cam, pt_wcs in zip(w_points.transpose(),
                                   points_ccs_hg.transpose(),
                                   points.transpose()):
        
        if 0 < pt_pix[0] < map_width and 0 < pt_pix[1] < map_height and pt_cam[2]>=0 : # only keep projected points that are within the actual map boundary, and if Z_ccs >0 (so we only get points INFORNT OF CAMERA)
            
            discretised_pt_pix = (int(pt_pix[0]), int(pt_pix[1])) # we cannot have decimals of pixels, so turn (x_proj_pixcs, y_proj_pixcs)  into ints 
            
            # need to convert the valid points back into LAS compatible format (uint32)
            x_wc_uint32 = (pt_wcs[0] - x_offset)/x_scale
            y_wc_uint32 = (pt_wcs[1] - y_offset)/y_scale
            z_wc_uint32 = (pt_wcs[2] - z_offset)/z_scale

            b_wc_uint8 = pt_wcs[4]
            g_wc_uint8 = pt_wcs[5]
            r_wc_uint8 = pt_wcs[6]
            greyscale = pt_wcs[7]

            # convert to uint16 instead of uint8 for LAS format
            b_wc_uint16 = b_wc_uint8*256
            g_wc_uint16 = g_wc_uint8*256
            r_wc_uint16 = r_wc_uint8*256
            
            """        
            existing_depth_val = depth_map[discretised_pt_pix[1], discretised_pt_pix[0]]
            depth_val = depth_factor * np.sqrt(np.sum(np.square(pt_cam))) 
        
            if existing_depth_val == 0:
                depth_map[discretised_pt_pix[1], discretised_pt_pix[0]] = depth_val
            else:
                depth_map[discretised_pt_pix[1], discretised_pt_pix[0]] = (existing_depth_val+depth_val)/2
            """

            # Colour depth map (rgbd) - projected points onto black plain image with the correct colours and distance from cam center 
            existing_depth = rgbd_map[discretised_pt_pix[1], discretised_pt_pix[0]][3]
            depth = np.sqrt(np.sum(np.square(pt_cam[:3]))) # depth from the camera centre, as these pts are defined in cam coord system
            
            if existing_depth == 0:
                LAS_new.append([r_wc_uint8,g_wc_uint8,b_wc_uint8,depth])
                rgbd_map[discretised_pt_pix[1], discretised_pt_pix[0]] = [r_wc_uint8,g_wc_uint8,b_wc_uint8,depth] # if using pil - RGB in arrays
                LAS_data_map[discretised_pt_pix[1], discretised_pt_pix[0]] = [x_wc_uint32,y_wc_uint32,z_wc_uint32,r_wc_uint16,g_wc_uint16,b_wc_uint16]

            elif depth < existing_depth: # we want the pt closest to the cam i.e. highest up so (lowest depth) to be the visible one
                LAS_new.append([r_wc_uint8,g_wc_uint8,b_wc_uint8,depth])
                rgbd_map[discretised_pt_pix[1], discretised_pt_pix[0]] = [r_wc_uint8,g_wc_uint8,b_wc_uint8,depth]
                LAS_data_map[discretised_pt_pix[1], discretised_pt_pix[0]] = [x_wc_uint32,y_wc_uint32,z_wc_uint32,r_wc_uint16,g_wc_uint16,b_wc_uint16]
            
    projected_img = rgbd_map[:,:,:3] # same as rgbd_map, but removed depth value from all the arrays, so just an image
    
    return rgbd_map, projected_img, LAS_data_map

def interpolateImage(): 
    # interpolates the availaible projected points on the image array grid
    
    pass

def append_to_las(in_las, out_las):
    with laspy.open(out_las, mode='a') as outlas: # mode a stands for appender https://laspy.readthedocs.io/en/latest/api/index.html#laspy.open
        with laspy.open(in_las) as inlas:
            for points in inlas.chunk_iterator(2_000_000):
                outlas.append_points(points)

def main():
    
    
    start_time = time.time()
    
    # paths to folders and data
    rootdir_path = os.path.dirname(__file__) # gives path to registration full code folder (i.e folder where this file is stored)
    csv_path = (glob.glob(os.path.join(rootdir_path,"Data/csv/*.csv")))[0]
    imgs_path_list = sorted(glob.glob(os.path.join(rootdir_path,'Data/rgb images/*')),reverse=True) # returns lsit of images, dont constrain it to be a specific filetype
    LAS_path = (glob.glob(os.path.join(rootdir_path,'Data/LAS files/*.LAS')))[0]
    projimg_folder = os.path.join(rootdir_path,"Result","projected images")
    # fetched all csv rows for the imgs present in data folder
    csvdata = read_csv(csv_path,imgs_path_list)
    csv_img_data = csvdata.copy()

    
    im_count = 0
    
    
    
    
    # as shown in read_csv function, the values in the columns can be foudn by doing row['file_name']
    
    no_imgs = len(imgs_path_list)
    no_rows_extracted = len(csv_img_data) #i.e. the number of imgs we are actually oeprating on
    not_present = []
    LAS_points = convertLAS2numpy(LAS_path)
    

    
    # SETTING WHETHER WE WANT TO SKIP POINTS FOR LOWER DENSIFICATION
    skip_pts = True
    # every nth column taken, and every nth row, hence nxn times less points
    n = 3
    
    # ERROR CORRECTIONS - bestEC means best error correction (for first lane of A11 Red lodge, Lane 1)
    bestEC = [-1,-0.5,1.5]
    noEC = [0,0,0]
    error_correct = bestEC
    testname = f"finalLASresult"

    # for if you want to view the denser PC overlaid onto original - because of interpolation the points conflict as they have same depth so its sometimes hard to see the coinciding points 
    increase_z = 0

    
    

    for path2img in imgs_path_list:
        
        
        img_name = (os.path.splitext(path2img)[0]).split('/')[-1] # name of img e.g. A11redlodge0006200010Lane1
        img_type = os.path.splitext(path2img)[1] # jpg
        filename = path2img.split('/')[-1]
        data = None

        for row in csv_img_data: #extract the correct row for correspondign img
            if row['file_name'] == img_name:
                data = row
                break #there shouls be only one row that correponds to an image, so exit once we have foudn the row
                
        if data == None: # if images exist in data folder that are not present in csv file, dont let it go through rest of the code, skip to next iteration
            not_present.append(filename)
            continue

        im_count+=1
        progress = int(im_count/no_rows_extracted*100)
        
        roll = float(data['roll[deg]'])
        pitch = float(data['pitch[deg]'])
        heading = float(data['heading[deg]'])
        x = float(data['projectedX[m]'])
        y = float(data['projectedY[m]'])
        z = float(data['projectedZ[m]'])
        rph = [roll,pitch,heading]
        xyz = [x,y,z]
        
        f = 8.5 #mm
        exMat = extrinsicsMat(rph,xyz,error_correct)
        K = np.hstack((intrinsicsMat(f),np.array([[0.],[0.],[0.]])))
        
        imArray_uint32 = read_img_uint32(path2img) 
        im_height = imArray_uint32.shape[0]
        im_width = imArray_uint32.shape[1]
        
        rgbd, projectedimg, LAS_data_array = projection_WCS2PCS(K, exMat, LAS_points, im_width, im_height, LAS_path)
        
        
        mapped_LAS_pts_wc = LAS_data_array[:,:,:3] # creates array with just x_wc_uint32, r_wc_uin32, z_wc_uin32
        mapped_LAS_pts_rgb = LAS_data_array[:,:,3:] # creates array with just r_wc_uint16,g_wc_uint16,b_wc_uint16

        # now interpolate    # ALWAYS REMEBER - THE FIRST INDEX IS THE ROW I.E. Y COORD!
        x_cols = np.arange(0,mapped_LAS_pts_wc.shape[1],1)
        y_rows = np.arange(0,mapped_LAS_pts_wc.shape[0],1)
        x_grid, y_grid =np.meshgrid(x_cols,y_rows) # makes two grids, both of the shape of the img or x_cols x y_cols and fills the first with all the xvalues and the secodn one with all the y vals
        
        # the below argwhere usualy gives [ [x1,y1], [x3,y3], ... [xN,yN]] etc, i.e. a list of coords where we dont have [0,0,0](black pixel). the next two lines slplit this into the x values, then the y values
        y_las = np.argwhere((mapped_LAS_pts_wc !=0).any(axis=2))[:,0] # x coords of img pix plane where we have data pts from projected las
        x_las = np.argwhere((mapped_LAS_pts_wc !=0).any(axis=2))[:,1] # y coords of img pix plane where we have data pts from projected las
        data_las = mapped_LAS_pts_wc[y_las,x_las] # gives all the array values [x_wc, y_wc, z_wc] that have non 0 value, i.e. the x_wc_16,y_wc_16,z_wc_16 values of the projected points, i.e. all the poitns stored at each xlas, ylas
        

       
        # create interpolated array with full XYZ_wc values at every img pixel
        # interLinear_map = interpolate.griddata((x_las,y_las),data_las,(x_grid,y_grid),method='linear') #for soem reason interlinear does not work
        interNearest_map = interpolate.griddata((x_las,y_las),data_las,(x_grid,y_grid),method='linear')
        
        # print(np.min(data_las[:,0]))
        # input()
        # print(interNearest_map)
        # input()
        
        
        interNearest_map_x = np.clip(interNearest_map[:,:,0], np.nanmin(data_las[:,0]),np.nanmax(data_las[:,0]))
        interNearest_map_y = np.clip(interNearest_map[:,:,1], np.nanmin(data_las[:,1]),np.nanmax(data_las[:,1]))
        # for clipping:
        # interNearest_map_z = np.clip(interNearest_map[:,:,2], np.nanmin(data_las[:,2]),np.nanmin(data_las[:,2]) + 0.1*(np.nanmax(data_las[:,2]) - np.nanmin(data_las[:,2])))
        
        interNearest_map_z = np.clip(interNearest_map[:,:,2], np.nanmin(data_las[:,2]),np.nanmin(data_las[:,2]))

        interNearest_map = np.dstack((interNearest_map_x,interNearest_map_y,interNearest_map_z))
        
   
    
        # print(f"internearest map:{interNearest_map.shape}")
        imArray_uint32 = read_img_uint32(path2img) *256
        interpolated_im_depth_map = np.dstack((interNearest_map,imArray_uint32)) # combining the orginal rgb vals from the 2D img with the XYZ coords extracted from LAS and interpolated that proiject to the image area 
        if skip_pts == True:
            interpolated_im_depth_map = interpolated_im_depth_map[::n,::n,:]

        
        
        interpolated_im_depth_map = np.round(interpolated_im_depth_map)
        
        # np.savetxt("tmp_array_end.csv",interpolated_im_depth_map[5,:,:], delimiter = ",")
        # print(np.max(interpolated_im_depth_map[:,:,0]),np.min(interpolated_im_depth_map[:,:,0]))
        # print(np.max(interpolated_im_depth_map[:,:,1]),np.min(interpolated_im_depth_map[:,:,1]))
        # print(np.max(interpolated_im_depth_map[:,:,2]),np.min(interpolated_im_depth_map[:,:,2]))
        # interpolated_im_depth_map = interpolated_im_depth_map[0:interpolated_im_depth_map.shape[0]:10,0:interpolated_im_depth_map.shape[1]:10,:,:,:]
        
        
        
        
        no_pts = interpolated_im_depth_map.shape[0]*interpolated_im_depth_map.shape[1] # number of new las points should just be h x w of the image depth map array
        new_LAS_points = interpolated_im_depth_map.reshape((no_pts,6)).T # convert 3D array into 2D array of all the points, as converting to LAS doesnt need it in the image structure so [x1,x2,x3...], [y1,y2,y3,...], [z1,z2,z3,....],[r1,r2,r3],....etc
        nan_row_array = ~np.isnan(new_LAS_points).any(axis = 0)
        
        new_LAS_points = new_LAS_points[:,nan_row_array]
        
        if im_count == 1:
            all_las_pts = new_LAS_points
        
        else:
            all_las_pts = np.concatenate((all_las_pts,new_LAS_points),axis=1)  
        
        print(f"{progress}% done ------- {im_count}/{no_rows_extracted} images completed")
        
    print("Now creating the LAS file -------->")
    # now we have las points of all images, create a LAS file
    lasfile = laspy.read(LAS_path)
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
    header.offsets = lasfile.header.offsets
    header.scales = lasfile.header.scales
    print(all_las_pts[2,:][0])
    all_las_pts[2,:]=all_las_pts[2,:]+increase_z
    print(all_las_pts[2,:][0])
    newlas = laspy.LasData(header)
    
    newlas.X = all_las_pts[0,:]
    newlas.Y = all_las_pts[1,:]
    newlas.Z = all_las_pts[2,:]
    newlas.red = all_las_pts[3,:]
    newlas.green = all_las_pts[4,:]
    newlas.blue = all_las_pts[5,:]

    # newlasname = rootdir_path +f"/Result/LAS result/projected_las"+ LAS_path.split('/')[-1]
    
    newlasname = os.path.join(rootdir_path,"Result/LAS result",testname+'.las')
    
    
    newlas.write(newlasname)

    
    
    if no_rows_extracted != no_imgs:
        print(f"WARNING: there are {no_imgs} images present but only {no_rows_extracted} images were found in the existing csv file." )
        print("The follwing images present in the data folder were not found in CSV file, please check images:")
        print(not_present)    
    
    end_time = time.time()
    print(f'Total runtime: {end_time-start_time}s')
    
if __name__ == '__main__':
    main()
    
    
    
# CHECKS
# need to make sure the images actually go into given point cloud
# can we automate this?
# e.g search a LAS and find the min and max gps time - if the image doesnt fall within this then we discard?
# for now assume a;l; images concide in the LAS


