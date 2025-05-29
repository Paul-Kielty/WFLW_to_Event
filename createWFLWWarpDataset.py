import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
from v2e import EventEmulator
from utils.sim_utils import *
import os
from tqdm import tqdm
from PIL import Image
import cv2 
from glob import glob
from utils.planar_motion_stream import CameraPoseGenerator
import matplotlib.animation as animation


def generate_WFLW_warp_dataset():

    ''' FILE PATH OPTIONS'''
    image_dir = "WFLW_images"
    annot_paths = "WFLW_annotations\\list_98pt_rect_attr_train_test\\list_98pt_rect_attr_test.txt"
    annotations = open(annot_paths).readlines()

    save_dir = "WFLW_events_33ms_warp"
    log_file = os.path.join(save_dir,"log.txt")
        
    ''' RGB MOTION SIMULATION OPTIONS'''
    base_crop_size = 256 # Determines the base size of the crop that moves over the full image
    output_size = base_crop_size # size of the output images. If different to base_crop_size, the images will be resized to this after warping

    scale_crop_above_face = 1.2 # if base crop size is smaller than the bbox, scale the crop size by this factor above the face bounding box
    min_face_pad_in_crop = 30 # Enforce a minimum size difference between the face bounding box and the output frame size (crop size),
                              # to ensure that the face has space to move around in the crop
    pause_probability = 0.5 # probability that the sequence contains a pause that will last around 10% of the frames. Few events, if any, will be generated over the pause
    
    min_face = 32 # minimum face size in pixels to consider the image for warping (to prevent too small faces)
    max_face = None # maximum face size in pixels to consider the image for warping
    # max_skew = 0 # TODO: (Not working correctly) Maximum skew allowed in the homography matrix (to prevent too much perspective distortion)
    show_warp_animation = False # if True, shows an animation of the warped RGB frames during generation
    rgb_event_fps_ratio = 5 # ratio of the number of RGB frames to the number of event frames (e.g. 5 means 5 RGB frames for every 1 event frame, replaces need for slomo)
    min_speed = 1e-4
    max_speed = 3e-2
    
    ''' EVENT SIMULATION OPTIONS '''
    num_frames = 15 # total number of event frames to generate (requires generating num_frames+1 RGB frames)
    fps = 30
    save_windows = True # if True, saves individual event windows as separate .npy files (based on the fps). If False, saves a single .npy file with all events
    clip_val = 8 # Clip value for visualised event output MP4
    
    v2e_params_preset = None # "noisy", "clean" or None if using custom parameters below:
    pos_thres=0.2
    neg_thres=0.2
    sigma_thres=0.03
    cutoff_hz= 15 # default 0
    leak_rate_hz=0.1
    refractory_period_s=0
    shot_noise_rate_hz=0.5 # default = 0 # rate in hz of temporal noise events
    use_slomo = False

    landmarks = []
    names = []
    bboxes = []
    attrs = []
    image_paths = []

    for i in range(len(annotations)):
    # for i in range(50):
        label = annotations[i].split(" ")
        landmark = [float(l) for l in label[0:196]]
        bbox = [float(l) for l in label[196:200]]
        attr = [int(l) for l in label[200:206]]
        name = label[-1].split("\n")[0]
        image = image_dir + "\\" + name
        image = image.replace("/", "\\")
        image_paths.append(image)
        landmarks.append(landmark)
        names.append(name)
        bboxes.append(bbox)
        attrs.append(attr)

    # already_completed = glob(save_dir + "\\*\\*event_video.avi")
    # already_completed_indices = [int(i.split("\\")[-2]) for i in already_completed]
    # start_i = max(already_completed_indices) + 1 if len(already_completed_indices) > 0 else 0
    
    simulated_indices = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                index = int(line.split("\n")[0])
                simulated_indices.append(index)

    for index in tqdm(range(0,len(image_paths),1)):
        if index in simulated_indices:
            continue
        
        ''' Loading images, landmarks'''
        save_path = save_dir + f"\\{index}"
        # if index in already_completed_indices:
        #     print("Deleting: ", save_path)
        #     if os.path.exists(save_path+"\\event_video.avi"):
        #         print("**** but found event video?")
        #         exit()
        #     del_files = glob(save_path+"\\*")
        #     for file in del_files:
        #         os.remove(file)
        #     os.rmdir(save_path)
            
        image = image_paths[index]
        landmark = np.array(landmarks[index]) 
        landmark = np.reshape(landmark, (98, 2))
        attr = attrs[index]
        img = plt.imread(image)
        
        print(f"\n\nImage {index}, occlusion = {attr[4]}, blur = {attr[5]}: ({image_paths[index]})")
        
        ''' Check if face size if big enough, else break '''
        height, width, _ = img.shape
        # if height <= (base_crop_size + min_face_pad_in_crop*2) or width <= (base_crop_size + min_face_pad_in_crop*2):
        #     print("\tImage too small, skipping.")
        #     continue
        x1 = (min(landmark[:,0])) 
        y1 = (min(landmark[:,1]))
        x2 = (max(landmark[:,0])) 
        y2 = (max(landmark[:,1]))
        w = x2-x1
        h = y2-y1
        center = (x1 + (x2-x1)/2, y1 + (y2-y1)/2)
        if w < min_face or h < min_face:
            print("\tFace too small, skipping.")
            continue
        if max_face is not None:
            if w >= max_face or h >= max_face:
                print("\tFace too large, skipping.")
                continue
        if w >= base_crop_size or h >= base_crop_size:
            crop_size = int(max(w,h) * scale_crop_above_face)
            print("\tFace bigger than base crop, crop size scaled to:", crop_size)

        else:
            crop_size = base_crop_size
    
        
        warp_output = run_camera_warping(height, width, num_frames, img, landmark, crop_size, 
                                         min_face_pad_in_crop=min_face_pad_in_crop, 
                                        #  max_skew=max_skew, 
                                         min_speed=min_speed, max_speed=max_speed,
                                         pause_probability=pause_probability,
                                         show_warp_animation=show_warp_animation,
                                         rgb_event_fps_ratio = rgb_event_fps_ratio,
                                        )
        if warp_output is None:
            continue
        new_images, new_landmarks, new_bboxes, crop_size = warp_output

        if output_size != crop_size or output_size != crop_size: #resize images
            # Crop and resize images
            resized_images = []
            resized_landmarks = []
            resized_bboxes = []

            for i,img in enumerate(new_images):
                img = Image.fromarray(img)
                img = img.resize((output_size, output_size))
                resized_images.append(np.array(img))

                if i%rgb_event_fps_ratio == 0:

                    resized_landmarks.append(new_landmarks[i//rgb_event_fps_ratio] * (output_size/crop_size))#= [landmark * (256/size) for landmark in landmarks[i]]
                    resized_bboxes.append(new_bboxes[i//rgb_event_fps_ratio] * (output_size/crop_size))
                
            new_images = resized_images
            new_landmarks = resized_landmarks
            new_bboxes = resized_bboxes
        
        device = torch.device("cuda:0")
        emulator = EventEmulator(
                    pos_thres=pos_thres,
                    neg_thres=neg_thres,
                    sigma_thres=sigma_thres,
                    cutoff_hz=cutoff_hz,
                    leak_rate_hz=leak_rate_hz,
                    shot_noise_rate_hz=shot_noise_rate_hz,
                    refractory_period_s=refractory_period_s,
                    seed=0,
                    output_folder=None, dvs_h5=None, dvs_aedat2=None,
                    dvs_text=None, device=device)
        if v2e_params_preset is not None:
            emulator.set_dvs_params(v2e_params_preset)

        os.makedirs(save_path, exist_ok=True)
        
        timestamps = np.array([(1/fps)*(t) for t in range(len(new_bboxes))])
        
        # drop t=0 as there will be no events until second frame:
        new_landmarks = new_landmarks[1:]
        new_bboxes = new_bboxes[1:]
        timestamps = timestamps[1:]

        if save_windows:
            sim_window_iter = EventSimulatorWindowIterator(new_images.copy(), emulator, fps=fps*rgb_event_fps_ratio)
            # events = simulateEventsSaveWindows(new_images, emulator, save_path, fps=fps)
            
            for i, timestamp in enumerate(timestamps):
                output_window = []
                for j in range(rgb_event_fps_ratio):
                    new_events = next(sim_window_iter)
                    output_window.append(new_events)
                output_window = np.concatenate(output_window, axis=0)
                np.save(save_path + f"\\{str(i).zfill(3)}_events.npy", new_events.astype(np.float32))
        else:
            events = simulateEventsV2e(new_images, emulator, fps=fps, use_slomo=use_slomo)
            np.save(save_path + "\\events.npy", events) # Save events to numpy
            
        emulator.cleanup()
        
        ''' Saving Data '''
        label = {
                "bboxes": new_bboxes,
                "landmarks": new_landmarks,
                "pose": attr[0],
                "expression": attr[1],
                "illumination": attr[2],
                "make-up": attr[3],
                "occlusion": attr[4],
                "blur": attr[5]
            }
        
        # np.savetxt(save_path + "\\events.txt", events) # Save NIR timestamp of this sequence 
        np.savetxt(save_path + "\\timestamps.txt", timestamps) # Save NIR timestamp of this sequence 
        np.save(save_path + "\\labels.npy", label)

        ''' Saving RGB Example '''
        fig, ax = plt.subplots(figsize=(6,6))
        plot = plt.imshow(new_images[-1])
        plt.scatter(new_landmarks[-1][:,0], new_landmarks[-1][:,1], s=4, c="red")
        plt.axis('off')

        x1,y1,w,h = new_bboxes[-1]
        rect = patches.Rectangle((x1,y1), w, h, linewidth=3, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        plt.savefig(save_path + "\\rgb.png")
        plt.close()
        ''' Saving Events Example '''    
        if save_windows:
            skip = False
            for i in range(num_frames):
                if not os.path.exists(save_path+f"\\{str(i).zfill(3)}_events.npy"):
                    print("Error, file not found:",save_path+'\\'+f'{str(i).zfill(3)}_events.npy')
                    skip = True
            if skip:
                # del_files = glob(save_path+"\\*")
                # for file in del_files:
                #     os.remove(file)
                # os.rmdir(save_path)
                continue      
                    
            # leaky_model = LeakyIntegrator(0, output_size, output_size, device) # 0 decay for basic voxel grid, otherwise timesurface
            out_path = save_path+"\\event_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, int(fps/4), (output_size,output_size))
            
            for i in range(len(new_landmarks)):
                event = np.load(save_path+f"\\{str(i).zfill(3)}_events.npy")
                # grid = leaky_model.integrateEvents(event)
                grid = voxel(event, height=output_size, width=output_size, device=device, esim=False)
                img = grid.clamp(-clip_val,clip_val)
                img = ((img[0,:,:]+clip_val)/(clip_val*2))#.unsqueeze(0)
                img = img.detach().cpu().numpy() #img.squeeze(0).detach().cpu().numpy()
                img = np.stack((img,img,img), axis=2)
                img = (img*255).astype(np.uint8)
                
                x1, y1, w, h = new_bboxes[i]
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (255,0,0), 1)
                for x,y in new_landmarks[i]:
                    img = cv2.circle(img, (int(x), int(y)), 2, (0,0,255), -1)
                    
                out.write(img)
            out.release()
            plt.close('all')

        else:
            ev_st = np.searchsorted(events[:,0], timestamps[-3], side="left") # viz last voxel
            ev_en = np.searchsorted(events[:,0], timestamps[-2], side="left") # viz last voxel
            event = events[ev_st:ev_en] 
            landmark = new_landmarks[-3]
            grid = voxel(event, height=output_size, width=output_size, device=device)
            img = grid.clamp(-clip_val,clip_val)
            # img = (img - torch.min(img)) / (torch.max(img)-torch.min(img))
            img = ((img[0,:,:]+clip_val)/(clip_val*2))
            fig, ax = plt.subplots(figsize=(8,8))
            plt.axis('off')
            img = img.squeeze(0).detach().cpu().numpy()
            plot = plt.imshow(img, cmap='gray')
            plt.scatter(landmark[:,0], landmark[:,1], s=4, c="red")
            plt.savefig(save_path + f"/voxel.png")
            plt.close('all')
            
        with open(log_file, "a") as f:
            f.write(f"{index}\n")
        

def voxel(events, height, width, device, esim=False, polarity_mapping=(-1, 1)):
    events_torch = torch.from_numpy(events.astype(np.float32)).to(device)
    voxel_grid = torch.zeros(1, height, width, dtype=torch.float32, device=device).flatten()

    if esim:
        xs = events_torch[:, 0].long()
        ys = events_torch[:, 1].long()
    else:
        xs = events_torch[:, 1].long()
        ys = events_torch[:, 2].long()
    pols = events_torch[:, 3].float()
    pols[pols == 0] = polarity_mapping[0]
    index1 = (xs + ys*width).long()
    voxel_grid.index_add_(dim=0, index=index1, source=pols) 
    voxel_grid = voxel_grid.view(1, height, width)

    return voxel_grid

def gen_new_bbox(landmarks, return_dims=False):
    x1 = (min(landmarks[:,0])) 
    y1 = (min(landmarks[:,1]))
    x2 = (max(landmarks[:,0])) 
    y2 = (max(landmarks[:,1]))
    w = x2-x1
    h = y2-y1
    if return_dims:
        return x1, y1, w, h
    return x1, y1, x2, y2

def warp_point(point, H):
    x,y=point
    new_x = (H[0,0]*x + H[0,1]*y + H[0,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
    new_y = (H[1,0]*x + H[1,1]*y + H[1,2]) / (H[2,0]*x + H[2,1]*y + H[2,2])
    return (new_x, new_y)

def limit_skew(H, max_skew=0.1):
    # QR decomposition of the homography matrix
    Q, R = np.linalg.qr(H)
    
    # Limiting the skew in the R matrix
    R[0, 1] = np.clip(R[0, 1], -max_skew, max_skew)
    R[1, 0] = np.clip(R[1, 0], -max_skew, max_skew)
    
    # Recompose the matrix
    H_limited_skew = np.dot(Q, R)
    return H_limited_skew

def run_camera_warping(height, width, num_frames, img, landmark, crop_size, 
                       min_speed = 1e-3, max_speed = 1e-1,
                       min_face_pad_in_crop=30, max_skew=0.1, show_warp_animation=False, 
                       retry_limit=20, pause_probability=0, rgb_event_fps_ratio=1):
    retry_count = 0
    retries_exceeded = False
    face_within_bounds = False
    while not face_within_bounds and not retries_exceeded:
        if retry_count >= retry_limit:
            print(f"Couldn't generate warps with all crops within image bounds (after {retry_count} retries) - Skipping this image.")
            retries_exceeded = True
            continue
        retry_count += 1
        
        new_landmarks = []
        new_bboxes = []
        new_images = []
        
        camera = CameraPoseGenerator(height, width, pause_probability=pause_probability, 
                                     max_frames=(num_frames+1)*rgb_event_fps_ratio, min_speed=min_speed/rgb_event_fps_ratio, 
                                     max_speed=max_speed/rgb_event_fps_ratio, max_interp_consecutive_frames=2)
        landmark_warped = np.zeros_like(landmark)

        while len(new_images) < num_frames*rgb_event_fps_ratio+1:
            
            H,ts = camera()
            # H = limit_skew(H, max_skew=max_skew) # Needs revisiting
            warp = cv2.warpPerspective(img, H, (width, height), borderMode=cv2.BORDER_REFLECT)
            
            if len(new_images)%rgb_event_fps_ratio == 0:
                for j,point in enumerate(landmark):
                    new_x, new_y = warp_point(point, H)
                    landmark_warped[j,0] = new_x
                    landmark_warped[j,1] = new_y
                    
                new_landmarks.append(landmark_warped.copy())
                
                x1, y1, w, h = gen_new_bbox(landmark_warped, return_dims=True)
                new_bboxes.append(np.array([x1,y1,w,h]))

            new_images.append(warp)
        
        
        new_bboxes = np.array(new_bboxes)
        
        max_x1, max_y1, max_x2, max_y2 = np.min(new_bboxes[:,0]), np.min(new_bboxes[:,1]), np.max(new_bboxes[:,0]+new_bboxes[:,2]), np.max(new_bboxes[:,1]+new_bboxes[:,3])
        max_bbox_w, max_bbox_h = max_x2-max_x1, max_y2-max_y1
        crop_centre = (max_x1 + max_bbox_w/2, max_y1 + max_bbox_h/2)
        new_crop_size = crop_size

        # Check if the new bboxes are within the set crop by min amount
        if max_bbox_w + min_face_pad_in_crop >= crop_size or max_bbox_h + min_face_pad_in_crop >= crop_size:
            # not within crop, increase crop size
            new_crop_size = int(max(max_bbox_w, max_bbox_h)+min_face_pad_in_crop)
            # print("\tBbox moves too much for current crop, increasing crop size to", new_crop_size)
            
        new_crop_top_left = (int(crop_centre[0]-new_crop_size/2), int(crop_centre[1]-new_crop_size/2))
        crop_corners = np.array([new_crop_top_left[0], new_crop_top_left[1], new_crop_top_left[0]+new_crop_size, new_crop_top_left[1]+new_crop_size], dtype=int)

        # Check if the crop is within the image bounds
        face_within_bounds = True
        if crop_corners[0] < 0 or crop_corners[1] < 0 or crop_corners[2] >= width or crop_corners[3] >= height:
            # print("\tCrop out of bounds, regenerating warp...")
            face_within_bounds = False
        
    if retries_exceeded:
        return None
    
    if show_warp_animation:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8,4))
        fig.tight_layout(pad=1.2)
        ani_frames = []
        for new_image, new_landmark, new_bbox in zip(new_images[::rgb_event_fps_ratio], new_landmarks, new_bboxes):
            warp_ani = new_image.copy()
            x1, y1, w, h = new_bbox
            warp_ani = cv2.rectangle(warp_ani, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (255,0,0), 2)
            warp_ani = cv2.rectangle(warp_ani, (crop_corners[0], crop_corners[1]), (crop_corners[2], crop_corners[3]), (0,255,0), 2)

            for x,y in new_landmark:
                warp_ani = cv2.circle(warp_ani, (int(x), int(y)), 3, (0,0,255), -1)
            ani_frame = ax.imshow(warp_ani)
            ani_frames.append([ani_frame])
        ani = animation.ArtistAnimation(fig, ani_frames, interval=66, blit=True)
        plt.show()
        
    new_landmarks = np.array(new_landmarks)
    new_landmarks = new_landmarks - crop_corners[0:2]
    new_bboxes = new_bboxes - np.array([crop_corners[0], crop_corners[1], 0, 0])
    new_images = [new_image[crop_corners[1]:crop_corners[3], crop_corners[0]:crop_corners[2]] for new_image in new_images]
    return new_images, new_landmarks, new_bboxes, new_crop_size

if __name__ == "__main__":
    generate_WFLW_warp_dataset()