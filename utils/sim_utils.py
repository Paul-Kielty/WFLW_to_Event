import numpy as np
from tempfile import TemporaryDirectory
import cv2
from glob import glob
import os
from tqdm import tqdm
# import torch
# import tonic#.transforms import Denoise
from v2e import SuperSloMo 

upsampling_factor = 5
slomo_model = "C:\\Users\\pkielty\\Documents\\v2e\\input\\SuperSloMo39.ckpt"
slomo = SuperSloMo(
            model=slomo_model, upsampling_factor=upsampling_factor, auto_upsample=False,
            video_path=None, vid_orig=None, vid_slomo=None,
            preview=False, batch_size=1)

def simulateEventsV2e(images, emulator, use_slomo=False, fps=30, denoise_time=0, slomo_factor=None, timestamp_resolution=None):
    total_events = np.empty((0, 4), float)
    timestamps = np.array([(1/fps)*(t) for t in range(len(images))])
    
    if use_slomo:
        if slomo_factor is None and timestamp_resolution is not None:
            # calculate slomo_factor
            slomo_factor = int(np.ceil((1/fps)/timestamp_resolution))
            slomo = SuperSloMo(model=slomo_model, upsampling_factor=slomo_factor, auto_upsample=False,
                video_path=None, vid_orig=None, vid_slomo=None,
                preview=False, batch_size=1)
            
        elif slomo_factor is not None:
            # use given slomo_factor
            slomo = SuperSloMo(model=slomo_model, upsampling_factor=slomo_factor, auto_upsample=False,
                    video_path=None, vid_orig=None, vid_slomo=None,
                    preview=False, batch_size=1)
        else:    
            # use auto_upsample, max 20x slowdown
            slomo = SuperSloMo(model=slomo_model, upsampling_factor=20, auto_upsample=True, video_path=None, vid_orig=None, vid_slomo=None, preview=False)

        # max_img, min_img = 1023, 64 # NIR range
        batchFrames = []
        ts0 = 0
        ts1 = 1. / fps  # timestamps of src frames

        # for i in range(len(images)):
        for i in tqdm(range(len(images))): 
            image = images[i]
            inputHeight, inputWidth = image.shape[0], image.shape[1]

            inputVideoFrame = np.asarray(image)
            batchFrames.append(inputVideoFrame)
            if len(batchFrames) < 2: 
                continue  # need at least 2 frames

            ''' Slomo frame interpolation '''
            output_width = inputWidth
            output_height = inputHeight
            srcNumFramesToBeProccessed = 2
            with TemporaryDirectory() as source_frames_dir:
                for i in range(srcNumFramesToBeProccessed):
                    inputVideoFrame = batchFrames[i]
                    if len(inputVideoFrame.shape) > 2:
                        inputVideoFrame = cv2.cvtColor(inputVideoFrame, cv2.COLOR_BGR2GRAY)  # much faster
                    save_path = os.path.join(source_frames_dir, str(i).zfill(8) + ".npy") # save frame into numpy records
                    np.save(save_path, inputVideoFrame)
                with TemporaryDirectory() as interpFramesFolder:
                    interpTimes, avgUpsamplingFactor = slomo.interpolate(source_frames_dir, interpFramesFolder, (output_width, output_height))
                    # read back to memory
                    interpFramesFilenames = glob(interpFramesFolder + "\\*.png")
                    interpFramesFilenames = sorted(interpFramesFilenames, key = lambda x: int(x.split("\\")[-1].split(".")[0]))
                    interpFrames = [cv2.imread(i) for i in interpFramesFilenames]


            ''' Simulate Events '''
            n = len(interpFrames)
            events = np.empty((0, 4), float)
            interpTimes = np.linspace(start=ts0, stop=ts1, num=n+1, endpoint=False)
            if n == 1:  # no slowdown
                fr = interpFrames[0]
                new_events = emulator.generate_events(fr, ts0)
            else:
                for j in range(n):  # for each interpolated frame
                    fr = interpFrames[j]
                    new_events = emulator.generate_events(fr, interpTimes[j])
                    if new_events is not None and new_events.shape[0] > 0:
                        events = np.append(events, new_events, axis=0)
        
        
            events = np.array(events)  # remove first None element
            total_events = np.append(total_events, events, axis=0)
            ts0 = ts1
            ts1 += 1. / fps
            batchFrames = [inputVideoFrame]  # save last frame of input as 1st frame of new batch
        slomo.cleanup()
        return total_events
    else:
        # events = np.empty((0, 4), float)
        frame_events = []
        for i,image in enumerate(images):
            
            inputVideoFrame = np.asarray(image*255).astype(np.uint8)
            new_events = emulator.generate_events(inputVideoFrame, timestamps[i])
            
            if new_events is not None and new_events.shape[0] > 0:
                frame_events.append(new_events)
                # new_events = np.array(new_events)  # remove first None element
                # total_events = np.append(total_events, new_events, axis=0)
        total_events = np.concatenate(frame_events)
        # if denoise_time > 0:
        #     event_dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", np.bool_), ("t", np.int64)])
        #     # total_events = np.concatenate((total_events[:, 1:3], (total_events[:, 3]==True).reshape(-1, 1), (total_events[:, 0]*1e6).reshape(-1, 1)), axis=1)
        #     events_tonic = np.empty((len(total_events),), dtype=event_dtype)
        #     events_tonic["x"] = total_events[:, 1].astype(np.int16)
        #     events_tonic["y"] = total_events[:, 2].astype(np.int16)
        #     events_tonic["p"] = (total_events[:, 3]==True).astype(np.bool_)
        #     events_tonic["t"] = (total_events[:, 0]*1e6).astype(np.int64)
        #     # total_events = [(x,y,p,t) for x,y,p,t in total_events]
        #     # total_events = total_events.astype(event_dtype)
            
        #     # total_events.astype([('x', '<i2'), ('y', '<i2'), ('p', '?'), ('t', '<i8')])
        #     denoise_transform = tonic.transforms.Denoise(filter_time=int(denoise_time*1e6))
        #     events_tonic_denoise = denoise_transform(events_tonic)
        #     total_events_denoise = np.concatenate((events_tonic_denoise["t"].reshape(-1, 1)/1e6, 
        #                                            events_tonic_denoise["x"].reshape(-1, 1), 
        #                                            events_tonic_denoise["y"].reshape(-1, 1), 
        #                                            (events_tonic_denoise["p"]*2-1).reshape(-1, 1)), 
        #                                           axis=1, dtype=np.float32)
        #     # total_events_denoise = np.array([(t,x,y,p*2-1) for (x,y,p,t) in events_tonic_denoise],dtype=np.float32)          
        #     return total_events_denoise
        
        print("Num events:",len(total_events))
        return total_events
    
    
def simulateEventsSaveWindows(imgs, emulator, out_dir, fps=30):
    
    timestamps = np.array([(1/fps)*(t) for t in range(len(imgs))])

    for i in range(len(imgs)):
        
        image = imgs[i]
        # inputVideoFrame = np.asarray(image)
        # batchFrames.append(inputVideoFrame)
       
        newEvents = emulator.generate_events(image, timestamps[i])
        
        if newEvents is not None and newEvents.shape[0] > 0:
            np.save(out_dir+f"\\{str(i-1).zfill(3)}_events.npy", newEvents.astype(np.float32))
        else:
            if i != 0:
                print("No events generated")
    return newEvents

    
class EventSimulatorWindowIterator():
    
    def __init__(self, imgs, emulator, out_dir=None, fps=30):
    
        self.imgs = imgs
        self.emulator = emulator
        self.timestamps = [(1/fps)*(t) for t in range(len(imgs))]
        self.out_dir = out_dir
        self.i = 0

        # initialize with the first image
        self.emulator.generate_events(self.imgs.pop(0), self.timestamps.pop(0))

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.imgs) == 0:
            raise StopIteration
        img = self.imgs.pop(0)
        timestamp = self.timestamps.pop(0)
        newEvents = self.emulator.generate_events(img, timestamp)
        
        if newEvents is not None and self.out_dir is not None:
            if not newEvents.shape[0] > 0:
                print("No events generated")
            print(f"Saving events: {self.out_dir}\\{str(self.i).zfill(3)}_events.npy")
            np.save(self.out_dir+f"\\{str(self.i).zfill(3)}_events.npy", newEvents.astype(np.float32))
        if newEvents is None:
            newEvents = np.empty((0, 4), float)
        return newEvents