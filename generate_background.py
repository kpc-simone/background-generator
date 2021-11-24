# histogram_play.py

import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter.filedialog import askopenfilename
from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.morphology import disk, binary_erosion, binary_opening, binary_closing, closing
from skimage.segmentation import flood_fill
from skimage.measure import label, regionprops, regionprops_table
from matplotlib.gridspec import GridSpec
from PIL import Image
import scipy.ndimage as nd
import math
import sys,os
import time

# todo: make into a function
# pass 'roi', 'plot' as an argument

def getROI(frame,roi):
    
    xmin = roi['xmin']
    ymin = roi['ymin']
    width = roi['width']
    height = roi['height']
    
    roi = frame[ymin:ymin+height,xmin:xmin+width]
    
    return roi

if __name__ == '__main__':
    args = sys.argv
    
if '--plot' in args:
    plot = True
else:
    plot = False

videopath = askopenfilename(
    filetypes=[('Video Files', '*.avi'), ('All Files', '*.*')]
    )
videofilename = videopath.split('/')[-1]
print(videofilename)

vidcap = cv2.VideoCapture(videopath)

# generate a background frame
pos_0 = 300
FPS = int(vidcap.get(cv2.CAP_PROP_FPS))
vidcap.set(cv2.CAP_PROP_POS_FRAMES,int(pos_0*FPS))
success,bg_frame = vidcap.read()

cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL) 
box_s = cv2.selectROIs("ROI Selection", bg_frame, fromCenter=False)
((xmin,ymin,width,height),) = tuple(map(tuple, box_s))
roi = {
    'xmin'      : xmin,
    'ymin'      : ymin,
    'width'     : width,
    'height'    : height,
}
cv2.destroyAllWindows()

# compute algorithm steps
prev_frame = np.float32(getROI(bg_frame,roi))
wa = 0.01 * prev_frame
wa_mov = prev_frame

t_idx = 0

model_built = False
t_start = time.time()

ts = [0]
ws = [0]
dfms = [127]
while vidcap.isOpened():
    success,frame = vidcap.read()

    if frame is None:
        break
    
    frame = np.float32(getROI(frame,roi))
    change = np.absolute(frame - wa_mov)
    cv2.accumulateWeighted(frame,wa_mov,alpha = 0.001 )
    
    weighting_factor = 0.01 * ( change.sum().sum().sum() / ( width * height * 765) )
    cv2.accumulateWeighted(frame,wa,alpha = weighting_factor)
    
    diff = wa_mov - wa
    diff_metric = diff.mean().mean().mean()
    
    ts.append(ts[-1] + 1/FPS)
    ws.append(weighting_factor)
    dfms.append(diff_metric)
    
    #print(ts)
    #print(ws)
    
    if abs(diff_metric) < 5.0:
        bg_model = wa
        model_built = True
        
        t_built = vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000
        print('built background model in {} s of video'.format(t_built - pos_0))
        
        fig, ax = plt.subplots(1,1)
        ax.imshow(cv2.convertScaleAbs(bg_model))
        ax.set_title('Background model generated:')
        ax.set_axis_off()
        plt.show()
        
        cv2.imwrite('output/{}_bg.png'.format(videofilename),bg_model)
        break
        
        if plot:
            plot_dict['plot_bg'] = ax_bg.imshow(bg_model)

    if plot:

        fr_disp = cv2.convertScaleAbs(frame)    
        ma_disp = cv2.convertScaleAbs(wa_mov)    
        ch_disp = cv2.convertScaleAbs(change)    
        wa_disp = cv2.convertScaleAbs(wa)    
        df_disp = cv2.convertScaleAbs(diff)    
        
        if t_idx == 0:
            
            # initialize figure
            fig = plt.figure( figsize = (16,6) )
            gs = GridSpec(3,4,figure=fig)
            ax_fr = fig.add_subplot(gs[0,0])
            ax_ma = fig.add_subplot(gs[1,0])
            ax_ch = fig.add_subplot(gs[2,0])
            ax_wa = fig.add_subplot(gs[0,2])
            ax_df = fig.add_subplot(gs[0,3])
            
            ax_wt = fig.add_subplot(gs[1,1:])
            ax_dfm = fig.add_subplot(gs[2,1:])
            
            ax_fr.set_title('Current Frame')
            ax_ma.set_title('Moving Average')
            ax_ch.set_title('Change from Moving Average')
            ax_wa.set_title('Background Model in Progress')
            ax_df.set_title('Current Frame less Background')

            for ax in (ax_fr,ax_ma,ax_ch,ax_wa,ax_df):
                ax.set_axis_off()
                
            for ax in (ax_wt,ax_dfm):
                ax.spines['top'].set_visible(False)    
                ax.spines['right'].set_visible(False)    
                ax.set_xlim(0,150)
                
            ax_wt.set_ylabel('weighting factor')
            ax_wt.set_ylim(0.002,0.004)
            ax_dfm.set_ylabel('difference metric')
            ax_dfm.set_ylim(1,255)
            ax_dfm.set_yscale('log')
            ax_dfm.set_xlabel('time (s, of video)')
            # plot the data
            plot_fr  = ax_fr.imshow(fr_disp)
            plot_ma  = ax_ma.imshow(ma_disp)
            plot_ch  = ax_ch.imshow(ch_disp)
            plot_wa  = ax_wa.imshow(wa_disp)
            plot_df  = ax_df.imshow(df_disp)
            
            plot_wt,  = ax_wt.plot(ts,ws,color='firebrick')
            plot_dfm,  = ax_dfm.plot(ts,dfms,color='steelblue')
            
            plot_dict = {
                'plot_fr'           : plot_fr,
                'plot_ma'           : plot_ma,
                'plot_ch'           : plot_ch,
                'plot_wa'           : plot_wa,
                'plot_df'           : plot_df,
                
                'plot_wt'           : plot_wt,
                'plot_dfm'           : plot_dfm,
            }
            
        else:
            plot_dict['plot_fr'].set_data(fr_disp)
            plot_dict['plot_ma'].set_data(ma_disp)
            plot_dict['plot_ch'].set_data(ch_disp)
            plot_dict['plot_wa'].set_data(wa_disp)
            plot_dict['plot_df'].set_data(df_disp)
            
            plot_dict['plot_wt'].set_data(ts,ws)
            plot_dict['plot_dfm'].set_data(ts,dfms)
            
            if model_built:
                plot_dict['plot_bg'].set_data(bg_model)
        
        fig.tight_layout()
        plt.draw()
        plt.pause(1e-17)
        t_idx += 1













