# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 12:20:10 2022

@author: rusty
"""

import subprocess, os

out='cimarron_v1.mp4'

def mkanim(out,frame_path,fps=16,ffmpeg='ffmpeg',overwrite=True):
    """
    out: path to output mp4 file
    frame_path: pattern for image paths, including %04d or similar for
    frame number.
    """
    if os.path.exists(out):
        if overwrite:
            os.unlink(out)
        else:
            raise Exception(f"{out} exists")
                
    res=subprocess.run(['ffmpeg',
                        '-framerate',str(fps),
                        '-i',frame_path,
                        # this deals with odd dimensions
                        '-vf',"crop=trunc(iw/2)*2:trunc(ih/2)*2", 
                        '-c:v','libx264',
                        '-preset','slow',
                        '-profile:v','high',
                        '-level:v','4.0',
                        '-pix_fmt','yuv420p',
                        '-crf','20','-r',str(fps),
                        out], capture_output=True)
    print(res.stderr.decode())
    