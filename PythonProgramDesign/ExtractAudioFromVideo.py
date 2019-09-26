#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: ExtractAudioFromVideo.py
@Time: 2019/8/26 上午10:36
@Overview:
"""
import ffmpeg
import moviepy.editor as mp
import os


filepath = '/Volumes/Document/Others/BBC.The.Story.of.the.Jews.720p.HDTV.x264.AAC-MVGroup/'
filenames= ['BBC.The.Story.of.the.Jews.1of5.In.the.Beginning.720p.HDTV.x264.AAC.MVGroup.org.mkv',	'BBC.The.Story.of.the.Jews.4of5.Over.the.Rainbow.720p.HDTV.x264.AAC.MVGroup.org.mkv',
'BBC.The.Story.of.the.Jews.2of5.Among.Believers.720p.HDTV.x264.AAC.MVGroup.org.mkv',	'BBC.The.Story.of.the.Jews.5of5.Return.720p.HDTV.x264.AAC.MVGroup.org.mkv',
'BBC.The.Story.of.the.Jews.3of5.A.Leap.of.Faith.720p.HDTV.x264.AAC.MVGroup.org.mkv']
destination = '/Users/yang/Desktop/BBC.The.Story.of.the.Jews.720p.HDTV.x264.AAC-MVGroup'
# stream = ffmpeg.input(filename)


def ExtractAudioFromVideo(file_path, file_type):
    audio_types = ['wav', 'mp3']

    if not os.path.exists(file_path):
        raise ValueError('File doesn\'t exist!')

    audio = file_path.replace('.mkv', '.'+file_type)

    clip = mp.VideoFileClip(file_path)
    clip.audio.write_audiofile(audio)




ind = 0
for filename in filenames:
    file = os.path.join(filepath, filename)
    if not os.path.exists(file):
        raise ValueError('File doesn\'t exist!')
    clip = mp.VideoFileClip(file)
    if not os.path.exists(destination):
        os.makedirs(destination)

    audio = filename.replace('.mkv', '.mp3')

    clip.audio.write_audiofile(os.path.join(destination, audio))
    ind+=1
    print('\r\33Finished {}/{} of Video extraction.'.format(ind, len(filenames)))

