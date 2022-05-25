from __future__ import division
from __future__ import print_function
#####################################################################
# -*- coding: iso-8859-1 -*-                                        #
#                                                                   #
# Frets on Fire                                                     #
# Copyright (C) 2006 Sami Kyöstilä                                  #
#               2008 myfingershurt                                  #
#                                                                   #
# This program is free software; you can redistribute it and/or     #
# modify it under the terms of the GNU General Public License       #
# as published by the Free Software Foundation; either version 2    #
# of the License, or (at your option) any later version.            #
#                                                                   #
# This program is distributed in the hope that it will be useful,   #
# but WITHOUT ANY WARRANTY; without even the implied warranty of    #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the     #
# GNU General Public License for more details.                      #
#                                                                   #
# You should have received a copy of the GNU General Public License #
# along with this program; if not, write to the Free Software       #
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,        #
# MA  02110-1301, USA.                                              #
#####################################################################

from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import pygame
MusicFinished = pygame.USEREVENT
import Log
import time
import struct
from Task import Task
import Config
import glob
import pdb
# from scipy import signal
import numpy as np
try:
  from scikits.samplerate import resample
  resampleOk = True
except ImportError:
    Log.warn("resample not supported for small sounds")
    resampleOk = False

#stump: check for pitch bending support
try:
  import pygame.pitchbend
  pitchBendSupported = True
  if not hasattr(pygame.pitchbend, 'ALL'):
    Log.warn("Your pitchbend module is too old; upgrade it to r7 or higher for pitch bending to work.")
    pitchBendSupported = False
except ImportError:
  Log.warn("Pitch bending is not supported; install john.stumpo's pitchbend module (r7 or higher) if you want it.")
  pitchBendSupported = False

#stump: get around some strangeness in pygame when py2exe'd...
if not hasattr(pygame.mixer, 'music'):
  import sys
  __import__('pygame.mixer_music')
  pygame.mixer.music = sys.modules['pygame.mixer_music']

try:
  import pyogg as ogg
  # pyogg is not finished, this vorbis interface should be merged !
  from pyogg import vorbis
  ogg.pyoggSetStreamBufferSize(4096)
except ImportError:
  Log.warn("PyOGG not found. OGG files will be fully decoded prior to playing; expect absurd memory usage.")
  ogg = None

class Audio(object):
  def pre_open(self, frequency = 22050, bits = 16, stereo = True, bufferSize = 1024):
    pygame.mixer.pre_init(frequency, -bits, stereo and 2 or 1, bufferSize)
    return True

  def open(self, frequency = 22050, bits = 16, stereo = True, bufferSize = 1024):
    try:
      pygame.mixer.quit()
    except:
      pass

    try:
      # for some reason pygame.mixer.init really expects a minus bits number -16 here... !
      pygame.mixer.init(frequency, -bits, stereo and 2 or 1, bufferSize)
    except:
      Log.warn("Audio setup failed. Trying with default configuration.")
      pygame.mixer.init()

    Log.debug("Audio configuration: %s" % str(pygame.mixer.get_init()))

    #myfingershurt: ensuring we have enough audio channels!
    pygame.mixer.set_num_channels(10)

    return True

  #myfingershurt:
  def findChannel(self):
    return pygame.mixer.find_channel()


  def getChannelCount(self):
    return pygame.mixer.get_num_channels()

  def getChannel(self, n):
    return Channel(n)

  def close(self):
    try:
      pygame.mixer.quit()
    except:
      pass
    #pygame.mixer.quit()

  def pause(self):
    pygame.mixer.pause()

  def unpause(self):
    pygame.mixer.unpause()

class OneSound(object):
  def __init__(self, fileName):
    self.file = fileName
    self.sound = None
    self.volume = None
    self.event = None
    self.channel = None
    stream        = OggStream(self.file)
    self.freq = stream.freq()

  def setPos(self,pos):
      # here just for inheritance, not used on these
      pass

  def isPlaying(self):  #MFH - adding function to check if sound is playing
    return self.sound.get_num_channels()

  def setEndEvent(self,event = None):
    self.event = event
    if self.channel:
      self.channel.set_endevent(event)

  def play(self):
    if not self.sound:
        self.sound   = pygame.mixer.Sound(self.file)
    if resampleOk:
        (freq,format,channels) = pygame.mixer.get_init()
        if freq != self.freq:
          print("reload sound ",self.file)
          self.freq = self.stream.freq()
          self.sound   = pygame.mixer.Sound(file = self.file)
        if self.freq != freq:
            snd_array = pygame.sndarray.array(self.sound)
            samples = old_div(len(snd_array),2)
            samples = int(samples*freq*1.0/(self.frequency))
            print("start resampling ",self.file," from ",self.frequency," to ",freq," len ",old_div(len(snd_array),2)," visée ",samples)
    #        if samples != len(snd_array):
    #          snd_array = np.resize(snd_array,(samples,2))
            snd_array = resample(snd_array, freq*1.0/self.frequency, "sinc_fastest").astype(snd_array.dtype)
            # datal = signal.resample(snd_array[0::2],samples).astype(snd_array.dtype)
            # datar = signal.resample(snd_array[1::2],samples).astype(snd_array.dtype)
            # snd_array = np.resize(snd_array,(len(datal)*2,2))
            # snd_array[0::2] = datal
            # snd_array[1::2] = datar
            # print "end resampling ",snd_array
            self.sound = pygame.sndarray.make_sound(snd_array)
            self.freq = freq

    if self.volume:
      self.sound.set_volume(self.volume)
    self.channel = self.sound.play()
    self.playTime = time.time()
    if self.event:
        self.channel.set_endevent(self.event)

  def stop(self):
    if self.sound:
      self.sound.stop()
    self.sound = None
    self.channel = None

  def setVolume(self, volume):
    self.volume = volume
    if self.sound:
      self.sound.set_volume(volume)

  def fadeout(self, time):
    if self.sound:
      self.sound.fadeout(time)

class Sound(object):
  def __init__(self,  fileName, engine = None, speed = 1):
    files = fileName.split(";")
    self.fileName = fileName
    self.sounds = []
    self.channel = self.event = None
    self.pausePos = 0.0
    self.playTime = 0
    self.isPause = False
    self.Playing = False
    self.speed = speed
    for file in files:
        for f in glob.glob(file):
            duration = 0
            if f.lower().endswith(".ogg"):
              file = ogg.VorbisFileStream(f)
              duration = file.vf.pcmlengths[1]/file.frequency
            # Both classes below are compatible and can both play ogg files
            # the difference is that OneSound is specialized on small
            # sounds, and the Streaming one on big audio tracks which need
            # to be played synchronously. So it's better to reserve the
            # streaming one on the tracks of more than 30s, just to save
            # memory.
            if duration > 12:
              self.sounds.append(StreamingOggSound(f, engine,speed=self.speed))
            else:
              self.sounds.append(OneSound(f))

  def setPos(self,pos):
      for sound in self.sounds:
          sound.setPos(pos)

  def play(self):
      self.Playing = True
      for sound in self.sounds:
          sound.play()
      self.playTime = time.time()

  def pause(self):
    pygame.mixer.pause()
    self.pausePos = self.getPosition()
    self.isPause = True

  def unpause(self):
    self.pausePos /= 1000*self.speed # return to time
    self.pausePos += self.playTime
    self.playTime += time.time() - self.pausePos
    self.isPause = False
    pygame.mixer.unpause()

  def getPosition(self):
    # There is no way to tell precisely where we are in the song because of
    # the buffers used to playback (no way to know where we are in the
    # buffer and the buffers are big). So the way around the problem is
    # just to calculate the time since the playback started and return it.
    # It's adjusted if there was a pause.
    if not self.Playing:
        return 0
    if self.isPause:
        return self.pausePos
    t = (time.time() - self.playTime)*1000*self.speed # number of ms !
    return t

  def isPlaying(self):
    if not self.isPause:
        # if the song ends on a fadeout, or reaches its end, the stop
        # command never arrives here
        # which was dumb, the code was updated to use the event instead, it
        # should never call this again, but I keep it just in case !
        found = False
        for sound in self.sounds:
            if sound.channel and sound.channel.get_busy():
                found = True
                break
        if not found:
          print("updated Playing to false")
          self.Playing = False
    return self.Playing

  def stop(self):
      self.Playing = False
      for sound in self.sounds:
          sound.stop()

  def setEndEvent(self,event = None):
    for sound in self.sounds:
      sound.setEndEvent(event)

  def setVolume(self, volume):
    for sound in self.sounds:
      sound.setVolume(volume)

  def fadeout(self, time):
    for sound in self.sounds:
      sound.fadeout(time)

import GameEngine

if ogg:
  class OggStream(object):
    def __init__(self, inputFileName):
      self.file = ogg.VorbisFileStream(inputFileName)

    def read(self, bytes = 4096):
        ogg.pyoggSetStreamBufferSize(bytes)
        try:
          (data, bytes) = self.file.get_buffer()
          return data
        except:
          return None

    def freq(self):
        return self.file.frequency

    def chans(self):
        return self.file.channels

    def time_seek(self,pos):
        return vorbis.ov_time_seek(self.file.vf,pos)

  class StreamingOggSound(OneSound, Task):

      # This class inherits from OneSound, but is totally different :
      # - it doesn't decode the whole ogg in 1 time, but uses Task to
      # decode it chunk by chunk which saves a vast amount of memory
      # so it requires engine to be able to add its task when starting
      # playback
      # As the name implies, it's streaming, so it pre-decodes 8 buffers
      # when initialized even before starting playback just to be sure to
      # be able to start to play immediately.

    def __init__(self, fileName,engine,speed = 1):
      Task.__init__(self)
      if not engine:
        engine = GameEngine.getEngine()
      self.engine       = engine
      self.fileName     = fileName
      self.channel      = None
      self.playing      = False
      self.bufferSize   = 1024 * 64
      self.bufferCount  = 8
      self.volume       = 1.0
      self.event = None
      self.speed = speed

        #myfingershurt: buffer is 2D array (one D for each channel) of 16-bit UNSIGNED integers / samples
        #  2*1024*64 = 131072 samples per channel
      self.buffer       = np.zeros((2 * self.bufferSize, 2),dtype=np.int16)

      self.decodingRate = 4
      self._reset()

    def getPosition(self):
        # not used - see Sound.getPosition instead
        return 0

    def _reset(self):

      self.stream        = OggStream(self.fileName)
      self.fadeoutTime = None
      (self.freq,self.format,self.channels) = pygame.mixer.get_init()
      if self.stream.freq()*self.speed != self.freq:
          # try to re-open the mixer with this file's frequency
          # the idea is to try to minimize the resampling work which is cpu
          # intensive, it should work if different frequencies are not
          # mixed on the same song
          self.engine.audio.close()    #MFH - ensure no audio is playing during the switch!
          print("reopen mixer at ",int(self.frequency*self.speed))
          self.engine.audio.pre_open(frequency = int(self.frequency*self.speed), bits = self.engine.bits, stereo = self.engine.stereo, bufferSize = self.engine.bufferSize)
          self.engine.audio.open(frequency = int(self.stream.freq()*self.speed), bits = self.engine.bits, stereo = self.engine.stereo, bufferSize = self.engine.bufferSize)
          (self.freq,self.format,self.channels) = pygame.mixer.get_init()

      self.buffersIn     = [pygame.sndarray.make_sound(zeros((self.bufferSize,2 ))) for i in range(self.bufferCount + 1)]
      self.buffersOut    = []
      self.buffersBusy   = []
      self.bufferPos     = 0
      self.done          = False
      # do not call prebuffer here in case the caller wants to set the position
      # 1st, which will happen for normal song tracks
      # self.prebuffer()

    def prebuffer(self):
      while len(self.buffersOut) < self.bufferCount and not self.done:
        #myfingershurt: while there are less than 8 sets of 65k sample 2 channel buffers in the buffersOut list,
        # continue to decode and fill them.
        self._produceSoundBuffers()

    def __del__(self):
      self.engine.removeTask(self)

    def streamIsPlaying(self):  #MFH - adding function to check if sound is playing
      return self.playing

    def setPos(self,pos):
        self.stream.time_seek(pos)
        self.buffersOut = []
        if self.channel:
            self.channel.stop() # clear the queue !
        self.prebuffer()

    def isPlaying(self):
        return self.playing

    def setEndEvent(self,event = None):
      self.event = event
      # if self.channel:
      #   self.channel.set_endevent(event)

    def play(self):
      if self.playing:
        return

        #myfingershurt: 2D buffer (L,R) of 16-bit unsigned integer samples, each channel 65536 samples long
        #.... buffersIn = a list of 9 of these.

      self.engine.addTask(self, synchronized = False)
      self.playing = True
      self.channel = pygame.mixer.find_channel()
      self.channel.set_volume(self.volume)

      self.prebuffer()

      #once all 8 output buffers are filled, play the first one.
      self.channel.play(self.buffersOut.pop())

    def stop(self):
      self.playing = False
      if self.channel:
        self.channel.stop()
      self.engine.removeTask(self)
      self._reset()

    def setVolume(self, volume):
      self.volume = volume
      if self.channel:
        self.channel.set_volume(self.volume)

    #stump: pitch bending
    def setPitchBend(self, factor):
      self.channel.setPitchBend(factor)

    def stopPitchBend(self):
      self.channel.stopPitchBend()

    def fadeout(self, ms):
      if self.channel:
        self.fadeoutTime = time.time() + old_div(ms,1000)
        # The channel alone is not enough here, because buffers are just
        # queued, so after the fadeout time it just continues playing what
        # arrives...
        self.channel.fadeout(ms)

    def _decodeStream(self):
      # No available buffers to fill?
      if not self.buffersIn or self.done:
        return

      data = self.stream.read()

      if not data:
        self.done = True
      else:
        if self.channels == 2:
          # data = struct.unpack("%dh" % (len(data) / 2), data)
          data = np.frombuffer(data, dtype=np.int16)
          samples = old_div(len(data), 2)
#          if self.freq != self.info.rate*self.speed:
#         samples = int(samples*self.freq/(self.info.rate*self.speed))
#              datal = signal.resample(data[0::2],samples)
#         datar = signal.resample(data[1::2],samples)
#         self.buffer[self.bufferPos:self.bufferPos + samples, 0] = datal
#         self.buffer[self.bufferPos:self.bufferPos + samples, 1] = datar
#     else:
          self.buffer[self.bufferPos:self.bufferPos + samples, 0] = data[0::2]
          self.buffer[self.bufferPos:self.bufferPos + samples, 1] = data[1::2]
          self.bufferPos += samples
        elif self.channels == 1:
          samples = old_div(len(data),2)
          # data = struct.unpack("%dh" % (samples), data)
          data = np.frombuffer(data, dtype=np.int16)
#          if self.freq != self.info.rate*self.speed:
#         samples = int(samples*self.freq/(self.info.rate*self.speed))
#         data = signal.resample(data,samples)
          self.buffer[self.bufferPos:self.bufferPos + samples,0] = data
          self.buffer[self.bufferPos:self.bufferPos + samples,1] = data
          self.bufferPos += samples
        else:
            raise "Number of channels ? %d" % (self.channels)

      # If we have at least one full buffer decode, claim a buffer and copy the
      # data over to it.
      if self.bufferPos >= self.bufferSize or (self.done and self.bufferPos):
        # Claim the sound buffer and copy the data
        if self.bufferPos < self.bufferSize:
          self.buffer[self.bufferPos:]  = 0
        soundBuffer = self.buffersIn.pop()
        pygame.sndarray.samples(soundBuffer)[:] = self.buffer[0:self.bufferSize]

        # Discard the copied sound data
        n = max(0, self.bufferPos - self.bufferSize)
        self.buffer[0:n] = self.buffer[self.bufferSize:self.bufferSize+n]
        self.bufferPos   = n

        return soundBuffer

    def _produceSoundBuffers(self):
      # Decode enough that we have at least one full sound buffer
      # ready in the queue if possible
      while not self.done:
        for i in range(self.decodingRate):
          soundBuffer = self._decodeStream()
          if soundBuffer:
            self.buffersOut.insert(0, soundBuffer)
        if self.buffersOut:
          break

    def run(self, ticks):
      if not self.playing:
        return

      if self.fadeoutTime and time.time() >= self.fadeoutTime:
        if self.event:
            pygame.event.post(pygame.event.Event(self.event,{}))
        self.stop()
        return
      #myfingershurt: this is now done directly when called.
      #self.channel.setVolume(self.volume)

      if len(self.buffersOut) < self.bufferCount:
        self._produceSoundBuffers()

      if not self.channel.get_queue() and self.buffersOut:
        # Queue one decoded sound buffer and mark the previously played buffer as free
        soundBuffer = self.buffersOut.pop()
        self.buffersBusy.insert(0, soundBuffer)
        self.channel.queue(soundBuffer)
        if len(self.buffersBusy) > 2:
          self.buffersIn.insert(0, self.buffersBusy.pop())

      if not self.buffersOut and self.done and not self.channel.get_busy():
        if self.event:
            # Send the event only when the sound ends normally
            pygame.event.post(pygame.event.Event(self.event,{}))
        self.stop()

# Debian has no Python Numeric module anymore
if tuple(int(i) for i in pygame.__version__[:5].split('.')) < (1, 9, 0):
  # Must use Numeric instead of numpy, since PyGame 1.7.1 is
  # not compatible with the latter, and 1.8.x isn't either (though it claims to be).
  import Numeric
  def zeros(size):
    return Numeric.zeros(size, typecode='s')   #myfingershurt: typecode s = short = int16
else:
  import numpy
  def zeros(size):
    return numpy.zeros(size, dtype="h")

#stump: mic passthrough
class MicrophonePassthroughStream(Sound, Task):
  def __init__(self, engine, mic):
    Task.__init__(self)
    self.engine = engine
    self.channel = None
    self.mic = mic
    self.playing = False
    self.volume = 1.0
  def __del__(self):
    self.stop()
  def play(self):
    if not self.playing:
      self.engine.addTask(self, synchronized=False)
      self.playing = True
  def stop(self):
    if self.playing:
      self.channel.stop()
      self.engine.removeTask(self)
      self.playing = False
  def setVolume(self, vol):
    self.volume = vol
  def run(self, ticks):
    chunk = ''.join(self.mic.passthroughQueue)
    self.mic.passthroughQueue = []
    if chunk == '':
      return
    samples = old_div(len(chunk),4)
    data = tuple(int(s * 32767) for s in struct.unpack('%df' % samples, chunk))
    playbuf = zeros((samples, 2))
    playbuf[:, 0] = data
    playbuf[:, 1] = data
    snd = pygame.sndarray.make_sound(playbuf)
    if self.channel is None or not self.channel.get_busy():
      self.channel = snd.play()
    else:
      self.channel.queue(snd)
    self.channel.set_volume(self.volume)

