#####################################################################
# -*- coding: iso-8859-1 -*-                                        #
#                                                                   #
# Frets on Fire                                                     #
# Copyright (C) 2006 Sami Ky�stil�                                  #
#               2008 myfingershurt                                  #
#               2008 Glorandwarf                                    #
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

import pygame
import Log
import Audio

ports = None
midi = []
midiin = None
portCount = 0

midiOutPorts = None
midiOutList = []
midiout = None
midiOutPortCount = 0

try:
  import rtmidi
  haveMidi = True
except ImportError:
  haveMidi = False


#haveMidi = False  #this line disables the rtmidi module for computers with 0 midi ports...has to be this way for now to avoid crashes.

if haveMidi:
  
  #MFH - check for and test MIDI output ports by playing a note
  try:
    Log.debug("Checking MIDI output ports for a wavetable or synth for sound generation...")
    midiout = rtmidi.RtMidiOut()
    midiOutPortCount = midiout.getPortCount()
    Log.debug("MIDI output port count = " + str(midiOutPortCount) )
    if midiOutPortCount > 0:
      midiOutPortNumber = 0
      midiOutPorts = range(midiOutPortCount)
      for x in midiOutPorts:
        midiOutList.append( rtmidi.RtMidiOut() )
        midiOutPortName = midiOutList[midiOutPortNumber].getPortName(midiOutPortNumber)
        Log.debug("MIDI Output port %d found: %s" % (midiOutPortNumber,midiOutPortName) )
        
        #Log.debug("Testing MIDI Output port %d (%s)..." % (midiOutPortNumber,midiOutPortName) )
        #midiOutList[midiOutPortNumber].openPort(midiOutPortNumber)
        #midiOutList[midiOutPortNumber].sendMessage(144, 64, 90)
        ##midiOutList[midiOutPortNumber].closePort(midiOutPortNumber)


        
        midiOutPortNumber += 1
  except Exception, e:
    Log.error(str(e))
    midiOutPorts = None


  #MFH - check for MIDI input ports
  try:
    midiin = rtmidi.RtMidiIn()
    portCount = midiin.getPortCount()
    #Log.debug("MIDI port count = " + str(portCount) )
    if portCount > 0:
      ports = range(portCount)
      for x in ports:
        midi.append( rtmidi.RtMidiIn() )
  except Exception, e:
    Log.error(str(e))
    ports = None



#-  // Program change: 192, 5
#-  message.push_back( 192 );
#-  message.push_back( 5 );
#-  midiout->sendMessage( &message );
#-
#-  // Control Change: 176, 7, 100 (volume)
#-  message[0] = 176;
#-  message[1] = 7;
#-  message.push_back( 100 );
#-  midiout->sendMessage( &message );
#-
#-  // Note On: 144, 64, 90
#-  message[0] = 144;
#-  message[1] = 64;
#-  message[2] = 90;
#-  midiout->sendMessage( &message );
#-
#-  SLEEP( 500 ); // Platform-dependent ... see example in tests directory.
#-
#-  // Note Off: 128, 64, 40
#-  message[0] = 128;
#-  message[1] = 64;
#-  message[2] = 40;
#-  midiout->sendMessage( &message );
#-
#-  // Clean up
#- cleanup:
#-  delete midiout;



from Task import Task
import Player
from Player import Controls

import Config   #MFH

class KeyListener:
  def keyPressed(self, key, unicode):
    pass
    
  def keyReleased(self, key):
    pass
  
  def lostFocus(self):
    pass
  
  def exitRequested(self):
    pass

class MouseListener:
  def mouseButtonPressed(self, button, pos):
    pass
    
  def mouseButtonReleased(self, button, pos):
    pass
    
  def mouseMoved(self, pos, rel):
    pass
    
class SystemEventListener:
  def screenResized(self, size):
    pass
    
  def restartRequested(self):
    pass
    
  def musicFinished(self):
    pass
    
  def quit(self):
    pass

MusicFinished = pygame.USEREVENT

try:
  reversed
except:
  def reversed(seq):
    seq = seq[:]
    seq.reverse()
    return seq

class Input(Task):
  def __init__(self):

    self.logClassInits = Config.get("game", "log_class_inits")
    if self.logClassInits == 1:
      Log.debug("Input class init (Input.py)...")
  
    Task.__init__(self)
    self.mouse                = pygame.mouse
    self.mouseListeners       = []
    self.keyListeners         = []
    self.systemListeners      = []
    self.priorityKeyListeners = []
    self.controls             = Controls()
    self.activeGameControls   = []
    self.p2Nav                = self.controls.p2Nav
    self.type1                = self.controls.type[0]
    self.keyCheckerMode       = Config.get("game","key_checker_mode")
    self.disableKeyRepeat()
    
    self.gameGuitars = 0
    self.gameDrums   = 0
    self.gameMics    = 0
    self.gameBots    = 0

    # Initialize joysticks
    pygame.joystick.init()
    self.joystickAxes = {}
    self.joystickHats = {}
    self.joyOff = False

    self.joysticks = [pygame.joystick.Joystick(id) for id in range(pygame.joystick.get_count())]
    for j in self.joysticks:
      j.init()
      self.joystickAxes[j.get_id()] = [0] * j.get_numaxes() 
      self.joystickHats[j.get_id()] = [(0, 0)] * j.get_numhats() 
    joyNum = len(self.joysticks)
    Log.debug("%d joysticks found." % (joyNum))
    oldJoyNum = Config.get("game", "joysticks")
    if joyNum != oldJoyNum:
      self.joyOff = True
    Config.set("game", "joysticks", joyNum)

    # Enable music events
    Audio.Music.setEndEvent(MusicFinished)
    #Audio.Music.setEndEvent()   #MFH - no event required?

    # Custom key names
    self.getSystemKeyName = pygame.key.name
    pygame.key.name       = self.getKeyName

    if haveMidi:
      if ports:
        Log.debug("%d MIDI inputs found." % (len(ports)))
        try:
          for i in ports:
            midi[i].openPort(i, False)
        except Exception, e:
          Log.error("Error opening MIDI port %d: %s" % (i,str(e)) )
      else:
        Log.warn("No MIDI input ports found.")


  def reloadControls(self):
    self.controls = Controls()

  def pluginControls(self):
    self.gameDrums = 0
    self.gameGuitars = 0
    self.gameMics = 0
    Player.pluginControls(self.activeGameControls)
    for i in self.activeGameControls:
      if self.controls.type[i] == -1:
        self.gameBots += 1
      elif self.controls.type[i] in Player.DRUMTYPES:
        self.gameDrums += 1
      elif self.controls.type[i] in Player.MICTYPES:
        self.gameMics += 1
      elif self.controls.type[i] in Player.GUITARTYPES:
        self.gameGuitars += 1
  
  def getAnalogKill(self, player):
    return self.controls.analogKill[self.activeGameControls[player]]
  
  def getAnalogSP(self, player):
    return self.controls.analogSP[self.activeGameControls[player]]
  
  def getAnalogSPThresh(self, player):
    return self.controls.analogSPThresh[self.activeGameControls[player]]
  
  def getAnalogSPSense(self, player):
    return self.controls.analogSPSense[self.activeGameControls[player]]
  
  def getAnalogSlide(self, player):
    return self.controls.analogSlide[self.activeGameControls[player]]
  
  def getAnalogFX(self, player):
    return self.controls.analogFX[self.activeGameControls[player]]
  
  def getTwoChord(self, player):
    return self.controls.twoChord[self.activeGameControls[player]]

  def disableKeyRepeat(self):
    pygame.key.set_repeat(0, 0)

  def enableKeyRepeat(self):
    pygame.key.set_repeat(300, 30)

  def addMouseListener(self, listener):
    if not listener in self.mouseListeners:
      self.mouseListeners.append(listener)

  def removeMouseListener(self, listener):
    if listener in self.mouseListeners:
      self.mouseListeners.remove(listener)

  def addKeyListener(self, listener, priority = False):
    if priority:
      if not listener in self.priorityKeyListeners:
        self.priorityKeyListeners.append(listener)
    else:
      if not listener in self.keyListeners:
        self.keyListeners.append(listener)

  def removeKeyListener(self, listener):
    if listener in self.keyListeners:
      self.keyListeners.remove(listener)
    if listener in self.priorityKeyListeners:
      self.priorityKeyListeners.remove(listener)

  def addSystemEventListener(self, listener):
    if not listener in self.systemListeners:
      self.systemListeners.append(listener)
      
  def removeSystemEventListener(self, listener):
    if listener in self.systemListeners:
      self.systemListeners.remove(listener)
      
  def broadcastEvent(self, listeners, function, *args):
    for l in reversed(listeners):
      if getattr(l, function)(*args):
        return True
    else:
      return False
    
  def broadcastSystemEvent(self, name, *args):
    return self.broadcastEvent(self.systemListeners, name, *args)

  def encodeMidiButton(self, midi, button):
    return 0x40000 + (midi << 8 ) + button

  def decodeMidiButton(self, id):
    id -= 0x40000
    return (id >> 8, id & 0xff)

  def encodeJoystickButton(self, joystick, button):
    return 0x10000 + (joystick << 8) + button

  def encodeJoystickAxis(self, joystick, axis, end):
    return 0x20000 + (joystick << 8) + (axis << 4) + end
  
  def encodeJoystickHat(self, joystick, hat, pos):
    v = int((pos[1] + 1) * 3 + (pos[0] + 1))
    return 0x30000 + (joystick << 8) + (hat << 4) + v 
  
  def decodeJoystickButton(self, id):
    id -= 0x10000
    return (id >> 8, id & 0xff)

  def decodeJoystickAxis(self, id):
    id -= 0x20000
    return (id >> 8, (id >> 4) & 0xf, id & 0xf)

  def decodeJoystickHat(self, id):
    id -= 0x30000
    v = id & 0xf
    x, y = (v % 3) - 1, (v / 3) - 1
    return (id >> 8, (id >> 4) & 0xf, (x, y))

  #myfingershurt: new function specifically for detecting an analog whammy input:
  def getWhammyAxis(self, id):
    if id < 0x30000 and id >= 0x20000:
      joy, axis, end = self.decodeJoystickAxis(id)
      return (True, joy, axis)
    else:
      return (False, 0, 0)
  
  def getJoysticksUsed(self, keys):
    midis = []
    joys  = []
    for id in keys:
      if id >= 0x40000:
        midi, but = self.decodeMidiButton(id)
        if midi not in midis:
          midis.append(midi)
      elif id >= 0x30000:
        joy, axis, pos = self.decodeJoystickHat(id)
        if joy not in joys:
          joys.append(joy)
      elif id >= 0x20000:
        joy, axis, end = self.decodeJoystickAxis(id)
        if joy not in joys:
          joys.append(joy)
      elif id >= 0x10000:
        joy, but = self.decodeJoystickButton(id)
        if joy not in joys:
          joys.append(joy)
      return [joys, midis]
  
  def getKeyName(self, id):
    if id >= 0x40000:
      midi, but = self.decodeMidiButton(id)
      return "Midi #%d-%d" % (midi + 1, but)
    elif id >= 0x30000:
      joy, axis, pos = self.decodeJoystickHat(id)
      return "Joy #%d, hat %d %s" % (joy + 1, axis, pos)
    elif id >= 0x20000:
      joy, axis, end = self.decodeJoystickAxis(id)
      return "Joy #%d, axis %d %s" % (joy + 1, axis, (end == 1) and "high" or "low")
    elif id >= 0x10000:
      joy, but = self.decodeJoystickButton(id)
      return "Joy #%d, %s" % (joy + 1, chr(ord('A') + but))
    return self.getSystemKeyName(id)

  def run(self, ticks):
    pygame.event.pump()
    for event in pygame.event.get():
      if event.type == pygame.KEYDOWN:
        if not self.broadcastEvent(self.priorityKeyListeners, "keyPressed", event.key, event.unicode):
          self.broadcastEvent(self.keyListeners, "keyPressed", event.key, event.unicode)
      elif event.type == pygame.KEYUP:
        if not self.broadcastEvent(self.priorityKeyListeners, "keyReleased", event.key):
          self.broadcastEvent(self.keyListeners, "keyReleased", event.key)
      elif event.type == pygame.MOUSEMOTION:
        self.broadcastEvent(self.mouseListeners, "mouseMoved", event.pos, event.rel)
      elif event.type == pygame.MOUSEBUTTONDOWN:
        self.broadcastEvent(self.mouseListeners, "mouseButtonPressed", event.button, event.pos)
      elif event.type == pygame.MOUSEBUTTONUP:
        self.broadcastEvent(self.mouseListeners, "mouseButtonReleased", event.button, event.pos)
      elif event.type == pygame.VIDEORESIZE:
        self.broadcastEvent(self.systemListeners, "screenResized", event.size)
      elif event.type == pygame.QUIT:
        self.broadcastEvent(self.systemListeners, "quit")
      elif event.type == pygame.ACTIVEEVENT: # akedrou - catch for pause onLoseFocus
        if (event.state == 2 or event.state == 6) and event.gain == 0:
          self.broadcastEvent(self.keyListeners, "lostFocus") # as a key event, since Scene clients don't handle system events
      elif event.type == MusicFinished:
        self.broadcastEvent(self.systemListeners, "musicFinished")
      elif event.type == pygame.JOYBUTTONDOWN: # joystick buttons masquerade as keyboard events
        id = self.encodeJoystickButton(event.joy, event.button)
        if not self.broadcastEvent(self.priorityKeyListeners, "keyPressed", id, u'\x00'):
          self.broadcastEvent(self.keyListeners, "keyPressed", id, u'\x00')
      elif event.type == pygame.JOYBUTTONUP:
        id = self.encodeJoystickButton(event.joy, event.button)
        if not self.broadcastEvent(self.priorityKeyListeners, "keyReleased", id):
          self.broadcastEvent(self.keyListeners, "keyReleased", id)
      elif event.type == pygame.JOYAXISMOTION:
        try:
          threshold = .8
          state     = self.joystickAxes[event.joy][event.axis]
          keyEvent  = None

          if event.value > threshold and state != 1:
            self.joystickAxes[event.joy][event.axis] = 1
            keyEvent = "keyPressed"
            args     = (self.encodeJoystickAxis(event.joy, event.axis, 1), u'\x00')
            state    = 1
          elif event.value < -threshold and state != -1:
            keyEvent = "keyPressed"
            args     = (self.encodeJoystickAxis(event.joy, event.axis, 0), u'\x00')
            state    = -1
          elif state != 0:
            keyEvent = "keyReleased"
            args     = (self.encodeJoystickAxis(event.joy, event.axis, (state == 1) and 1 or 0), )
            state    = 0

          if keyEvent:
            self.joystickAxes[event.joy][event.axis] = state
            if not self.broadcastEvent(self.priorityKeyListeners, keyEvent, *args):
              self.broadcastEvent(self.keyListeners, keyEvent, *args)
        except KeyError:
          pass
      elif event.type == pygame.JOYHATMOTION:
        try:
          state     = self.joystickHats[event.joy][event.hat]
          keyEvent  = None

          #if event.value != (0, 0) and state == (0, 0):

          # Stump's PS3 GH3 up-and-down-strumming patch
          if event.value != (0, 0) and state != (0, 0):
            keyEvent = "keyReleased"
            args     = (self.encodeJoystickHat(event.joy, event.hat, state), )
            state    = (0, 0)
            pygame.event.post(event)
          elif event.value != (0, 0) and state == (0, 0):

            self.joystickHats[event.joy][event.hat] = event.value
            keyEvent = "keyPressed"
            args     = (self.encodeJoystickHat(event.joy, event.hat, event.value), u'\x00')
            state    = event.value
          else:
            keyEvent = "keyReleased"
            args     = (self.encodeJoystickHat(event.joy, event.hat, state), )
            state    = (0, 0)

          if keyEvent:
            self.joystickHats[event.joy][event.hat] = state
            if not self.broadcastEvent(self.priorityKeyListeners, keyEvent, *args):
              self.broadcastEvent(self.keyListeners, keyEvent, *args)
        except KeyError:
          pass

    if ports:
      for i in ports:
        midimsg = midi[i].getMessage()
        if len(midimsg) > 0:
          id = self.encodeMidiButton(x, midimsg[1])
          #MFH - must check for 0x80 - 0x8F for Note Off events (keyReleased) and 0x90 - 0x9F for Note On events (keyPressed)
          #if midimsg[0] == 153:
          noteOn = False
          noteOff = False
          
          if (midimsg[0] >= 0x90) and (midimsg[0] <= 0x9F):   #note ON range
            if midimsg[2] > 0:  #velocity > 0, confirmed note on
              noteOn = True
            else:   #velocity is 0 - this is pretty much a note off.
              noteOff = True
          elif (midimsg[0] >= 0x80) and (midimsg[0] <= 0x8F):  #note OFF range
            noteOff = True

          if noteOn:
            if not self.broadcastEvent(self.priorityKeyListeners, "keyPressed", id, u'\x00'):
              self.broadcastEvent(self.keyListeners, "keyPressed", id, u'\x00')
          
          elif noteOff:
            if not self.broadcastEvent(self.priorityKeyListeners, "keyReleased", id):
              self.broadcastEvent(self.keyListeners, "keyReleased", id)


  #-------------------------------------------
  # glorandwarf: check that there are no control conflicts
  def checkControls(self):
    if self.controls.isKeyMappingOK() == False:
      Log.warn("Conflicting player controls, resetting to defaults")
      self.controls.restoreDefaultKeyMappings()
      self.reloadControls()
  
  # glorandwarf: sets the new key mapping and checks for a conflict
  def setNewKeyMapping(self, section, option, key, control):
    return Player.setNewKeyMapping(section, option, key, control)
  #-------------------------------------------
