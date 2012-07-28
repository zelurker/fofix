#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
#####################################################################
# Frets on Fire X (FoFiX)                                           #
# Copyright (C) 2006 Sami Ky�stil�                                  #
#               2008 evilynux <evilynux@gmail.com>                  #
#               2009 FoFiX Team                                     #
#               2009 akedrou                                        #
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

##@package FoFiX
# Main game executable.

# Register the latin-1 encoding
import codecs
import encodings.iso8859_1
import encodings.utf_8
codecs.register(lambda encoding: encodings.iso8859_1.getregentry())
codecs.register(lambda encoding: encodings.utf_8.getregentry())
assert codecs.lookup("iso-8859-1")
assert codecs.lookup("utf-8")

#stump: pygst eats --help, so process the command line before that.
# Also do this before we import any heavyweight stuff so --help goes through
# as efficiently as possible and we can disable pyOpenGL error checking
# if we are not asked for it.
import getopt
import sys
import os
import Version

## Display command-line usage and exit.
# Outputs to stdout unless py2exe'd, in which case the usage is presented
# using a MessageBox().
# @param errmsg    Optional error message.
def _usage(errmsg=None):
  usage = """Usage: %(prog)s [options]

Options:
  --help,    -h                       Show this help.
  --config=, -c [configfile]          Use this configuration file instead of
                                      fofix.ini from its standard location on
                                      your platform.  Use "reset" to use the
                                      usual fofix.ini but clear it first.
  --fullscreen=, -f [true/false]      Force (non-)usage of full-screen mode.
  --resolution=, -r [resolution]      Force a specific resolution to be used.
  --theme=,  -t [theme]               Force the specified theme to be used.
                                      Remember to quote the theme name if it
                                      contains spaces (e.g.
                                        %(prog)s -t "Guitar Hero III")
  --song=,   -s [songdir]             Play a song in one-shot mode.
                                      (See "One-shot mode options" below.)

Advanced options:
  --verbose, -v                       Verbose messages
  --debug,   -d                       Write debug file
  --opengl-error-checking             Enable OpenGL error checking

One-shot mode options (ignored unless in one-shot mode):
  --part=,   -p [part number]         0: Guitar, 1: Rhythm, 2: Bass, 3: Lead
                                      4: Drum,   5: Vocals
  --diff=,   -l [level of difficulty] 0: Expert, 1: Hard, 2: Medium, 3: Easy
                                      (Only applies if "part" is set)
  --mode=,   -m [game mode]           0: Quickplay, 1: Practice, 2: Career
""" % {"prog": sys.argv[0]}
  if errmsg is not None:
    usage = '%s: %s\n\n%s' % (sys.argv[0], errmsg, usage)
  if hasattr(sys, 'frozen') and os.name == 'nt':
    import win32api
    import win32con
    win32api.MessageBox(0, usage, '%s %s' % (Version.appNameSexy(), Version.version()), win32con.MB_OK)
  else:
    print usage
  sys.exit(1)

try:
  opts, args = getopt.getopt(sys.argv[1:], "hvdc:f:r:t:s:l:p:m:n:", ["help", "verbose", "debug", "config=", "fullscreen=", "resolution=", "theme=", "song=", "diff=", "part=", "mode=", "nbrplayers=", "opengl-error-checking"])
except getopt.GetoptError, e:
  _usage(str(e))  # str(e): error message from getopt, e.g. "option --some-invalid-option not recognized"
if ('-h', '') in opts or ('--help', '') in opts:
  _usage()

#stump: disable pyOpenGL error checking if we are not asked for it.
# This must be before *anything* that may import pyOpenGL!
assert 'OpenGL' not in sys.modules
if ('--opengl-error-checking', '') not in opts:
  import OpenGL
  if OpenGL.__version__ >= '3':
    OpenGL.ERROR_CHECKING = False

import Log
import Config
from GameEngine import GameEngine
from MainMenu import MainMenu
from Language import _
import Resource
import pygame
import traceback

## Main function.
def main():
  playing = None
  configFile = None
  fullscreen = None
  resolution = None
  theme = None
  debug = False
  difficulty = None
  part = None
  mode = 0
  nbrplayers = 1
  for opt, arg in opts:
    if opt in ["--verbose", "-v"]:
      Log.quiet = False
    if opt in ["--debug", "-d"]:
      debug = True
    if opt in ["--config", "-c"]:
      configFile = arg
    if opt in ["--fullscreen", "-f"]:
      fullscreen = arg
    if opt in ["--resolution", "-r"]:
      resolution = arg
    if opt in ["--theme", "-t"]:
      theme = arg
    if opt in ["--song", "-s"]:
      playing = arg
    if opt in ["--diff", "-l"]:
      difficulty = arg      
    if opt in ["--part", "-p"]:
      part = arg
    #evilynux - Multiplayer and mode selection support
    if opt in ["--mode", "-m"]:
      mode = int(arg)
    if opt in ["--nbrplayers", "-n"]:
      nbrplayers = int(arg)

  # Load the configuration file.
  if configFile is not None:
    if configFile.lower() == "reset":
      fileName = os.path.join(Resource.getWritableResourcePath(), Version.appName() + ".ini")
      os.remove(fileName)
      config = Config.load(Version.appName() + ".ini", setAsDefault = True)
    else:
      config = Config.load(configFile, setAsDefault = True)
  else:
    config = Config.load(Version.appName() + ".ini", setAsDefault = True)

  #Lysdestic - Allow support for manipulating fullscreen via CLI
  if fullscreen is not None:
    Config.set("video", "fullscreen", fullscreen)

  #Lysdestic - Change resolution from CLI
  if resolution is not None:
    Config.set("video", "resolution", resolution)

  #Lysdestic - Alter theme from CLI
  if theme is not None:
    Config.set("coffee", "themename", theme)

  engine = GameEngine(config)
  engine.cmdPlay = 0

  # Check for a valid invocation of one-shot mode.
  if playing is not None:
    Log.debug('Validating song directory for one-shot mode.')
    library = Config.get("game","base_library")
    basefolder = os.path.join(Version.dataPath(),library,"songs",playing)
    if not (os.path.exists(os.path.join(basefolder, "song.ini")) and (os.path.exists(os.path.join(basefolder, "notes.mid")) or os.path.exists(os.path.join(basefolder, "notes-unedited.mid"))) and (os.path.exists(os.path.join(basefolder, "song.ogg")) or os.path.exists(os.path.join(basefolder, "guitar.ogg")))):
      Log.warn("Song directory provided ('%s') is not a valid song directory. Starting up FoFiX in standard mode." % playing)
      engine.startupMessages.append(_("Song directory provided ('%s') is not a valid song directory. Starting up FoFiX in standard mode.") % playing)
      playing = None

  # Set up one-shot mode if the invocation is valid for it.
  if playing is not None:
    Log.debug('Entering one-shot mode.')
    Config.set("game", "selected_library", "songs")
    Config.set("game", "selected_song", playing)
    engine.cmdPlay = 1
    if difficulty is not None:
      engine.cmdDiff = int(difficulty)
    if part is not None:
      engine.cmdPart = int(part)
    #evilynux - Multiplayer and mode selection support
    Config.set("game", "players", nbrplayers)
    if nbrplayers == 1:
      Config.set("game", "game_mode", mode)
    else:
      Config.set("game", "game_mode", 0)
      Config.set("game", "multiplayer_mode", mode)

  if debug:
    engine.setDebugModeEnabled(not engine.isDebugModeEnabled())
    engine.debugLayer.debugOut(engine)
    engine.quit()
    return

  encoding = Config.get("game", "encoding")
  if encoding is not None:
    #stump: XXX: Everything I have seen indicates that this is a
    # horrible, horrible hack.  Is there another way?  Do we even need this?
    reload(sys)
    sys.setdefaultencoding(encoding)

  engine.setStartupLayer(MainMenu(engine))

  #stump: make psyco optional
  if Config.get("performance", "use_psyco"):
    try:
      import psyco
      psyco.profile()
    except:
      Log.error("Unable to enable psyco as requested: ")

  # Run the main game loop.
  try:
    engine.ticksAtStart = pygame.time.get_ticks()
    while engine.run():
      pass
  except KeyboardInterrupt:
    Log.notice("Left mainloop due to KeyboardInterrupt.")
    # don't reraise

  # Restart the program if the engine is asking that we do so.
  if engine.restartRequested:
    Log.notice("Restarting.")
    engine.audio.close()
    try:
      # Extra arguments to insert between the executable we call and our
      # command line arguments.
      args = []
      # Figure out what executable to call.
      if hasattr(sys, "frozen"):
        if os.name == "nt":
          # When py2exe'd, sys.executable is the name of the EXE.
          exe = os.path.abspath(unicode(sys.executable, sys.getfilesystemencoding()))
        elif sys.frozen == "macosx_app":
          # When py2app'd, sys.executable is a Python interpreter copied
          # into the same dir where we live.
          exe = os.path.join(os.path.dirname(sys.executable), 'FoFiX')  # FIXME: don't hard-code "FoFiX" here
        else:
          raise RuntimeError, "Don't know how to restart when sys.frozen is %s" % repr(sys.frozen)
      else:
        # When running from source, sys.executable is the Python interpreter
        # being used to run the program.
        exe = sys.executable
        # Pass the optimization level on.
        if sys.flags.optimize > 0:
          args.append('-%s' % ('O' * sys.flags.optimize))
        args.append(__file__)
      os.execv(exe, [sys.executable] + args + sys.argv[1:])
    except:
      Log.error("Restart failed: ")
      raise

  # evilynux - MainMenu class already calls this - useless?
  engine.quit()


if __name__ == '__main__':
  try:
    main()
  except (KeyboardInterrupt, SystemExit):
    raise
  except:
    Log.error("Terminating due to unhandled exception: ")
    _logname = os.path.abspath(Log.logFile.name)
    _errmsg = "%s\n\n%s\n%s\n%s\n%s" % (
      _("Terminating due to unhandled exception:"),
      traceback.format_exc(),
      _("If you make a bug report about this error, please include the contents of the following log file:"),
      _logname,
      _("The log file already includes the traceback given above."))

    if os.name == 'nt':
      import win32api
      import win32con
      if win32api.MessageBox(0, "%s\n\n%s" % (_errmsg, _("Open the logfile now?")), "%s %s" % (Version.appNameSexy(), Version.version()), win32con.MB_YESNO|win32con.MB_ICONSTOP) == win32con.IDYES:
        Log.logFile.close()
        os.startfile(_logname)
      if hasattr(sys, 'frozen'):
        sys.exit(1)  # don't reraise if py2exe'd so the "Errors occurred" box won't appear after this and confuse the user as to which logfile we actually want
    else:
      print >>sys.stderr, _errmsg
    raise
