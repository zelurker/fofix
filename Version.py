#####################################################################
# -*- coding: iso-8859-1 -*-                                        #
#                                                                   #
# Frets on Fire                                                     #
# Copyright (C) 2006 Sami Ky�stil�                                  #
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

import sys
import os
VERSION = '3.121'
URL = 'http://fofix.googlecode.com'

def appName():
  return "fofix"

def appNameSexy():
  return "FoFiX"

def revision():
  import svntag
  try:
    revision = " Maintenance (r%d)" % int(svntag.get_svn_info(os.path.dirname(__file__))['revnum'])
  except:
    revision = " Final"
  return revision

# evilynux: Returns version number w.r.t. frozen state
def version():
  if hasattr(sys, 'frozen'):
    # stump: if we've been py2exe'd, read our version string from the exe.
    if sys.frozen == 'windows_exe':
      import win32api
      us = os.path.abspath(unicode(sys.executable, sys.getfilesystemencoding()))
      version = win32api.GetFileVersionInfo(us, r'\StringFileInfo\%04x%04x\ProductVersion' % win32api.GetFileVersionInfo(us, r'\VarFileInfo\Translation')[0])
    else:
      version = "%s%s" % ( VERSION, revision() )
  else:
    version = "%s%s" % ( VERSION, revision() )
  return version

def dataPath():
  # Determine whether were running from an exe or not
  if hasattr(sys, "frozen"):
    if os.name == "posix":
      dataPath = os.path.join(os.path.dirname(sys.argv[0]), "../lib/fofix")
      if not os.path.isdir(dataPath):
        dataPath = "data"
    else:
      dataPath = "data"
  else:
    dataPath = os.path.join("..", "data")
  return dataPath
  
