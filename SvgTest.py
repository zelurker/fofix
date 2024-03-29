from __future__ import division
#####################################################################
# -*- coding: iso-8859-1 -*-                                        #
#                                                                   #
# Frets on Fire                                                     #
# Copyright (C) 2006 Sami Kyöstilä                                  #
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

from past.utils import old_div
import unittest
from GameEngine import GameEngine
from Texture import Texture
import Config
import Version
from OpenGL.GL import *
from OpenGL.GLU import *

class SvgTest(unittest.TestCase):
  def testRendering(self):
    self.svg.transform.translate(256, 256)
    self.svg.transform.rotate(3.141592)
    self.svg.draw()
    self.e.video.flip()

  def testRenderToTexture(self):
    scale = 4
    fullwidth, fullheight = 512, 512
    width, height = int(old_div(fullwidth, scale)), int(old_div(fullheight, scale))
    t = Texture()
    self.e.svg.setProjection((0, 0, fullwidth, fullheight))
    
    glViewport(0, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    self.svg.transform.translate(old_div(width * scale, 2), old_div(height * scale, 2))
    self.svg.transform.rotate(3.141592)
    self.svg.draw()

    glViewport(0, 0, fullwidth, fullheight)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0.0, 1.0, 0.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    t.bind()
    glEnable(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glClear(GL_COLOR_BUFFER_BIT)
    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_TRIANGLE_STRIP)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(0.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(1.0, 0.0)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(0.0, 1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(1.0, 1.0)
    glEnd()

    self.e.video.flip()
    import time
    time.sleep(2)

  def setUp(self):
    config = Config.load(Version.appName() + ".ini", setAsDefault = True)
    self.e = GameEngine(config)
    self.e.loadImgDrawing(self, "svg", "mfhlogo.png")

    while not self.svg:
      self.e.run()

    glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

if __name__ == "__main__":
  unittest.main()
