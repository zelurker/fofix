# -*- coding: ISO-8859-1 -*-

from __future__ import absolute_import
from builtins import object
from .MidiOutStream import MidiOutStream

class MidiInStream(object):

    """
    Takes midi events from the midi input and calls the apropriate
    method in the eventhandler object
    """

    def __init__(self, midiOutStream, device):

        """

        Sets a default output stream, and sets the device from where
        the input comes

        """

        if midiOutStream is None:
            self.midiOutStream = MidiOutStream()
        else:
            self.midiOutStream = midiOutStream


    def close(self):

        """
        Stop the MidiInstream
        """


    def read(self, time=0):

        """

        Start the MidiInstream.

        "time" sets timer to specific start value.

        """


    def resetTimer(self, time=0):
        """

        Resets the timer, probably a good idea if there is some kind
        of looping going on

        """
