""" Read Gimp .ggr gradient files.
    Ned Batchelder, http://nedbatchelder.com
    This code is in the public domain.
"""

__version__ = '1.0.20070915'

import colorsys, math

class GimpGradient:
    """ Read and interpret a Gimp .ggr gradient file.
    """
    def __init__(self, f=None):
        if f:
            self.read(f)
        
    class _segment:
        pass
    
    def read(self, f):
        """ Read a .ggr file from f (either an open file or a file path).
        """
        if isinstance(f, basestring):
            f = file(f)
        if f.readline().strip() != "GIMP Gradient":
            raise Exception("Not a GIMP gradient file")
        line = f.readline().strip()
        if not line.startswith("Name: "):
            raise Exception("Not a GIMP gradient file")
        self.name = line.split(": ", 1)[1]
        nsegs = int(f.readline().strip())
        self.segs = []
        for i in range(nsegs):
            line = f.readline().strip()
            seg = self._segment()
            (seg.l, seg.m, seg.r,
                seg.rl, seg.gl, seg.bl, _,
                seg.rr, seg.gr, seg.br, _,
             seg.fn, seg.space) = map(float, line.split())[:13]
            self.segs.append(seg)
            
    def color(self, x):
        """ Get the color for the point x in the range [0..1).
            The color is returned as an rgb triple, with all values in the range
            [0..1).
        """
        # Find the segment.
        for seg in self.segs:
            if seg.l <= x <= seg.r:
                break
        else:
            # No segment applies! Return black I guess.
            return (0,0,0)

        # Normalize the segment geometry.
        mid = (seg.m - seg.l)/(seg.r - seg.l)
        pos = (x - seg.l)/(seg.r - seg.l)
        
        # Assume linear (most common, and needed by most others).
        if pos <= mid:
            f = pos/mid/2
        else:
            f = (pos - mid)/(1 - mid)/2 + 0.5

        # Find the correct interpolation factor.
        if seg.fn == 1:   # Curved
            f = math.pow(pos, math.log(0.5) / math.log(mid));
        elif seg.fn == 2:   # Sinusoidal
            f = (math.sin((-math.pi/2) + math.pi*f) + 1)/2
        elif seg.fn == 3:   # Spherical increasing
            f -= 1
            f = math.sqrt(1 - f*f)
        elif seg.fn == 4:   # Spherical decreasing
            f = 1 - math.sqrt(1 - f*f);

        # Interpolate the colors
        if seg.space == 0:
            c = (
                seg.rl + (seg.rr-seg.rl) * f,
                seg.gl + (seg.gr-seg.gl) * f,
                seg.bl + (seg.br-seg.bl) * f
                )
        elif seg.space in (1,2):
            hl, sl, vl = colorsys.rgb_to_hsv(seg.rl, seg.gl, seg.bl)
            hr, sr, vr = colorsys.rgb_to_hsv(seg.rr, seg.gr, seg.br)

            if seg.space == 1 and hr < hl:
                hr += 1
            elif seg.space == 2 and hr > hl:
                hr -= 1

            c = colorsys.hsv_to_rgb(
                (hl + (hr-hl) * f) % 1.0,
                sl + (sr-sl) * f,
                vl + (vr-vl) * f
                )
        return c
    
if __name__ == '__main__':
    import sys, wx

    class GgrView(wx.Frame):
        def __init__(self, ggr, chunks):
            """ Display the ggr file as a strip of colors.
                If chunks is non-zero, then also display the gradient quantized
                into that many chunks.
            """
            super(GgrView, self).__init__(None, -1, 'Ggr: %s' % ggr.name)
            self.ggr = ggr
            self.chunks = chunks
            self.SetSize((600, 100))
            self.panel = wx.Panel(self)
            self.panel.Bind(wx.EVT_PAINT, self.on_paint)
            self.panel.Bind(wx.EVT_SIZE, self.on_size)

        def on_paint(self, event):
            dc = wx.PaintDC(self.panel)
            cw, ch = self.GetClientSize()
            if self.chunks:
                self.paint_some(dc, 0, 0, ch/2)
                self.paint_some(dc, self.chunks, ch/2, ch)
            else:
                self.paint_some(dc, 0, 0, ch)
                
        def paint_some(self, dc, chunks, y0, y1):
            cw, ch = self.GetClientSize()
            chunkw = 1
            if chunks:
                chunkw = (cw // chunks) or 1
            for x in range(0, cw, chunkw):
                c = map(lambda x:int(255*x), ggr.color(float(x)/cw))
                dc.SetPen(wx.Pen(wx.Colour(*c), 1))
                dc.SetBrush(wx.Brush(wx.Colour(*c), wx.SOLID))
                dc.DrawRectangle(x, y0, chunkw, y1-y0)
        
        def on_size(self, event):
            self.Refresh()

    app = wx.PySimpleApp()
    ggr = GimpGradient(sys.argv[1])
    chunks = 0
    if len(sys.argv) > 2:
        chunks = int(sys.argv[2])
    f = GgrView(ggr, chunks)
    f.Show()
    app.MainLoop()
