from mayavi import mlab
import numpy as np
import geoprobe
import scipy.ndimage

hor = geoprobe.horizon('data/seismic/Horizons/channels.hzn')
vol = geoprobe.volume('data/seismic/Volumes/example.vol')
data = vol.load()

# Because we're not working in inline/crossline, convert to "pixel" coords.
z = hor.grid
z = vol.model2index(z, axis='z', int_conversion=False)

# Clip out some spikes and smooth the surface a bit...
z = scipy.ndimage.median_filter(z, 4)
z = scipy.ndimage.gaussian_filter(z, 1)

# Quirks due to the way hor.grid is defined...
mlab.surf(np.arange(z.shape[0]), np.arange(z.shape[1]), -z.T,
          colormap='gist_earth')

source = mlab.pipeline.scalar_field(data)
source.spacing = [1, 1, -1]

for axis in ['x', 'y']:
    plane = mlab.pipeline.image_plane_widget(source, 
                                    plane_orientation='{}_axes'.format(axis),
                                    slice_index=100, colormap='gray')
    plane.module_manager.scalar_lut_manager.reverse_lut = True

mlab.show()
