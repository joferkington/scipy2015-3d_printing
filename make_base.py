import tempfile

from mayavi import mlab
import Image
import numpy as np
import geoprobe
import scipy.ndimage
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import utils

def main():
    vol, top = load_data()
    downsample = 3

    fig = mlab.figure(bgcolor=(1,1,1))

    x, y, z = hor2xyz(top, vol, downsample)
    build_sides(vol, x, y, z, vol.nz)

    # Build top
    seafloor = top_texture(top, vol)
    top_mesh = mlab.mesh(x, y, z)
    texture(top_mesh, np.flipud(seafloor.T), cm.gray)

    build_base(x, y, z, vol)
    utils.present(fig)

def load_data():
    top = geoprobe.horizon('data/seismic/Horizons/channels.hzn')
    vol = geoprobe.volume('data/seismic/Volumes/example.vol')
    vol.data = vol.load()

    top = smooth_horizon(top)
    return vol, top

def build_base(x, y, z, vol, wall_thick=5):
    z0 = -vol.nz * np.ones_like(z)
    sl = np.s_[wall_thick:-wall_thick, wall_thick:-wall_thick]
    z0[sl] = z[sl] - wall_thick
    return mlab.mesh(x, y, z0, color=(1, 1, 1))

def build_sides(vol, x, y, z, base, zbase=None):
    for axis in [0, 1]:
        for val in [0, -1]:
            slices = [slice(None), slice(None), slice(None, base)]
            slices[axis] = val
            slices = tuple(slices)
            build_side(vol, slices, x, y, z, zbase)

def build_side(vol, sl, x, y, z, zbase=None):
    data = vol.data
    full = sl
    sl = sl[:2]
    z0, x0, y0 = z[sl], x[sl], y[sl]

    if zbase is None:
        base = -np.arange(data.shape[2])[full[-1]].max()
        base = base * np.ones_like(z0)
    else:
        base = zbase[sl]

    z0 = np.vstack([z0, base])
    x0 = np.vstack([x0, x0])
    y0 = np.vstack([y0, y0])
    mesh = mlab.mesh(x0, y0, z0)

    sl = slice(-z0.max(), -z0.min(), full[-1].step)
    full = tuple([full[0], full[1], sl])
    dat = data[full].T

    cmap = geoprobe.colormap('data/seismic/Colormaps/brown_black').as_matplotlib
    texture(mesh, dat, cmap)
    return mesh

def hor2xyz(hor, vol, downsample=1):
    z = hor.grid
    z = vol.model2index(z, axis='z', int_conversion=False)
    z = -z.T

    ds = downsample
    y, x = np.mgrid[:z.shape[0], :z.shape[1]]
    x,y,z = x[::ds, ::ds], y[::ds, ::ds], z[::ds, ::ds]
    return x, y, z

def top_texture(hor, vol):
    """RMS Amplitude Extraction on Bottom Horizon."""
    chan = geoprobe.utilities.extractWindow(hor, vol, 0, 4)
    chan = (chan.astype(float) - 128.0)**2
    chan = np.sqrt(chan.mean(axis=-1))
    return chan

def texture(mesh, data, cmap, vmin=None, vmax=None):
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    dat = scipy.ndimage.zoom(data, 3)
    norm = mcolors.Normalize(vmin, vmax)
    rgba = cmap(norm(dat))
    rgba = (255 * rgba).astype(np.uint8)
    im = Image.fromarray(rgba).convert('RGB')

    # Evil, ugly hack. Still don't understand why RGB texturing isn't working
    # correctly without bringing in an image. Fix later!
    _, fname = tempfile.mkstemp()
    with open(fname, 'w') as f:
        im.save(f, 'PNG')

    utils.texture(mesh, fname=fname)

def smooth_horizon(hor):
    z = hor.grid
    z = scipy.ndimage.median_filter(z.astype(float), 4)
    z = scipy.ndimage.gaussian_filter(z, 1.5)
    xmin, xmax, ymin, ymax = hor.grid_extents
    y, x = np.mgrid[ymin:ymax+1, xmin:xmax+1]
    return geoprobe.horizon(x.flatten(), y.flatten(), z.flatten())

main()
