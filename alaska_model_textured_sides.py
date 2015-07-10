import numpy as np
from osgeo import gdal
import mayavi.mlab as mlab

import utils
import shapeways_io

gdal.UseExceptions()

def main():
    z, x, y = read('data/alaska/clipped_elev.tif')
    rgb, _, _ = read('data/alaska/clipped_map.tif')
    rgb = np.swapaxes(rgb.T, 0, 1)

    fig = mlab.figure()

    surf = mlab.mesh(x[::2, ::2], y[::2, ::2], z[::2, ::2])
    utils.texture(surf, rgb)
    build_sides(x, y, z, -1000)
    build_bottom(x, y, z, -1000)

    
    utils.scale(fig, (1, 1, 2.5))
    utils.scale(fig, 0.00001)
#    shapeways_io.save_vrml(fig, 'models/alaska_textured_sides.zip')
    utils.present(fig)

def read(filename):
    ds = gdal.Open(filename)
    elev = ds.ReadAsArray()

    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
    i, j = np.mgrid[:elev.shape[0], :elev.shape[1]]
    x = x0 + dx * j + dxdy * i
    y = y0 + dy * i + dydx * j

    return ds.ReadAsArray(), x, y

def build_bottom(x, y, z, level):
    i = [-1, -1, 0, 0]
    j = [0, -1, 0, -1]
    corners = lambda item: item[i, j].reshape(2, 2)
    bottom = mlab.mesh(corners(x), corners(y), level * np.ones((2,2)))

    utils.texture(bottom, fname='data/alaska/bottom_annotated.png', clamp=False)
    return bottom

def build_sides(x, y, z, level):
    images = ['left_annotated_halo.png', 'right_annotated_halo.png',
              'back_annotated_halo.png', 'front_annotated_halo.png']

    slices = [np.s_[:,0], np.s_[:,-1], np.s_[0,:], np.s_[-1,:]]

    for sl, im in zip(slices, images):
        image = 'data/alaska/' + im
        x_side = np.vstack([x[sl], x[sl]])
        y_side = np.vstack([y[sl], y[sl]])
        z_side = np.vstack([z[sl], level * np.ones_like(z[sl])])

        mesh = mlab.mesh(x_side, y_side, z_side)
        utils.texture(mesh, fname=image)

main()
