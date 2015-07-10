import numpy as np
from mayavi import mlab
import mayavi.tools
from tvtk.api import tvtk

def scale(fig, ratio, reset=False):
    """
    Scales a Mayavi figure and resets the camera.

    Parameters
    ----------
    """
    for actor in fig.scene.renderer.actors:
        if reset:
            actor.scale = np.ones(3) * ratio
        else:
            actor.scale = actor.scale * ratio
    mayavi.tools.camera.view(distance='auto', focalpoint='auto', figure=fig)

def present(fig):
    """Makes a mayavi scene full screen and exits when "a" is pressed."""
    fig.scene.full_screen = True
    Closer(fig)
    mlab.show()

class Closer:
    """Close mayavi window when "a" is pressed."""
    def __init__(self, fig):
        self.fig = fig
        fig.scene.interactor.add_observer('KeyPressEvent', self)
    def __call__(self, vtk_obj, event):
        if vtk_obj.GetKeyCode() == 'a':
            self.fig.parent.close_scene(self.fig)

def texture(mesh, data=None, fname=None, clamp=1):
    """
    Apply a texture to a mayavi module (as produced by mlab.mesh, etc).

    Parameters:
    -----------
    mesh : 
        The mesh/surface to apply the texture to
    """
    if data is not None:
        img = image_from_array(data)
    elif fname is not None:
        img = tvtk.PNGReader(file_name=fname).output
    else:
        raise ValueError('You must specify either "data" or "fname".')

    t = tvtk.Texture(input=img, interpolate=1, edge_clamp=clamp)

    mesh.actor.enable_texture = True
    mesh.actor.tcoord_generator_mode = 'plane'
    mesh.actor.actor.texture = t
    mesh.actor.mapper.scalar_visibility = False

def image_from_array(ary):
    """ 
    Create a VTK image object that references the data in ary.  The array is
    either 2D or 3D with.  The last dimension is always the number of channels.
    It is only tested with 3 (RGB) or 4 (RGBA) channel images.

    Parameters
    ----------
    ary : 2D or 3D ndarray
        The texture data

    Returns
    -------
    img : a tvtk.ImageData instance

    Notes:
    ------
    Note: This works no matter what the ary type is (except probably
    complex...).  uint8 gives results that make since to me.  Int32 and Float
    types give colors that I am not so sure about.  Need to look into this...

    Taken from the mayavi examples.
    # Authors: Prabhu Ramachandran, Eric Jones
    # Copyright (c) 2006, Enthought, Inc.
    # License: BSD Style.
    """
    # Expects top of image as first row...
    ary = np.ascontiguousarray(np.flipud(ary))
    sz = ary.shape
    dims = len(sz)
    # create the vtk image data
    img = tvtk.ImageData()

    if dims == 2:
        # 1D array of pixels.
        img.extent = (0, sz[0]-1, 0, 0, 0, 0)
        img.dimensions = sz[0], 1, 1
        img.point_data.scalars = ary

    elif dims == 3:
        # 2D array of pixels.
        img.extent = (0, sz[0]-1, 0, sz[1]-1, 0, 0)
        img.dimensions = sz[1], sz[0], 1

        # create a 2d view of the array
        ary_2d = ary[:]
        ary_2d.shape = sz[0]*sz[1],sz[2]
        img.point_data.scalars = ary_2d

    else:
        raise ValueError, "ary must be 3 dimensional."

    return img


