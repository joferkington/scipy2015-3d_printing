import os
import binascii
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from cStringIO import StringIO

import numpy as np
import Image

def save_vrml(fig, output_filename):
    """
    Saves a Mayavi figure as shapeways-formatted VRML in a zip file.

    Parameters
    ----------
    fig : a Mayavi/mlab figure
    output_filename : string
    """
    _, fname = tempfile.mkstemp()
    fig.scene.save_vrml(fname)

    wrl_name = os.path.basename(output_filename).rstrip('.zip')
    vrml2shapeways(fname, output_filename, wrl_name)

    os.remove(fname)

def vrml2shapeways(filename, output_filename, wrl_name=None):
    """
    Un-embededs images from a vrml file and creates a zip archive with the
    images saved as .png's and the vrml file with links to the images.

    Parameters
    ----------
    filename : string
        The name of the input VRML file
    output_filename : string
        The filename of the zip archive that will be created.
    wrl_name : string or None (optional)
        The name of the VRML file in the zip archive. If None, this will be
        taken from *filename*.
    """
    if not output_filename.endswith('.zip'):
        output_filename += '.zip'

    with ZipFile(output_filename, 'w', ZIP_DEFLATED) as z:
        if wrl_name is None:
            wrl_name = os.path.basename(filename)
        if not wrl_name.endswith('.wrl'):
            wrl_name += '.wrl'

        outfile = StringIO()
        with open(filename, 'r') as infile:
            images = unembed_wrl_images(infile, outfile)
        z.writestr(wrl_name, outfile.getvalue())

        for fname, im in images.iteritems():
            outfile = StringIO()
            im.save(outfile, format='png')
            z.writestr(fname, outfile.getvalue())

def unembed_wrl_images(infile, outfile):
    """
    Converts embedded images in a VRML file to linked .png's.

    Parameters
    ----------
    infile : file-like object
    outfile: file-like object

    Returns
    -------
    images : a dict of filename : PIL Image pairs

    Notes:
    -----
    Should use a proper parser instead of just iterating line-by-line...
    """
    i = 1
    images = {}
    for line in infile:
        if 'texture' in line:
            data, width, height = read_texture_wrl(infile)
            image_filename = 'texture_{}.png'.format(i)
            im = ascii2image_wrl(data, width, height)
            line = '            texture ImageTexture {{ url ["{}"]}}'
            line = line.format(image_filename)
            images[image_filename] = im
            i += 1
        outfile.write(line)
    return images

def read_texture_wrl(infile):
    """
    Reads hexlified image data from the current position in a VRML file.
    """
    header = next(infile).strip().split()
    width, height, nbands = map(int, header[1:])

    data = []
    for line in infile:
        line = line.strip().split()
        for item in line:
            if item.startswith('0x'):
                data.append(item)
            else:
                return data, width, height

def ascii2image_wrl(data, width, height):
    """
    Converts hexlified data in VRML to a PIL image.
    """
    if len(data[0]) == 8:
        nbands = 3
    elif len(data[0]) == 10:
        nbands = 4
    else:
        raise ValueError('Unrecognized data type for image data')

    results = []
    for item in data:
        results.append(binascii.unhexlify(item[2:]))
    data = results
    data = ''.join(data)
    dat = np.fromstring(data, dtype=np.uint8).reshape(height, width, nbands)
    dat = np.roll(dat, nbands, -1)
    dat = np.flipud(dat)
    im = Image.fromarray(dat)
    return im
