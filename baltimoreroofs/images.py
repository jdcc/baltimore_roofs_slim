import glob
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union, Optional

import h5py
import numpy as np
import pyproj
from psycopg2 import sql
import rasterio
import rasterio.mask
import rasterio.warp
import shapely
from shapely.ops import transform
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from tqdm.auto import tqdm

from .utils import split_blocklot

logger = logging.getLogger(__name__)


class ImageCropper:
    def __init__(self, db, root_path: str, file_ext="tif"):
        self._db = db
        self.root_path = Path(root_path)
        self._filenames: list[str] = []
        self._file_bounds: dict[str, Polygon] = {}
        self._file_ext = file_ext
        self._load_images()

    def _load_images(self):
        """Load image filename and bounds from disk into memory.

        This does not store the actual images in memory (so we don't run out of memory).
        """
        self._filenames = self._get_image_filenames(self.root_path, self._file_ext)
        self._file_bounds = {}
        projections = set()

        for filename in tqdm(self._filenames, desc="Reading bounds", smoothing=0):
            raster = rasterio.open(filename)
            projections.add(raster.crs)
            self._file_bounds[filename] = raster_to_bounds(raster)
        logger.debug("Loaded %s images", len(self._filenames))
        logger.debug("Projections found: %s", "\n".join(str(p) for p in projections))

    @staticmethod
    def _get_image_filenames(path, file_ext):
        return glob.glob(str(path / f"*Ortho*.{file_ext}"))

    def _find_images_for_shape(
        self, shape: Union[Polygon, MultiPolygon]
    ) -> list[rasterio.DatasetReader]:
        """Find all images for a given shape

        Args:
            shape (Union[Polygon, MultiPolygon]): The Shapely shape for which to find
                images.

        Returns:
            A list of rasterio images
        """
        shape_images = []
        for image, bounds in self._file_bounds.items():
            # TODO Handle block lots that break across tile bounds
            if bounds.contains(shape):
                shape_images.append(rasterio.open(image))
        return shape_images

    def images_for_blocklot(self, blocklot: str) -> list[rasterio.DatasetReader]:
        """Return image files for a given blocklot.

        Args:
            blocklot (str): The blocklot id of the tax parcel

        Returns:
            A list of rasterio datasets (images) that cover the given blocklot
        """
        return self._find_images_for_shape(blocklot_to_shape(self._db, blocklot))

    def pixels_for_blocklot(
        self, blocklot: str, *to_shape_args, **to_shape_kwargs
    ) -> Optional[np.ndarray]:
        """Return the pixel values for an aerial image of a blocklot.

        Args:
            blocklot (str): The blocklot id of the tax parcel
            year (int): The year of aerial data

        Returns:
            np.ndarray: The pixel values for the first aerial image.
        """ """"""
        shape = blocklot_to_shape(self._db, blocklot, *to_shape_args, **to_shape_kwargs)
        shape_tiles = self.images_for_blocklot(blocklot)
        if len(shape_tiles) == 0:
            logging.debug("Shape for blocklot {} not found in tiles".format(blocklot))
            return None
        # TODO Handle block lots that break across tile bounds
        tile = shape_tiles[0]
        return self._mask_image_to_shape(tile, shape)

    def _mask_image_to_shape(
        self, tile: rasterio.DatasetReader, shape: Polygon
    ) -> np.ndarray:
        """Given an image and a shape, return just the pixels for the shape.

        Args:
            tile (rasterio.DatasetReader): The image
            shape (Polygon): The shape

        Returns:
            np.ndarray: The pixel values for the masked region
        """
        try:
            mask, _, window = rasterio.mask.raster_geometry_mask(
                tile, [reproject_shape(shape, "EPSG:4326", tile.crs)], crop=True
            )
        except ValueError as e:
            logger.warning("Error pulling out shape from image: %s", e)
            return np.ndarray([])
        # TODO clean this up
        image = tile.read([1, 2, 3], window=window)
        i = image.astype(np.float16)
        i[:, mask] = np.nan
        return np.transpose(
            i,
            [1, 2, 0],
        )

    def write_h5py(self, output_file: Path, blocklots: list[str], overwrite=False):
        f = h5py.File(str(output_file), "a")
        without_pixels = set()
        for this_blocklot in tqdm(blocklots, desc="Cropping", smoothing=0):
            block, lot = split_blocklot(this_blocklot)
            key = f"{block}/{lot}"
            if key not in f or overwrite:
                pixels = self.pixels_for_blocklot(this_blocklot)
                if pixels is None:
                    without_pixels.add(this_blocklot)
                    continue
                if key in f and overwrite:
                    del f[key]
                f.create_dataset(key, data=pixels)
        if len(without_pixels) > 0:
            logger.warn("Images were not created for %s blocklots", len(without_pixels))
            with NamedTemporaryFile("w", delete=False) as f:
                f.write("\n".join(f"{bl}" for bl in without_pixels))
                logger.warn("Blocklots without images written to %s", f.name)
        f.close()


def raster_to_bounds(raster: rasterio.DatasetReader) -> Polygon:
    """Return the bounds of the raster as a Shapely shape.

    Reprojects everything to EPSG:4326.

    Args:
        raster: A dataset (e.g. GeoTIFF) read in with rasterio.

    Returns:
        The geographic bounds of the dataset as a polygon
    """
    bounds = raster.bounds
    left, bottom, right, top = rasterio.warp.transform_bounds(
        raster.crs, "EPSG:4326", *bounds
    )
    return Polygon([(left, top), (right, top), (right, bottom), (left, bottom)])


class RecordNotFoundError(Exception):
    """We couldn't find a corresponding record in the database."""

    pass


def fetch_blocklot_geometry(db, blocklot: str) -> str:
    """Get the geometry of the blocklot from the database.

    Args:
        blocklot: The block lot in "bbbb lll" form.

    Returns:
        The geometry of the block as a well-known-binary hex string.

        The database stores these geometries in EPSG:2248, which is the
        projection for Maryland, and uses feet as its reference unit.
    """
    query = sql.SQL("SELECT shape FROM {tpa_table} WHERE blocklot = %s").format(
        tpa_table=db.TPA
    )
    results = db.run_query(query, (blocklot,))
    if len(results) == 0:
        raise RecordNotFoundError("No records found for blocklot {}".format(blocklot))
    return results[0][0]


def wkb_to_shape(
    wkb: str, buffer: int = 0, simplify: bool = True, hull: bool = True
) -> Polygon:
    """Turn a well-known-binary represenation of a geometry into a shape

    Args:
        wkb: A Well-Known-Binary representation of a geometry
        buffer: Make the shape a this many units bigger than the geometry all the way
            around. The unit is defined by the coordinate system of the geometry.
        simplify: Whether or not to simplify the resulting shape
        hull: Whether or not to return just the hull of the shape, i.e. remove the holes

    Returns:
        The Shapely polygon of the geometry
    """
    shape = shapely.from_wkb(wkb)
    # Buffering even with 0 helps clean up shape
    shape = shape.buffer(buffer)
    if simplify:
        shape = shape.simplify(2)
    if hull:
        if isinstance(shape, shapely.geometry.multipolygon.MultiPolygon):
            shape = shape.convex_hull
        else:
            shape = shapely.geometry.polygon.Polygon(shape.exterior)
    return shape


def blocklot_to_shape(db, blocklot: str, *to_shape_args, **to_shape_kwargs) -> Polygon:
    """Return the shape of a given blocklot

    Blocklot coordinates are in EPSG:2248 projection.

    Args:
        blocklot: The block lot in "bbbb lll" form.
        *to_shape_args: Arguments to pass to shape creation (`wkb_to_shape`)
        **to_shape_kwargs: Keyword arguments to pass to shape creation (`wkb_to_shape`)

    Returns:
        The geometry of the block as a Shapely polygon
    """
    return reproject_shape(
        wkb_to_shape(
            fetch_blocklot_geometry(db, blocklot), *to_shape_args, **to_shape_kwargs
        ),
        "EPSG:2248",
        "EPSG:4326",
    )


def reproject_shape(shape, src_crs, dst_crs):
    src_proj = pyproj.CRS(src_crs)
    dst_proj = pyproj.CRS(dst_crs)

    project = pyproj.Transformer.from_crs(src_proj, dst_proj, always_xy=True).transform
    return transform(project, shape)


def fetch_image_from_hdf5(blocklot, f=None, hdf5_filename=None):
    assert (
        f is not None or hdf5_filename is not None
    ), "Must pass either a file handle or a filename"
    if hdf5_filename:
        f = h5py.File(hdf5_filename)
    block, lot = split_blocklot(blocklot)
    data = f[f"{block}/{lot}"]
    arr = np.empty_like(data)
    data.read_direct(arr)
    if hdf5_filename:
        f.close()
    return arr


def count_datasets_in_hdf5(h5file):
    def is_dataset(name):
        obj = h5file[name]
        if isinstance(obj, h5py.Dataset):
            n_datasets[0] += 1

    n_datasets = [0]
    h5file.visit(is_dataset)

    return n_datasets[0]


def blocklots_in_hdf5(file: h5py.File):
    blocklots = []
    for block in file.keys():
        for lot in file[block].keys():
            blocklots.append(f"{block:5}{lot}")
    return blocklots
