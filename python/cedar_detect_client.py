# Copyright (c) 2023 Steven Rosenthal smr@dt3.org
# See LICENSE file in root directory for license terms.

"""
This example loads the tetra3 default database and solves an image using CedarDetect's
centroid finding and Tetra3's solve_from_centroids().

Note: Requires PIL (pip install Pillow)
"""

import sys
sys.path.append('../..')

import numpy as np
from tetra3 import Tetra3
from PIL import Image
from pathlib import Path
from time import perf_counter as precision_timestamp

import grpc
from multiprocessing import shared_memory
import cedar_detect_pb2
import cedar_detect_pb2_grpc

def extract_centroids(stub, image):
    cr = cedar_detect_pb2.CentroidsRequest(
        input_image=image, sigma=8.0, max_size=5, return_binned=False,
        use_binned_for_star_candidates=True)
    return stub.ExtractCentroids(cr)

# Create instance and load default_database.
t3 = Tetra3('database_auto_30_10_002')

# Set up to make gRPC calls to CedarDetect centroid finder (it must be running
# already).
channel = grpc.insecure_channel('localhost:50051')
stub = cedar_detect_pb2_grpc.CedarDetectStub(channel)

# Use shared memory to make the gRPC calls faster. This works only when the
# client (this program) and the CedarDetect gRPC server are running on the same
# machine.
USE_SHMEM = True

# Path where test images are.
path = Path('../test_data/')
for impath in list(path.glob('*.jpg')) + list(path.glob('*.bmp')) + list(path.glob('*.png')):
    print('Solving for image at: ' + str(impath))
    with Image.open(str(impath)) as img:
        img = img.convert(mode='L')
        (width, height) = (img.width, img.height)
        image = np.asarray(img, dtype=np.uint8)

        centroids_result = None
        rpc_duration_secs = None
        if USE_SHMEM:
            # Using shared memory. The image data is passed in a shared memory
            # object, with the gRPC request giving the name of the shared memory
            # object.

            # Set up shared memory object for passing input image to CedarDetect.
            shmem = shared_memory.SharedMemory(
                "/cedar_detect_image", create=True, size=height*width)
            try:
                # Create numpy array backed by shmem.
                shimg = np.ndarray(image.shape, dtype=image.dtype, buffer=shmem.buf)
                # Copy image into shimg. This is much cheaper than passing image
                # over the gRPC call.
                shimg[:] = image[:]

                im = cedar_detect_pb2.Image(width=width, height=height,
                                            shmem_name=shmem.name)
                rpc_start = precision_timestamp()
                centroids_result = extract_centroids(stub, im)
                rpc_duration_secs = precision_timestamp() - rpc_start
            finally:
                shmem.close()
                shmem.unlink()
        else:
            # Not using shared memory. The image data is passed as part of the
            # gRPC request.
            im = cedar_detect_pb2.Image(width=width, height=height,
                                        image_data=image.tobytes())
            rpc_start = precision_timestamp()
            centroids_result = extract_centroids(stub, im)
            rpc_duration_secs = precision_timestamp() - rpc_start

        if len(centroids_result.star_candidates) == 0:
            print('Found no stars!')
        else:
            tetra_centroids = []  # List of (y, x).
            for sc in centroids_result.star_candidates:
                tetra_centroids.append((sc.centroid_position.y,
                                        sc.centroid_position.x))
            solved = t3.solve_from_centroids(tetra_centroids, (height, width),
                                             fov_estimate=11)
            algo_duration_secs = (centroids_result.algorithm_time.seconds +
                                  centroids_result.algorithm_time.nanos / 1e9)
            print('Centroids: %s. Solution: %s. %.2fms in centroiding (%.2fms rpc overhead)' %
                  (len(tetra_centroids),
                   solved,
                   rpc_duration_secs * 1000,
                   (rpc_duration_secs - algo_duration_secs) * 1000))
