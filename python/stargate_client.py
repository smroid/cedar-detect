"""
This example loads the tetra3 default database and solves an image using StarGate's
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
import star_gate_pb2
import star_gate_pb2_grpc

def tiff_force_8bit(image):
    if image.format == 'TIFF' and image.mode == 'I;16':
        array = np.array(image)
        normalized = (array.astype(np.uint16)) / 256
        image = Image.fromarray(normalized.astype(np.uint8))
    return image

# Create instance and load default_database (built with max_fov=12 and the rest
# as default).
t3 = Tetra3('default_database')

# Set up to make gRPC calls to StarGate centroid finder (it must be running
# already).
channel = grpc.insecure_channel('localhost:50051')
stub = star_gate_pb2_grpc.StarGateStub(channel)

# Use shared memory to make the gRPC calls faster. This works only when the
# client (this program) and the StarGate gRPC server are running on the same
# machine.
USE_SHMEM = True

# Path where test images are.
path = Path('../test_data/')
for impath in path.glob('*.tiff'):
    print('Solving for image at: ' + str(impath))
    with Image.open(str(impath)) as img:
        (width, height) = (img.width, img.height)
        img_u8 = tiff_force_8bit(img)
        image = np.asarray(img_u8, dtype=np.uint8)

        centroids_result = None
        rpc_duration_secs = None
        if USE_SHMEM:
            # Using shared memory. The image data is passed in a shared memory
            # object, with the gRPC request giving the name of the shared memory
            # object.

            # Set up shared memory object for passing input image to StarGate.
            shmem = shared_memory.SharedMemory(
                "/stargate_image", create=True, size=height*width)
            try:
                # Create numpy array backed by shmem.
                shimg = np.ndarray(image.shape, dtype=image.dtype, buffer=shmem.buf)
                # Copy image into shimg. This is much cheaper than passing image
                # over the gRPC call.
                shimg[:] = image[:]

                im = star_gate_pb2.Image(width=width, height=height,
                                         shmem_name=shmem.name)
                cr = star_gate_pb2.CentroidsRequest(
                    input_image=im, sigma=8.0, max_size=5, return_binned=False,
                    use_binned_for_star_candidates=False)
                rpc_start = precision_timestamp()
                centroids_result = stub.ExtractCentroids(cr)
                rpc_duration_secs = precision_timestamp() - rpc_start
            finally:
                shmem.close()
                shmem.unlink()
        else:
            # Not using shared memory. The image data is passed as part of the
            # gRPC request.
            im = star_gate_pb2.Image(width=width, height=height,
                                     image_data=image.tobytes())
            cr = star_gate_pb2.CentroidsRequest(
                input_image=im, sigma=8.0, max_size=5, return_binned=False,
                use_binned_for_star_candidates=False)
            rpc_start = precision_timestamp()
            centroids_result = stub.ExtractCentroids(cr)
            rpc_duration_secs = precision_timestamp() - rpc_start

        tetra_centroids = []  # List of (y, x).
        for sc in centroids_result.star_candidates:
            tetra_centroids.append((sc.centroid_position.y,
                                    sc.centroid_position.x))
        solved = t3.solve_from_centroids(tetra_centroids, (height, width))
        algo_duration_secs = (centroids_result.algorithm_time.seconds +
                              centroids_result.algorithm_time.nanos / 1e9)
        print('Solution: %s. %.2fms in centroiding (%.2fms rpc overhead)' %
              (str(solved),
               rpc_duration_secs * 1000,
               (rpc_duration_secs - algo_duration_secs) * 1000))
