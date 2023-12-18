# Overview

StarGate provides efficient and accurate detection of stars in sky images.
Given an image, StarGate returns a list of detected star centroids expressed
in image pixel coordinates.

Features:

* Employs localized thresholding to tolerate changes in background levels
  across the image.
* Adapts to different image exposure levels.
* Estimates noise in the image and adapts the star detection threshold
  accordingly.
* Automatically classifies and rejects hot pixels.
* Rejects trailed objects such as aircraft lights or satellites.
* Tolerates the presence of bright interlopers such as the moon, streetlights,
  or illuminated foreground objects.
* Simple function call interface with few parameters aside from the input
  image.
* Fast! On a Raspberry Pi 4B, the execution time per 1M image pixels is
  usually less than 10ms, even when several dozen stars are present in the
  image.

For more information, see the crate documentation in src/algorithm.rs.

# Usage

## Rust

See the sample program at src/bin/test_star_gate.rs for example usages of
StarGate called directly from Rust program logic.

## Python

There is (currently) no option to link StarGate with Python. Instead, a
microservice invocation API is provided using gRPC.

The src/bin/stargate_server.rs binary runs as a gRPC service, providing a simple
image &rarr; centroids RPC interface. See src/proto/star_gate.proto for the gRPC
service definition.

In python/stargate_client.py, a simple Python script demonstrates how to invoke
StarGate using its gRPC server. This program reads some test images, uses
StarGate to find centroids, and then plate-solves these centroids using Tetra3.

To try it out:

0. Install Python and all needed dependencies.
1. Install Tetra3.
2. With python/ as your current directory, execute:

        python -m grpc_tools.protoc -I../src/proto --python_out=. --pyi_out=. --grpc_python_out=. ../src/proto/star_gate.proto
3. Run the stargate gRPC server in background:

        cargo run --release --bin stargate-server &
4. Run the python program:

        python stargate_client.py
## Other languages

Any language that can be a gRPC client should be able to invoke StarGate.
