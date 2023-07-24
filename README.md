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
* Tolerates the presence of bright interlopers such as the moon or
  streetlights.
* Simple function call interface with few parameters aside from the input
  image.
* Fast! On a Raspberry Pi 4B, the execution time per 1M image pixels is
  around 5ms, even when several dozens stars are present in the image.

For more information, see the crate documentation in src/lib.rs.
