# Virtual camera (use the swapped feed in Zoom / Teams / Discord / OBS)

Status: implemented via [pyvirtualcam](https://github.com/letmaik/pyvirtualcam). **Not yet verified on
real hardware** — the development sandbox had no virtual-camera driver. Please report results.

## Windows
1. Install **OBS Studio 26 or newer** (https://obsproject.com). This registers the "OBS Virtual Camera" device.
   You do not need to keep OBS open.
2. In FaceFusion Live > LIVE SWAP, tick **Send to virtual camera** before pressing **Start Live Swap**.
3. In the other app choose **OBS Virtual Camera** as the camera.

If you see "Virtual camera backend not found", OBS is not installed (or is older than 26). Alternatively
install the Unity Capture driver.

## Using OBS directly (no virtual camera)
Add a **Window Capture** source in OBS pointing at the FaceFusion Live window, then crop to the preview.

## Linux
```
sudo apt install v4l2loopback-dkms
sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="FaceFusion Live"
```
