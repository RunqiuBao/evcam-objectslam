# evcam-objectslam
Estimate camera trajectory by stereo object detection in event cameras:<br>
- Rapid camera motion tracking.
- Object-level compact map.

<table style="width: 100%;">
  <tr>
    <td>
      <img src="assets/seq0.gif" height="200px">
    </td>
    <td>
      <img src="assets/evobjslam2.gif" height="200px">
    </td>
  </tr>
</table>


## Installation
```
# Dependencies
git clone --recurse-submodules git@github.com:RunqiuBao/evcam-objectslam.git
(cd 3rdparty && ./build3rdparty.sh)
# Build this project
source runBuild.sh -a -i $(pwd)/tdoinstall
```

## Quick Start
```
# After installation
cd tdoinstall/bin/
rsync ../../assets/seq0.zip ./ && unzip seq0.zip
./slamRunner ./seq0/ ./seq0/sysconfig.json
```

## License
MIT license

## Our Supplementary Video
<video src="https://github.com/user-attachments/assets/8699a359-94d8-4c84-8ab8-90dd1f507612" width="600" controls="controls"></video>

