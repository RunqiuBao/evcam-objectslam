# evcam-objectslam
Estimate camera trajectory with stereo object detection in event cameras:<br>
<p align="left">
  <img src="assets/evobjslam.gif" height="200px"/>
  <img src="assets/seq0.gif" height="200px"/>
</p>

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