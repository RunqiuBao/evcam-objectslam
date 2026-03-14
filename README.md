# evcam-objectslam
Estimate camera trajectory by stereo object detection in event cameras:<br>
- Rapid camera motion tracking.
- Object-level compact map.

<img src="assets/evobjslam.gif" height="200"/>
<img src="assets/seq0.gif" height="200"/>


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