(cd ..
git submodule update --init --recursive)

read -p "DO you want to build log4cxx?(y/n)" answer
if [[ "$answer" == [yY] ]]; then
  # build log4cxx
  (cd log4cxx && mkdir build
  # install apr1
  sudo apt install libapr1 libaprutil1 libapr1-dev libaprutil1-dev -y
  cd build
  cmake -DCMAKE_INSTALL_PREFIX:STRING=/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/3rdparty/log4cxx/install/ ..
  make -j8
  make install)
fi

read -p "Do you want to build rapidjson?(y/n)" answer
if [[ "$answer" == [yY] ]]; then
  # build rapidjson
  (cd rapidjson
  rm -r build/
  mkdir -p build && cd build
  mkdir -p rapidjsoninstall
  cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=${PWD}/rapidjsoninstall/ -DRAPIDJSON_BUILD_DOC=OFF -DRAPIDJSON_BUILD_EXAMPLES=OFF -DRAPIDJSON_BUILD_TESTS=OFF ..
  make
  make install)
fi

read -p "Do you want to build opencv?(y/n)" answer
if [[ "$answer" == [yY] ]]; then
  # build opencv
  (cd opencv
  mkdir -p opencvinstall
  mkdir -p build && cd build
  cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=../opencvinstall/ -DWITH_TBB=ON -DBUILD_NEW_PYTHON_SUPPORT=ON -DWITH_V4L=ON -DWITH_OPENGL=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=0 -DWITH_CUBLAS=0 -DBUILD_TIFF=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.7.0/modules ..
  make -j8
  make install)
fi

read -p "Do you want to build pybind11?(y/n)" answer
if [[ "$answer" == [yY] ]]; then
  # build pybind11
  (cd pybind11
  mkdir -p build && cd build
  cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=./pybind11install/ ..
  make -j8
  make install)
fi

read -p "Do you want to build pangolin?(y/n)" answer
if [[ "$answer" == [yY] ]]; then
  # build pangolin
  (cd Pangolin
  cmake -B build -GNinja
  cmake --build build
  echo -e "-------- build python stuff --------"
  cmake --build build -t pypangolin_pip_install)
fi
 
