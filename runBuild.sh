#!/bin/bash

read -p "Do you want to build or source existing libraries?(b/s) " answer

if [[ "$answer" == [yY] ]]; then
  echo "Yes, continuing..."
  # install python dev headers
  sudo apt-get install python3-dev
  
  # install deps
  pip install -r requirementspy.txt
  
  cd python
  
  python3 -m build
  cd dist/
  if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("tool_eventdetectobjects"))'; then
      echo 'delete existing one'
      python3 -m pip uninstall tool_eventdetectobjects
  fi
  echo 'installing tool_eventdetectobjects...'
  python3 -m pip install tool_eventdetectobjects-0.0.1-py3-none-any.whl
else
  echo -e "(1) \nexport LD_LIBRARY_PATH=/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/build/tdoinstall/lib:\${LD_LIBRARY_PATH}"
  echo -e "(2) \nexport LD_LIBRARY_PATH=/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/3rdparty/log4cxx/install/lib:\${LD_LIBRARY_PATH}"
  echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
  echo -e "(3) \ncmake -DCMAKE_INSTALL_PREFIX=./tdoinstall/ .."
  echo -e "(4) \nmake"
  echo -e "(5) \nmake install"
fi

