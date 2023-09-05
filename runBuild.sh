#!/bin/bash
readonly GREEN="\033[0;32m"
readonly YELLOW="\033[1;33m"
readonly WHITE="\033[1;37m"
readonly NC="\033[0m" # No Color

INSTALL_PATH="/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/tdoinstall"
LOG4CXX_LIB_PATH="/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/3rdparty/log4cxx/install/lib"
RAPIDJSON_LIB_PATH="/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/3rdparty/rapidjson/rapidjsoninstall/lib"
OPENCV4_LIB_PATH="/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/3rdparty/opencv/opencvinstall/lib"
PANGOLIN_LIB_PATH="/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/3rdparty/Pangolin/pangolininstall/lib"
PYTHONPACKAGE_SRC_PATH="/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/python/src"
isAddPaths=false
isInstallPythonPackage=false

# ================================================================================================================================================== #
# ================================================================= Displays usage ================================================================= #
# ================================================================================================================================================== #
function display_help() {
    echo "Build the executables and set environment variables for testing this library (tool_detectobjects)."
    echo -e "${YELLOW}Note: assuming all 3rdparty libs are built in ./3rdparty; assume python3.8 exists. ${NC}"
    echo
    echo "Syntax: runBuild.bash [-h] [-i installpath] "
    echo "options:"
    echo "-h,    --help                     Display this help"
    echo "-i,    --installpath              Define the path to install the library. By default is '/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/tdoinstall'"
    echo "       --log4cxxlibpath           Define the path to the installed lib path of log4cxx. By default is ${LOG4CXX_LIB_PATH}"
    echo "       --rapidjsonlibpath         Define the path to the installed lib path of rapidjson. By default is ${RAPIDJSON_LIB_PATH}"
    echo "       --pangolinlibpath          Define the path to the installed lib path of Pangolin. By default is ${PANGOLIN_LIB_PATH}"
    echo "       --opencv4libpath           Define the path to the installed lib path of opencv4. By default is ${OPENCV4_LIB_PATH}"
    echo "       --pypackagepath            Define the path to the python package source dir. By default is ${PYTHONPACKAGE_SRC_PATH}"
    echo "       --installpythonpackage     Whether to install the python package or not. By default False"
    echo "-a,    --addpaths                 Add shared library paths to LD_LIBRARY_PATH. By default is false."
    echo
    echo -e "${GREEN}Example usages: ${NC}"
    echo -e "${GREEN}  source runBuild.sh -a --installpythonpackage -i /home/runqiu/code/event_camera_repo/tools/tool_detectobjects/tdoinstall ${NC}"
    echo -e "${GREEN}  Note: source runBuild.sh at the first build to make addPaths global; -a to add shared libs paths to LD_LIBRARY_PATH ${NC}"
    echo -e "${GREEN}  Note: If you want to run executable in one terminal, need to source this script in that terminal. ${NC}"

}

# ================================================================================================================================================== #
# ================================================================ Parses arguments ================================================================ #
# ================================================================================================================================================== #
function parse_args() {
    LONGOPTS=installpath:,log4cxxlibpath:,addpaths,help
    OPTIONS=i:l:ah
    PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        exit 2  # if parsing failed, exit.
    fi

    eval set -- "$PARSED"

    while true; do
        case "$1" in
            -h|--help)
                display_help
                exit 0
                ;;
            -i|--installpath)
                INSTALL_PATH="$2"
                shift 2
                ;;
            -a|--addpaths)
                isAddPaths=true
                shift 1  # only need to shift by 1 as this is a store_true arguments.
                ;;
            --log4cxxlibpath)
                LOG4CXX_LIB_PATH="$2"
                shift 2
                ;;
            --rapidjsonlibpath)
                RAPIDJSON_LIB_PATH="$2"
                shift 2
                ;;
            --pangolinlibpath)
                PANGOLIN_LIB_PATH="$2"
                shift 2
                ;;
            --opencv4libpath)
                OPENCV4_LIB_PATH="$2"
                shift 2
                ;;
            --pypackagepath)
                PYTHONPACKAGE_SRC_PATH="$2"
                shift 2
                ;;
            --installpythonpackage)
                isInstallPythonPackage=true
                shift 1  # only need to shift by 1 as this is a store_true arguments.
                ;;
            --)
                shift
                break
                ;;
            *)
                display_help
                exit 3
                ;;
        esac
    done
}

parse_args $@

# install python dev headers
sudo apt-get install python3-dev

# install deps
pip install -r requirementspy.txt

# read -p "DO you want to install the python package 'tool_eventdetectobjects' with pip?(y/n)" answer
# if [[ "$answer" == [yY] ]]; then
if $isInstallPythonPackage; then
  (cd python
  python3 -m build
  cd dist/
  if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("tool_eventdetectobjects"))'; then
      echo 'delete existing one'
      python3 -m pip uninstall tool_eventdetectobjects
  fi
  echo '[Install python package] installing tool_eventdetectobjects...'
  python3 -m pip install tool_eventdetectobjects-0.0.1-py3-none-any.whl)
else
  echo '[Install python package] skipping pip install 'tool_eventdetectobjects''
fi


# source paths
if $isAddPaths; then
    export LD_LIBRARY_PATH=${INSTALL_PATH}/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${LOG4CXX_LIB_PATH}:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${OPENCV4_LIB_PATH}:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${PANGOLIN_LIB_PATH}:${LD_LIBRARY_PATH}
    echo -e "[Add lib path & python src path] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    # add paths for python packages
    export PYTHONPATH="${PYTHONPACKAGE_SRC_PATH}:$PYTHONPATH"
else
    echo -e "[Add lib path & python src path] Skipping add paths."
fi

(mkdir -p build/ && cd build
cmake \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
  -DINSTALLED_LOG4CXX_LIB_PATH=${LOG4CXX_LIB_PATH} \
  -DINSTALLED_RAPIDJSON_LIB_PATH=${RAPIDJSON_LIB_PATH} \
  -DINSTALLED_PANGOLIN_LIB_PATH=${PANGOLIN_LIB_PATH} \
  -DINSTALLED_OPENCV4_LIB_PATH=${OPENCV4_LIB_PATH} \
  -DINSTALL_PATH_ALLTARGETS=${INSTALL_PATH} \
  ..
make
make install)
