#!/bin/bash
readonly GREEN="\033[0;32m"
readonly YELLOW="\033[1;33m"
readonly WHITE="\033[1;37m"
readonly NC="\033[0m" # No Color

INSTALL_PATH="/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/tdoinstall"
LOG4CXX_LIB_PATH="/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/3rdparty/log4cxx/install/lib"
PYTHONPACKAGE_SRC_PATH="/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/python/src"
isAddPaths=false

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
    echo "-l,    --log4cxxlibpath           Define the path to the installed lib path of log4cxx. By default is '/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/3rdparty/log4cxx/install/lib'"
    echo "-p,    --pypackagepath            Define the path to the python package source dir. By default is '/home/runqiu/code/event_camera_repo/tools/tool_detectobjects/python/src'"
    echo "-a,    --addpaths                 Add shared library paths to LD_LIBRARY_PATH. By default is false."
    echo
    echo -e "${GREEN}Example usages: ${NC}"
    echo -e "${GREEN}  source runBuild.sh -a -i /home/runqiu/code/event_camera_repo/tools/tool_detectobjects/tdoinstall ${NC}"
    echo -e "${GREEN}  Note: source runBuild.sh to make change public, -a to add shared libs paths to LD_LIBRARY_PATH ${NC}"
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
            -l|--log4cxxlibpath)
                LOG4CXX_LIB_PATH="$2"
                shift 2
                ;;
            -s|--addpaths)
                isAddPaths=true
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

read -p "DO you want to install the python package 'tool_eventdetectobjects' with pip?(y/n)" answer
if [[ "$answer" == [yY] ]]; then
  (cd python
  python3 -m build
  cd dist/
  if python3 -c 'import pkgutil; exit(not pkgutil.find_loader("tool_eventdetectobjects"))'; then
      echo 'delete existing one'
      python3 -m pip uninstall tool_eventdetectobjects
  fi
  echo 'installing tool_eventdetectobjects...'
  python3 -m pip install tool_eventdetectobjects-0.0.1-py3-none-any.whl)
else
  echo 'skipping pip install 'tool_eventdetectobjects''
fi


# source paths
if $isAddPaths; then
    export LD_LIBRARY_PATH=${INSTALL_PATH}/lib:${LD_LIBRARY_PATH}
    export LD_LIBRARY_PATH=${LOG4CXX_LIB_PATH}:${LD_LIBRARY_PATH}
    echo -e "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    # add paths for python packages
    export PYTHONPATH="${PYTHONPACKAGE_SRC_PATH}:$PYTHONPATH"
else
    echo -e "Skipping source paths"
fi

(mkdir -p build/ && cd build
cmake \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
  -DINSTALLED_LOG4CXX_LIB_PATH=${LOG4CXX_LIB_PATH} \
  -DINSTALL_PATH_ALLTARGETS=${INSTALL_PATH} \
  ..
make
make install)
