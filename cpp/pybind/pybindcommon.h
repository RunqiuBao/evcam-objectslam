#ifndef __TDO_PYBINDCOMMON_H__
#define __TDO_PYBINDCOMMON_H__

#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/Core>

namespace py = pybind11;

namespace tooldetectobject{

void aaaaa();

namespace pybindutils{

const Eigen::MatrixXf GetEigenMatrixFromPyObject(const PyObject* pPyObj);

};

};

#endif