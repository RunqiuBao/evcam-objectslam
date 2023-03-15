#ifndef __TDO_PYBINDCOMMON_H__
#define __TDO_PYBINDCOMMON_H__

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

namespace py = pybind11;

namespace pybindutils{

const Eigen::MatrixXf GetEigenMatrixFromPyObject(const py::object inputMatObj);

const Eigen::MatrixXf GetEigenMatrixFromPyObject(const py::object inputMatObj)  // let it copy in case of small matrix
{
    const py::buffer_info& matInfo = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(*const_cast<py::object*>(&inputMatObj)).request(false);
    float* data = static_cast<float*>(matInfo.ptr);
    std::vector<ssize_t> matrixShape = matInfo.shape;

    Eigen::MatrixXf outputMatrix;
    outputMatrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, matrixShape[0], matrixShape[1]);
    return outputMatrix;
}

};

#endif