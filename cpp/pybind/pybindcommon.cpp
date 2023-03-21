#include "pybindcommon.h"


const Eigen::MatrixXf tooldetectobject::pybindutils::GetEigenMatrixFromPyObject(const PyObject* pPyObj)  // let it copy in case of small matrix
{
    py::object pyObj = py::reinterpret_borrow<py::object>(py::cast(pPyObj));
    const py::buffer_info& matInfo = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(*const_cast<py::object*>(&pyObj)).request(false);
    float* data = static_cast<float*>(matInfo.ptr);
    std::vector<ssize_t> matrixShape = matInfo.shape;

    Eigen::MatrixXf outputMatrix;
    outputMatrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, matrixShape[0], matrixShape[1]);
    return outputMatrix;
}


void tooldetectobject::aaaaa(){
    std::cout << "fuck you!" << std::endl;
}