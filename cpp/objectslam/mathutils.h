#ifndef MATHUTILS_H
#define MATHUTILS_H

#include <Eigen/Core>
#include <cassert>


namespace mathutils {

template<typename MatrixType>
MatrixType TransformPoints(const MatrixType& transform, const MatrixType& points){
    // Note: points should be n*3 or 3*n shape, transform is 4*4 Eigen matrix
    assert(transform.rows() == 4 && transform.cols() == 4);
    assert(points.rows() == 3 || points.cols() == 3);
    if (points.rows() == 3){
        return (transform.block(0, 0, 3, 3) * points).colwise() + transform.block(0, 3, 3, 1).col(0);
    }
    else{
        return (transform.block(0, 0, 3, 3) * points.transpose()).colwise() + transform.block(0, 3, 3, 1).col(0);
    }
}

}

#endif  // MATHUTILS_H