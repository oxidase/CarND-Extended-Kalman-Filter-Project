#include <iostream>
#include <limits>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{

    VectorXd rmse(4);
    rmse << 0,0,0,0;

    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){
        // ... your code here
        VectorXd res = estimations[i]-ground_truth[i];
        res = res.array()*res.array();
        rmse += res;
    }

    //calculate the mean
    // ... your code here
    rmse /= estimations.size();

    //calculate the squared root
    return rmse.array().sqrt();
}

// https://stackoverflow.com/a/4609795
template <typename T> T sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    float d = px * px + py * py;
    float d12 = sqrt(d);
    float d32 = d12 * d;

    MatrixXd jac(3, 4);
    if (d12 < std::numeric_limits<float>::epsilon())
    {
        jac <<
            sgn(px), sgn(py), 0., 0.,
            0.,  0.,  0., 0.,
            0., 0., sgn(px), sgn(py);

    }
    else
    {
        jac <<
            px/d12, py/d12, 0., 0.,
            -py/d,  px/d,  0., 0.,
            py*(vx*py-vy*px)/d32, px*(vy*px-vx*py)/d32, px/d12, py/d12;
    }

    return jac;
}
