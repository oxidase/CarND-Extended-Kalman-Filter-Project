#include "kalman_filter.h"
#include <iostream>
#include <cmath>

// https://stackoverflow.com/a/11126083
// Bring the 'difference' between two angles into [-pi; pi]
template <int K, typename T>
T normalize(T rad) {
  // Copy the sign of the value in radians to the value of pi.
  T signed_pi = std::copysign(M_PI, rad);
  // Set the value of difference to the appropriate signed value between pi and -pi.
  rad = std::fmod(rad + K * signed_pi,(2 * M_PI)) - K * signed_pi;
  return rad;
}

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    UpdateRes(z - H_ * x_);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    float px = x_(0), py = x_(1), vx = x_(2), vy = x_(3);
    float rho = std::sqrt(px * px + py * py);
    float theta = std::atan2(py, px);
    float rho_dot = (px * vx + py * vy) / rho;

    VectorXd y(3);
    y << z(0)-rho, normalize<1>(z(1)-theta), z(2)-rho_dot;

    UpdateRes(y);
}


void KalmanFilter::UpdateRes(const VectorXd &y)
{
    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    MatrixXd K = P_ * H_.transpose() * S.inverse();

		// new state
    x_ = x_ + K * y;
    P_ = P_ - K * H_ * P_;
}
