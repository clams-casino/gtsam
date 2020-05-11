#include <iostream>
#include <gtsam/slam/OrientedPlane3Factor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <math.h>


using namespace gtsam;

int main()
{
  // Start frame and "World" frame
  Symbol x0('x', 0);
  Pose3 x0_pose (Rot3::Ypr(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

  Symbol x1('x', 1);
  Pose3 x1_pose (Rot3::Ypr(M_PI_4, 0.0, 0.0), Point3(0.0, 0.0, 0.0));


  // Three orthogonal planes
  Symbol l0('l', 0);
  OrientedPlane3 p0(Unit3(-1.0, 0.0, 0.0), 1.0);

  Symbol l1('l', 1);
  OrientedPlane3 p1(Unit3(0.0, -1.0, 0.0), 1.0);

  Symbol l2('l', 2);
  OrientedPlane3 p2(Unit3(0.0, 0.0, -1.0), 1.0);


  Values initial_estimate;
  NonlinearFactorGraph graph;

  // Prior on the first pose
  Vector prior_sigma(6);
  prior_sigma << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001;
  PriorFactor<Pose3> x0_prior(x0, x0_pose,
      noiseModel::Diagonal::Sigmas(prior_sigma));
  initial_estimate.insert(x0, x0_pose);
  graph.add(x0_prior);

  // Initialize the second pose as the first pose
  initial_estimate.insert(x1, x0_pose);

  // Set initial guesses for planes to their true values
  initial_estimate.insert(l0, p0);
  initial_estimate.insert(l1, p1);
  initial_estimate.insert(l2, p2);


  Vector meas_sigmas(3);
  meas_sigmas << 0.1, 0.1, 0.1;
  auto meas_noise = noiseModel::Diagonal::Sigmas(meas_sigmas);

  // Insert measurement factors to the FIRST pose
  Vector meas_x0_l0 = p0.planeCoefficients();
  OrientedPlane3Factor x0_l0(meas_x0_l0, meas_noise, x0, l0);
  graph.add(x0_l0);

  Vector meas_x0_l1 = p1.planeCoefficients();
  OrientedPlane3Factor x0_l1(meas_x0_l1, meas_noise, x0, l1);
  graph.add(x0_l1);

  Vector meas_x0_l2 = p2.planeCoefficients();
  OrientedPlane3Factor x0_l2(meas_x0_l2, meas_noise, x0, l2);
  graph.add(x0_l2);


  // Insert Measurement factors for the SECOND pose
  Vector meas_x1_l0 = p0.transform(x1_pose).planeCoefficients();
  OrientedPlane3Factor x1_l0(meas_x1_l0, meas_noise, x1, l0);
  graph.add(x1_l0);

  Vector meas_x1_l1 = p1.transform(x1_pose).planeCoefficients();
  OrientedPlane3Factor x1_l1(meas_x1_l1, meas_noise, x1, l1);
  graph.add(x1_l1);

  Vector meas_x1_l2 = p2.transform(x1_pose).planeCoefficients();
  OrientedPlane3Factor x1_l2(meas_x1_l2, meas_noise, x1, l2);
  graph.add(x1_l2);


  // std::cout << meas_x1_l0 << std::endl;
  // std::cout << meas_x1_l1 << std::endl;
  // std::cout << meas_x1_l2 << std::endl;

  LevenbergMarquardtParams params;
  params.setMaxIterations(100);
  
  LevenbergMarquardtOptimizer optimizer(graph, initial_estimate, params);
  
  Values results = optimizer.optimize();

  Pose3 x1_optimized = results.at<Pose3>(x1);

  std::cout << x1_optimized << std::endl;

  return 0;

}
