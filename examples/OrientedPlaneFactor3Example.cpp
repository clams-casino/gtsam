#include <iostream>
#include <gtsam/slam/OrientedPlane3Factor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <math.h>
#include <memory>
#include <random>

using namespace gtsam;

// Define ground truth for pose to be estimated
constexpr double X = 0.5;
constexpr double Y = -0.3;
constexpr double Z = 0.4;

constexpr double YAW = (M_PI/180.0) * 45;
constexpr double PITCH = (M_PI/180.0) * -45;
constexpr double ROLL = (M_PI/180.0) * 32;

// Define measurement uncertainties
constexpr double SIGMA_N = 0.01;
constexpr double SIGMA_D = 0.05;


// Generate noise from a gaussian distribution
class GaussianNoiseGenerator
{
  using nd = std::normal_distribution<double>; 
  std::default_random_engine re;
  boost::scoped_ptr<nd> n_gen;
  boost::scoped_ptr<nd> d_gen;

  public:
    GaussianNoiseGenerator(double n_gen_std, double d_gen_std) 
    : n_gen(new nd(0.0, n_gen_std)),
      d_gen(new nd(0.0, d_gen_std))
    {
      re.seed(0); 
    }

    Vector3 operator() ()
    {
      Vector3 noise((*n_gen)(re), (*n_gen)(re), (*d_gen)(re));
      return noise;
    }
};


int main()
{
  GaussianNoiseGenerator noise_gen(0.01, 0.01);

  Values initial_estimate;
  NonlinearFactorGraph graph;

  // Start frame and "World" frame
  Symbol x0('x', 0);
  Pose3 x0_pose (Rot3::Ypr(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

  Symbol x1('x', 1);
  Pose3 x1_pose (Rot3::Ypr(YAW, PITCH, ROLL), Point3(X, Y, Z));


  // Three orthogonal planes
  Symbol l0('l', 0);
  OrientedPlane3 p0(Unit3(-1.0, 0.0, 0.0), 1.0);

  Symbol l1('l', 1);
  OrientedPlane3 p1(Unit3(0.0, -1.0, 0.0), 1.0);

  Symbol l2('l', 2);
  OrientedPlane3 p2(Unit3(0.0, 0.0, -1.0), 1.0);


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
  meas_sigmas << SIGMA_N, SIGMA_N, SIGMA_D;
  auto meas_covariance = noiseModel::Diagonal::Sigmas(meas_sigmas);

  // Insert measurement factors to the FIRST pose
  Vector meas_x0_l0 = p0.retract(noise_gen()).planeCoefficients();
  OrientedPlane3Factor x0_l0(meas_x0_l0, meas_covariance, x0, l0);
  graph.add(x0_l0);

  Vector meas_x0_l1 = p1.retract(noise_gen()).planeCoefficients();
  OrientedPlane3Factor x0_l1(meas_x0_l1, meas_covariance, x0, l1);
  graph.add(x0_l1);

  Vector meas_x0_l2 = p2.retract(noise_gen()).planeCoefficients();
  OrientedPlane3Factor x0_l2(meas_x0_l2, meas_covariance, x0, l2);
  graph.add(x0_l2);


  // Insert Measurement factors for the SECOND pose
  Vector meas_x1_l0 = p0.transform(x1_pose).retract(noise_gen()).planeCoefficients();
  OrientedPlane3Factor x1_l0(meas_x1_l0, meas_covariance, x1, l0);
  graph.add(x1_l0);

  Vector meas_x1_l1 = p1.transform(x1_pose).retract(noise_gen()).planeCoefficients();
  OrientedPlane3Factor x1_l1(meas_x1_l1, meas_covariance, x1, l1);
  graph.add(x1_l1);

  Vector meas_x1_l2 = p2.transform(x1_pose).retract(noise_gen()).planeCoefficients();
  OrientedPlane3Factor x1_l2(meas_x1_l2, meas_covariance, x1, l2);
  graph.add(x1_l2);


  // Solver
  LevenbergMarquardtParams params;
  params.setMaxIterations(100);
  
  LevenbergMarquardtOptimizer optimizer(graph, initial_estimate, params);
  

  // Get results
  Values results = optimizer.optimize();

  Pose3 x1_optimized = results.at<Pose3>(x1);

  std::cout << "Actual translation:" << std::endl;
  x1_pose.translation().print(); std::cout << std::endl << std::endl;
  std::cout << "Estimated translation:" << std::endl;
  x1_optimized.translation().print(); std::cout << std::endl << std::endl;

  std::cout << "Actual Rotation:" << std::endl;
  x1_pose.rotation().print(); std::cout << std::endl << std::endl;
  std::cout << "Estimated Rotation:" << std::endl;
  x1_optimized.rotation().print(); std::cout << std::endl << std::endl;


  return 0;

}
