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

// Define ACTUAL measurement uncertainties
constexpr double SIGMA_N = 0.01;
constexpr double SIGMA_D = 0.05;

// Define FACTOR measurement uncertainties
constexpr double F_SIGMA_N = 0.01;
constexpr double F_SIGMA_D = 0.05;


// Example B, noise for perturbed landmark retraction
constexpr double PL_N = 0.5; 
constexpr double PL_D = 0.5;

// Generate noise from a gaussian distribution
class GaussianNoiseGenerator
{
  using nd = std::normal_distribution<double>; 
  std::default_random_engine re_;
  boost::scoped_ptr<nd> n_gen_;
  boost::scoped_ptr<nd> d_gen_;

  public:
    GaussianNoiseGenerator(double n_gen_std, double d_gen_std) 
    : n_gen_(new nd(0.0, n_gen_std)),
      d_gen_(new nd(0.0, d_gen_std))
    {
      re_.seed(0); 
    }

    Vector3 operator() ()
    {
      Vector3 noise((*n_gen_)(re_), (*n_gen_)(re_), (*d_gen_)(re_));
      return noise;
    }
};


struct CreatePlaneLandmark
{
  using PlaneLandmark = std::pair<Symbol, OrientedPlane3>;

  PlaneLandmark operator() (const Unit3& n, double d)
  {
    return std::make_pair(Symbol('l', landmark_count_++), OrientedPlane3(n, d));
  }

  PlaneLandmark operator() (OrientedPlane3 p)
  {
    return std::make_pair(Symbol('l', landmark_count_++), p);
  }

  private:
    int landmark_count_ = 0;
};


void exampleA()
{
  std::cout << std::endl
  << "****************** Example A ******************" 
  << std::endl << std::endl;

  GaussianNoiseGenerator meas_noise_gen(SIGMA_N, SIGMA_D);

  Values initial_estimate;
  NonlinearFactorGraph graph;

  // Start frame and World frame
  Symbol x0('x', 0);
  Pose3 x0_pose (Rot3::Ypr(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

  Symbol x1('x', 1);
  Pose3 x1_pose (Rot3::Ypr(YAW, PITCH, ROLL), Point3(X, Y, Z));

  // Prior on the first pose
  Vector prior_sigma(6);
  prior_sigma << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001;
  PriorFactor<Pose3> x0_prior(x0, x0_pose,
      noiseModel::Diagonal::Sigmas(prior_sigma));
  initial_estimate.insert(x0, x0_pose);
  graph.add(x0_prior);

  // Initialize the second pose as the first pose
  initial_estimate.insert(x1, x0_pose);

  // Create the true plane_landmarks in the World frame
  // and a landmark for each plane
  CreatePlaneLandmark create_plane_landmark;
  std::vector<std::pair<Symbol, OrientedPlane3>> plane_landmarks;
  plane_landmarks.push_back(create_plane_landmark(Unit3(-1.0, 0.0, 0.0), 1.0));
  plane_landmarks.push_back(create_plane_landmark(Unit3(0.0, -1.0, 0.0), 1.0));
  plane_landmarks.push_back(create_plane_landmark(Unit3(0.0, 0.0, -1.0), 1.0));


  // Noise model for measurements
  Vector meas_sigmas(3);
  meas_sigmas << F_SIGMA_N, F_SIGMA_N, F_SIGMA_D;
  auto meas_covariance = noiseModel::Diagonal::Sigmas(meas_sigmas);

  auto add_measurement = [&](const std::pair<Symbol, OrientedPlane3> pl, const Symbol& pose_symbol, const Pose3& pose)
  {
    Vector4 meas = pl.second.transform(pose).retract(meas_noise_gen()).planeCoefficients();
    OrientedPlane3Factor factor(meas, meas_covariance, pose_symbol, pl.first);
    graph.add(factor);
  };

  
  for (const auto& pl : plane_landmarks)
  {
    // Set initial estimate for plane to their true values
    initial_estimate.insert(pl.first, pl.second);
    
    // Add measurements to each plane from the two poses
    add_measurement(pl, x0, x0_pose);
    add_measurement(pl, x1, x1_pose);
  }

  // Solver
  LevenbergMarquardtParams params;
  params.setMaxIterations(10);
  params.linearSolverType = NonlinearOptimizerParams::MULTIFRONTAL_QR;
  LevenbergMarquardtOptimizer optimizer(graph, initial_estimate, params);
  
  // Get results
  Values results = optimizer.optimize();
  Pose3 x1_optimized = results.at<Pose3>(x1);

  std::cout << "Actual translation:" << std::endl;
  x1_pose.translation().print(); std::cout << std::endl << std::endl;
  std::cout << "Estimated translation:" << std::endl;
  x1_optimized.translation().print(); std::cout << std::endl;

  std::cout << "Translation error: " 
            << x1_pose.translation().distance(x1_optimized.translation())
            << std::endl << std::endl;

  // displacement error

  std::cout << "Actual Rotation:" << std::endl;
  x1_pose.rotation().print(); std::cout << std::endl << std::endl;
  std::cout << "Estimated Rotation:" << std::endl;
  x1_optimized.rotation().print(); std::cout << std::endl << std::endl;

  // orientation error as an angle
  Rot3 R_err = x1_optimized.rotation().inverse() * x1_pose.rotation();
  std::cout << "Rotation error in degrees: " << (180.0/M_PI) * R_err.axisAngle().second << std::endl;
}


void exampleB()
{
    std::cout << std::endl
    << "****************** Example B ******************" 
    << std::endl << std::endl;

  GaussianNoiseGenerator meas_noise_gen(SIGMA_N, SIGMA_D);
  GaussianNoiseGenerator pl_noise_gen(PL_N, PL_D);

  Values initial_estimate;
  NonlinearFactorGraph graph;

  // Start frame and World frame
  Symbol x0('x', 0);
  Pose3 x0_pose (Rot3::Ypr(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

  Symbol x1('x', 1);
  Pose3 x1_pose (Rot3::Ypr(YAW, PITCH, ROLL), Point3(X, Y, Z));

  // Prior on the first pose
  Vector prior_sigma(6);
  prior_sigma << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001;
  PriorFactor<Pose3> x0_prior(x0, x0_pose,
      noiseModel::Diagonal::Sigmas(prior_sigma));
  initial_estimate.insert(x0, x0_pose);
  graph.add(x0_prior);

  // Initialize the second pose as the first pose
  initial_estimate.insert(x1, x0_pose);

  // Create the true plane_landmarks in the World frame
  std::vector<OrientedPlane3> planes =
    {
      OrientedPlane3(Unit3(-1.0, 0.0, 0.0), 1.0), 
      OrientedPlane3(Unit3(0.0, -1.0, 0.0), 1.0),
      OrientedPlane3(Unit3(0.0, 0.0, -1.0), 1.0)
    };


  // Create a ton of repeat landmarks which are just
  // the actual planes offset by a bit
  CreatePlaneLandmark create_plane_landmark;
  std::vector<std::pair<Symbol, OrientedPlane3>> plane_landmarks;
  for (const auto& plane : planes)
  {
    for (size_t i = 0; i < 100; i++)
    {
      auto perturbed_landmark = 
        create_plane_landmark(plane);
      perturbed_landmark.second = perturbed_landmark.second.retract(pl_noise_gen());
      plane_landmarks.push_back(perturbed_landmark);
    }
  }
  

  // Noise model for measurements
  Vector meas_sigmas(3);
  meas_sigmas << F_SIGMA_N, F_SIGMA_N, F_SIGMA_D;
  auto meas_covariance = noiseModel::Diagonal::Sigmas(meas_sigmas);

  auto add_measurement = [&](const std::pair<Symbol, OrientedPlane3> pl, const Symbol& pose_symbol, const Pose3& pose)
  {
    Vector4 meas = pl.second.transform(pose).retract(meas_noise_gen()).planeCoefficients();
    OrientedPlane3Factor factor(meas, meas_covariance, pose_symbol, pl.first);
    graph.add(factor);
  };

  
  for (const auto& pl : plane_landmarks)
  {
    // Set initial estimate for plane to their true values
    initial_estimate.insert(pl.first, pl.second);
    
    // Add measurements to each plane from the two poses
    add_measurement(pl, x0, x0_pose);
    add_measurement(pl, x1, x1_pose);
  }

  // Solver
  LevenbergMarquardtParams params;
  params.setMaxIterations(10);
  params.linearSolverType = NonlinearOptimizerParams::MULTIFRONTAL_QR;
  LevenbergMarquardtOptimizer optimizer(graph, initial_estimate, params);
  
  // Get results
  Values results = optimizer.optimize();
  Pose3 x1_optimized = results.at<Pose3>(x1);

  std::cout << "Actual translation:" << std::endl;
  x1_pose.translation().print(); std::cout << std::endl << std::endl;
  std::cout << "Estimated translation:" << std::endl;
  x1_optimized.translation().print(); std::cout << std::endl;

  std::cout << "Translation error: " 
            << x1_pose.translation().distance(x1_optimized.translation())
            << std::endl << std::endl;

  // displacement error

  std::cout << "Actual Rotation:" << std::endl;
  x1_pose.rotation().print(); std::cout << std::endl << std::endl;
  std::cout << "Estimated Rotation:" << std::endl;
  x1_optimized.rotation().print(); std::cout << std::endl << std::endl;

  // orientation error as an angle
  Rot3 R_err = x1_optimized.rotation().inverse() * x1_pose.rotation();
  std::cout << "Rotation error in degrees: " << (180.0/M_PI) * R_err.axisAngle().second << std::endl;
}



int main()
{
  exampleA();
  exampleB();

  return 0;

}
