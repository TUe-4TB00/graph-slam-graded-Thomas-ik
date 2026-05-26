import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer
    params = gtsam.LevenbergMarquardtParams()
    optimizzatonnn = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result
    result = optimizzatonnn.optimize()

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    # TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.

    lowest_sum = float("inf")
    best_pose = None
    best_landmark = None

    # Run optimization on the base graph first to get the correct 
    # landmark locations needed by the measurement helper function
    baseline_result = optimize(graph, initial_estimate)

    for pose in pose_options:
        for landmark in [1, 2]:

            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_initial_estimate = gtsam.Values(initial_estimate)

            pose_5 = pose_options[pose]

            # 1. Add the 5th pose to the graph and initial estimate
            test_graph, test_initial_estimate = add_pose(
                test_graph,
                test_initial_estimate,
                pose_5
            )

            # 2. Add the landmark measurement, passing baseline_result to fulfill the helper function's expectations
            test_graph = add_landmark_measurement(
                test_graph,
                baseline_result,
                pose_5,
                landmark
            )

            # 3. Perform final optimization on the fully built graph
            Result = optimize(test_graph, test_initial_estimate)

            # Calculate marginal covariances for the relevant variables
            marginals = gtsam.Marginals(test_graph, Result)

            # Sum the variance diagonals using .trace()
            sum_of_marginals = (
                marginals.marginalCovariance(L(1)).trace()
                + marginals.marginalCovariance(L(2)).trace()
            )

            if sum_of_marginals < lowest_sum:
                lowest_sum = sum_of_marginals
                best_pose = pose
                best_landmark = landmark

    return best_pose, best_landmark, lowest_sum

def minimize_errors(graph, initial_estimate, pose_options):
    
    return graph, initial_estimate, pose_options
