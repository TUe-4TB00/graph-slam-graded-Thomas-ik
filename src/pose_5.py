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

    for pose in pose_options:
        for landmark in [1, 2]:

            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_initial_estimate = gtsam.Values(initial_estimate)

            pose_5 = pose_options[pose]

            test_graph, test_initial_estimate = add_pose(
                test_graph,
                test_initial_estimate,
                pose_5
            )

            Result = optimize(test_graph, test_initial_estimate)

            test_graph = add_landmark_measurement(
                test_graph,
                Result,
                pose_5,
                landmark
            )

            Result = optimize(test_graph, test_initial_estimate)

            # TODO: Calculate marginal covariances for the relevant variables
            marginals = gtsam.Marginals(test_graph, Result)

            sum_of_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            if sum_of_marginals < lowest_sum:
                lowest_sum = sum_of_marginals
                best_pose = pose
                best_landmark = landmark

    return best_pose, best_landmark, lowest_sum

def minimize_errors(graph, initial_estimate, pose_options):
    # TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.

    lowest_error = float("inf")

    for pose in pose_options:
        for landmark in [1, 2]:

            test_graph = gtsam.NonlinearFactorGraph(graph)
            test_initial_estimate = gtsam.Values(initial_estimate)

            pose_5 = pose_options[pose]

            test_graph, test_initial_estimate = add_pose(
                test_graph,
                test_initial_estimate,
                pose_5
            )

            result = optimize(test_graph, test_initial_estimate)

            test_graph = add_landmark_measurement(
                test_graph,
                result,
                pose_5,
                landmark
            )

            result = optimize(test_graph, test_initial_estimate)

            # collect errors per pose
            list_of_errors = []

            ground_truth = {X(1): (0.0, 0.0), X(2): (2.0, 0.0), X(3): (4.0, 0.0)}

            for key in [X(1), X(2), X(3)]:
                tx, ty = ground_truth[key]
                err = (result.atPose2(key).x() - tx)**2 + (result.atPose2(key).y() - ty)**2
                list_of_errors.append(err)

            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < lowest_error:
                lowest_error = sum_of_errors
                best_pose = pose
                best_landmark = landmark

    return best_pose, best_landmark, lowest_error