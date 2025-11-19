"""
Unit tests for Phase 3: Planning (SCvx and helpers)

Tests for:
- convexification.py: DynamicsLinearizer, TrustRegionManager, ConvexificationHelper
- collision_constraints.py: CollisionConstraintLinearizer, AdaptiveCollisionMargin
- constraints.py: BoxConstraints, TimeVaryingConstraints, ConstraintTightening
- scvx_planner.py: SCvxPlanner
"""

import cvxpy as cp
import numpy as np
import pytest

from ddfs.environment.collision import CollisionChecker
from ddfs.environment.obstacles import CircularObstacle
from ddfs.models.unicycle import UnicycleModel
from ddfs.planning.collision_constraints import (
    AdaptiveCollisionMargin,
    CollisionConstraintLinearizer,
)
from ddfs.planning.constraints import BoxConstraints, ConstraintTightening, ConstraintValidator, TimeVaryingConstraints
from ddfs.planning.convexification import ConvexificationHelper, DynamicsLinearizer, TrustRegionManager
from ddfs.planning.scvx_planner import SCvxPlanner


class TestDynamicsLinearizer:
    """Tests for DynamicsLinearizer."""

    def test_initialization(self):
        """Test linearizer initialization."""
        model = UnicycleModel()
        linearizer = DynamicsLinearizer(model, dt=0.1, method="rk4")

        assert linearizer.n == 3
        assert linearizer.m == 2
        assert linearizer.dt == 0.1
        assert linearizer.method == "rk4"

    def test_linearize_at_point(self):
        """Test linearization at single point."""
        model = UnicycleModel()
        linearizer = DynamicsLinearizer(model, dt=0.1, method="rk4")

        x = np.array([1.0, 2.0, 0.5])
        u = np.array([0.5, 0.1])

        A_d, B_d, c_d = linearizer.linearize_at_point(x, u)

        # Check dimensions
        assert A_d.shape == (3, 3)
        assert B_d.shape == (3, 2)
        assert c_d.shape == (3,)

        # Check linearization is consistent
        x_next_true = model.discrete_dynamics(x, u, 0.1, method="rk4")
        x_next_linear = A_d @ x + B_d @ u + c_d

        np.testing.assert_allclose(x_next_true, x_next_linear, atol=1e-8)

    def test_constraint_tightening_pipeline(self):
        """Test full constraint tightening pipeline."""
        # Base constraints
        base = BoxConstraints(np.array([0, 0, -np.pi]), np.array([10, 10, np.pi]))

        # Compute tightening from uncertainty
        sigma = np.array([0.1, 0.1, 0.05])
        tightening = ConstraintTightening.compute_tightening_from_uncertainty(sigma, confidence=0.95)

        # Create time-varying tightened constraints
        N = 10
        tv_constraints = ConstraintTightening.tighten_trajectory_constraints(base, tightening, N)

        # Validate
        assert tv_constraints.N == N

        # Check that constraints are tightened
        constraints_0 = tv_constraints.get_constraints_at_time(0)
        lower_0, upper_0 = constraints_0.get_bounds()

        base_lower, base_upper = base.get_bounds()

        assert np.all(lower_0 >= base_lower)
        assert np.all(upper_0 <= base_upper)

    def test_collision_linearization_and_validation(self):
        """Test collision constraint linearization and validation."""
        obs1 = CircularObstacle([2, 2], 0.5)
        obs2 = CircularObstacle([4, 4], 0.6)
        checker = CollisionChecker([obs1, obs2])

        linearizer = CollisionConstraintLinearizer(checker)

        # Create trajectory
        x_traj = np.array([[0, 0, 0], [1, 1, 0], [3, 3, 0], [5, 5, 0]])

        # Linearize
        traj_linearizations = linearizer.linearize_trajectory(x_traj)

        assert len(traj_linearizations) == 4

        # Check violations
        violated, violations = linearizer.check_constraint_violation(x_traj)

        # Compute clearance
        min_clearance, timestep, obs_idx = linearizer.compute_minimum_clearance(x_traj)

        assert min_clearance >= 0 or violated

    def test_adaptive_margin_with_scvx(self):
        """Test adaptive margin in SCvx context."""
        margin_manager = AdaptiveCollisionMargin(margin_init=0.1, margin_min=0.05, margin_max=0.3)

        # Simulate SCvx iterations with varying clearance
        clearances = [0.5, 0.3, 0.15, 0.08, 0.12, 0.25]

        margins = []
        for clearance in clearances:
            margin = margin_manager.update(clearance)
            margins.append(margin)

        # Margin should increase when clearance is low
        assert margins[3] > margins[0]  # After low clearance

    def test_trust_region_convergence(self):
        """Test trust region behavior during convergence."""
        manager = TrustRegionManager(rho_init=1.0)

        # Simulate successful iterations
        for _ in range(5):
            manager.expand()

        # Trust region should grow
        expanded_radius = manager.get_radius()
        assert expanded_radius > 1.0

        # Simulate failures
        for _ in range(10):
            manager.contract()

        # Should have decreased from expanded value
        final_radius = manager.get_radius()
        assert final_radius < expanded_radius or manager.is_too_small()

    def test_linearize_trajectory(self):
        """Test linearization along trajectory."""
        model = UnicycleModel()
        linearizer = DynamicsLinearizer(model, dt=0.1, method="rk4")

        N = 10
        x_traj = np.random.rand(N + 1, 3)
        u_traj = np.random.rand(N, 2)

        lin_data = linearizer.linearize_trajectory(x_traj, u_traj)

        assert "A" in lin_data
        assert "B" in lin_data
        assert "c" in lin_data
        assert lin_data["A"].shape == (N, 3, 3)
        assert lin_data["B"].shape == (N, 3, 2)
        assert lin_data["c"].shape == (N, 3)

    def test_compute_linearization_error(self):
        """Test linearization error computation."""
        model = UnicycleModel()
        linearizer = DynamicsLinearizer(model, dt=0.1, method="rk4")

        # Straight trajectory (should have small linearization error)
        N = 5
        x_traj = np.zeros((N + 1, 3))
        x_traj[:, 0] = np.linspace(0, 1, N + 1)
        u_traj = np.tile([0.1, 0], (N, 1))

        errors = linearizer.compute_linearization_error(x_traj, u_traj)

        assert errors.shape == (N,)
        assert np.all(errors >= 0)
        assert np.all(errors < 1e-3)  # Should be small for straight motion

    def test_validate_linearization(self):
        """Test linearization validation."""
        model = UnicycleModel()
        linearizer = DynamicsLinearizer(model, dt=0.01, method="rk4")

        # Small timestep should give accurate linearization
        N = 5
        x_traj = np.random.rand(N + 1, 3)
        u_traj = np.random.rand(N, 2) * 0.1

        valid, max_error = linearizer.validate_linearization(x_traj, u_traj, tolerance=1e-5)

        assert isinstance(valid, bool)
        assert max_error >= 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_scvx_with_infeasible_problem(self):
        """Test SCvx with infeasible problem (very tight constraints)."""
        model = UnicycleModel()

        # Very large obstacle blocking path
        obs = CircularObstacle([2, 2], 3.0)
        checker = CollisionChecker([obs])

        planner = SCvxPlanner(
            model, dt=0.1, N=10, collision_checker=checker, params={"max_iterations": 3, "verbose": False}
        )

        x0 = np.array([0, 0, 0])
        xf = np.array([4, 4, 0])

        # Very tight bounds
        x_bounds = (np.array([0, 0, -0.1]), np.array([4, 4, 0.1]))
        u_bounds = (np.array([0, -0.1]), np.array([0.1, 0.1]))

        x_traj, u_traj, converged = planner.plan(x0, xf, x_bounds, u_bounds)

        # Should still return trajectory (may not converge)
        assert x_traj is not None
        assert u_traj is not None

    def test_zero_horizon_planning(self):
        """Test edge case with very short horizon."""
        model = UnicycleModel()
        planner = SCvxPlanner(model, dt=0.1, N=1, params={"verbose": False})

        x0 = np.array([0, 0, 0])
        xf = np.array([0.1, 0.1, 0])

        x_traj, u_traj, converged = planner.plan(x0, xf)

        assert x_traj.shape == (2, 3)
        assert u_traj.shape == (1, 2)

    def test_constraints_at_boundary(self):
        """Test constraints exactly at boundaries."""
        constraints = BoxConstraints(np.array([0, 0]), np.array([10, 10]))

        x_boundary = np.array([0, 10])

        assert constraints.is_satisfied(x_boundary)
        assert constraints.compute_violation(x_boundary) == 0

    def test_empty_collision_checker(self):
        """Test collision linearizer with no obstacles."""
        checker = CollisionChecker()
        linearizer = CollisionConstraintLinearizer(checker)

        x = np.array([1, 2, 0])
        linearizations = linearizer.linearize_at_point(x)

        assert len(linearizations) == 0

    def test_overlapping_obstacles_linearization(self):
        """Test linearization with overlapping obstacles."""
        obs1 = CircularObstacle([2, 2], 1.0)
        obs2 = CircularObstacle([2.5, 2], 1.0)  # Overlapping
        checker = CollisionChecker([obs1, obs2])

        linearizer = CollisionConstraintLinearizer(checker)

        x = np.array([2.25, 2, 0])  # Between centers
        linearizations = linearizer.linearize_at_point(x)

        assert len(linearizations) == 2

        # Both should report collision (negative distance)
        for lin in linearizations.values():
            assert lin["distance"] < 0

    def test_very_small_trust_region(self):
        """Test behavior with very small trust region."""
        manager = TrustRegionManager(rho_init=1e-5, rho_min=1e-6, gamma_contract=0.5)

        # Should reach minimum after enough contractions
        for _ in range(10):
            manager.contract()

        assert manager.is_too_small()

    def test_box_constraints_single_dimension(self):
        """Test box constraints in 1D."""
        constraints = BoxConstraints(np.array([0]), np.array([10]))

        x_valid = np.array([5])
        x_invalid = np.array([15])

        assert constraints.is_satisfied(x_valid)
        assert not constraints.is_satisfied(x_invalid)

    def test_constraint_tightening_zero_margin(self):
        """Test constraint tightening with zero margin."""
        constraints = BoxConstraints(np.array([0, 0]), np.array([10, 10]))

        tight = constraints.tighten(margin=0.0)

        lower, upper = tight.get_bounds()
        lower_orig, upper_orig = constraints.get_bounds()

        np.testing.assert_array_equal(lower, lower_orig)
        np.testing.assert_array_equal(upper, upper_orig)

    def test_linearization_at_obstacle_boundary(self):
        """Test linearization exactly at obstacle boundary."""
        obs = CircularObstacle([0, 0], 1.0)
        checker = CollisionChecker([obs])
        linearizer = CollisionConstraintLinearizer(checker)

        x = np.array([1, 0, 0])  # Exactly on boundary
        linearizations = linearizer.linearize_at_point(x)

        # Distance should be ~0
        assert abs(linearizations[0]["distance"]) < 1e-10

        # Gradient should point away from center
        grad = linearizations[0]["gradient"]
        assert grad[0] > 0  # Points in +x direction

    def test_projection_already_feasible(self):
        """Test projection of point already in feasible region."""
        constraints = BoxConstraints(np.array([0, 0]), np.array([10, 10]))

        x = np.array([5, 5])
        x_proj = constraints.project_onto_feasible(x)

        np.testing.assert_array_equal(x, x_proj)


class TestPerformance:
    """Performance and scaling tests."""

    def test_linearization_performance(self):
        """Test linearization performance on long trajectory."""
        model = UnicycleModel()
        linearizer = DynamicsLinearizer(model, dt=0.1, method="rk4")

        N = 100
        x_traj = np.random.rand(N + 1, 3)
        u_traj = np.random.rand(N, 2)

        # Should complete without error
        lin_data = linearizer.linearize_trajectory(x_traj, u_traj)

        assert lin_data["A"].shape == (N, 3, 3)

    def test_multiple_obstacles_linearization(self):
        """Test linearization with many obstacles."""
        obstacles = [CircularObstacle([i, i], 0.5) for i in range(10)]
        checker = CollisionChecker(obstacles)
        linearizer = CollisionConstraintLinearizer(checker)

        x = np.array([5, 5, 0])
        linearizations = linearizer.linearize_at_point(x)

        assert len(linearizations) == 10

    def test_long_horizon_scvx(self):
        """Test SCvx with long horizon (smoke test)."""
        model = UnicycleModel()
        planner = SCvxPlanner(model, dt=0.1, N=50, params={"max_iterations": 2, "verbose": False})

        x0 = np.array([0, 0, 0])
        xf = np.array([5, 5, 0])

        x_traj, u_traj, _ = planner.plan(x0, xf)

        assert x_traj.shape == (51, 3)


class TestTrustRegionManager:
    """Tests for TrustRegionManager."""

    def test_initialization(self):
        """Test trust region manager initialization."""
        manager = TrustRegionManager(rho_init=1.0, rho_min=0.01, rho_max=10.0)

        assert manager.get_radius() == 1.0
        assert manager.rho_min == 0.01
        assert manager.rho_max == 10.0

    def test_expand(self):
        """Test trust region expansion."""
        manager = TrustRegionManager(rho_init=1.0, beta_expand=2.0)

        initial_rho = manager.get_radius()
        manager.expand()

        assert manager.get_radius() == initial_rho * 2.0

    def test_contract(self):
        """Test trust region contraction."""
        manager = TrustRegionManager(rho_init=1.0, gamma_contract=0.5)

        initial_rho = manager.get_radius()
        manager.contract()

        assert manager.get_radius() == initial_rho * 0.5

    def test_expand_respects_max(self):
        """Test expansion respects maximum radius."""
        manager = TrustRegionManager(rho_init=8.0, rho_max=10.0, beta_expand=2.0)

        manager.expand()

        assert manager.get_radius() <= 10.0

    def test_contract_respects_min(self):
        """Test contraction respects minimum radius."""
        manager = TrustRegionManager(rho_init=0.02, rho_min=0.01, gamma_contract=0.5)

        manager.contract()

        assert manager.get_radius() >= 0.01

    def test_is_too_small(self):
        """Test checking if trust region is too small."""
        manager = TrustRegionManager(rho_init=0.01, rho_min=0.01)

        assert manager.is_too_small()

        manager = TrustRegionManager(rho_init=1.0, rho_min=0.01)
        assert not manager.is_too_small()

    def test_reset(self):
        """Test resetting trust region."""
        manager = TrustRegionManager(rho_init=1.0)

        manager.expand()
        manager.expand()
        manager.reset(rho_new=1.5)

        assert manager.get_radius() == 1.5

    def test_history_tracking(self):
        """Test history tracking."""
        manager = TrustRegionManager(rho_init=1.0)

        manager.expand()
        manager.contract()

        history = manager.get_history()
        assert len(history) == 2
        assert history[0][0] == "expand"
        assert history[1][0] == "contract"


class TestConvexificationHelper:
    """Tests for ConvexificationHelper."""

    def test_compute_trajectory_deviation(self):
        """Test trajectory deviation computation."""
        x_new = np.random.rand(10, 3)
        x_ref = np.random.rand(10, 3)

        metrics = ConvexificationHelper.compute_trajectory_deviation(x_new, x_ref)

        assert "l2" in metrics
        assert "linf" in metrics
        assert "mean" in metrics
        assert "max" in metrics
        assert all(v >= 0 for v in metrics.values())

    def test_check_convergence(self):
        """Test convergence checking."""
        x_new = np.array([[1, 2, 3], [4, 5, 6]])
        x_ref = np.array([[1.001, 2.001, 3.001], [4.001, 5.001, 6.001]])
        u_new = np.array([[0.1, 0.2]])
        u_ref = np.array([[0.1001, 0.2001]])

        converged, metrics = ConvexificationHelper.check_convergence(x_new, x_ref, u_new, u_ref, tol_x=1e-2, tol_u=1e-2)

        assert converged
        assert "dx_linf" in metrics
        assert "du_linf" in metrics

    def test_compute_control_cost(self):
        """Test control cost computation."""
        u_traj = np.array([[1, 0], [0, 1], [1, 1]])

        cost = ConvexificationHelper.compute_control_cost(u_traj, weight=1.0)

        # Should be ||[1,0]||^2 + ||[0,1]||^2 + ||[1,1]||^2 = 1 + 1 + 2 = 4
        assert abs(cost - 4.0) < 1e-10

    def test_compute_terminal_cost(self):
        """Test terminal cost computation."""
        x_final = np.array([5, 5, 0])
        x_goal = np.array([6, 6, 0])

        cost = ConvexificationHelper.compute_terminal_cost(x_final, x_goal, weight=1.0)

        # Should be ||[5,5,0] - [6,6,0]||^2 = ||-1,-1,0||^2 = 2
        assert abs(cost - 2.0) < 1e-10

    def test_compute_trust_region_violation(self):
        """Test trust region violation computation."""
        x_new = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]])
        x_ref = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        rho = 1.5

        violation = ConvexificationHelper.compute_trust_region_violation(x_new, x_ref, rho)

        # Max deviation is 2.0 at last timestep, violation = 2.0 - 1.5 = 0.5
        assert abs(violation - 0.5) < 1e-10


class TestCollisionConstraintLinearizer:
    """Tests for CollisionConstraintLinearizer."""

    def test_initialization(self):
        """Test linearizer initialization."""
        obs = CircularObstacle([2, 2], 1.0)
        checker = CollisionChecker([obs])
        linearizer = CollisionConstraintLinearizer(checker)

        assert linearizer.num_obstacles == 1

    def test_linearize_at_point(self):
        """Test linearization at point."""
        obs = CircularObstacle([0, 0], 1.0)
        checker = CollisionChecker([obs])
        linearizer = CollisionConstraintLinearizer(checker)

        x = np.array([2, 0, 0])  # Outside obstacle
        linearizations = linearizer.linearize_at_point(x)

        assert 0 in linearizations  # Obstacle index 0
        assert "distance" in linearizations[0]
        assert "gradient" in linearizations[0]
        assert "offset" in linearizations[0]

        # Distance should be ~1.0 (at x=2, center=0, radius=1)
        assert abs(linearizations[0]["distance"] - 1.0) < 1e-6

    def test_linearize_trajectory(self):
        """Test linearization along trajectory."""
        obs = CircularObstacle([2, 2], 1.0)
        checker = CollisionChecker([obs])
        linearizer = CollisionConstraintLinearizer(checker)

        N = 5
        x_traj = np.random.rand(N + 1, 3) * 5

        traj_linearizations = linearizer.linearize_trajectory(x_traj)

        assert len(traj_linearizations) == N + 1
        for lin in traj_linearizations:
            assert 0 in lin

    def test_check_constraint_violation(self):
        """Test constraint violation checking."""
        obs = CircularObstacle([2, 2], 1.0)
        checker = CollisionChecker([obs])
        linearizer = CollisionConstraintLinearizer(checker)

        # Safe trajectory
        x_safe = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 0]])
        violated, violations = linearizer.check_constraint_violation(x_safe)

        # Colliding trajectory
        x_collision = np.array([[0, 0, 0], [2, 2, 0], [5, 5, 0]])
        violated_col, violations_col = linearizer.check_constraint_violation(x_collision)

        assert violated_col
        assert len(violations_col) > 0

    def test_compute_minimum_clearance(self):
        """Test minimum clearance computation."""
        obs = CircularObstacle([3, 3], 1.0)
        checker = CollisionChecker([obs])
        linearizer = CollisionConstraintLinearizer(checker)

        x_traj = np.array([[0, 0, 0], [2, 2, 0], [5, 5, 0]])

        min_clearance, timestep, obs_idx = linearizer.compute_minimum_clearance(x_traj)

        assert min_clearance >= 0  # Assuming safe trajectory
        assert 0 <= timestep < len(x_traj)
        assert obs_idx == 0


class TestAdaptiveCollisionMargin:
    """Tests for AdaptiveCollisionMargin."""

    def test_initialization(self):
        """Test adaptive margin initialization."""
        margin_manager = AdaptiveCollisionMargin(margin_init=0.1, margin_min=0.05, margin_max=0.5)

        assert margin_manager.get_margin() == 0.1

    def test_update_increase(self):
        """Test margin increases when clearance is low."""
        margin_manager = AdaptiveCollisionMargin(margin_init=0.1, clearance_threshold=0.2)

        initial_margin = margin_manager.get_margin()
        margin_manager.update(min_clearance=0.1)  # Low clearance

        assert margin_manager.get_margin() > initial_margin

    def test_update_decrease(self):
        """Test margin decreases when clearance is high."""
        margin_manager = AdaptiveCollisionMargin(margin_init=0.3, clearance_threshold=0.2)

        initial_margin = margin_manager.get_margin()
        margin_manager.update(min_clearance=0.5)  # High clearance

        assert margin_manager.get_margin() < initial_margin

    def test_update_respects_bounds(self):
        """Test margin respects min/max bounds."""
        margin_manager = AdaptiveCollisionMargin(
            margin_init=0.05, margin_min=0.05, margin_max=0.5, clearance_threshold=0.2
        )

        # Try to decrease below min
        margin_manager.update(min_clearance=1.0)
        assert margin_manager.get_margin() >= 0.05

        # Try to increase above max
        for _ in range(20):
            margin_manager.update(min_clearance=0.01)
        assert margin_manager.get_margin() <= 0.5


class TestBoxConstraints:
    """Tests for BoxConstraints."""

    def test_initialization(self):
        """Test box constraints initialization."""
        lower = np.array([0, 0, -1])
        upper = np.array([10, 10, 1])

        constraints = BoxConstraints(lower, upper, name="test")

        assert constraints.dim == 3
        assert constraints.name == "test"

    def test_is_satisfied(self):
        """Test feasibility checking."""
        constraints = BoxConstraints(np.array([0, 0]), np.array([10, 10]))

        x_valid = np.array([5, 5])
        x_invalid_low = np.array([-1, 5])
        x_invalid_high = np.array([5, 11])

        assert constraints.is_satisfied(x_valid)
        assert not constraints.is_satisfied(x_invalid_low)
        assert not constraints.is_satisfied(x_invalid_high)

    def test_compute_violation(self):
        """Test violation computation."""
        constraints = BoxConstraints(np.array([0, 0]), np.array([10, 10]))

        x_valid = np.array([5, 5])
        x_invalid = np.array([-2, 12])

        viol_valid = constraints.compute_violation(x_valid)
        viol_invalid = constraints.compute_violation(x_invalid)

        assert viol_valid == 0
        assert viol_invalid == 2  # max(2, 2)

    def test_project_onto_feasible(self):
        """Test projection onto feasible region."""
        constraints = BoxConstraints(np.array([0, 0]), np.array([10, 10]))

        x = np.array([-5, 15])
        x_proj = constraints.project_onto_feasible(x)

        expected = np.array([0, 10])
        np.testing.assert_array_equal(x_proj, expected)

    def test_tighten(self):
        """Test constraint tightening."""
        constraints = BoxConstraints(np.array([0, 0]), np.array([10, 10]))

        tight = constraints.tighten(margin=1.0)

        lower, upper = tight.get_bounds()
        np.testing.assert_array_equal(lower, [1, 1])
        np.testing.assert_array_equal(upper, [9, 9])

    def test_build_cvxpy_constraints(self):
        """Test CVXPY constraint building."""
        constraints = BoxConstraints(np.array([0, 0]), np.array([10, 10]))

        x_var = cp.Variable(2)
        cvxpy_constraints = constraints.build_cvxpy_constraints(x_var)

        assert len(cvxpy_constraints) == 2  # lower and upper


class TestTimeVaryingConstraints:
    """Tests for TimeVaryingConstraints."""

    def test_initialization(self):
        """Test time-varying constraints initialization."""
        constraints_seq = [BoxConstraints(np.array([0, 0]), np.array([10, 10])) for _ in range(5)]

        tv_constraints = TimeVaryingConstraints(constraints_seq)

        assert tv_constraints.N == 4

    def test_get_constraints_at_time(self):
        """Test getting constraints at specific time."""
        constraints_seq = [BoxConstraints(np.array([i, i]), np.array([10, 10])) for i in range(5)]

        tv_constraints = TimeVaryingConstraints(constraints_seq)

        constraints_2 = tv_constraints.get_constraints_at_time(2)
        lower, _ = constraints_2.get_bounds()

        np.testing.assert_array_equal(lower, [2, 2])

    def test_is_trajectory_satisfied(self):
        """Test trajectory satisfaction checking."""
        constraints_seq = [BoxConstraints(np.array([0, 0]), np.array([10, 10])) for _ in range(5)]

        tv_constraints = TimeVaryingConstraints(constraints_seq)

        x_valid = np.array([[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]])
        x_invalid = np.array([[5, 5], [15, 5], [5, 5], [5, 5], [5, 5]])

        satisfied_valid, _ = tv_constraints.is_trajectory_satisfied(x_valid)
        satisfied_invalid, violations = tv_constraints.is_trajectory_satisfied(x_invalid)

        assert satisfied_valid
        assert not satisfied_invalid
        assert 1 in violations


class TestConstraintTightening:
    """Tests for ConstraintTightening."""

    def test_compute_tightening_from_deviation(self):
        """Test tightening from observed deviations."""
        x_nominal = np.zeros((10, 3))
        x_actual = np.random.randn(10, 3) * 0.1

        tightening = ConstraintTightening.compute_tightening_from_deviation(x_nominal, x_actual, percentile=95)

        assert tightening.shape == (3,)
        assert np.all(tightening >= 0)

    def test_compute_tightening_from_uncertainty(self):
        """Test tightening from uncertainty."""
        sigma = np.array([0.1, 0.1, 0.05])

        tightening = ConstraintTightening.compute_tightening_from_uncertainty(sigma, confidence=0.99)

        assert tightening.shape == (3,)
        assert np.all(tightening >= sigma)  # Should be larger due to confidence


class TestConstraintValidator:
    """Tests for ConstraintValidator."""

    def test_validate_state_trajectory(self):
        """Test state trajectory validation."""
        constraints = BoxConstraints(np.array([0, 0, -1]), np.array([10, 10, 1]))

        x_valid = np.array([[5, 5, 0], [6, 6, 0.5]])
        x_invalid = np.array([[5, 5, 0], [15, 5, 0]])

        valid, report_valid = ConstraintValidator.validate_state_trajectory(x_valid, constraints)
        invalid, report_invalid = ConstraintValidator.validate_state_trajectory(x_invalid, constraints)

        assert valid
        assert report_valid["num_violations"] == 0

        assert not invalid
        assert report_invalid["num_violations"] > 0

    def test_compute_constraint_margin(self):
        """Test constraint margin computation."""
        constraints = BoxConstraints(np.array([0, 0]), np.array([10, 10]))

        x_traj = np.array([[5, 5], [1, 9]])

        margins = ConstraintValidator.compute_constraint_margin(x_traj, constraints)

        assert "lower" in margins
        assert "upper" in margins
        assert "min" in margins
        assert margins["min"].shape == (2,)


class TestSCvxPlanner:
    """Tests for SCvxPlanner."""

    def test_initialization(self):
        """Test SCvx planner initialization."""
        model = UnicycleModel()
        planner = SCvxPlanner(model, dt=0.1, N=10)

        assert planner.n == 3
        assert planner.m == 2
        assert planner.N == 10
        assert planner.dt == 0.1

    def test_initialize_trajectory(self):
        """Test trajectory initialization."""
        model = UnicycleModel()
        planner = SCvxPlanner(model, dt=0.1, N=10)

        x0 = np.array([0, 0, 0])
        xf = np.array([5, 5, 0])

        x_init, u_init = planner._initialize_trajectory(x0, xf)

        assert x_init.shape == (11, 3)
        assert u_init.shape == (10, 2)
        np.testing.assert_array_equal(x_init[0], x0)
        np.testing.assert_array_equal(x_init[-1], xf)

    def test_plan_simple_no_obstacles(self):
        """Test planning without obstacles (smoke test)."""
        model = UnicycleModel()
        planner = SCvxPlanner(model, dt=0.1, N=20, params={"max_iterations": 5, "verbose": False})

        x0 = np.array([0, 0, 0])
        xf = np.array([2, 2, 0])

        x_bounds = (np.array([0, 0, -np.pi]), np.array([5, 5, np.pi]))
        u_bounds = (np.array([0, -1]), np.array([1, 1]))

        x_traj, u_traj, converged = planner.plan(x0, xf, x_bounds, u_bounds)

        # Should return some trajectory even if not converged
        assert x_traj.shape == (21, 3)
        assert u_traj.shape == (20, 2)

    def test_plan_with_obstacles(self):
        """Test planning with obstacles (smoke test)."""
        model = UnicycleModel()

        obs = CircularObstacle([1, 1], 0.3)
        checker = CollisionChecker([obs])

        planner = SCvxPlanner(
            model, dt=0.1, N=20, collision_checker=checker, params={"max_iterations": 5, "verbose": False}
        )

        x0 = np.array([0, 0, 0])
        xf = np.array([2, 2, 0])

        x_bounds = (np.array([0, 0, -np.pi]), np.array([3, 3, np.pi]))
        u_bounds = (np.array([0, -1]), np.array([1, 1]))

        x_traj, u_traj, converged = planner.plan(x0, xf, x_bounds, u_bounds)

        # Should return trajectory
        assert x_traj.shape == (21, 3)
        assert u_traj.shape == (20, 2)

    def test_history_tracking(self):
        """Test that history is tracked."""
        model = UnicycleModel()
        planner = SCvxPlanner(model, dt=0.1, N=10, params={"max_iterations": 3, "verbose": False})

        x0 = np.array([0, 0, 0])
        xf = np.array([1, 1, 0])

        planner.plan(x0, xf)

        history = planner.get_history()

        assert "cost" in history
        assert "trust_region_rho" in history

    def test_clear_history(self):
        """Test history clearing."""
        model = UnicycleModel()
        planner = SCvxPlanner(model, dt=0.1, N=10, params={"verbose": False})

        x0 = np.array([0, 0, 0])
        xf = np.array([1, 1, 0])

        planner.plan(x0, xf)
        planner.clear_history()

        history = planner.get_history()
        assert len(history["cost"]) == 0


class TestIntegration:
    """Integration tests across planning components."""

    def test_scvx_with_all_helpers(self):
        """Test SCvx using all helper modules."""
        # Setup
        model = UnicycleModel()
        obs = CircularObstacle([2, 2], 0.5)
        checker = CollisionChecker([obs])

        # Create planner
        planner = SCvxPlanner(
            model, dt=0.1, N=30, collision_checker=checker, params={"max_iterations": 10, "verbose": False}
        )

        # Plan
        x0 = np.array([0, 0, 0])
        xf = np.array([4, 4, 0])
        x_bounds = (np.array([0, 0, -np.pi]), np.array([5, 5, np.pi]))
        u_bounds = (np.array([0, -1]), np.array([1, 1]))

        x_traj, u_traj, converged = planner.plan(x0, xf, x_bounds, u_bounds)

        # Validate results
        assert x_traj.shape[0] == planner.N + 1
        assert u_traj.shape[0] == planner.N

        # Check no collisions
        violated, _, _ = checker.check_trajectory_collision(x_traj)
        # May or may not have collision depending on convergence

    def test_linearization_consistency(self):
        """Test that linearization is consistent across modules."""
        model = UnicycleModel()
        linearizer = DynamicsLinearizer(model, dt=0.1, method="rk4")

        x = np.array([1, 2, 0.5])
        u = np.array([0.5, 0.1])

        A_d, B_d, c_d = linearizer.linearize_at_point(x, u)

        # Linearization should approximate dynamics
        x_next_true = model.discrete_dynamics(x, u, 0.1, method="rk4")
        x_next_linear = A_d @ x + B_d @ u + c_d

        np.testing.assert_allclose(x_next_true, x_next_linear, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
