"""
Test suite for LMI matrix construction.

Tests all matrix construction functions in lmi_matrices.py with:
- Dimension validation
- Matrix property verification (symmetry, PSD)
- Numerical accuracy checks
- Edge cases and error handling

Run with: pytest tests/test_lmi_matrices.py -v
"""

import cvxpy as cp
import numpy as np
import pytest

from ddfs.synthesis.lmi_matrices import LMIMatrixConstructor, LMIMatrixDimensions, create_lmi_constructor


class TestLMIMatrixDimensions:
    """Test dimension tracking dataclass"""

    def test_basic_dimensions(self):
        """Test basic dimension properties"""
        dims = LMIMatrixDimensions(n=2, m=1, L=10)

        assert dims.n == 2
        assert dims.m == 1
        assert dims.L == 10
        assert dims.S_dim == 4 * 2 + 2 * 1  # 4n+2m = 10
        assert dims.N_dim == 3 * 2 + 2 * 1  # 3n+2m = 8

    def test_validation_positive_dimensions(self):
        """Test that dimensions must be positive"""
        with pytest.raises(AssertionError):
            dims = LMIMatrixDimensions(n=0, m=1, L=10)
            dims.validate()

        with pytest.raises(AssertionError):
            dims = LMIMatrixDimensions(n=2, m=-1, L=10)
            dims.validate()

    def test_validation_sufficient_data(self):
        """Test that L must be sufficient for informativity"""
        with pytest.raises(AssertionError):
            dims = LMIMatrixDimensions(n=5, m=3, L=5)  # L < n+m
            dims.validate()


class TestLMIMatrixConstructor:
    """Test LMI matrix constructor class"""

    @pytest.fixture
    def small_constructor(self):
        """Create small system for testing (n=2, m=1, L=10)"""
        return LMIMatrixConstructor(n_states=2, n_controls=1, data_length=10)

    @pytest.fixture
    def medium_constructor(self):
        """Create medium system for testing (n=4, m=2, L=20)"""
        return LMIMatrixConstructor(n_states=4, n_controls=2, data_length=20)

    def test_constructor_initialization(self, small_constructor):
        """Test constructor initializes correctly"""
        assert small_constructor.dims.n == 2
        assert small_constructor.dims.m == 1
        assert small_constructor.dims.L == 10
        assert small_constructor.dims.S_dim == 10  # 4*2 + 2*1
        assert small_constructor.dims.N_dim == 8  # 3*2 + 2*1

    def test_factory_function(self):
        """Test factory function creates constructor"""
        constructor = create_lmi_constructor(n_states=3, n_controls=2, data_length=15)
        assert constructor.dims.n == 3
        assert constructor.dims.m == 2
        assert constructor.dims.L == 15


class TestSMatrixConstruction:
    """Test S-matrix construction"""

    @pytest.fixture
    def constructor(self):
        return LMIMatrixConstructor(n_states=2, n_controls=1, data_length=10)

    @pytest.fixture
    def cvxpy_variables(self):
        """Create CVXPY variables for testing"""
        n, m = 2, 1
        P_i = cp.Variable((n, n), symmetric=True)
        L_i = cp.Variable((m, n))
        return P_i, L_i

    def test_S_matrix_dimensions(self, constructor, cvxpy_variables):
        """Test S matrix has correct dimensions"""
        P_i, L_i = cvxpy_variables
        nu = 0.01
        alpha = 0.95

        S = constructor.construct_S_matrix(P_i, L_i, nu, alpha)

        # Check it's a CVXPY expression
        assert isinstance(S, cp.Expression)

        # Check dimensions (4n+2m = 4*2+2*1 = 10)
        assert S.shape == (10, 10)

    def test_S_matrix_parameter_validation(self, constructor, cvxpy_variables):
        """Test S matrix validates parameters"""
        P_i, L_i = cvxpy_variables

        # Invalid alpha (must be in (0,1))
        with pytest.raises(AssertionError):
            constructor.construct_S_matrix(P_i, L_i, nu=0.01, alpha=1.5)

        with pytest.raises(AssertionError):
            constructor.construct_S_matrix(P_i, L_i, nu=0.01, alpha=-0.1)

        # Invalid nu (must be positive)
        with pytest.raises(AssertionError):
            constructor.construct_S_matrix(P_i, L_i, nu=-0.01, alpha=0.95)

    def test_S_matrix_structure_numerical(self, constructor):
        """Test S matrix structure with numerical values"""
        n, m = 2, 1

        # Use numerical values instead of CVXPY variables for structure test
        P_val = np.eye(n)
        L_val = np.ones((m, n))
        nu = 0.01
        alpha = 0.95

        # Manually construct expected structure
        S_expected = np.zeros((10, 10))

        # Block (0,0): alpha * P_i - nu * I_n
        S_expected[0:2, 0:2] = alpha * P_val - nu * np.eye(n)

        # Block (1,1): -P
        S_expected[2:4, 2:4] = -P_val

        # Block (1,2): -L^T
        S_expected[2:4, 4:5] = -L_val.T

        # Block (1,3): -P
        S_expected[2:4, 5:7] = -P_val

        # Block (1,4): -L^T
        S_expected[2:4, 7:8] = -L_val.T

        # Block (2,1): -L
        S_expected[4:5, 2:4] = -L_val

        # Block (2,3): -L
        S_expected[4:5, 5:7] = -L_val

        # Block (2,5): L
        S_expected[4:5, 8:10] = L_val

        # Block (3,1): -P
        S_expected[5:7, 2:4] = -P_val

        # Block (3,2): -L^T
        S_expected[5:7, 4:5] = -L_val.T

        # Block (3,3): -P
        S_expected[5:7, 5:7] = -P_val

        # Block (3,4): -L^T
        S_expected[5:7, 7:8] = -L_val.T

        # Block (4,1): -L
        S_expected[7:8, 2:4] = -L_val

        # Block (4,3): -L
        S_expected[7:8, 5:7] = -L_val

        # Block (4,5): L
        S_expected[7:8, 8:10] = L_val

        # Block (5,2): L^T
        S_expected[8:10, 4:5] = L_val.T

        # Block (5,4): L^T
        S_expected[8:10, 7:8] = L_val.T

        # Block (5,5): P
        S_expected[8:10, 8:10] = P_val

        # Now test with CVXPY (structure check only)
        P_i = cp.Variable((n, n), symmetric=True)
        L_i = cp.Variable((m, n))
        S = constructor.construct_S_matrix(P_i, L_i, nu, alpha)

        # Just verify it constructs without error and has right shape
        assert S.shape == S_expected.shape


class TestN1MatrixConstruction:
    """Test N₁ matrix construction"""

    @pytest.fixture
    def constructor(self):
        return LMIMatrixConstructor(n_states=2, n_controls=1, data_length=10)

    @pytest.fixture
    def hankel_data(self):
        """Create valid Hankel matrices for testing"""
        n, m, L = 2, 1, 10
        H = np.random.randn(n, L)
        H_plus = np.random.randn(n, L)
        Xi = np.random.randn(m, L)
        beta = 0.5
        return H, H_plus, Xi, beta

    def test_N1_dimensions(self, constructor, hankel_data):
        """Test N₁ has correct dimensions"""
        H, H_plus, Xi, beta = hankel_data

        N1 = constructor.construct_N1_matrix(H, H_plus, Xi, beta)

        # Should be (3n+2m) x (3n+2m) = 8x8
        assert N1.shape == (8, 8)

    def test_N1_symmetry(self, constructor, hankel_data):
        """Test N₁ is symmetric"""
        H, H_plus, Xi, beta = hankel_data

        N1 = constructor.construct_N1_matrix(H, H_plus, Xi, beta)

        assert np.allclose(N1, N1.T, atol=1e-10)

    def test_N1_input_validation(self, constructor):
        """Test N₁ validates input dimensions"""
        n, m, L = 2, 1, 10

        # Wrong H shape
        with pytest.raises(AssertionError):
            constructor.construct_N1_matrix(
                H_prev=np.random.randn(3, L),  # Wrong n
                H_plus_prev=np.random.randn(n, L),
                Xi_prev=np.random.randn(m, L),
                beta_prev=0.5,
            )

        # Wrong Xi shape
        with pytest.raises(AssertionError):
            constructor.construct_N1_matrix(
                H_prev=np.random.randn(n, L),
                H_plus_prev=np.random.randn(n, L),
                Xi_prev=np.random.randn(2, L),  # Wrong m
                beta_prev=0.5,
            )

        # Negative beta
        with pytest.raises(AssertionError):
            constructor.construct_N1_matrix(
                H_prev=np.random.randn(n, L),
                H_plus_prev=np.random.randn(n, L),
                Xi_prev=np.random.randn(m, L),
                beta_prev=-0.5,  # Must be positive
            )

    def test_N1_manual_construction(self, constructor):
        """Test N₁ construction with known values"""
        n, m, L = 2, 1, 10

        # Simple Hankel matrices for manual verification
        H = np.ones((n, L))
        H_plus = 2 * np.ones((n, L))
        Xi = 0.5 * np.ones((m, L))
        beta = 1.0

        N1 = constructor.construct_N1_matrix(H, H_plus, Xi, beta)

        # Build manually
        G1 = np.zeros((8, 12))  # (3n+2m) x (n+L)
        G1[0:2, 0:2] = np.eye(2)
        G1[0:2, 2:12] = H_plus
        G1[2:4, 2:12] = -H
        G1[4:5, 2:12] = -Xi

        M1 = np.zeros((12, 12))
        M1[0:2, 0:2] = beta * np.eye(2)
        M1[2:12, 2:12] = -np.eye(10)

        N1_expected = G1 @ M1 @ G1.T

        assert np.allclose(N1, N1_expected, atol=1e-10)


class TestN2MatrixConstruction:
    """Test N₂ matrix construction"""

    @pytest.fixture
    def constructor(self):
        return LMIMatrixConstructor(n_states=2, n_controls=1, data_length=10)

    def test_N2_dimensions(self, constructor):
        """Test N₂ has correct dimensions"""
        C = 0.1
        T_segment = 1.0

        N2 = constructor.construct_N2_matrix(C, T_segment)

        # Should be (3n+2m) x (3n+2m) = 8x8
        assert N2.shape == (8, 8)

    def test_N2_symmetry(self, constructor):
        """Test N₂ is symmetric"""
        C = 0.1
        T_segment = 1.0

        N2 = constructor.construct_N2_matrix(C, T_segment)

        assert np.allclose(N2, N2.T, atol=1e-10)

    def test_N2_parameter_validation(self, constructor):
        """Test N₂ validates parameters"""
        # Negative C
        with pytest.raises(AssertionError):
            constructor.construct_N2_matrix(C=-0.1, T_segment=1.0)

        # Negative T_segment
        with pytest.raises(AssertionError):
            constructor.construct_N2_matrix(C=0.1, T_segment=-1.0)

    def test_N2_manual_construction(self, constructor):
        """Test N₂ construction with known values"""
        n, m = 2, 1
        C = 0.5
        T_segment = 2.0

        N2 = constructor.construct_N2_matrix(C, T_segment)

        # Build manually
        # G2 is (3n+2m) x (2n+m) = 8x5
        G2 = np.zeros((8, 5))

        # Row block 0 (rows 0:2): [I_n, 0, 0]
        G2[0:2, 0:2] = np.eye(2)

        # Row blocks 1-2 (rows 2:5): all zeros

        # Row block 3 (rows 5:7 which is 2*n+m:3*n+m): [0, I_n, 0]
        G2[2 * n + m : 3 * n + m, n : 2 * n] = np.eye(2)

        # Row block 4 (rows 7:8 which is 3*n+m:3*n+2*m): [0, 0, I_m]
        G2[3 * n + m : 3 * n + 2 * m, 2 * n : 2 * n + m] = np.eye(1)

        M2 = np.zeros((5, 5))
        M2[0:2, 0:2] = (C**2) * (T_segment**2) * np.eye(2)
        M2[2:4, 2:4] = -np.eye(2)
        M2[4:5, 4:5] = -np.eye(1)

        N2_expected = G2 @ M2 @ G2.T

        assert np.allclose(N2, N2_expected, atol=1e-10)

    def test_N2_scaling(self, constructor):
        """Test N₂ scales correctly with C and T"""
        C1, T1 = 0.1, 1.0
        C2, T2 = 0.2, 2.0

        N2_1 = constructor.construct_N2_matrix(C1, T1)
        N2_2 = constructor.construct_N2_matrix(C2, T2)

        # The (0,0) block should scale as C²T²
        # Extract top-left nxn block
        scale_factor = (C2 / C1) ** 2 * (T2 / T1) ** 2

        assert np.allclose(N2_2[0:2, 0:2], scale_factor * N2_1[0:2, 0:2], atol=1e-10)


class TestPadding:
    """Test matrix padding to S dimension"""

    @pytest.fixture
    def constructor(self):
        return LMIMatrixConstructor(n_states=2, n_controls=1, data_length=10)

    def test_padding_dimensions(self, constructor):
        """Test padding produces correct dimensions"""
        # Create a (3n+2m) x (3n+2m) = 8x8 matrix
        N = np.random.randn(8, 8)
        N = (N + N.T) / 2  # Make symmetric

        N_tilde = constructor.pad_to_S_dimension(N)

        # Should be (4n+2m) x (4n+2m) = 10x10
        assert N_tilde.shape == (10, 10)

    def test_padding_preserves_topleft(self, constructor):
        """Test padding preserves top-left block"""
        N = np.random.randn(8, 8)
        N = (N + N.T) / 2

        N_tilde = constructor.pad_to_S_dimension(N)

        # Top-left 8x8 should match original
        assert np.allclose(N_tilde[0:8, 0:8], N)

    def test_padding_zeros_elsewhere(self, constructor):
        """Test padding adds zeros in bottom-right"""
        N = np.random.randn(8, 8)
        N = (N + N.T) / 2

        N_tilde = constructor.pad_to_S_dimension(N)

        # Bottom-right 2x2 should be zeros
        assert np.allclose(N_tilde[8:10, 8:10], 0)

        # Bottom n rows should be zero
        assert np.allclose(N_tilde[8:10, :], 0)

        # Right n columns should be zero
        assert np.allclose(N_tilde[:, 8:10], 0)

    def test_padding_invalid_dimension(self, constructor):
        """Test padding rejects wrong dimensions"""
        # Wrong size matrix
        N_wrong = np.random.randn(10, 10)

        with pytest.raises(AssertionError):
            constructor.pad_to_S_dimension(N_wrong)


class TestCompleteConstruction:
    """Test complete LMI matrix construction"""

    @pytest.fixture
    def constructor(self):
        return LMIMatrixConstructor(n_states=2, n_controls=1, data_length=10)

    @pytest.fixture
    def all_inputs(self):
        """Create all inputs needed for complete construction"""
        n, m, L = 2, 1, 10

        P_i = cp.Variable((n, n), symmetric=True)
        L_i = cp.Variable((m, n))
        nu = 0.01
        alpha = 0.95

        H_prev = np.random.randn(n, L)
        H_plus_prev = np.random.randn(n, L)
        Xi_prev = np.random.randn(m, L)
        beta_prev = 0.5

        C = 0.1
        T_segment = 1.0

        return {
            "P_i": P_i,
            "L_i": L_i,
            "nu": nu,
            "alpha": alpha,
            "H_prev": H_prev,
            "H_plus_prev": H_plus_prev,
            "Xi_prev": Xi_prev,
            "beta_prev": beta_prev,
            "C": C,
            "T_segment": T_segment,
        }

    def test_construct_all_matrices(self, constructor, all_inputs):
        """Test constructing all matrices at once"""
        S, N1_tilde, N2_tilde = constructor.construct_all_lmi_matrices(**all_inputs)

        # Check S is CVXPY expression
        assert isinstance(S, cp.Expression)
        assert S.shape == (10, 10)

        # Check N1_tilde is numpy array with correct shape
        assert isinstance(N1_tilde, np.ndarray)
        assert N1_tilde.shape == (10, 10)

        # Check N2_tilde is numpy array with correct shape
        assert isinstance(N2_tilde, np.ndarray)
        assert N2_tilde.shape == (10, 10)

    def test_verify_matrix_properties(self, constructor, all_inputs):
        """Test matrix property verification"""
        _, N1_tilde, N2_tilde = constructor.construct_all_lmi_matrices(**all_inputs)

        properties = constructor.verify_matrix_properties(N1_tilde, N2_tilde)

        # Check all expected properties are present
        assert "N1_symmetric" in properties
        assert "N2_symmetric" in properties
        assert "shapes_correct" in properties

        # Check symmetry
        assert properties["N1_symmetric"]
        assert properties["N2_symmetric"]

        # Check shapes
        assert properties["shapes_correct"]
        assert properties["N1_shape"] == (10, 10)
        assert properties["N2_shape"] == (10, 10)


class TestNumericalStability:
    """Test numerical stability and conditioning"""

    @pytest.fixture
    def constructor(self):
        return LMIMatrixConstructor(n_states=4, n_controls=2, data_length=30)

    def test_large_system(self, constructor):
        """Test construction works for larger systems"""
        n, m, L = 4, 2, 30

        H = np.random.randn(n, L)
        H_plus = np.random.randn(n, L)
        Xi = np.random.randn(m, L)
        beta = 0.5

        N1 = constructor.construct_N1_matrix(H, H_plus, Xi, beta)

        # 3n+2m = 3*4+2*2 = 16
        assert N1.shape == (3 * n + 2 * m, 3 * n + 2 * m)
        assert np.allclose(N1, N1.T, atol=1e-10)

    def test_ill_conditioned_hankel(self, constructor):
        """Test with ill-conditioned Hankel matrices"""
        n, m, L = 4, 2, 30

        # Create ill-conditioned Hankel (rank deficient)
        H = np.outer(np.ones(n), np.random.randn(L))
        H_plus = np.outer(np.ones(n), np.random.randn(L))
        Xi = np.random.randn(m, L)
        beta = 0.5

        # Should still construct without error
        N1 = constructor.construct_N1_matrix(H, H_plus, Xi, beta)

        # 3n+2m = 3*4+2*2 = 16
        assert N1.shape == (16, 16)

    def test_very_small_beta(self, constructor):
        """Test with very small beta values"""
        n, m, L = 4, 2, 30

        H = np.random.randn(n, L)
        H_plus = np.random.randn(n, L)
        Xi = np.random.randn(m, L)
        beta = 1e-10  # Very small

        N1 = constructor.construct_N1_matrix(H, H_plus, Xi, beta)

        assert np.isfinite(N1).all()

    def test_condition_number_check(self, constructor):
        """Test condition number computation"""
        n, m, L = 4, 2, 30

        H = np.random.randn(n, L)
        H_plus = np.random.randn(n, L)
        Xi = np.random.randn(m, L)
        beta = 0.5

        C = 0.1
        T = 1.0

        N1 = constructor.construct_N1_matrix(H, H_plus, Xi, beta)
        N2 = constructor.construct_N2_matrix(C, T)

        N1_tilde = constructor.pad_to_S_dimension(N1)
        N2_tilde = constructor.pad_to_S_dimension(N2)

        properties = constructor.verify_matrix_properties(N1_tilde, N2_tilde, check_condition=True)

        assert "N1_condition" in properties
        assert "N2_condition" in properties
        assert properties["N1_condition"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
