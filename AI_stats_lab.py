import numpy as np

# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return the bivariate Gaussian PDF f_XY(x,y).
    """
    norm = 1.0 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    Q = (
        ((x - mu_x)**2 / sigma_x**2)
        - 2 * rho * ((x - mu_x) * (y - mu_y)) / (sigma_x * sigma_y)
        + ((y - mu_y)**2 / sigma_y**2)
    )
    return norm * np.exp(-Q / (2 * (1 - rho**2)))


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    """
    Return marginal Gaussian PDF of X.
    """
    return (1.0 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(-0.5 * ((x - mu_x) / sigma_x)**2)


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    """
    Return marginal Gaussian PDF of Y.
    """
    return (1.0 / (np.sqrt(2 * np.pi) * sigma_y)) * np.exp(-0.5 * ((y - mu_y) / sigma_y)**2)


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return covariance matrix:
    [[sigma_x^2, rho*sigma_x*sigma_y],
     [rho*sigma_x*sigma_y, sigma_y^2]]
    """
    return np.array([
        [sigma_x**2,              rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2             ]
    ])


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):
    """
    Numerically approximate integral of joint Gaussian PDF
    over the rectangle:
      [mu_x - 4*sigma_x, mu_x + 4*sigma_x]
      x
      [mu_y - 4*sigma_y, mu_y + 4*sigma_y]
    Uses trapezoidal integration.
    """
    x = np.linspace(mu_x - 4 * sigma_x, mu_x + 4 * sigma_x, n)
    y = np.linspace(mu_y - 4 * sigma_y, mu_y + 4 * sigma_y, n)
    X, Y = np.meshgrid(x, y)
    Z = joint_gaussian_pdf(X, Y, mu_x, mu_y, sigma_x, sigma_y, rho)
    # Trapezoidal rule in both dimensions (compatible with numpy >= 2.0)
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return trapz(trapz(Z, x, axis=1), y)


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    """
    Generate n samples from a jointly Gaussian distribution.
    Return two arrays: x_samples, y_samples
    """
    rng = np.random.default_rng(seed)
    mean = [mu_x, mu_y]
    cov = covariance_matrix(sigma_x, sigma_y, rho)
    samples = rng.multivariate_normal(mean, cov, size=n)
    return samples[:, 0], samples[:, 1]


def sample_means(x_samples, y_samples):
    """
    Return sample means of X and Y.
    """
    return np.mean(x_samples), np.mean(y_samples)


def sample_covariance_matrix(x_samples, y_samples):
    """
    Return 2x2 sample covariance matrix using denominator n-1.
    """
    data = np.vstack([x_samples, y_samples])   # shape (2, n)
    return np.cov(data, ddof=1)


def sample_correlation(x_samples, y_samples):
    """
    Return sample correlation coefficient.
    """
    cov_mat = sample_covariance_matrix(x_samples, y_samples)
    return cov_mat[0, 1] / np.sqrt(cov_mat[0, 0] * cov_mat[1, 1])


def gaussian_independence_check(rho):
    """
    For jointly Gaussian variables:
    return True if rho is zero, otherwise False.
    """
    return rho == 0


def zero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0 and check that
    sample covariance is approximately zero.
    Return True or False.
    """
    x_samples, y_samples = generate_joint_gaussian_samples(n=n, rho=0, seed=42)
    cov_mat = sample_covariance_matrix(x_samples, y_samples)
    sample_cov = cov_mat[0, 1]
    return bool(abs(sample_cov) < 0.1)   # should be very close to 0


def nonzero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0.6 and check that
    sample covariance is close to rho*sigma_x*sigma_y.
    Return True or False.
    """
    rho, sigma_x, sigma_y = 0.6, 2, 3
    x_samples, y_samples = generate_joint_gaussian_samples(n=n, rho=rho, seed=42)
    cov_mat = sample_covariance_matrix(x_samples, y_samples)
    sample_cov = cov_mat[0, 1]
    expected_cov = rho * sigma_x * sigma_y   # 3.6
    return bool(abs(sample_cov - expected_cov) < 0.1)