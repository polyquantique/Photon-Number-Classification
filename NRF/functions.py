import numpy as np
from scipy.stats import poisson, geom
from scipy.signal import convolve2d
from lmfit import Minimizer, Parameters

def loss_mat(eta, cutoff):  # pragma: no cover
    r"""Constructs a binomial loss matrix with transmission eta up to n photons.

    Args:
        eta (float): Transmission coefficient. ``eta=0.0`` corresponds to complete loss and ``eta=1.0`` corresponds to no loss.
        cutoff (int): cutoff in Fock space.

    Returns:
        array: :math:`n\times n` matrix representing the loss.
    """
    # If full transmission return the identity

    if eta < 0.0 or eta > 1.0:
        raise ValueError("The transmission parameter eta should be a number between 0 and 1.")

    if eta == 1.0:
        return np.identity(cutoff)

    # Otherwise construct the matrix elements recursively
    lm = np.zeros((cutoff, cutoff))
    mu = 1.0 - eta
    lm[:, 0] = mu ** (np.arange(cutoff))
    for i in range(cutoff):
        for j in range(1, i + 1):
            lm[i, j] = lm[i, j - 1] * (eta / mu) * (i - j + 1) / (j)
    return lm

def twinbeam_pmf(params, cutoff=50, sq_label="sq_", noise_label="noise"):
    r"""Contructs the joint probability mass function of a conjugate source for a total
    of n photons in both signal idler and for an overall loss after generation
    characterized by the transmissions etas and etai.
    The source is described by either conjugate (correlated) and uncorrelated parts.

    Args:
        params (dict): Parameter dictionary, with possible keys "noise_s", "noise_i" for the
        Poisson noise mean photon numbers, "eta_s", "eta_i" for the transmission of the twin_beams,
        "n_modes" describing the number of twin_beams and the parameters sq_0,..,sq_n where
        n = n_modes giving the mean photon numbers of the different twin_beams.
        cutoff (int): Fock cutoff.
        sq_label (string): label for the squeezing parameters.
        noise_label (string): label for the noise parameters.

    Returns:
        (array): `n\times n` matrix representing the joint probability mass function
    """
    if noise_label + "_s" in params:
        noise_s = float(params[noise_label + "_s"])
    else:
        noise_s = 0.0

    if noise_label + "_i" in params:
        noise_i = float(params[noise_label + "_i"])
    else:
        noise_i = 0.0

    if "eta_s" in params:
        eta_s = float(params["eta_s"])
    else:
        eta_s = 1.0

    if "eta_i" in params:
        eta_i = float(params["eta_i"])
    else:
        eta_i = 1.0

    # First convolve all the 1-d distributions.
    ns = poisson.pmf(np.arange(cutoff), noise_s)
    ni = poisson.pmf(np.arange(cutoff), noise_i)

    joint_pmf = np.outer(ns, ni)
    # Then convolve with the twin beam distributions if there are any.
    if "n_modes" in params:
        n_modes = int(params["n_modes"])
        sq = [float(params[sq_label + str(i)]) for i in range(n_modes)]
        loss_mat_ns = loss_mat(eta_s, cutoff).T
        loss_mat_ni = loss_mat(eta_i, cutoff)
        twin_pmf = np.zeros([cutoff, cutoff])
        twin_pmf[0, 0] = 1.0
        for nmean in sq:
            twin_pmf = convolve2d(
                twin_pmf,
                np.diag(
                    geom.pmf(
                        np.arange(1, cutoff + 1),
                        1 / (1.0 + nmean),
                    )
                ),
            )[0:cutoff, 0:cutoff]
        twin_pmf = loss_mat_ns @ twin_pmf @ loss_mat_ni
        joint_pmf = convolve2d(twin_pmf, joint_pmf)[:cutoff, :cutoff]

    return joint_pmf

def two_schmidt_mode_guess(jpd_data, sq_label="sq_", noise_label="noise"):
    """Given a two mode histogram, this function generates a "physically" motivated guess for the loss, Schmidt occupations
    and dark counts parameters.

    This model is sensible only if the average g2 of signal and idler is above 1.5.

    Args:
        jpd_data (array): rectangular array with the probability mass functions of the photon events
        sq_label (string): label for the squeezing parameters.
        noise_label (string): label for the noise parameters.

    Returns:
        dict: dictionary containing a set of "reasonable" model parameters
    """
    res = marginal_calcs_2d(jpd_data)
    g2avg = np.max([0.5 * (res["g2_s"] + res["g2_i"]), 1.5])
    Nbar = 1.0 / (res["g11"] - g2avg)
    n0 = 0.5 * (1 + np.sqrt(2 * g2avg - 3)) * Nbar
    n1 = 0.5 * (1 - np.sqrt(2 * g2avg - 3)) * Nbar
    etas = res["n_s"] / Nbar
    etai = res["n_i"] / Nbar
    noise = np.abs(res["g2_s"] - res["g2_i"]) * Nbar
    return {
        "eta_s": etas,
        "eta_i": etai,
        sq_label + "0": n0,
        sq_label + "1": n1,
        noise_label + "_s": noise * etas,
        noise_label + "_i": noise * etai,
        "n_modes": 2,
    }


def marginal_calcs_2d(jpd_data, as_dict=True):
    """Given a two dimensional array of probabilities it calculates
    the mean photon numbers, their g2's and g11.

    It returns these values as a dictionary or as an array.

    Args:
        jpd_data (array): probability mass function of the photon events
        as_dict (boolean): whether to return the results as a dictionary

    Returns:
        dict or array: values of the mean photons number for signal and idlers,
            their corresponding g2, their g11 and their noise reduction factor (nrf).
    """
    inta, intb = jpd_data.shape
    na = np.arange(inta)
    nb = np.arange(intb)
    ps = np.sum(jpd_data, axis=1)
    pi = np.sum(jpd_data, axis=0)
    ns = ps @ na
    ni = pi @ nb
    ns2 = ps @ (na ** 2)
    ni2 = pi @ (nb ** 2)
    ns3 = ps @ (na ** 3)
    ni3 = pi @ (nb ** 3)
    g2s = (ns2 - ns) / ns ** 2
    g2i = (ni2 - ni) / ni ** 2
    g3s = (ns3 - 3 * ns2 + 2 * ns) / ns ** 3
    g3i = (ni3 - 3 * ni2 + 2 * ni) / ni ** 3
    g11 = (na @ jpd_data @ nb) / (ns * ni)
    nrf = np.sum([[((i - j) ** 2) * jpd_data[i, j] for i in range(inta)] for j in range(intb)])
    nrf -= np.sum([[(i - j) * jpd_data[i, j] for i in range(inta)] for j in range(intb)]) ** 2
    nrf /= ns + ni
    if as_dict is True:
        return {
            "n_s": ns,
            "n_i": ni,
            "g11": g11,
            "g2_s": g2s,
            "g2_i": g2i,
            "nrf": nrf,
            "g3_s": g3s,
            "g3_i": g3i,
        }
    return np.array([ns, ni, g11, g2s, g2i, nrf, g3s, g3i])


def gen_hist_2d(beam1, beam2):
    """Calculate the joint probability mass function of events.

    Args:
        beam1 (array): 1D events array containing the raw click events of first beam
        beam2 (array): 1D events array containing the raw click events of second beam

    Returns:
        array: probability mass function of the click patterns for the two beams
    """
    nx = np.max(beam1)
    ny = np.max(beam2)
    xedges = np.arange(nx + 2)
    yedges = np.arange(ny + 2)
    mass_fun, _, _ = np.histogram2d(beam1, beam2, bins=(xedges, yedges), normed=True)
    return mass_fun


def threshold_2d(ps, nmax, mmax):
    """Thresholds a 2D probability distribution by assigning events with more than nmax (mmax) photons
    in the first (second) axis to the nmax (mmax) bin.

    Args:
        ps (array): probability distribution
        nmax (int): threshold for first axis
        mmax (int): threshold for second axis

    Returns:
        array: thresholded probability distribution
    """
    probs = np.copy(ps[:nmax, :mmax])
    for i in range(mmax - 1):
        probs[nmax - 1, i] += np.sum(ps[nmax:, i])
    for i in range(nmax - 1):
        probs[i, mmax - 1] += np.sum(ps[i, mmax:])
    probs[nmax - 1, mmax - 1] = np.sum(ps[nmax - 1 :, mmax - 1 :])
    return probs


def fit_2d(
    pd_data,
    guess,
    do_not_vary=None,
    method="leastsq",
    threshold=False,
    cutoff=50,
    sq_label="sq_",
    noise_label="noise",
):
    """Returns a model fit from the parameter guess and the data

    Args:
        pd_data (array): one dimensional array of the probability distribution of the data
        guess (dict): dictionary with the guesses for the different parameters
        method (string): method to be used by the optimizer
        threshold (boolean or list): threshold photon number for the signal and idlers
        do_not_vary (list): list of variables that should be held constant during optimization
        cutoff (int): internal cutoff
        sq_label (string): label for the squeezing parameters.
        noise_label (string): label for the noise parameters.

    Returns:
        Object: object containing the optimized parameter and several goodness-of-fit statistics
    """
    if do_not_vary is None:
        do_not_vary = []
    pars_model = Parameters()
    n_modes = guess["n_modes"]
    pars_model.add("n_modes", value=n_modes, vary=False)
    for i in range(n_modes):
        pars_model.add(sq_label + str(i), value=guess["sq_" + str(i)], min=0.0)

    if "eta_s" in do_not_vary:
        pars_model.add("eta_s", value=guess["eta_s"], vary=False)
    else:
        pars_model.add("eta_s", value=guess["eta_s"], min=0.0, max=1.0)

    if "eta_i" in do_not_vary:
        pars_model.add("eta_i", value=guess["eta_i"], vary=False)
    else:
        pars_model.add("eta_i", value=guess["eta_i"], min=0.0, max=1.0)

    if noise_label + "_s" in do_not_vary:
        pars_model.add(noise_label + "_s", value=guess[noise_label + "_s"], vary=False)
    else:
        pars_model.add(noise_label + "_s", value=guess[noise_label + "_s"], min=0.0)

    if noise_label + "_i" in do_not_vary:
        pars_model.add(noise_label + "_i", value=guess[noise_label + "_i"], vary=False)
    else:
        pars_model.add(noise_label + "_i", value=guess[noise_label + "_i"], min=0.0)

    if threshold:

        def model_2d(params, jpd_data):
            (dim_s, dim_i) = pd_data.shape
            joint_pmf = twinbeam_pmf(params, cutoff=cutoff)
            return threshold_2d(joint_pmf, dim_s, dim_i) - threshold_2d(jpd_data, dim_s, dim_i)

    else:

        def model_2d(params, jpd_data):
            (dim_s, dim_i) = pd_data.shape
            joint_pmf = twinbeam_pmf(params, cutoff=cutoff)[:dim_s, :dim_i]
            return joint_pmf - jpd_data

    minner_model = Minimizer(model_2d, pars_model, fcn_args=([pd_data]))
    result_model = minner_model.minimize(method=method)

    return result_model


def compute_nrf(joint_dist: np.ndarray) -> float:
    r"""Computes the Noise Reduction Factor (NRF) from a joint probability mass function."""
    # Photon number values for each mode
    n1 = np.arange(joint_dist.shape[0])[:, None]  # shape (N1, 1)
    n2 = np.arange(joint_dist.shape[1])[None, :]  # shape (1, N2)

    # Mean photon numbers
    mean_n1 = np.sum(n1 * joint_dist)
    mean_n2 = np.sum(n2 * joint_dist)

    # Variance of the difference (n1 - n2)
    mean_diff = mean_n1 - mean_n2
    mean_diff2 = np.sum((n1 - n2) ** 2 * joint_dist)
    var_diff = mean_diff2 - mean_diff ** 2

    # NRF: variance of difference divided by sum of means
    nrf = var_diff / (mean_n1 + mean_n2 + 1e-8)
    return nrf


def compute_g2(dist: np.ndarray) -> float:
    r"""Computes the second-order correlation function g2 from a probability distribution."""
    n = np.arange(len(dist))
    mean_n = np.sum(n * dist)
    mean_n2 = np.sum(n * (n - 1) * dist)
    if mean_n == 0:
        return np.nan
    return mean_n2 / (mean_n ** 2 + 1e-8)