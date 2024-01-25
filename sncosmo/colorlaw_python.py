import numpy as np

B_WAVELENGTH = 4302.57
V_WAVELENGTH = 5428.55
colorlaw_coeffs = [-0.504294, 0.787691, -0.461715, 0.0815619]
colorlaw_range = (2800., 7000.)

# old python implementation
def colorlaw_python(wave):
    v_minus_b = V_WAVELENGTH - B_WAVELENGTH

    l = (wave - B_WAVELENGTH) / v_minus_b
    l_lo = (colorlaw_range[0] - B_WAVELENGTH) / v_minus_b
    l_hi = (colorlaw_range[1] - B_WAVELENGTH) / v_minus_b

    alpha = 1. - sum(colorlaw_coeffs)
    coeffs = [0., alpha]
    coeffs.extend(colorlaw_coeffs)
    coeffs = np.array(coeffs)
    prime_coeffs = (np.arange(len(coeffs)) * coeffs)[1:]

    extinction = np.empty_like(wave)

    # Blue side
    idx_lo = l < l_lo
    p_lo = np.polyval(np.flipud(coeffs), l_lo)
    pprime_lo = np.polyval(np.flipud(prime_coeffs), l_lo)
    extinction[idx_lo] = p_lo + pprime_lo * (l[idx_lo] - l_lo)

    # Red side
    idx_hi = l > l_hi
    p_hi = np.polyval(np.flipud(coeffs), l_hi)
    pprime_hi = np.polyval(np.flipud(prime_coeffs), l_hi)
    extinction[idx_hi] = p_hi + pprime_hi * (l[idx_hi] - l_hi)

    # In between
    idx_between = np.invert(idx_lo | idx_hi)
    extinction[idx_between] = np.polyval(np.flipud(coeffs), l[idx_between])

    return -extinction

