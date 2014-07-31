"""
Conversion between color spaces.

.. note:: This module makes extensive use of imports within functions.
    That stinks.
"""

import math
import logging

import numpy

from colormath import color_constants
from colormath import spectral_constants
from colormath.color_objects import ColorBase, XYZColor, sRGBColor, LCHabColor, \
    LCHuvColor, LabColor, xyYColor, LuvColor, HSVColor, HSLColor, CMYColor, \
    CMYKColor, BaseRGBColor, IPTColor, ICHptColor, hIPTColor, hICHptColor
from colormath.chromatic_adaptation import apply_chromatic_adaptation
from colormath.color_exceptions import InvalidIlluminantError, UndefinedConversionError


logger = logging.getLogger(__name__)


# noinspection PyPep8Naming
def apply_RGB_matrix(var1, var2, var3, rgb_type, convtype="xyz_to_rgb"):
    """
    Applies an RGB working matrix to convert from XYZ to RGB.
    The arguments are tersely named var1, var2, and var3 to allow for the passing
    of XYZ _or_ RGB values. var1 is X for XYZ, and R for RGB. var2 and var3
    follow suite.
    """

    convtype = convtype.lower()
    # Retrieve the appropriate transformation matrix from the constants.
    rgb_matrix = rgb_type.conversion_matrices[convtype]

    logger.debug("  \* Applying RGB conversion matrix: %s->%s",
                 rgb_type.__class__.__name__, convtype)
    # Stuff the RGB/XYZ values into a NumPy matrix for conversion.
    var_matrix = numpy.array((
        var1, var2, var3
    ))
    # Perform the adaptation via matrix multiplication.
    result_matrix = numpy.dot(var_matrix, rgb_matrix)
    return result_matrix[0], result_matrix[1], result_matrix[2]


# noinspection PyPep8Naming,PyUnusedLocal
def Spectral_to_XYZ(cobj, illuminant_override=None, *args, **kwargs):
    """
    Converts spectral readings to XYZ.
    """

    # If the user provides an illuminant_override numpy array, use it.
    if illuminant_override:
        reference_illum = illuminant_override
    else:
        # Otherwise, look up the illuminant from known standards based
        # on the value of 'illuminant' pulled from the SpectralColor object.
        try:
            reference_illum = spectral_constants.REF_ILLUM_TABLE[cobj.illuminant]
        except KeyError:
            raise InvalidIlluminantError(cobj.illuminant)

    # Get the spectral distribution of the selected standard observer.
    if cobj.observer == '10':
        std_obs_x = spectral_constants.STDOBSERV_X10
        std_obs_y = spectral_constants.STDOBSERV_Y10
        std_obs_z = spectral_constants.STDOBSERV_Z10
    else:
        # Assume 2 degree, since it is theoretically the only other possibility.
        std_obs_x = spectral_constants.STDOBSERV_X2
        std_obs_y = spectral_constants.STDOBSERV_Y2
        std_obs_z = spectral_constants.STDOBSERV_Z2

    # This is a NumPy array containing the spectral distribution of the color.
    sample = cobj.get_numpy_array()

    # The denominator is constant throughout the entire calculation for X,
    # Y, and Z coordinates. Calculate it once and re-use.
    denom = std_obs_y * reference_illum

    # This is also a common element in the calculation whereby the sample
    # NumPy array is multiplied by the reference illuminant's power distribution
    # (which is also a NumPy array).
    sample_by_ref_illum = sample * reference_illum

    # Calculate the numerator of the equation to find X.
    x_numerator = sample_by_ref_illum * std_obs_x
    y_numerator = sample_by_ref_illum * std_obs_y
    z_numerator = sample_by_ref_illum * std_obs_z

    xyz_x = x_numerator.sum() / denom.sum()
    xyz_y = y_numerator.sum() / denom.sum()
    xyz_z = z_numerator.sum() / denom.sum()

    return XYZColor(
        xyz_x, xyz_y, xyz_z, observer=cobj.observer, illuminant=cobj.illuminant)


# noinspection PyPep8Naming,PyUnusedLocal
def Lab_to_LCHab(cobj, *args, **kwargs):
    """
    Convert from CIE Lab to LCH(ab).
    """

    lch_l = cobj.lab_l
    lch_c = math.sqrt(math.pow(float(cobj.lab_a), 2) + math.pow(float(cobj.lab_b), 2))
    lch_h = math.atan2(float(cobj.lab_b), float(cobj.lab_a))

    if lch_h > 0:
        lch_h = (lch_h / math.pi) * 180
    else:
        lch_h = 360 - (math.fabs(lch_h) / math.pi) * 180

    return LCHabColor(
        lch_l, lch_c, lch_h, observer=cobj.observer, illuminant=cobj.illuminant)


# noinspection PyPep8Naming,PyUnusedLocal
def Lab_to_XYZ(cobj, *args, **kwargs):
    """
    Convert from Lab to XYZ
    """

    illum = cobj.get_illuminant_xyz()
    xyz_y = (cobj.lab_l + 16.0) / 116.0
    xyz_x = cobj.lab_a / 500.0 + xyz_y
    xyz_z = xyz_y - cobj.lab_b / 200.0

    if math.pow(xyz_y, 3) > color_constants.CIE_E:
        xyz_y = math.pow(xyz_y, 3)
    else:
        xyz_y = (xyz_y - 16.0 / 116.0) / 7.787

    if math.pow(xyz_x, 3) > color_constants.CIE_E:
        xyz_x = math.pow(xyz_x, 3)
    else:
        xyz_x = (xyz_x - 16.0 / 116.0) / 7.787

    if math.pow(xyz_z, 3) > color_constants.CIE_E:
        xyz_z = math.pow(xyz_z, 3)
    else:
        xyz_z = (xyz_z - 16.0 / 116.0) / 7.787

    xyz_x = (illum["X"] * xyz_x)
    xyz_y = (illum["Y"] * xyz_y)
    xyz_z = (illum["Z"] * xyz_z)

    return XYZColor(
        xyz_x, xyz_y, xyz_z, observer=cobj.observer, illuminant=cobj.illuminant)


# noinspection PyPep8Naming,PyUnusedLocal
def Luv_to_LCHuv(cobj, *args, **kwargs):
    """
    Convert from CIE Luv to LCH(uv).
    """

    lch_l = cobj.luv_l
    lch_c = math.sqrt(math.pow(cobj.luv_u, 2.0) + math.pow(cobj.luv_v, 2.0))
    lch_h = math.atan2(float(cobj.luv_v), float(cobj.luv_u))

    if lch_h > 0:
        lch_h = (lch_h / math.pi) * 180
    else:
        lch_h = 360 - (math.fabs(lch_h) / math.pi) * 180
    return LCHuvColor(
        lch_l, lch_c, lch_h, observer=cobj.observer, illuminant=cobj.illuminant)


# noinspection PyPep8Naming,PyUnusedLocal
def Luv_to_XYZ(cobj, *args, **kwargs):
    """
    Convert from Luv to XYZ.
    """

    illum = cobj.get_illuminant_xyz()
    # Without Light, there is no color. Short-circuit this and avoid some
    # zero division errors in the var_a_frac calculation.
    if cobj.luv_l <= 0.0:
        xyz_x = 0.0
        xyz_y = 0.0
        xyz_z = 0.0
        return XYZColor(
            xyz_x, xyz_y, xyz_z, observer=cobj.observer, illuminant=cobj.illuminant)

    # Various variables used throughout the conversion.
    cie_k_times_e = color_constants.CIE_K * color_constants.CIE_E
    u_sub_0 = (4.0 * illum["X"]) / (illum["X"] + 15.0 * illum["Y"] + 3.0 * illum["Z"])
    v_sub_0 = (9.0 * illum["Y"]) / (illum["X"] + 15.0 * illum["Y"] + 3.0 * illum["Z"])
    var_u = cobj.luv_u / (13.0 * cobj.luv_l) + u_sub_0
    var_v = cobj.luv_v / (13.0 * cobj.luv_l) + v_sub_0

    # Y-coordinate calculations.
    if cobj.luv_l > cie_k_times_e:
        xyz_y = math.pow((cobj.luv_l + 16.0) / 116.0, 3.0)
    else:
        xyz_y = cobj.luv_l / color_constants.CIE_K

    # X-coordinate calculation.
    xyz_x = xyz_y * 9.0 * var_u / (4.0 * var_v)
    # Z-coordinate calculation.
    xyz_z = xyz_y * (12.0 - 3.0 * var_u - 20.0 * var_v) / (4.0 * var_v)

    return XYZColor(
        xyz_x, xyz_y, xyz_z, illuminant=cobj.illuminant, observer=cobj.observer)


# noinspection PyPep8Naming,PyUnusedLocal
def LCHab_to_Lab(cobj, *args, **kwargs):
    """
    Convert from LCH(ab) to Lab.
    """

    lab_l = cobj.lch_l
    lab_a = math.cos(math.radians(cobj.lch_h)) * cobj.lch_c
    lab_b = math.sin(math.radians(cobj.lch_h)) * cobj.lch_c
    return LabColor(
        lab_l, lab_a, lab_b, illuminant=cobj.illuminant, observer=cobj.observer)


# noinspection PyPep8Naming,PyUnusedLocal
def LCHuv_to_Luv(cobj, *args, **kwargs):
    """
    Convert from LCH(uv) to Luv.
    """

    luv_l = cobj.lch_l
    luv_u = math.cos(math.radians(cobj.lch_h)) * cobj.lch_c
    luv_v = math.sin(math.radians(cobj.lch_h)) * cobj.lch_c
    return LuvColor(
        luv_l, luv_u, luv_v, illuminant=cobj.illuminant, observer=cobj.observer)


# noinspection PyPep8Naming,PyUnusedLocal
def xyY_to_XYZ(cobj, *args, **kwargs):
    """
    Convert from xyY to XYZ.
    """
    # avoid division by zero
    if cobj.xyy_y==0.0:
        xyz_x=0.0
        xyz_y=0.0
        xyz_z=0.0
    else:
        xyz_x = (cobj.xyy_x * cobj.xyy_Y) / cobj.xyy_y
        xyz_y = cobj.xyy_Y
        xyz_z = ((1.0 - cobj.xyy_x - cobj.xyy_y) * xyz_y) / cobj.xyy_y

    return XYZColor(
        xyz_x, xyz_y, xyz_z, illuminant=cobj.illuminant, observer=cobj.observer)


# noinspection PyPep8Naming,PyUnusedLocal
def XYZ_to_xyY(cobj, *args, **kwargs):
    """
    Convert from XYZ to xyY.
    """
    xyz_sum = cobj.xyz_x + cobj.xyz_y + cobj.xyz_z
    # avoid division by zero
    if xyz_sum == 0.0:
        xyy_x = 0.0
        xyy_y = 0.0
    else:
        xyy_x = cobj.xyz_x / xyz_sum
        xyy_y = cobj.xyz_y / xyz_sum
    xyy_Y = cobj.xyz_y

    return xyYColor(
        xyy_x, xyy_y, xyy_Y, observer=cobj.observer, illuminant=cobj.illuminant)


# noinspection PyPep8Naming,PyUnusedLocal
def XYZ_to_Luv(cobj, *args, **kwargs):
    """
    Convert from XYZ to Luv
    """

    temp_x = cobj.xyz_x
    temp_y = cobj.xyz_y
    temp_z = cobj.xyz_z
    denom = temp_x + (15.0 * temp_y) + (3.0 * temp_z)
    # avoid division by zero
    if denom == 0.0:
        luv_u = 0.0
        luv_v = 0.0
    else:
        luv_u = (4.0 * temp_x) / denom
        luv_v = (9.0 * temp_y) / denom

    illum = cobj.get_illuminant_xyz()
    temp_y = temp_y / illum["Y"]
    if temp_y > color_constants.CIE_E:
        temp_y = math.pow(temp_y, (1.0 / 3.0))
    else:
        temp_y = (7.787 * temp_y) + (16.0 / 116.0)

    ref_U = (4.0 * illum["X"]) / (illum["X"] + (15.0 * illum["Y"]) + (3.0 * illum["Z"]))
    ref_V = (9.0 * illum["Y"]) / (illum["X"] + (15.0 * illum["Y"]) + (3.0 * illum["Z"]))

    luv_l = (116.0 * temp_y) - 16.0
    luv_u = 13.0 * luv_l * (luv_u - ref_U)
    luv_v = 13.0 * luv_l * (luv_v - ref_V)

    return LuvColor(
        luv_l, luv_u, luv_v, observer=cobj.observer, illuminant=cobj.illuminant)


# noinspection PyPep8Naming,PyUnusedLocal
def XYZ_to_Lab(cobj, *args, **kwargs):
    """
    Converts XYZ to Lab.
    """

    illum = cobj.get_illuminant_xyz()
    temp_x = cobj.xyz_x / illum["X"]
    temp_y = cobj.xyz_y / illum["Y"]
    temp_z = cobj.xyz_z / illum["Z"]

    if temp_x > color_constants.CIE_E:
        temp_x = math.pow(temp_x, (1.0 / 3.0))
    else:
        temp_x = (7.787 * temp_x) + (16.0 / 116.0)

    if temp_y > color_constants.CIE_E:
        temp_y = math.pow(temp_y, (1.0 / 3.0))
    else:
        temp_y = (7.787 * temp_y) + (16.0 / 116.0)

    if temp_z > color_constants.CIE_E:
        temp_z = math.pow(temp_z, (1.0 / 3.0))
    else:
        temp_z = (7.787 * temp_z) + (16.0 / 116.0)

    lab_l = (116.0 * temp_y) - 16.0
    lab_a = 500.0 * (temp_x - temp_y)
    lab_b = 200.0 * (temp_y - temp_z)
    return LabColor(
        lab_l, lab_a, lab_b, observer=cobj.observer, illuminant=cobj.illuminant)


def sign_safe_power(x, y):
    if x==0.0:
        return 0.0
    if x>0:
        return math.pow(x,y)
    return -math.pow(abs(x),y)

# noinspection PyPep8Naming,PyUnusedLocal
def XYZ_to_RGB(cobj, target_rgb, *args, **kwargs):
    """
    XYZ to RGB conversion.
    """

    temp_X = cobj.xyz_x
    temp_Y = cobj.xyz_y
    temp_Z = cobj.xyz_z

    logger.debug("  \- Target RGB space: %s", target_rgb)
    target_illum = target_rgb.native_illuminant
    logger.debug("  \- Target native illuminant: %s", target_illum)
    logger.debug("  \- XYZ color's illuminant: %s", cobj.illuminant)

    # If the XYZ values were taken with a different reference white than the
    # native reference white of the target RGB space, a transformation matrix
    # must be applied.
    if cobj.illuminant != target_illum:
        logger.debug("  \* Applying transformation from %s to %s ",
                     cobj.illuminant, target_illum)
        # Get the adjusted XYZ values, adapted for the target illuminant.
        temp_X, temp_Y, temp_Z = apply_chromatic_adaptation(
            temp_X, temp_Y, temp_Z,
            orig_illum=cobj.illuminant, targ_illum=target_illum)
        logger.debug("  \*   New values: %.3f, %.3f, %.3f",
                     temp_X, temp_Y, temp_Z)

    # Apply an RGB working space matrix to the XYZ values (matrix mul).
    rgb_r, rgb_g, rgb_b = apply_RGB_matrix(
        temp_X, temp_Y, temp_Z,
        rgb_type=target_rgb, convtype="xyz_to_rgb")

    # v
    linear_channels = dict(r=rgb_r, g=rgb_g, b=rgb_b)
    # V
    nonlinear_channels = {}
    if target_rgb == sRGBColor:
        for channel in ['r', 'g', 'b']:
            v = linear_channels[channel]
            if v <= 0.0031308:
                nonlinear_channels[channel] = v * 12.92
            else:
                nonlinear_channels[channel] = 1.055 * math.pow(v, 1 / 2.4) - 0.055
    else:
        # If it's not sRGB...
        for channel in ['r', 'g', 'b']:
            v = linear_channels[channel]
            nonlinear_channels[channel]= sign_safe_power(v, 1 /
                                                         target_rgb.rgb_gamma)

    return target_rgb(
        nonlinear_channels['r'], nonlinear_channels['g'], nonlinear_channels['b'])


# noinspection PyPep8Naming,PyUnusedLocal
def RGB_to_XYZ(cobj, target_illuminant=None, *args, **kwargs):
    """
    RGB to XYZ conversion. Expects 0-255 RGB values.

    Based off of: http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    """

    # Will contain linearized RGB channels (removed the gamma func).
    linear_channels = {}

    if isinstance(cobj, sRGBColor):
        for channel in ['r', 'g', 'b']:
            V = getattr(cobj, 'rgb_' + channel)
            if V <= 0.04045:
                linear_channels[channel] = V / 12.92
            else:
                linear_channels[channel] = math.pow((V + 0.055) / 1.055, 2.4)
    else:
        # If it's not sRGB...
        gamma = cobj.rgb_gamma

        for channel in ['r', 'g', 'b']:
            V = getattr(cobj, 'rgb_' + channel)
            linear_channels[channel] = math.pow(V, gamma)

    # Apply an RGB working space matrix to the XYZ values (matrix mul).
    xyz_x, xyz_y, xyz_z = apply_RGB_matrix(
        linear_channels['r'], linear_channels['g'], linear_channels['b'],
        rgb_type=cobj, convtype="rgb_to_xyz")

    if target_illuminant is None:
        target_illuminant = cobj.native_illuminant

    # The illuminant of the original RGB object. This will always match
    # the RGB colorspace's native illuminant.
    illuminant = cobj.native_illuminant
    xyzcolor = XYZColor(xyz_x, xyz_y, xyz_z, illuminant=illuminant)
    # This will take care of any illuminant changes for us (if source
    # illuminant != target illuminant).
    xyzcolor.apply_adaptation(target_illuminant)

    return xyzcolor


# noinspection PyPep8Naming,PyUnusedLocal
def __RGB_to_Hue(var_R, var_G, var_B, var_min, var_max):
    """
    For RGB_to_HSL and RGB_to_HSV, the Hue (H) component is calculated in
    the same way.
    """

    if var_max == var_min:
        return 0.0
    elif var_max == var_R:
        return (60.0 * ((var_G - var_B) / (var_max - var_min)) + 360) % 360.0
    elif var_max == var_G:
        return 60.0 * ((var_B - var_R) / (var_max - var_min)) + 120
    elif var_max == var_B:
        return 60.0 * ((var_R - var_G) / (var_max - var_min)) + 240.0


# noinspection PyPep8Naming,PyUnusedLocal
def RGB_to_HSV(cobj, *args, **kwargs):
    """
    Converts from RGB to HSV.

    H values are in degrees and are 0 to 360.
    S values are a percentage, 0.0 to 1.0.
    V values are a percentage, 0.0 to 1.0.
    """

    var_R = cobj.rgb_r
    var_G = cobj.rgb_g
    var_B = cobj.rgb_b

    var_max = max(var_R, var_G, var_B)
    var_min = min(var_R, var_G, var_B)

    var_H = __RGB_to_Hue(var_R, var_G, var_B, var_min, var_max)

    if var_max == 0:
        var_S = 0
    else:
        var_S = 1.0 - (var_min / var_max)

    var_V = var_max

    hsv_h = var_H
    hsv_s = var_S
    hsv_v = var_V

    return HSVColor(
        var_H, var_S, var_V)


# noinspection PyPep8Naming,PyUnusedLocal
def RGB_to_HSL(cobj, *args, **kwargs):
    """
    Converts from RGB to HSL.

    H values are in degrees and are 0 to 360.
    S values are a percentage, 0.0 to 1.0.
    L values are a percentage, 0.0 to 1.0.
    """

    var_R = cobj.rgb_r
    var_G = cobj.rgb_g
    var_B = cobj.rgb_b

    var_max = max(var_R, var_G, var_B)
    var_min = min(var_R, var_G, var_B)

    var_H = __RGB_to_Hue(var_R, var_G, var_B, var_min, var_max)
    var_L = 0.5 * (var_max + var_min)

    if var_max == var_min:
        var_S = 0
    elif var_L <= 0.5:
        var_S = (var_max - var_min) / (2.0 * var_L)
    else:
        var_S = (var_max - var_min) / (2.0 - (2.0 * var_L))

    return HSLColor(
        var_H, var_S, var_L)


# noinspection PyPep8Naming,PyUnusedLocal
def __Calc_HSL_to_RGB_Components(var_q, var_p, C):
    """
    This is used in HSL_to_RGB conversions on R, G, and B.
    """

    if C < 0:
        C += 1.0
    if C > 1:
        C -= 1.0

    # Computing C of vector (Color R, Color G, Color B)
    if C < (1.0 / 6.0):
        return var_p + ((var_q - var_p) * 6.0 * C)
    elif (1.0 / 6.0) <= C < 0.5:
        return var_q
    elif 0.5 <= C < (2.0 / 3.0):
        return var_p + ((var_q - var_p) * 6.0 * ((2.0 / 3.0) - C))
    else:
        return var_p


# noinspection PyPep8Naming,PyUnusedLocal
def HSV_to_RGB(cobj, target_rgb, *args, **kwargs):
    """
    HSV to RGB conversion.

    H values are in degrees and are 0 to 360.
    S values are a percentage, 0.0 to 1.0.
    V values are a percentage, 0.0 to 1.0.
    """

    H = cobj.hsv_h
    S = cobj.hsv_s
    V = cobj.hsv_v

    h_floored = int(math.floor(H))
    h_sub_i = int(h_floored / 60) % 6
    var_f = (H / 60.0) - (h_floored // 60)
    var_p = V * (1.0 - S)
    var_q = V * (1.0 - var_f * S)
    var_t = V * (1.0 - (1.0 - var_f) * S)

    if h_sub_i == 0:
        rgb_r = V
        rgb_g = var_t
        rgb_b = var_p
    elif h_sub_i == 1:
        rgb_r = var_q
        rgb_g = V
        rgb_b = var_p
    elif h_sub_i == 2:
        rgb_r = var_p
        rgb_g = V
        rgb_b = var_t
    elif h_sub_i == 3:
        rgb_r = var_p
        rgb_g = var_q
        rgb_b = V
    elif h_sub_i == 4:
        rgb_r = var_t
        rgb_g = var_p
        rgb_b = V
    elif h_sub_i == 5:
        rgb_r = V
        rgb_g = var_p
        rgb_b = var_q
    else:
        raise ValueError("Unable to convert HSL->RGB due to value error.")

    # In the event that they define an HSV color and want to convert it to
    # a particular RGB space, let them override it here.
    if target_rgb is not None:
        rgb_type = target_rgb
    else:
        rgb_type = cobj.rgb_type

    return target_rgb(rgb_r, rgb_g, rgb_b)


# noinspection PyPep8Naming,PyUnusedLocal
def HSL_to_RGB(cobj, target_rgb, *args, **kwargs):
    """
    HSL to RGB conversion.
    """

    H = cobj.hsl_h
    S = cobj.hsl_s
    L = cobj.hsl_l

    if L < 0.5:
        var_q = L * (1.0 + S)
    else:
        var_q = L + S - (L * S)

    var_p = 2.0 * L - var_q

    # H normalized to range [0,1]
    h_sub_k = (H / 360.0)

    t_sub_R = h_sub_k + (1.0 / 3.0)
    t_sub_G = h_sub_k
    t_sub_B = h_sub_k - (1.0 / 3.0)

    rgb_r = __Calc_HSL_to_RGB_Components(var_q, var_p, t_sub_R)
    rgb_g = __Calc_HSL_to_RGB_Components(var_q, var_p, t_sub_G)
    rgb_b = __Calc_HSL_to_RGB_Components(var_q, var_p, t_sub_B)

    # In the event that they define an HSV color and want to convert it to
    # a particular RGB space, let them override it here.
    if target_rgb is not None:
        rgb_type = target_rgb
    else:
        rgb_type = cobj.rgb_type

    return target_rgb(rgb_r, rgb_g, rgb_b)


# noinspection PyPep8Naming,PyUnusedLocal
def RGB_to_CMY(cobj, *args, **kwargs):
    """
    RGB to CMY conversion.

    NOTE: CMYK and CMY values range from 0.0 to 1.0
    """

    cmy_c = 1.0 - cobj.rgb_r
    cmy_m = 1.0 - cobj.rgb_g
    cmy_y = 1.0 - cobj.rgb_b

    return CMYColor(cmy_c, cmy_m, cmy_y)


# noinspection PyPep8Naming,PyUnusedLocal
def CMY_to_RGB(cobj, target_rgb, *args, **kwargs):
    """
    Converts CMY to RGB via simple subtraction.

    NOTE: Returned values are in the range of 0-255.
    """

    rgb_r = 1.0 - cobj.cmy_c
    rgb_g = 1.0 - cobj.cmy_m
    rgb_b = 1.0 - cobj.cmy_y

    return target_rgb(rgb_r, rgb_g, rgb_b)


# noinspection PyPep8Naming,PyUnusedLocal
def CMY_to_CMYK(cobj, *args, **kwargs):
    """
    Converts from CMY to CMYK.

    NOTE: CMYK and CMY values range from 0.0 to 1.0
    """

    var_k = 1.0
    if cobj.cmy_c < var_k:
        var_k = cobj.cmy_c
    if cobj.cmy_m < var_k:
        var_k = cobj.cmy_m
    if cobj.cmy_y < var_k:
        var_k = cobj.cmy_y

    if var_k == 1:
        cmyk_c = 0.0
        cmyk_m = 0.0
        cmyk_y = 0.0
    else:
        cmyk_c = (cobj.cmy_c - var_k) / (1.0 - var_k)
        cmyk_m = (cobj.cmy_m - var_k) / (1.0 - var_k)
        cmyk_y = (cobj.cmy_y - var_k) / (1.0 - var_k)
    cmyk_k = var_k

    return CMYKColor(cmyk_c, cmyk_m, cmyk_y, cmyk_k)


# noinspection PyPep8Naming,PyUnusedLocal
def CMYK_to_CMY(cobj, *args, **kwargs):
    """
    Converts CMYK to CMY.

    NOTE: CMYK and CMY values range from 0.0 to 1.0
    """

    cmy_c = cobj.cmyk_c * (1.0 - cobj.cmyk_k) + cobj.cmyk_k
    cmy_m = cobj.cmyk_m * (1.0 - cobj.cmyk_k) + cobj.cmyk_k
    cmy_y = cobj.cmyk_y * (1.0 - cobj.cmyk_k) + cobj.cmyk_k

    return CMYColor(cmy_c, cmy_m, cmy_y)


def ipt_hdr_lightness_func(exponent=0.59):
    def f(lms_values):
        aa=numpy.power(numpy.abs(lms_values), exponent)
        return numpy.sign(lms_values)*(2.46*aa/(aa+2**exponent) + 0.0002)
    return f


def ipt_hdr_inv_lightness_func(exponent=0.59):
    """ aa = x**exponent
        c = 2**exponent
        k = 0.0002
        m = 1/exponent
        y = b*aa/(aa+c) + k
        (y-k)*(aa+c) = b*aa
        (y-k)*aa +(y-k)*c = b*aa
        (y-k)*c = aa*(b -(y-k))
        (y-k)*c = aa*(b-y+k)
        aa = (y-k)*c/(b-y+k)
        log(aa) = log(y-k)+log(c)-log(b-y+k)
        log(x) = (log(y-k)+log(c)-log(b-y+k))/exponent
        x = (c*(y-k)/(b-y+k))**exponent
    """
    m = 1.0/exponent
    def f_inv(ipt_values):
        return 2*numpy.sign(ipt_values)*abs((ipt_values-0.0002)/(
            2.46-ipt_values+0.0002))**m
    return f_inv


def ipt_lightness(lms_values):
    return numpy.sign(lms_values) * numpy.abs(lms_values) ** 0.43


def ipt_inv_lightness(ipt_values):
    return numpy.sign(ipt_values) * numpy.abs(ipt_values) ** (1 / 0.43)


def XYZ_to_IPT_func(lightness_func):
    # noinspection PyPep8Naming,PyUnusedLocal
    def XYZ_to_IPT(cobj, *args, **kwargs):
        """
        Converts XYZ to IPT.

        NOTE: XYZ values need to be adapted to 2 degree D65

        Reference:
        Fairchild, M. D. (2013). Color appearance models, 3rd Ed. (pp. 271-272). John Wiley & Sons.
        """

        if cobj.illuminant != 'd65' or cobj.observer != '2':
            raise ValueError('XYZColor for XYZ->IPT conversion needs to be '
                              'D65 adapted.')
        xyz_values = numpy.array(cobj.get_value_tuple())
        lms_values = numpy.dot(IPTColor.conversion_matrices['xyz_to_lms'], xyz_values)

        lms_prime = lightness_func(lms_values)

        ipt_values = numpy.dot(IPTColor.conversion_matrices['lms_to_ipt'], lms_prime)
        return IPTColor(*ipt_values)
    return XYZ_to_IPT


XYZ_to_IPT = XYZ_to_IPT_func(ipt_lightness)
XYZ_to_hdrIPT = XYZ_to_IPT_func(ipt_hdr_lightness_func())

def IPT_to_XYZ_func(inv_lightness):
    # noinspection PyPep8Naming,PyUnusedLocal
    def IPT_to_XYZ(cobj, *args, **kwargs):
        """
        Converts XYZ to IPT.
        """

        ipt_values = numpy.array(cobj.get_value_tuple())
        lms_values = numpy.dot(numpy.linalg.inv(IPTColor.conversion_matrices['lms_to_ipt']), ipt_values)

        lms_prime = inv_lightness(lms_values)

        xyz_values = numpy.dot(numpy.linalg.inv(IPTColor.conversion_matrices['xyz_to_lms']), lms_prime)
        return XYZColor(*xyz_values, observer='2', illuminant='d65')
    return IPT_to_XYZ

IPT_to_XYZ = IPT_to_XYZ_func(ipt_inv_lightness)
hIPT_to_XYZ = IPT_to_XYZ_func(ipt_hdr_inv_lightness_func())
XYZ_to_IPT_ = XYZ_to_IPT_func(ipt_lightness)
XYZ_to_hdrIPT = XYZ_to_IPT_func(ipt_hdr_lightness_func())


def IPT_to_ICHpt_func(ICHptClass):
    # noinspection PyPep8Naming,PyUnusedLocal
    def IPT_to_ICHpt(cobj, *args, **kwargs):
        """
        Converts IPT to ICHpt.
        """
        ich_c = math.sqrt(math.pow(cobj.ipt_p, 2.0) + math.pow(cobj.ipt_t, 2.0))
        ich_h = math.atan2(float(cobj.ipt_t), float(cobj.ipt_p))

        if ich_h > 0:
            ich_h = (ich_h / math.pi) * 180
        else:
            ich_h = 360 - (math.fabs(ich_h) / math.pi) * 180
        return ICHptClass(cobj.ipt_i, ich_c, ich_h)
    return IPT_to_ICHpt


IPT_to_ICHpt = IPT_to_ICHpt_func(ICHptColor)
hIPT_to_hICHpt = IPT_to_ICHpt_func(hICHptColor)

def ICHpt_to_IPT_func(IPTClass):
    # noinspection PyPep8Naming,PyUnusedLocal
    def ICHpt_to_IPT(cobj, *args, **kwargs):
        """
        Convert from ICHpt to IPT(uv).
        """

        ipt_i = cobj.ich_i
        ipt_p = math.cos(math.radians(cobj.ich_h)) * cobj.ich_c
        ipt_t = math.sin(math.radians(cobj.ich_h)) * cobj.ich_c
        return IPTClass(ipt_i, ipt_p, ipt_t )
    return ICHpt_to_IPT

ICHpt_to_IPT = ICHpt_to_IPT_func(IPTColor)
hICHpt_to_hIPT = ICHpt_to_IPT_func(hIPTColor)


_all_color_classes = ["AdobeRGBColor", "sRGBColor", "XYZColor", "xyYColor",
                      "LabColor", "LuvColor",
                "LCHabColor", "LCHuvColor",
                "HSLColor", "HSVColor",
                "CMYColor", "CMYKColor",
                "IPTColor", "ICHptColor",
                "hIPTColor", "hICHptColor"]

_TO_XYZ = {
    "AdobeRGBColor": [RGB_to_XYZ],
    "sRGBColor": [RGB_to_XYZ],
    "XYZColor": [],
    "xyYColor": [xyY_to_XYZ],
    "LabColor": [Lab_to_XYZ],
    "LuvColor": [Luv_to_XYZ],
    "LCHabColor": [LCHab_to_Lab, Lab_to_XYZ],
    "LCHuvColor": [LCHuv_to_Luv, Luv_to_XYZ],
    "IPTColor": [IPT_to_XYZ],
    "ICHptColor": [ICHpt_to_IPT, IPT_to_XYZ],
    "hIPTColor": [hIPT_to_XYZ],
    "hICHptColor": [hICHpt_to_hIPT, hIPT_to_XYZ]
}

_TO_RGB ={
    "AdobeRGBColor": [],
    "sRGBColor": [],
    "HSLColor": [HSL_to_RGB],
    "HSVColor": [HSV_to_RGB],
    "CMYColor": [CMY_to_RGB],
    "CMYKColor": [CMYK_to_CMY, CMY_to_RGB]
}

_FROM_XYZ = {
    "XYZColor": [],
    "xyYColor": [XYZ_to_xyY],
    "LabColor": [XYZ_to_Lab],
    "LuvColor": [XYZ_to_Luv],
    "LCHabColor": [XYZ_to_Lab, Lab_to_LCHab ],
    "LCHuvColor": [XYZ_to_Luv, Luv_to_LCHuv],
    "IPTColor": [XYZ_to_IPT],
    "ICHptColor": [XYZ_to_IPT, IPT_to_ICHpt],
    "hIPTColor": [XYZ_to_hdrIPT],
    "hICHptColor": [XYZ_to_hdrIPT, hIPT_to_hICHpt]
}

_FROM_RGB ={
    "AdobeRGBColor": [],
    "sRGBColor": [],
    "HSLColor": [RGB_to_HSL],
    "HSVColor": [RGB_to_HSV],
    "CMYColor": [RGB_to_CMY],
    "CMYKColor": [RGB_to_CMY, CMY_to_CMYK]
}


def make_conv_chain(src_class, dest_class):
    if src_class == dest_class: return [None]


    s2x = _TO_XYZ.get(src_class, None)
    s2r = _TO_RGB.get(src_class, None)
    x2d = _FROM_XYZ.get(dest_class,None)
    r2d = _FROM_RGB.get(dest_class,None)

    if src_class in ("AdobeRGBColor", "sRGBColor"):
        if r2d is not None:
            return [] + r2d  # force copy
    elif src_class in ("XYZ_Color",):
        if x2d is not None:
            return [] + x2d
    """
    if dest_class in ("AdobeRGBColor", "sRGBColor"):
        if s2r is not None:
            return [] +s2r
    elif dest_class in ("XYZ_Color",):
        if s2x is not None:
            return [] +s2x
    """
    print((src_class,dest_class))
    print((s2x, s2r,x2d, r2d))


    if s2x is None:
        assert s2r is not None
        if r2d is None:
            assert x2d is not None
            return s2r + [RGB_to_XYZ] + x2d
        return s2r+r2d
    if x2d is None:
        assert r2d is not None
        return s2x + [XYZ_to_RGB] +r2d
    return s2x + x2d

CONVERSION_TABLE = dict()

for src in _all_color_classes:
    CONVERSION_TABLE[src] = dict()
    for dest in _all_color_classes:
        CONVERSION_TABLE[src][dest] = make_conv_chain(src, dest)

# Handle Spectral colors separately, since it is not a valid destination
sct = {"SpectralColor": [None]}
for dest in _all_color_classes:
    sct[dest] = ([Spectral_to_XYZ] + make_conv_chain("XYZColor", dest))
CONVERSION_TABLE["SpectralColor"] = sct

print('\n\n'+('-'*80))
for (s,t) in CONVERSION_TABLE.items():
    print('\n\n'+s+':')
    for (k,c) in t.items():
        print('    %s: %s'%(k,str(c)))
print(('-'*80)+'\n\n')

"""
from . import color_objects
color_class_names = [v for v in dir(color_objects) if v.endswith('Color')]
color_name_to_class = { c: getattr(color_objects,c) for c in color_class_names}
print(color_class_names)
print(color_name_to_class)
color_class_abbrev = set([c[0:-5] for c in color_class_names])
RGB_CLASSES=[c for c in color_class_names if 'RGB' in c]
color_class_abbrev.difference_update(RGB_CLASSES)
#color_class_abbrev.add('RGB')

_CONVERSIONS = {c: dict([None]) for c in color_class_abbrev}

for func_name, func in locals.items():
    source, _to_, dest = func_name.split('_',2)
    if (_to_=='to' and
            (source in color_class_abbrev) and
            (dest in color_class_abbrev)):
        _CONVERSIONS[source][dest]=[func]

for s in color_class_abbrev:
    for d in color_class_abbrev:
        pass
"""



def convert_color(color, target_cs, through_rgb_type=sRGBColor,
                  target_illuminant=None, *args, **kwargs):
    """
    Converts the color to the designated color space.

    :param color: A Color instance to convert.
    :param target_cs: The Color class to convert to. Note that this is not
        an instance, but a class.
    :keyword BaseRGBColor through_rgb_type: If during your conversion between
        your original and target color spaces you have to pass through RGB,
        this determines which kind of RGB to use. For example, XYZ->HSL.
        You probably don't need to specify this unless you have a special
        usage case.
    :type target_illuminant: None or str
    :keyword target_illuminant: If during conversion from RGB to a reflective
        color space you want to explicitly end up with a certain illuminant,
        pass this here. Otherwise the RGB space's native illuminant will be used.
    :returns: An instance of the type passed in as ``target_cs``.
    :raises: :py:exc:`colormath.color_exceptions.UndefinedConversionError`
        if conversion between the two color spaces isn't possible.
    """

    if isinstance(target_cs, str):
        raise ValueError("target_cs parameter must be a Color object.")
    if not issubclass(target_cs, ColorBase):
        raise ValueError("target_cs parameter must be a Color object.")

    # Find the origin color space's conversion table.
    cs_table = CONVERSION_TABLE[color.__class__.__name__]
    try:
        # Look up the conversion path for the specified color space.
        conversions = cs_table[target_cs.__name__]
    except KeyError:
        raise UndefinedConversionError(
            color.__class__.__name__,
            target_cs.__name__,
        )

    logger.debug('Converting %s to %s', color, target_cs)
    logger.debug(' @ Conversion path: %s', conversions)

    # Start with original color in case we convert to the same color space.
    new_color = color

    if issubclass(target_cs, BaseRGBColor):
        # If the target_cs is an RGB color space of some sort, then we
        # have to set our through_rgb_type to make sure the conversion returns
        # the expected RGB colorspace (instead of defaulting to sRGBColor).
        through_rgb_type = target_cs

    # We have to be careful to use the same RGB color space that created
    # an object (if it was created by a conversion) in order to get correct
    # results. For example, XYZ->HSL via Adobe RGB should default to Adobe
    # RGB when taking that generated HSL object back to XYZ.
    # noinspection PyProtectedMember
    if through_rgb_type != sRGBColor:
        # User overrides take priority over everything.
        # noinspection PyProtectedMember
        target_rgb = through_rgb_type
    elif color._through_rgb_type:
        # Otherwise, a value on the color object is the next best thing,
        # when available.
        # noinspection PyProtectedMember
        target_rgb = color._through_rgb_type
    else:
        # We could collapse this into a single if statement above,
        # but I think this reads better.
        target_rgb = through_rgb_type

    # Iterate through the list of functions for the conversion path, storing
    # the results in a dictionary via update(). This way the user has access
    # to all of the variables involved in the conversion.
    for func in conversions:
        # Execute the function in this conversion step and store the resulting
        # Color object.
        logger.debug(' * Conversion: %s passed to %s()',
                     new_color.__class__.__name__, func)
        logger.debug(' |->  in %s', new_color)

        if func:
            # This can be None if you try to convert a color to the color
            # space that is already in. IE: XYZ->XYZ.
            new_color = func(
                new_color,
                target_rgb=target_rgb,
                target_illuminant=target_illuminant,
                *args, **kwargs)

        logger.debug(' |-< out %s', new_color)

    # If this conversion had something other than the default sRGB color space
    # requested,
    if through_rgb_type != sRGBColor:
        new_color._through_rgb_type = through_rgb_type

    return new_color
