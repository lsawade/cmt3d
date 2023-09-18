from scipy.signal.windows import hann as taper
import numpy as np
import cmt3d


def test_dlna():

    # Generate random data
    np.random.seed(12345)
    d = np.random.random(100)

    # Compute dlna
    dlnA = cmt3d.dlna(d, d)

    # Check if computation is ok.
    assert abs(dlnA) <= 1E-12


def test_norm2():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])

    # Compute dlna
    norm = cmt3d.norm2(d)

    # Check if computation is ok.
    assert (norm-14.0) <= 1E-12


def test_norm1():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])

    # Compute dlna
    norm = cmt3d.norm1(d)

    # Check if computation is ok.
    assert (norm-6.0) <= 1E-12


def test_dnorm2():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])
    s = np.array([4, 5, 6])

    # Compute dlna
    norm = cmt3d.dnorm2(d, s)

    # Check if computation is ok.
    assert (norm-13.5) <= 1E-12


def test_dnorm1():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])
    s = np.array([4, 5, 6])

    # Compute dlna
    norm = cmt3d.dnorm1(d, s)

    # Check if computation is ok.
    assert (norm-9.0) <= 1E-12


def test_power_l1():

    # Generate random data
    np.random.seed(12345)
    d = np.array([-1, -2, -3])
    s = np.array([-4, -5, -6])

    # Compute dlna
    powerl1 = cmt3d.power_l1(d, s)

    # Check if computation is ok.
    assert (powerl1-10*np.log10(6/15)) <= 1E-12


def test_power_l2():

    # Generate random data
    np.random.seed(12345)
    d = np.array([1, 2, 3])
    s = np.array([4, 5, 6])

    # Compute dlna
    powerl2 = cmt3d.power_l2(d, s)

    # Check if computation is ok.
    assert (powerl2-10*np.log10(14/77)) <= 1E-12


def test_dlna():

    # Generate random data
    np.random.seed(12345)
    d = np.random.random(100)

    # Compute dlna
    max_cc, tshift = cmt3d.xcorr(d, d)

    # Check if computation is ok.
    assert abs(max_cc - 1.0) <= 1E-12
    assert abs(tshift) <= 1E-12


def test_correct_index():

    times = np.linspace(0, 2*np.pi,  1000)
    data = np.sin(3*times) * taper(len(times))
    model = np.array([.5])

    def forward(a):
        return a * np.sin(3*times + 0.25*np.pi) * taper(len(times))

    istart, iend = 300, 700
    _, nshift = cmt3d.xcorr(data, forward(model))
    nshift
    istart_d, iend_d, istart_s, iend_s = cmt3d.correct_window_index(
        istart, iend, nshift, len(times))

    d_test_i, d_test_f = (340, 740)
    s_test_i, s_test_f = (300, 700)

    assert istart_s == s_test_i
    assert iend_s == s_test_f
    assert istart_d == d_test_i
    assert iend_d == d_test_f
