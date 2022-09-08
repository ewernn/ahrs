# -*- coding: utf-8 -*-


"""
stuff I can still do:
-refactor Jacobian to save computations
-
"""

import numpy as np
from ..common.orientation import q2R
from ..common.orientation import ecompass
from ..common.orientation import acc2q
from ..common.mathfuncs import cosd
from ..common.mathfuncs import sind
from ..common.mathfuncs import skew

from ..utils.core import _assert_numerical_iterable


class EricEKF:
    """
    Extended Kalman Filter to estimate orientation as Quaternion.
    
    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in nT
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    frame : str, default: 'NED'
        Local tangent plane coordinate frame. Valid options are right-handed
        ``'NED'`` for North-East-Down and ``'ENU'`` for East-North-Up.
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).
    magnetic_ref : float or numpy.ndarray
        Local magnetic reference.
    noises : numpy.ndarray
        List of noise variances for each type of sensor. Default values:
        ``[0.3**2, 0.5**2, 0.8**2]``.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. NOT required
        if ``frequency`` value is given.

    """
    def __init__(self,
        gyr: np.ndarray = None,
        acc: np.ndarray = None,
        mag: np.ndarray = None,
        frequency: float = 100.0,
        frame: str = 'NED',
        **kwargs):

        self.gyr: np.ndarray = gyr
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.frequency: float = frequency
        self.frame: str = frame                          # Local tangent plane coordinate frame
        self.Dt: float = kwargs.get('Dt', (1.0/self.frequency) if self.frequency else 0.01)
        self.q0: np.ndarray = kwargs.get('q0')
        self.P: np.ndarray = kwargs.get('P', np.identity(4))    # Initial state covariance
        self.R: np.ndarray = self._set_measurement_noise_covariance(**kwargs)
        self._set_reference_frames(kwargs.get('magnetic_ref'), self.frame)
        self._assert_validity_of_inputs()

        # Process of data is given
        if self.gyr is not None and self.acc is not None:
            self.Q = self._compute_all(self.frame)

    def _set_measurement_noise_covariance(self, **kw) -> np.ndarray:
        default_noises = kw.get('noises', [0.3**2, 0.5**2, 0.8**2])
        _assert_numerical_iterable(default_noises, 'Spectral noise variances')
        default_noises = np.copy(default_noises)
        if default_noises.ndim != 1:
            raise ValueError(f"Spectral noise variances must be given in a 1-dimensional array. Got {default_noises.ndim} dimensions instead.")
        if default_noises.size != 3:
            raise ValueError(f"Spectral noise variances must be given in a 1-dimensional array with 3 elements. Got {default_noises.size} elements instead.")
        self.noises = [kw.get(label, value) for label, value in zip(['var_acc', 'var_gyr', 'var_mag'], default_noises)]
        self.g_noise, self.a_noise, self.m_noise = self.noises
        return np.diag(np.repeat(self.noises[1:], 3))

    def _set_reference_frames(self, mref: float, frame: str = 'NED') -> None:
        if not isinstance(frame, str):
            raise TypeError(f"Parameter 'frame' must be a string. Got {type(frame)}.")
        if frame.upper() not in ['NED', 'ENU']:
            raise ValueError(f"Invalid frame '{frame}'. Try 'NED' or 'ENU'")
        # Magnetic Reference Vector
        if mref is None:
            # Local magnetic reference of Munich, Germany
            from ..common.constants import MUNICH_LATITUDE, MUNICH_LONGITUDE, MUNICH_HEIGHT
            from ..utils.wmm import WMM
            wmm = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT)
            self.m_ref = np.array([wmm.X, wmm.Y, wmm.Z]) if frame.upper() == 'NED' else np.array([wmm.Y, wmm.X, -wmm.Z])
        elif isinstance(mref, bool):
            raise TypeError("Invalid magnetic reference. Try a float or a numpy.ndarray.")
        elif isinstance(mref, (int, float)):
            cd, sd = cosd(mref), sind(mref)
            self.m_ref = np.array([cd, 0.0, sd]) if frame.upper() == 'NED' else np.array([0.0, cd, -sd])
        elif isinstance(mref, (list, tuple, np.ndarray)):
            self.m_ref = np.copy(mref)
        else:
            raise TypeError(f"mref must be given as a float, list, tuple or NumPy array. Got {type(mref)}")
        if self.m_ref.ndim != 1:
            raise ValueError(f"mref must be given as a 1-dimensional array. Got {self.m_ref.ndim} dimensions instead.")
        if self.m_ref.size != 3:
            raise ValueError(f"mref must be given as a 1-dimensional array with 3 elements. Got {self.m_ref.size} elements instead.")
        for item in self.m_ref:
            if not isinstance(item, (int, float)):
                raise TypeError(f"mref must be given as a 1-dimensional array of floats. Got {type(item)} instead.")
        self.m_ref /= np.linalg.norm(self.m_ref)
        # Gravitational Reference Vector
        self.a_ref = np.array([0.0, 0.0, 1.0]) if frame.upper() == 'NED' else np.array([0.0, 0.0, -1.0])

    def _assert_validity_of_inputs(self):
        """Asserts the validity of the inputs."""
        for item in ["frequency", "Dt"]:
            if isinstance(self.__getattribute__(item), bool):
                raise TypeError(f"Parameter '{item}' must be numeric.")
            if not isinstance(self.__getattribute__(item), (int, float)):
                raise TypeError(f"Parameter '{item}' is not a non-zero number.")
            if self.__getattribute__(item) <= 0.0:
                raise ValueError(f"Parameter '{item}' must be a non-zero number.")
        for item in ['q0', 'P', 'R']:
            if self.__getattribute__(item) is not None:
                if isinstance(self.__getattribute__(item), bool):
                    raise TypeError(f"Parameter '{item}' must be an array of numeric values.")
                if not isinstance(self.__getattribute__(item), (list, tuple, np.ndarray)):
                    raise TypeError(f"Parameter '{item}' is not an array. Got {type(self.__getattribute__(item))}.")
                self.__setattr__(item, np.copy(self.__getattribute__(item)))
        if self.q0 is not None:
            if self.q0.shape != (4,):
                raise ValueError(f"Parameter 'q0' must be an array of shape (4,). It is {self.q0.shape}.")
            if not np.allclose(np.linalg.norm(self.q0), 1.0):
                raise ValueError(f"Parameter 'q0' must be a versor (norm equal to 1.0). Its norm is equal to {np.linalg.norm(self.q0)}.")
        for item in ['P', 'R']:
            if self.__getattribute__(item).ndim != 2:
                raise ValueError(f"Parameter '{item}' must be a 2-dimensional array.")
            m, n = self.__getattribute__(item).shape
            if m != n:
                raise ValueError(f"Parameter '{item}' must be a square matrix. It is {m}x{n}.")

    def _compute_all(self, frame: str) -> np.ndarray:
        """
        Estimate the quaternions given all sensor data.

        Attributes ``gyr``, ``acc`` MUST contain data. Attribute ``mag`` is
        optional.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        Q[0] = self.q0
        
        ###### Compute attitude with MARG architecture ######
        
        if self.q0 is None:
            Q[0] = ecompass(self.acc[0], self.mag[0], frame=frame, representation='quaternion')
        Q[0] /= np.linalg.norm(Q[0])

        # EKF Loop over all data
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def Omega(self, x: np.ndarray) -> np.ndarray:
        """
        Omega operator.

        Vector3 --> 4x4 matrix:

            0   & -x_1  & -x_2  & -x_3 \\\\
            x_1 & 0     & x_3   & -x_2 \\\\
            x_2 & -x_3  & 0     & x_1   \\\\
            x_3 & x_2   & -x_1  & 0

        This operator is constantly used at different steps of the EKF.
        """
        return np.array([
            [0.0,  -x[0], -x[1], -x[2]],
            [x[0],   0.0,  x[2], -x[1]],
            [x[1], -x[2],   0.0,  x[0]],
            [x[2],  x[1], -x[0],   0.0]])

    def f(self, q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Linearized function of Process Model (PREDICTION)

        PARAMS / RETURNS
        q_{t-1}, omega ndarray (rad/s), dt  -->  q_hat ndarray
        """
        Omega_t = self.Omega(omega)
        return (np.identity(4) + 0.5*dt*Omega_t) @ q

    def dfdq(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Jacobian of linearized predicted state.

        PARAMS / RETURN
        ----------
        omega ndarray, dt --> F ndarray(Jacobian of state)
        """
        x = 0.5*dt*omega
        return np.identity(4) + self.Omega(x)

    def h(self, q: np.ndarray) -> np.ndarray:
        """
        Measurement Model

        accelerometer/magnetometer: 6x1 matrix

        PARAM / RETURN
        ----------
        q (Predicted Quaternion) --> ndarray (Expected Measurements)
        """
        C = q2R(q).T
        if len(self.z) < 4:
            return C @ self.a_ref
        return np.r_[C @ self.a_ref, C @ self.m_ref]

    def dhdq(self, q: np.ndarray, mode: str = 'normal') -> np.ndarray:
        """
        Linearization of observations with Jacobian

        accelerometer/magnetometer  6x4 matrix

        Parameters
        ----------
        q : numpy.ndarray (Predicted state estimate) -->  H : numpy.ndarray (Jacobian of observations)
        """
        qw, qx, qy, qz = q
        
        v = np.r_[self.a_ref, self.m_ref]
        H = np.array([[-qy*v[2] + qz*v[1],                qy*v[1] + qz*v[2], -qw*v[2] + qx*v[1] - 2.0*qy*v[0],  qw*v[1] + qx*v[2] - 2.0*qz*v[0]],
                      [ qx*v[2] - qz*v[0],  qw*v[2] - 2.0*qx*v[1] + qy*v[0],                qx*v[0] + qz*v[2], -qw*v[0] + qy*v[2] - 2.0*qz*v[1]],
                      [-qx*v[1] + qy*v[0], -qw*v[1] - 2.0*qx*v[2] + qz*v[0],  qw*v[0] - 2.0*qy*v[2] + qz*v[1],  qx*v[0] + qy*v[1]]])
        if len(self.z) == 6:
            H_2 = np.array([[-qy*v[5] + qz*v[4],                qy*v[4] + qz*v[5], -qw*v[5] + qx*v[4] - 2.0*qy*v[3],  qw*v[4] + qx*v[5] - 2.0*qz*v[3]],
                            [ qx*v[5] - qz*v[3],  qw*v[5] - 2.0*qx*v[4] + qy*v[3],                qx*v[3] + qz*v[5], -qw*v[3] + qy*v[5] - 2.0*qz*v[4]],
                            [-qx*v[4] + qy*v[3], -qw*v[4] - 2.0*qx*v[5] + qz*v[3],  qw*v[3] - 2.0*qy*v[5] + qz*v[4],  qx*v[3] + qy*v[4]]])
            H = np.vstack((H, H_2))
        return 2.0*H

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray = None, dt: float = None) -> np.ndarray:
        """
        Perform an update of the state.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori orientation as quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in nT.
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Estimated a-posteriori orientation as quaternion.

        """
        _assert_numerical_iterable(q, 'Quaternion')
        _assert_numerical_iterable(gyr, 'Tri-axial gyroscope sample')
        _assert_numerical_iterable(acc, 'Tri-axial accelerometer sample')
        if mag is not None:
            _assert_numerical_iterable(mag, 'Tri-axial magnetometer sample')
        dt = self.Dt if dt is None else dt
        if not np.isclose(np.linalg.norm(q), 1.0):
            raise ValueError("A-priori quaternion must have a norm equal to 1.")
        # Current Measurements
        g = np.copy(gyr)                        # Gyroscope data (control vector)
        a = np.copy(acc)
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return q
        a /= a_norm
        self.z = np.copy(a)
        if mag is not None:
            m_norm = np.linalg.norm(mag)
            if m_norm == 0:
                raise ValueError("Invalid geomagnetic field. Its magnitude must be greater than zero.")
            self.z = np.r_[a, mag/m_norm]
        self.R = np.diag(np.repeat(self.noises[1:] if mag is not None else self.noises[1], 3))
        # ----- Prediction -----
        q_t = self.f(q, g, dt)                  # Predicted State
        F   = self.dfdq(g, dt)                  # Linearized Fundamental Matrix
        W   = 0.5*dt * np.r_[[-q[1:]], q[0]*np.identity(3) + skew(q[1:])]  # Jacobian W = df/dω
        Q_t = 0.5*dt * self.g_noise * W@W.T     # Process Noise Covariance
        P_t = F@self.P@F.T + Q_t                # Predicted Covariance Matrix
        # ----- Correction -----
        y   = self.h(q_t)                       # Expected Measurement function
        v   = self.z - y                        # Innovation (Measurement Residual)
        H   = self.dhdq(q_t)                    # Linearized Measurement Matrix
        S   = H@P_t@H.T + self.R                # Measurement Prediction Covariance
        K   = P_t@H.T@np.linalg.inv(S)          # Kalman Gain
        self.P = (np.identity(4) - K@H)@P_t     # Updated Covariance Matrix
        self.q = q_t + K@v                      # Corrected State
        self.q /= np.linalg.norm(self.q)
        return self.q
