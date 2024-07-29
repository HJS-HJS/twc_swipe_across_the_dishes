import numpy as np

class Ellipse(object):
    def __init__(self, x, y):
        self.coeffs = Ellipse.fit_ellipse(x, y)
        self.pole = Ellipse.cart_to_pol(self.coeffs)

    def get_ellipse_pts(self, npts:int=100, tmin:float=0, tmax:float=2*np.pi):
        """
        Return npts points on the ellipse described by the params = x0, y0, ap,
        bp, e, phi for values of the parametric variable t between tmin and tmax.

        """
        x0, y0, ap, bp, e, phi = self.pole
        # A grid of the parametric variable, t.
        t = np.linspace(tmin, tmax, npts)
        x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
        y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
        return x, y

    def resize_ratio(self, ratio:float = 1.0):
        x0, y0, ap, bp, e, phi = self.pole
        self.pole = (x0, y0, ratio * ap, ratio * bp, e, phi)

    def resize(self, a_value:float = 0.0, b_value:float = 0.0):
        x0, y0, ap, bp, e, phi = self.pole
        self.pole = (x0, y0, ap + a_value, bp + b_value, e, phi)

    @property
    def size(self):
        return self.pole[2], self.pole[3]
    @property
    def center(self):
        return np.array([self.pole[0], self.pole[1]])

    def point(self, angle:float):
        x0, y0, ap, bp, e, phi = self.pole
        angle = angle - phi
        r = np.sqrt(bp**2 / (1-e**2 * np.cos(angle)**2))
        x = x0 + r * np.cos(angle) * np.cos(phi) - r * np.sin(angle) * np.sin(phi)
        y = y0 + r * np.cos(angle) * np.sin(phi) + r * np.sin(angle) * np.cos(phi)
        return np.array([x, y])
    
    def lengh(self, angle:float):
        x0, y0, ap, bp, e, phi = self.pole
        angle = angle - phi
        r = np.sqrt(bp**2 / (1-e**2 * np.cos(angle)**2))
        return np.sqrt(bp**2 / (1-e**2 * np.cos(angle)**2))
    
    @staticmethod
    def fit_ellipse(x, y):
        """
        Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
        the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
        arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

        Based on the algorithm of Halir and Flusser, "Numerically stable direct
        least squares fitting of ellipses'.
        """
        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        return np.concatenate((ak, T @ ak)).ravel()
    
    @staticmethod
    def cart_to_pol(param):
        """

        Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
        ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
        The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
        ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
        respectively; e is the eccentricity; and phi is the rotation of the semi-
        major axis from the x-axis.

        """

        # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
        # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
        # Therefore, rename and scale b, d and f appropriately.
        a = param[0]
        b = param[1] / 2
        c = param[2]
        d = param[3] / 2
        f = param[4] / 2
        g = param[5]

        den = b**2 - a*c
        if den > 0:
            raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                            ' be negative!')

        # The location of the ellipse centre.
        x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        # The semi-major and semi-minor axis lengths (these are not sorted).
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))

        # Sort the semi-major and semi-minor axis lengths but keep track of
        # the original relative magnitudes of width and height.
        width_gt_height = True
        if ap < bp:
            width_gt_height = False
            ap, bp = bp, ap

        # The eccentricity.
        r = (bp/ap)**2
        if r > 1:
            r = 1/r
        e = np.sqrt(1 - r)

        # The angle of anticlockwise rotation of the major-axis from x-axis.
        if b == 0:
            phi = 0 if a < c else np.pi/2
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2
            if a > c:
                phi += np.pi/2
        if not width_gt_height:
            # Ensure that phi is the angle to rotate to the semi-major axis.
            phi += np.pi/2
        phi = phi % np.pi

        return x0, y0, ap, bp, e, phi

    @staticmethod
    def check_overlap(a, b):
        # check b is in a
        center_vector = b.center - a.center
        c_angle=np.arctan2(center_vector[1], center_vector[0])
        if (a.lengh(c_angle) + b.lengh(c_angle + np.pi) - np.linalg.norm(center_vector)) > 0:
            return True

        # check a and b is overlapped
        c_angle += np.pi
        angle_range = np.pi/1.5
        x, y = b.get_ellipse_pts(npts=50,tmin=c_angle - angle_range, tmax=c_angle + angle_range)
        angle = np.arctan2(y - a.pole[1], x - a.pole[0])
        lenghts = np.linalg.norm(np.array([x - a.center[0],y - a.center[1]]), axis=0).reshape(-1)
        points = a.point(angle).T
        lenghts2 = np.linalg.norm(points - a.center, axis=1)

        return np.any((lenghts - lenghts2) < 0)
