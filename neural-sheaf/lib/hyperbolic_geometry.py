
import torch
from torch.nn import Parameter
import numpy as np

# by ai @grok
# def hyperbolic_distance(p, q, c=1.0):
#     """Compute hyperbolic distance in Poincaré disk with curvature c."""
#     diff = p - q
#     norm_diff_sq = torch.sum(diff ** 2, dim=-1)
#     norm_p_sq = torch.sum(p ** 2, dim=-1)
#     norm_q_sq = torch.sum(q ** 2, dim=-1)
#     denom = (1 - c * norm_p_sq) * (1 - c * norm_q_sq)
#     arg = 1 + 2 * c * norm_diff_sq / denom
#     # Clamp to avoid numerical issues with arccosh
#     arg = torch.clamp(arg, min=1.0 + 1e-7)
#     return torch.acosh(arg) / torch.sqrt(c)

# def mobius_addition(p, q, c=1.0):
#     """Möbius addition in Poincaré disk with curvature c."""
#     inner = torch.sum(p * q, dim=-1)
#     p_sq = torch.sum(p ** 2, dim=-1)
#     q_sq = torch.sum(q ** 2, dim=-1)
#     num = (1 + 2 * c * inner + c * q_sq) * p + (1 - c * p_sq) * q
#     denom = 1 + 2 * c * inner + c * c * p_sq * q_sq
#     return num / (denom.unsqueeze(-1) + 1e-7)  # Avoid division by zero

# def logmap(p, q, c=1.0):
#     """Logarithmic map from p to q in Poincaré disk with curvature c."""
#     lambda_p = 1 - c * torch.sum(p ** 2, dim=-1)
#     diff = -mobius_addition(p, q, c)
#     norm_diff = torch.norm(diff, dim=-1)
#     coef = (2 / (torch.sqrt(c) * lambda_p)) * torch.atanh(torch.clamp(torch.sqrt(c) * norm_diff, max=1.0 - 1e-7))
#     return coef.unsqueeze(-1) * (diff / (norm_diff.unsqueeze(-1) + 1e-7))

# """Base manifold."""


def identity(x):
    return x
def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)

def tanimoto_score(A,B):
    a_mag =np.linalg.norm(A)
    b_mag =np.linalg.norm(B)
    return np.dot(A,B)/((b_mag**2)+(a_mag**2)-np.dot(A,B))

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5






class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p, c):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """Parallel transport of u from the origin to y."""
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """
    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature. 
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        return torch.clamp(sqdist, max=50.0)

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2 
        mask = torch.ones_like(x)
        mask[:, 0] = 0
        vals = torch.zeros_like(x)
        vals[:, 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))
        return vals + mask * x

    def proj_tan(self, u, x, c):
        K = 1. / c
        d = x.size(1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
        mask = torch.ones_like(u)
        mask[:, 0] = 0
        vals = torch.zeros_like(u)
        vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u, c):
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = cosh(theta) * x + sinh(theta) * u / theta
        return self.proj(result, c)
        
    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        x = u.narrow(-1, 1, d).view(-1, d)
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[:, 0:1] = sqrtK * cosh(theta)
        res[:, 1:] = sqrtK * sinh(theta) * x / x_norm
        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d).view(-1, d)
        y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        res[:, 1:] = sqrtK * arcosh(theta) * y / y_norm
        return res

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[:, 0:1] = - y_norm 
        v[:, 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)

# prefered
class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-12
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        ### BATCH

        ## here would be a good spot to normalize?-- no it wouldn't leave this shit alone.
        # print(p1,'p1?')
        # print(p2,'p1?')
        # print(c,'c?')
        # p1=torch.from_numpy(np.array(p1).astype(float))
        # p2=torch.from_numpy(np.array(p2).astype(float))
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )

        # print(dist_c,'DIST C')
        # print(sqrt_c,'sqrizzle')
        dist = dist_c * 2 / sqrt_c
        # print(dist)
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x, c):
        ##BATCH
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        # print(maxnorm,'MAX NORM',c)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        #print(f"expmap input shapes: p={p.shape}, u={u.shape}")
        if p.dim() == 2 and u.dim() == 3:
            p = p.unsqueeze(1)
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        assert p1.shape == p2.shape, f"logmap shape mismatch: p1={p1.shape}, p2={p2.shape}"
        ##BATCH
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u, c):
        ## BATCH
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p, c):
        ### BATCH
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        ### BATCH
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        #print(f"mobius_add input shapes: x={x.shape}, y={y.shape}")
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2

        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        ### BATCH
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        ## BATCH
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        print(d)
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

