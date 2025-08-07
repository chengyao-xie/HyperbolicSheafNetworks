import warnings
from enum import Enum
import abc
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.autograd.functional import vjp

# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6 


class Perturb(Enum):
    NONE = 0
    PREV = 1
    NEXT = 2

_all_callback_names = ['callback_step', 'callback_accept_step', 'callback_reject_step']
_all_adjoint_callback_names = [name + '_adjoint' for name in _all_callback_names]
_null_callback = lambda *args, **kwargs: None

def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__.__name__, unused_kwargs))

def find_event(interp_fn, sign0, t0, t1, event_fn, tol):
    with torch.no_grad():

        # Num iterations for the secant method until tolerance is within target.
        nitrs = torch.ceil(torch.log((t1 - t0) / tol) / math.log(2.0))

        for _ in range(nitrs.long()):
            t_mid = (t1 + t0) / 2.0
            y_mid = interp_fn(t_mid)
            sign_mid = torch.sign(event_fn(t_mid, y_mid))
            same_as_sign0 = (sign0 == sign_mid)
            t0 = torch.where(same_as_sign0, t_mid, t0)
            t1 = torch.where(same_as_sign0, t1, t_mid)
        event_t = (t0 + t1) / 2.0

    return event_t, interp_fn(event_t)

def combine_event_functions(event_fn, t0, y0):
    """
    We ensure all event functions are initially positive,
    so then we can combine them by taking a min.
    """
    with torch.no_grad():
        initial_signs = torch.sign(event_fn(t0, y0))

    def combined_event_fn(t, y):
        c = event_fn(t, y)
        return torch.min(c * initial_signs)

    return combined_event_fn

class _StitchGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, out):
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def _nextafter(x1, x2):
    with torch.no_grad():
        if hasattr(torch, "nextafter"):
            out = torch.nextafter(x1, x2)
        else:
            out = np_nextafter(x1, x2)
    return _StitchGradient.apply(x1, out)

def np_nextafter(x1, x2):
    warnings.warn("torch.nextafter is only available in PyTorch 1.7 or newer."
                  "Falling back to numpy.nextafter. Upgrade PyTorch to remove this warning.")
    x1_np = x1.detach().cpu().numpy()
    x2_np = x2.detach().cpu().numpy()
    out = torch.tensor(np.nextafter(x1_np, x2_np)).to(x1)
    return out

def _interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at the given time point.

    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar float64 Tensor giving the start of the interval.
        t1: scalar float64 Tensor giving the end of the interval.
        t: scalar float64 Tensor giving the desired interpolation point.

    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """

    assert (t0 <= t) & (t <= t1), 'invalid interpolation, fails `t0 <= t <= t1`: {}, {}, {}'.format(t0, t, t1)
    x = (t - t0) / (t1 - t0)
    x = x.to(coefficients[0].dtype)

    total = coefficients[0] + x * coefficients[1]
    x_power = x
    for coefficient in coefficients[2:]:
        x_power = x_power * x
        total = total + x_power * coefficient

    return total

class _TupleFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return torch.cat([f_.reshape(-1) for f_ in f])

class _ReverseFunc(torch.nn.Module):
    def __init__(self, base_func, mul=1.0):
        super(_ReverseFunc, self).__init__()
        self.base_func = base_func
        self.mul = mul

    def forward(self, t, y):
        return self.mul * self.base_func(-t, y)

class _TupleInputOnlyFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleInputOnlyFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        return self.base_func(t, _flat_to_shape(y, (), self.shapes))

class _PerturbFunc(torch.nn.Module):

    def __init__(self, base_func):
        super(_PerturbFunc, self).__init__()
        self.base_func = base_func

    def forward(self, t, y, *, perturb=Perturb.NONE):
        assert isinstance(perturb, Perturb), "perturb argument must be of type Perturb enum"
        # This dtype change here might be buggy.
        # The exact time value should be determined inside the solver,
        # but this can slightly change it due to numerical differences during casting.
        if torch.is_complex(t):
            t = t.real
        t = t.to(y.abs().dtype)
        if perturb is Perturb.NEXT:
            # Replace with next smallest representable value.
            t = _nextafter(t, t + 1)
        elif perturb is Perturb.PREV:
            # Replace with prev largest representable value.
            t = _nextafter(t, t - 1)
        else:
            # Do nothing.
            pass
        return self.base_func(t, y)




def _rms_norm(tensor):
    return tensor.abs().pow(2).mean().sqrt()


def _mixed_norm(tensor_tuple):
    if len(tensor_tuple) == 0:
        return 0.
    return max([_rms_norm(tensor) for tensor in tensor_tuple])


def _tuple_tol(name, tol, shapes):
    try:
        iter(tol)
    except TypeError:
        return tol
    tol = tuple(tol)
    assert len(tol) == len(shapes), "If using tupled {} it must have the same length as the tuple y0".format(name)
    tol = [torch.as_tensor(tol_).expand(shape.numel()) for tol_, shape in zip(tol, shapes)]
    return torch.cat(tol)

def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)

def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, torch.Tensor), '{} must be a torch.Tensor'.format(name)
    _assert_floating(name, timelike)
    assert timelike.ndimension() == 1, "{} must be one dimensional".format(name)
    if not can_grad:
        assert not timelike.requires_grad, "{} cannot require gradient".format(name)
    diff = timelike[1:] > timelike[:-1]
    assert diff.all() or (~diff).all(), '{} must be strictly increasing or decreasing'.format(name)

def _assert_floating(name, t):
    if not torch.is_floating_point(t):
        raise TypeError('`{}` must be a floating point Tensor but is a {}'.format(name, t.type()))
    
def _flip_option(options, option_name):
    try:
        option_value = options[option_name]
    except KeyError:
        pass
    else:
        if isinstance(option_value, torch.Tensor):
            options[option_name] = -option_value
        # else: an error will be raised when the option is attempted to be used in Solver.__init__, but we defer raising
        # the error until then to keep things tidy.

def _assert_increasing(name, t):
    assert (t[1:] > t[:-1]).all(), '{} must be strictly increasing or decreasing'.format(name)

def _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS):

    if event_fn is not None:
        if len(t) != 2:
            raise ValueError(f"We require len(t) == 2 when in event handling mode, but got len(t)={len(t)}.")

        # Combine event functions if the output is multivariate.
        event_fn = combine_event_functions(event_fn, t[0], y0)

    # Keep reference to original func as passed in
    original_func = func

    # Normalise to tensor (non-tupled) input
    shapes = None
    is_tuple = not isinstance(y0, torch.Tensor)
    if is_tuple:
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        shapes = [y0_.shape for y0_ in y0]
        rtol = _tuple_tol('rtol', rtol, shapes)
        atol = _tuple_tol('atol', atol, shapes)
        y0 = torch.cat([y0_.reshape(-1) for y0_ in y0])
        func = _TupleFunc(func, shapes)
        if event_fn is not None:
            event_fn = _TupleInputOnlyFunc(event_fn, shapes)

    # Normalise method and options
    if options is None:
        options = {}
    else:
        options = options.copy()
    if method is None:
        method = 'dopri5'
    if method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(method,
                                                                         '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))

    if is_tuple:
        # We accept tupled input. This is an abstraction that is hidden from the rest of odeint (exception when
        # returning values), so here we need to maintain the abstraction by wrapping norm functions.

        if 'norm' in options:
            # If the user passed a norm then get that...
            norm = options['norm']
        else:
            # ...otherwise we default to a mixed Linf/L2 norm over tupled input.
            norm = _mixed_norm

        # In either case, norm(...) is assumed to take a tuple of tensors as input. (As that's what the state looks
        # like from the point of view of the user.)
        # So here we take the tensor that the machinery of odeint has given us, and turn it in the tuple that the
        # norm function is expecting.
        def _norm(tensor):
            y = _flat_to_shape(tensor, (), shapes)
            return norm(y)
        options['norm'] = _norm

    else:
        if 'norm' in options:
            # No need to change the norm function.
            pass
        else:
            # Else just use the default norm.
            # Technically we don't need to set that here (RKAdaptiveStepsizeODESolver has it as a default), but it
            # makes it easier to reason about, in the adjoint norm logic, if we know that options['norm'] is
            # definitely set to something.
            options['norm'] = _rms_norm

    # Normalise time
    _check_timelike('t', t, True)
    t_is_reversed = False
    if len(t) > 1 and t[0] > t[1]:
        t_is_reversed = True

    if t_is_reversed:
        # Change the integration times to ascending order.
        # We do this by negating the time values and all associated arguments.
        t = -t

        # Ensure time values are un-negated when calling functions.
        func = _ReverseFunc(func, mul=-1.0)
        if event_fn is not None:
            event_fn = _ReverseFunc(event_fn)

        # For fixed step solvers.
        try:
            _grid_constructor = options['grid_constructor']
        except KeyError:
            pass
        else:
            options['grid_constructor'] = lambda func, y0, t: -_grid_constructor(func, y0, -t)

        # For RK solvers.
        _flip_option(options, 'step_t')
        _flip_option(options, 'jump_t')

    # Can only do after having normalised time
    _assert_increasing('t', t)

    # Tol checking
    if torch.is_tensor(rtol):
        assert not rtol.requires_grad, "rtol cannot require gradient"
    if torch.is_tensor(atol):
        assert not atol.requires_grad, "atol cannot require gradient"

    # Backward compatibility: Allow t and y0 to be on different devices
    if t.device != y0.device:
        warnings.warn("t is not on the same device as y0. Coercing to y0.device.")
        t = t.to(y0.device)
    # ~Backward compatibility

    # Add perturb argument to func.
    func = _PerturbFunc(func)

    # Add callbacks to wrapped_func
    callback_names = set()
    for callback_name in _all_callback_names:
        try:
            callback = getattr(original_func, callback_name)
        except AttributeError:
            setattr(func, callback_name, _null_callback)
        else:
            if callback is not _null_callback:
                callback_names.add(callback_name)
                # At the moment all callbacks have the arguments (t0, y0, dt).
                # These will need adjusting on a per-callback basis if that changes in the future.
                if is_tuple:
                    def callback(t0, y0, dt, _callback=callback):
                        y0 = _flat_to_shape(y0, (), shapes)
                        return _callback(t0, y0, dt)
                if t_is_reversed:
                    def callback(t0, y0, dt, _callback=callback):
                        return _callback(-t0, y0, dt)
            setattr(func, callback_name, callback)
    for callback_name in _all_adjoint_callback_names:
        try:
            callback = getattr(original_func, callback_name)
        except AttributeError:
            pass
        else:
            setattr(func, callback_name, callback)

    invalid_callbacks = callback_names - SOLVERS[method].valid_callbacks()
    if len(invalid_callbacks) > 0:
        warnings.warn("Solver '{}' does not support callbacks {}".format(method, invalid_callbacks))

    return shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed

def hyperbolic_rk4_step_func(func, t0, dt, t1, y0, manifold, perturb=False,c=1.0):
    """
    Implements one step of the hyperbolic 3/8 4th-order Runge-Kutta method.
    
    Args:
        func: Function defining the ODE dynamics, outputs manifold vectors.
        t0: Current time (scalar tensor).
        dt: Time step (scalar tensor).
        t1: Next time (scalar tensor).
        y0: Current state (point on the manifold, tensor).
        manifold: Object with expmap(p, v) and logmap(p, q) methods.
        perturb: Whether to perturb time evaluations (Perturb enum).
    
    Returns:
        y1: Next state (point on the manifold, tensor).
    """
    # refer to the appendix C of Continuous Geometry-Aware Graph Diffusion via Hyperbolic Neural PDE
    # Ensure dt is a scalar tensor with appropriate dtype

    # y0_h = manifold.proj(manifold.expmap0(y0, c), c)  # hyperbolic projection of y0

    dt = dt.to(y0.dtype)
    # k1: Hyperbolic Euler step to get a point on the manifold
    x_k1_hyperbolic = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)  # manifold (h_HEuler(t + τ)) vector at y0
    x_k1 = manifold.logmap(y0, x_k1_hyperbolic,c)
    k1 = manifold.proj(manifold.expmap(y0, dt * x_k1, c), c)

    # k2: Compute intermediate point and evaluate func
    x_k2 = manifold.logmap(y0, k1, c) / 3
    k2 = func(t0 + dt / 3, manifold.proj(manifold.expmap(y0, dt * x_k2, c), c)) 

    # k3: Compute intermediate point and evaluate func
    x_k3 = manifold.logmap(y0, k2, c) - manifold.logmap(y0, k1, c) / 3  # log(k2) - log(k1)/3
    k3 = func(t0 + 2 * dt / 3, manifold.proj(manifold.expmap(y0, dt * x_k3, c), c))
    
    # k4: Compute intermediate point and evaluate func
    x_k4 = manifold.logmap(y0, k1, c) - manifold.logmap(y0, k2, c) + manifold.logmap(y0, k3, c)  # log(k1) - log(k2) + log(k3)
    k4 = func(t1, manifold.proj(manifold.expmap(y0, dt * x_k4, c), c), perturb=Perturb.PREV if perturb else Perturb.NONE)
    
    # Combine tangent vectors with 3/8 weights
    # Note: k2, k3, k4 are in different tangent spaces (T_h2 M, T_h3 M, T_h4 M)
    # Following the paper, we approximate by combining in T_y0 M after adjustment
    x_rk4 = (manifold.logmap(y0, k1, c)  + 3 * manifold.logmap(y0, k2, c)  + 3 * manifold.logmap(y0, k3, c)  + manifold.logmap(y0, k4, c) ) * (dt / 8)
    
    # Update state using exponential map

    y1 = manifold.proj(manifold.expmap(y0, dt * x_rk4, c), c)
    
    return y1


def rk4_alt_step_func(func, t0, dt, t1, y0, f0=None, perturb=False):
    """Smaller error with slightly more compute."""
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)
    k2 = func(t0 + dt * _one_third, y0 + dt * k1 * _one_third)
    k3 = func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third))
    k4 = func(t1, y0 + dt * (k1 - k2 + k3), perturb=Perturb.PREV if perturb else Perturb.NONE)
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125

class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="hyperbolic", perturb=False, manifold=None, **unused_kwargs):
        self.atol = unused_kwargs.pop('atol')
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device
        self.step_size = step_size
        self.interp = interp
        self.perturb = perturb
        self.manifold = manifold

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_callbacks(cls):
        return {'callback_step'}

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t0, dt, t1, y0):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            self.func.callback_step(t0, y0, dt)
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    solution[j] = self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t[j])
                elif self.interp == "hyperbolic":
                    solution[j] = self._hyperbolic_interp(t0, y0, t1, y1, t[j])
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                j += 1
            y0 = y1

        return solution

    def integrate_until_event(self, t0, event_fn):
        assert self.step_size is not None, "Event handling for fixed step solvers currently requires `step_size` to be provided in options."

        t0 = t0.type_as(self.y0.abs())
        y0 = self.y0
        dt = self.step_size

        sign0 = torch.sign(event_fn(t0, y0))
        max_itrs = 20000
        itr = 0
        while True:
            itr += 1
            t1 = t0 + dt
            dy, f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            sign1 = torch.sign(event_fn(t1, y1))

            if sign0 != sign1:
                if self.interp == "linear":
                    interp_fn = lambda t: self._linear_interp(t0, t1, y0, y1, t)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    interp_fn = lambda t: self._cubic_hermite_interp(t0, y0, f0, t1, y1, f1, t)
                else:
                    raise ValueError(f"Unknown interpolation method {self.interp}")
                event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, float(self.atol))
                break
            else:
                t0, y0 = t1, y1

            if itr >= max_itrs:
                raise RuntimeError(f"Reached maximum number of iterations {max_itrs}.")
        solution = torch.stack([self.y0, y1], dim=0)
        return event_time, solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
    
    def _hyperbolic_interp(self, t0, y0, t1, y1, t):
        c = 1.0
        if t == t0:
            return y0
        if t == t1:
            return y1
        h_tan = (t1 - t) * self.manifold.logmap(y0, y1) / (t1 - t0)
        return self.proj(self.manifold.expmap(y0, h_tan, c), c)

class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0
    
class HyperbolicRK4(FixedGridODESolver):
    """
    Hyperbolic 3/8 4th-order Runge-Kutta solver for ODEs on a hyperbolic manifold.
    """
    order = 4
    
    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp="linear", perturb=False, manifold=None, **unused_kwargs):
        """
        Args:
            func: ODE dynamics function.
            y0: Initial state (tensor on the manifold).
            step_size: Fixed step size (optional).
            grid_constructor: Function to construct time grid (optional).
            interp: Interpolation method ("linear" or "cubic").
            perturb: Whether to perturb time evaluations.
            manifold: Manifold object with expmap and logmap methods.
        """
        super().__init__(func, y0, step_size=step_size, grid_constructor=grid_constructor, interp=interp, perturb=perturb, manifold=manifold,  **unused_kwargs)
        self.manifold = manifold
        #self.curvature = options.get("c", 1.0)
        if manifold is None:
            raise ValueError("Manifold object must be provided for hyperbolic integration.")
    
    def _step_func(self, func, t0, dt, t1, y0):
        """
        Performs one integration step.
        
        Args:
            func: ODE dynamics function.
            t0: Current time.
            dt: Time step.
            t1: Next time.
            y0: Current state.
        
        Returns:
            dy: Increment (tangent vector), f0: Initial derivative (tangent vector).
        """
        dy = hyperbolic_rk4_step_func(func, t0, dt, t1, y0, self.manifold, perturb=self.perturb, c=1.0)
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        # Compute dy as the tangent vector from y0 to y1 for compatibility
        # dy = self.manifold.logmap(y0, y1,1.0)
        return dy, f0



class Dopri5Solver():
    print("Dopri5Solver is not implemented yet. Please use RK4 or another solver.")

SOLVERS = {
    'rk4': RK4,
    'hyperbolic_rk4': HyperbolicRK4,
}



def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`, in either increasing or decreasing order. The first element of
            this sequence is taken to be the initial time point.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        event_fn: Function that maps the state `y` to a Tensor. The solve terminates when
            event_fn evaluates to zero. If this is not None, all but the first elements of
            `t` are ignored.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
    """

    shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)

    if event_fn is None:
        solution = solver.integrate(t)
    else:
        event_t, solution = solver.integrate_until_event(t[0], event_fn)
        event_t = event_t.to(t)
        if t_is_reversed:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def odeint_dense(func, y0, t0, t1, *, rtol=1e-7, atol=1e-9, method=None, options=None):

    assert torch.is_tensor(y0)  # TODO: handle tuple of tensors

    t = torch.tensor([t0, t1]).to(t0)

    shapes, func, y0, t, rtol, atol, method, options, _, _ = _check_inputs(func, y0, t, rtol, atol, method, options, None, SOLVERS)

    assert method == "dopri5"

    solver = Dopri5Solver(func=func, y0=y0, rtol=rtol, atol=atol, **options)    
    
    # The integration loop
    solution = torch.empty(len(t), *solver.y0.shape, dtype=solver.y0.dtype, device=solver.y0.device)
    solution[0] = solver.y0
    t = t.to(solver.dtype)
    solver._before_integrate(t)
    t0 = solver.rk_state.t0

    times = [t0]
    interp_coeffs = []

    for i in range(1, len(t)):
        next_t = t[i]
        while next_t > solver.rk_state.t1:
            solver.rk_state = solver._adaptive_step(solver.rk_state)
            t1 = solver.rk_state.t1

            if t1 != t0:
                # Step accepted.
                t0 = t1
                times.append(t1)
                interp_coeffs.append(torch.stack(solver.rk_state.interp_coeff))

        solution[i] = _interp_evaluate(solver.rk_state.interp_coeff, solver.rk_state.t0, solver.rk_state.t1, next_t)

    times = torch.stack(times).reshape(-1).cpu()
    interp_coeffs = torch.stack(interp_coeffs)

    def dense_output_fn(t_eval):
        idx = torch.searchsorted(times, t_eval, side="right")
        t0 = times[idx - 1]
        t1 = times[idx]
        coef = [interp_coeffs[idx - 1][i] for i in range(interp_coeffs.shape[1])]
        return _interp_evaluate(coef, t0, t1, t_eval)

    return dense_output_fn


def odeint_event(func, y0, t0, *, event_fn, reverse_time=False, odeint_interface=odeint, **kwargs):
    """Automatically links up the gradient from the event time."""

    if reverse_time:
        t = torch.cat([t0.reshape(-1), t0.reshape(-1).detach() - 1.0])
    else:
        t = torch.cat([t0.reshape(-1), t0.reshape(-1).detach() + 1.0])

    event_t, solution = odeint_interface(func, y0, t, event_fn=event_fn, **kwargs)

    # Dummy values for rtol, atol, method, and options.
    shapes, _func, _, t, _, _, _, _, event_fn, _ = _check_inputs(func, y0, t, 0.0, 0.0, None, None, event_fn, SOLVERS)

    if shapes is not None:
        state_t = torch.cat([s[-1].reshape(-1) for s in solution])
    else:
        state_t = solution[-1]

    # Event_fn takes in negated time value if reverse_time is True.
    if reverse_time:
        event_t = -event_t

    event_t, state_t = ImplicitFnGradientRerouting.apply(_func, event_fn, event_t, state_t)

    # Return the user expected time value.
    if reverse_time:
        event_t = -event_t

    if shapes is not None:
        state_t = _flat_to_shape(state_t, (), shapes)
        solution = tuple(torch.cat([s[:-1], s_t[None]], dim=0) for s, s_t in zip(solution, state_t))
    else:
        solution = torch.cat([solution[:-1], state_t[None]], dim=0)

    return event_t, solution


class ImplicitFnGradientRerouting(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, event_fn, event_t, state_t):
        """ event_t is the solution to event_fn """
        ctx.func = func
        ctx.event_fn = event_fn
        ctx.save_for_backward(event_t, state_t)
        return event_t.detach(), state_t.detach()

    @staticmethod
    def backward(ctx, grad_t, grad_state):
        func = ctx.func
        event_fn = ctx.event_fn
        event_t, state_t = ctx.saved_tensors

        event_t = event_t.detach().clone().requires_grad_(True)
        state_t = state_t.detach().clone().requires_grad_(True)

        f_val = func(event_t, state_t)

        with torch.enable_grad():
            c, (par_dt, dstate) = vjp(event_fn, (event_t, state_t))

        # Total derivative of event_fn wrt t evaluated at event_t.
        dcdt = par_dt + torch.sum(dstate * f_val)

        # Add the gradient from final state to final time value as if a regular odeint was called.
        grad_t = grad_t + torch.sum(grad_state * f_val)

        dstate = dstate * (-grad_t / (dcdt + 1e-12)).reshape_as(c)

        grad_state = grad_state + dstate

        return None, None, None, grad_state




class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):

        ctx.shapes = shapes
        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad
        ctx.event_mode = event_fn is not None

        with torch.no_grad():
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options, event_fn=event_fn)

            if event_fn is None:
                y = ans
                ctx.save_for_backward(t, y, *adjoint_params)
            else:
                event_t, y = ans
                ctx.save_for_backward(t, y, event_t, *adjoint_params)

        return ans

    @staticmethod
    def backward(ctx, *grad_y):
        with torch.no_grad():
            func = ctx.func
            adjoint_rtol = ctx.adjoint_rtol
            adjoint_atol = ctx.adjoint_atol
            adjoint_method = ctx.adjoint_method
            adjoint_options = ctx.adjoint_options
            t_requires_grad = ctx.t_requires_grad

            # Backprop as if integrating up to event time.
            # Does NOT backpropagate through the event time.
            event_mode = ctx.event_mode
            if event_mode:
                t, y, event_t, *adjoint_params = ctx.saved_tensors
                _t = t
                t = torch.cat([t[0].reshape(-1), event_t.reshape(-1)])
                grad_y = grad_y[1]
            else:
                t, y, *adjoint_params = ctx.saved_tensors
                grad_y = grad_y[0]

            adjoint_params = tuple(adjoint_params)

            ##################################
            #      Set up initial state      #
            ##################################

            # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
            aug_state = [torch.zeros((), dtype=y.dtype, device=y.device), y[-1], grad_y[-1]]  # vjp_t, y, vjp_y
            aug_state.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params

            ##################################
            #    Set up backward ODE func    #
            ##################################

            # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
            def augmented_dynamics(t, y_aug):
                # Dynamics of the original system augmented with
                # the adjoint wrt y, and an integrator wrt t and args.
                y = y_aug[1]
                adj_y = y_aug[2]
                # ignore gradients wrt time and parameters

                with torch.enable_grad():
                    t_ = t.detach()
                    t = t_.requires_grad_(True)
                    y = y.detach().requires_grad_(True)

                    # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
                    # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
                    # wrt t here means we won't compute that if we don't need it.
                    func_eval = func(t if t_requires_grad else t_, y)

                    # Workaround for PyTorch bug #39784
                    _t = torch.as_strided(t, (), ())  # noqa
                    _y = torch.as_strided(y, (), ())  # noqa
                    _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)  # noqa

                    vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                        func_eval, (t, y) + adjoint_params, -adj_y,
                        allow_unused=True, retain_graph=True
                    )

                # autograd.grad returns None if no gradient, set to zero.
                vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
                vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
                vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                              for param, vjp_param in zip(adjoint_params, vjp_params)]

                return (vjp_t, func_eval, vjp_y, *vjp_params)

            # Add adjoint callbacks
            for callback_name, adjoint_callback_name in zip(_all_callback_names, _all_adjoint_callback_names):
                try:
                    callback = getattr(func, adjoint_callback_name)
                except AttributeError:
                    pass
                else:
                    setattr(augmented_dynamics, callback_name, callback)

            ##################################
            #       Solve adjoint ODE        #
            ##################################

            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
            else:
                time_vjps = None
            for i in range(len(t) - 1, 0, -1):
                if t_requires_grad:
                    # Compute the effect of moving the current time measurement point.
                    # We don't compute this unless we need to, to save some computation.
                    func_eval = func(t[i], y[i])
                    dLd_cur_t = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                    aug_state[0] -= dLd_cur_t
                    time_vjps[i] = dLd_cur_t

                # Run the augmented system backwards in time.
                aug_state = odeint(
                    augmented_dynamics, tuple(aug_state),
                    t[i - 1:i + 1].flip(0),
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                )
                aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
                aug_state[1] = y[i - 1]  # update to use our forward-pass estimate of the state
                aug_state[2] += grad_y[i - 1]  # update any gradients wrt state at this time point

            if t_requires_grad:
                time_vjps[0] = aug_state[0]

            # Only compute gradient wrt initial time when in event handling mode.
            if event_mode and t_requires_grad:
                time_vjps = torch.cat([time_vjps[0].reshape(-1), torch.zeros_like(_t[1:])])

            adj_y = aug_state[2]
            adj_params = aug_state[3:]

        return (None, None, adj_y, time_vjps, None, None, None, None, None, None, None, None, None, None, *adj_params)


def odeint_adjoint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None,
                   adjoint_rtol=None, adjoint_atol=None, adjoint_method=None, adjoint_options=None, adjoint_params=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if adjoint_params is None and not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method

    if adjoint_method != method and options is not None and adjoint_options is None:
        raise ValueError("If `adjoint_method != method` then we cannot infer `adjoint_options` from `options`. So as "
                         "`options` has been passed then `adjoint_options` must be passed as well.")

    if adjoint_options is None:
        adjoint_options = {k: v for k, v in options.items() if k != "norm"} if options is not None else {}
    else:
        # Avoid in-place modifying a user-specified dict.
        adjoint_options = adjoint_options.copy()

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)  # in case adjoint_params is a generator.

    # Filter params that don't require gradients.
    oldlen_ = len(adjoint_params)
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)
    if len(adjoint_params) != oldlen_:
        # Some params were excluded.
        # Issue a warning if a user-specified norm is specified.
        if 'norm' in adjoint_options and callable(adjoint_options['norm']):
            warnings.warn("An adjoint parameter was passed without requiring gradient. For efficiency this will be "
                          "excluded from the adjoint pass, and will not appear as a tensor in the adjoint norm.")

    # Convert to flattened state.
    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    # Handle the adjoint norm function.
    state_norm = options["norm"]
    handle_adjoint_norm_(adjoint_options, shapes, state_norm)

    ans = OdeintAdjointMethod.apply(shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol,
                                    adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)

    if event_fn is None:
        solution = ans
    else:
        event_t, solution = ans
        event_t = event_t.to(t)
        if decreasing_time:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


def handle_adjoint_norm_(adjoint_options, shapes, state_norm):
    """In-place modifies the adjoint options to choose or wrap the norm function."""

    # This is the default adjoint norm on the backward pass: a mixed norm over the tuple of inputs.
    def default_adjoint_norm(tensor_tuple):
        t, y, adj_y, *adj_params = tensor_tuple
        # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
        return max(t.abs(), state_norm(y), state_norm(adj_y), _mixed_norm(adj_params))

    if "norm" not in adjoint_options:
        # `adjoint_options` was not explicitly specified by the user. Use the default norm.
        adjoint_options["norm"] = default_adjoint_norm
    else:
        # `adjoint_options` was explicitly specified by the user...
        try:
            adjoint_norm = adjoint_options['norm']
        except KeyError:
            # ...but they did not specify the norm argument. Back to plan A: use the default norm.
            adjoint_options['norm'] = default_adjoint_norm
        else:
            # ...and they did specify the norm argument.
            if adjoint_norm == 'seminorm':
                # They told us they want to use seminorms. Slight modification to plan A: use the default norm,
                # but ignore the parameter state
                def adjoint_seminorm(tensor_tuple):
                    t, y, adj_y, *adj_params = tensor_tuple
                    # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
                    return max(t.abs(), state_norm(y), state_norm(adj_y))
                adjoint_options['norm'] = adjoint_seminorm
            else:
                # And they're using their own custom norm.
                if shapes is None:
                    # The state on the forward pass was a tensor, not a tuple. We don't need to do anything, they're
                    # already going to get given the full adjoint state as (t, y, adj_y, adj_params)
                    pass  # this branch included for clarity
                else:
                    # This is the bit that is tuple/tensor abstraction-breaking, because the odeint machinery
                    # doesn't know about the tupled nature of the forward state. We need to tell the user's adjoint
                    # norm about that ourselves.

                    def _adjoint_norm(tensor_tuple):
                        t, y, adj_y, *adj_params = tensor_tuple
                        y = _flat_to_shape(y, (), shapes)
                        adj_y = _flat_to_shape(adj_y, (), shapes)
                        return adjoint_norm((t, *y, *adj_y, *adj_params))
                    adjoint_options['norm'] = _adjoint_norm
