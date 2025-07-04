import numpy as np
import matplotlib.pyplot as plt
from dolfinx import mesh, fem, io
from dolfinx.fem import Function, FunctionSpace, Constant
from ufl import TrialFunction, TestFunction, dx, grad, dot
from mpi4py import MPI
import basix

# Parameters
L = 1.0          # Length of the domain
T = 0.5          # Total time
num_steps = 200  # Number of time steps
dt = T / num_steps  # Time step size
c = 1.0          # Wave speed
rho = 1.0        # Density (for weak form)

# Newmark-beta parameters (average acceleration method)
beta = 0.25
gamma = 0.5

# Create mesh and function space
nx = 100  # Number of spatial elements
domain = mesh.create_interval(MPI.COMM_WORLD, nx, [0, L])
V = FunctionSpace(domain, ("Lagrange", 1))  # Linear finite elements

# Define boundary conditions (u = 0 at x = 0, L)
def boundary(x):
    return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], L))
dofs = fem.locate_dofs_geometrical(V, boundary)
bc = fem.dirichletbc(np.array(0.0, dtype=np.float64), dofs, V)

# Initial conditions
u_n = Function(V)  # Displacement at t_n
v_n = Function(V)  # Velocity at t_n
a_n = Function(V)  # Acceleration at t_n

# Set initial displacement: Gaussian pulse
def initial_u(x):
    return np.exp(-100 * (x[0] - 0.5)**2)
u_n.interpolate(initial_u)
v_n.interpolate(lambda x: np.zeros_like(x[0]))  # Zero initial velocity

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Weak form for acceleration
f = Constant(domain, 0.0)  # No external force
a = (1/rho) * (c**2 * dot(grad(u), grad(v)) - f * v) * dx

# Compute initial acceleration
problem = fem.LinearProblem(a, -a, [bc], u=a_n)
problem.solve()

# Time-stepping loop
u_np1 = Function(V)  # Solution at t_{n+1}
a_np1 = Function(V)  # Acceleration at t_{n+1}
u_values = []  # Store solutions for plotting
x_values = domain.geometry.x[:, 0]  # Mesh coordinates

for n in range(num_steps):
    t = (n + 1) * dt

    # Predictor step (Newmark-beta)
    u_pred = Function(V)
    v_pred = Function(V)
    u_pred.x.array[:] = u_n.x.array + dt * v_n.x.array + 0.5 * dt**2 * (1 - 2*beta) * a_n.x.array
    v_pred.x.array[:] = v_n.x.array + dt * (1 - gamma) * a_n.x.array

    # Corrector step: Solve for u_{n+1}
    a_form = (u - u_pred) / (beta * dt**2) * v * dx + (1/rho) * c**2 * dot(grad(u), grad(v)) * dx
    L = f * v * dx
    problem = fem.LinearProblem(a_form, L, [bc], u=u_np1)
    problem.solve()

    # Compute acceleration at t_{n+1}
    a_np1.interpolate(u_np1)
    a_np1.x.array[:] = (u_np1.x.array - u_pred.x.array) / (beta * dt**2)

    # Update velocity
    v_np1 = Function(V)
    v_np1.x.array[:] = v_pred.x.array + dt * gamma * a_np1.x.array

    # Update variables for next step
    u_n.x.array[:] = u_np1.x.array
    v_n.x.array[:] = v_np1.x.array
    a_n.x.array[:] = a_np1.x.array

    # Store solution for visualization
    u_values.append(u_np1.x.array.copy())

    # Optional: Print progress
    if n % 20 == 0:
        print(f"Time: {t:.3f}")

# Plot the solution at selected time steps
plt.figure(figsize=(10, 6))
times_to_plot = [0, 50, 100, 150, 199]
for idx in times_to_plot:
    plt.plot(x_values, u_values[idx], label=f't={idx*dt:.3f}')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('1D Wave Equation Solution using FEniCSx')
plt.legend()
plt.grid(True)
plt.show()