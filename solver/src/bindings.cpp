#include <vector>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void check_grid(const py::buffer_info& a, int rows, int cols, bool boolean = false) {
    const auto size = boolean ? sizeof(bool) : sizeof(double);
    const auto format = boolean ? py::format_descriptor<bool>::format() : py::format_descriptor<double>::format();
    if (a.ndim != 2 || a.shape[0] != rows || a.shape[1] != cols ||
        a.strides[1] != size || a.strides[0] != cols * size || a.format != format)
        throw py::value_error("MAC arrays must have matching staggered shapes, C layout, and float64/bool dtype");
}

void check_fields(const py::buffer_info& u, const py::buffer_info& v, const py::buffer_info& mask,
                  double dx, double dy) {
    if (mask.ndim != 2 || mask.shape[0] < 2 || mask.shape[1] < 2)
        throw py::value_error("mask must be a 2D grid with at least 2 cells per direction");
    const int ny = static_cast<int>(mask.shape[0]), nx = static_cast<int>(mask.shape[1]);
    check_grid(mask, ny, nx, true);
    check_grid(u, ny, nx + 1);
    check_grid(v, ny + 1, nx);
    if (!(std::isfinite(dx) && dx > 0 && std::isfinite(dy) && dy > 0))
        throw py::value_error("dx and dy must be finite and positive");
}

void check_step(const py::buffer_info& inlet, int ny, double dt, double rho) {
    if (inlet.ndim != 1 || inlet.shape[0] != ny || inlet.strides[0] != sizeof(double) ||
        inlet.format != py::format_descriptor<double>::format())
        throw py::value_error("inlet must be a contiguous float64 vector of length ny");
    if (!(std::isfinite(dt) && dt > 0 && std::isfinite(rho) && rho > 0))
        throw py::value_error("dt and rho must be finite and positive");
}

#define P(j, i) p[(j) * nx + (i)]
#define MASK(j, i) mask[(j) * nx + (i)]
#define RHS(j, i) rhs[(j) * nx + (i)]
#define PHI(j, i) phi[(j) * nx + (i)]
#define U(j, i) u[(j) * (nx + 1) + (i)]
#define U_STAR(j, i) u_star[(j) * (nx + 1) + (i)]
#define V(j, i) v[(j) * nx + (i)]
#define V_STAR(j, i) v_star[(j) * nx + (i)]

void predictor_rhs_cpp(
    py::array u_arr, py::array v_arr,
    py::array u_star_arr, py::array v_star_arr,
    py::array rhs_arr, py::array p_arr, py::array mask_arr,
    py::array u_inlet_arr,
    double dx, double dy, double dt, double nu, double rho, bool slip_walls, bool centered
) {
    py::buffer_info u_info = u_arr.request();
    py::buffer_info v_info = v_arr.request();
    py::buffer_info u_star_info = u_star_arr.request(true);
    py::buffer_info v_star_info = v_star_arr.request(true);
    py::buffer_info rhs_info = rhs_arr.request(true);
    py::buffer_info p_info = p_arr.request();
    py::buffer_info mask_info = mask_arr.request();
    py::buffer_info u_inlet_info = u_inlet_arr.request();

    check_fields(u_info, v_info, mask_info, dx, dy);
    int ny = static_cast<int>(mask_info.shape[0]);
    int nx = static_cast<int>(mask_info.shape[1]);
    check_grid(u_star_info, ny, nx + 1);
    check_grid(v_star_info, ny + 1, nx);
    check_grid(rhs_info, ny, nx);
    check_grid(p_info, ny, nx);
    check_step(u_inlet_info, ny, dt, rho);
    if (!(std::isfinite(nu) && nu >= 0)) throw py::value_error("nu must be finite and nonnegative");

    const double* u = static_cast<const double*>(u_info.ptr);
    const double* v = static_cast<const double*>(v_info.ptr);
    double* u_star = static_cast<double*>(u_star_info.ptr);
    double* v_star = static_cast<double*>(v_star_info.ptr);
    double* rhs = static_cast<double*>(rhs_info.ptr);
    const double* p = static_cast<const double*>(p_info.ptr);
    const bool* mask = static_cast<const bool*>(mask_info.ptr);
    const double* u_inlet = static_cast<const double*>(u_inlet_info.ptr);

    const double dx2 = dx * dx;
    const double dy2 = dy * dy;

    // 1. Predictor for u on vertical faces: j in [0, ny-1], i in [1, nx-1]
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < ny; ++j) {
        for (int i = 1; i < nx; ++i) {
            if (MASK(j, i - 1) || MASK(j, i)) {
                U_STAR(j, i) = 0.0;
                continue;
            }
            double u_c = U(j, i);

            // Upwind differencing for convective term u * dudx
            double dudx = (u_c > 0.0) ? (u_c - U(j, i - 1)) / dx : (U(j, i + 1) - u_c) / dx;

            // v_avg interpolated to the center of vertical face (j, i)
            double v_avg = 0.25 * (V(j, i - 1) + V(j, i) + V(j + 1, i - 1) + V(j + 1, i));

            // Upwind differencing for convective term v * dudy with wall ghost boundaries
            double u_bot, u_top;
            if (j == 0) {
                u_bot = slip_walls ? u_c : -u_c;
            } else {
                // Reflect across flat staircase faces. At a stair corner the
                // neighbouring normal face itself is a zero-velocity boundary.
                u_bot = (MASK(j - 1, i - 1) && MASK(j - 1, i)) ? -u_c : U(j - 1, i);
            }

            if (j == ny - 1) {
                u_top = slip_walls ? u_c : -u_c;
            } else {
                u_top = (MASK(j + 1, i - 1) && MASK(j + 1, i)) ? -u_c : U(j + 1, i);
            }

            double dudy = (v_avg > 0.0) ? (u_c - u_bot) / dy : (u_top - u_c) / dy;

            // Viscous diffusion (2nd-order central differencing)
            double lap_u = (U(j, i + 1) - 2.0 * u_c + U(j, i - 1)) / dx2 +
                           (u_top - 2.0 * u_c + u_bot) / dy2;

            // Pressure gradient dpdx
            double dpdx = (P(j, i) - P(j, i - 1)) / dx;

            double convection = u_c * dudx + v_avg * dudy;
            if (centered) {
                const double east = .5 * (u_c + U(j, i + 1));
                const double west = .5 * (u_c + U(j, i - 1));
                convection = (east*east - west*west)/dx +
                    .25*((u_c+u_top)*(V(j+1,i-1)+V(j+1,i)) -
                         (u_c+u_bot)*(V(j,i-1)+V(j,i)))/dy;
            }
            U_STAR(j, i) = u_c + dt * (-convection + nu * lap_u - dpdx / rho);
        }

        // Boundary vertical faces for u_star
        U_STAR(j, 0) = u_inlet[j];
        U_STAR(j, nx) = U(j, nx - 1); // Previous-step zero-gradient outlet prediction
    }

    // 2. Predictor for v on horizontal faces: j in [1, ny-1], i in [0, nx-1]
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (MASK(j - 1, i) || MASK(j, i)) {
                V_STAR(j, i) = 0.0;
                continue;
            }
            double v_c = V(j, i);

            // u_avg interpolated to the center of horizontal face (j, i)
            double u_avg = 0.25 * (U(j - 1, i) + U(j, i) + U(j - 1, i + 1) + U(j, i + 1));

            // dvdx with inlet/outlet ghost values
            double v_left, v_right;
            if (i == 0) {
                v_left = -v_c; // v=0 at inlet
            } else {
                v_left = (MASK(j - 1, i - 1) && MASK(j, i - 1)) ? -v_c : V(j, i - 1);
            }

            if (i == nx - 1) {
                v_right = v_c; // Zero gradient dv/dx = 0 at outlet
            } else {
                v_right = (MASK(j - 1, i + 1) && MASK(j, i + 1)) ? -v_c : V(j, i + 1);
            }

            double dvdx = (u_avg > 0.0) ? (v_c - v_left) / dx : (v_right - v_c) / dx;
            double dvdy = (v_c > 0.0) ? (v_c - V(j - 1, i)) / dy : (V(j + 1, i) - v_c) / dy;

            // Viscous diffusion
            double lap_v = (v_right - 2.0 * v_c + v_left) / dx2 +
                           (V(j + 1, i) - 2.0 * v_c + V(j - 1, i)) / dy2;

            // Pressure gradient dpdy
            double dpdy = (P(j, i) - P(j - 1, i)) / dy;

            double convection = u_avg * dvdx + v_c * dvdy;
            if (centered) {
                const double north = .5 * (v_c + V(j + 1, i));
                const double south = .5 * (v_c + V(j - 1, i));
                convection = (north*north-south*south)/dy +
                    .25*((v_c+v_right)*(U(j-1,i+1)+U(j,i+1)) -
                         (v_c+v_left)*(U(j-1,i)+U(j,i)))/dx;
            }
            V_STAR(j, i) = v_c + dt * (-convection + nu * lap_v - dpdy / rho);
        }
    }

    // Wall boundary faces for v_star
    for (int i = 0; i < nx; ++i) {
        V_STAR(0, i) = 0.0;
        V_STAR(ny, i) = 0.0;
    }

    // 3. Poisson RHS for every cell: j in [0, ny-1], i in [0, nx-1]
    const double inv_dt = rho / dt;
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (MASK(j, i)) {
                RHS(j, i) = 0.0;
            } else {
                double div_star = (U_STAR(j, i + 1) - U_STAR(j, i)) / dx +
                                  (V_STAR(j + 1, i) - V_STAR(j, i)) / dy;
                RHS(j, i) = inv_dt * div_star;
            }
        }
    }
}

void corrector_cpp(
    py::array u_arr, py::array v_arr,
    py::array u_star_arr, py::array v_star_arr,
    py::array phi_arr, py::array mask_arr,
    py::array u_inlet_arr,
    double dx, double dy, double dt, double rho, bool slip_walls
) {
    py::buffer_info u_info = u_arr.request(true);
    py::buffer_info v_info = v_arr.request(true);
    py::buffer_info u_star_info = u_star_arr.request();
    py::buffer_info v_star_info = v_star_arr.request();
    py::buffer_info phi_info = phi_arr.request();
    py::buffer_info mask_info = mask_arr.request();
    py::buffer_info u_inlet_info = u_inlet_arr.request();

    check_fields(u_info, v_info, mask_info, dx, dy);
    int ny = static_cast<int>(mask_info.shape[0]);
    int nx = static_cast<int>(mask_info.shape[1]);
    check_grid(u_star_info, ny, nx + 1);
    check_grid(v_star_info, ny + 1, nx);
    check_grid(phi_info, ny, nx);
    check_step(u_inlet_info, ny, dt, rho);

    double* u = static_cast<double*>(u_info.ptr);
    double* v = static_cast<double*>(v_info.ptr);
    const double* u_star = static_cast<const double*>(u_star_info.ptr);
    const double* v_star = static_cast<const double*>(v_star_info.ptr);
    const double* phi = static_cast<const double*>(phi_info.ptr);
    const bool* mask = static_cast<const bool*>(mask_info.ptr);
    const double* u_inlet = static_cast<const double*>(u_inlet_info.ptr);

    const double dt_over_rho = dt / rho;

    // 1. Correct U on vertical faces: j in [0, ny-1], i in [0, nx]
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < ny; ++j) {
        // Inlet boundary face
        U(j, 0) = u_inlet[j];

        // Interior vertical faces
        for (int i = 1; i < nx; ++i) {
            if (MASK(j, i - 1) || MASK(j, i)) {
                U(j, i) = 0.0;
            } else {
                U(j, i) = U_STAR(j, i) - dt_over_rho * (PHI(j, i) - PHI(j, i - 1)) / dx;
            }
        }

        // Outlet vertical face: Exact Dirichlet pressure projection (Phi = 0 at distance dx/2)
        // dPhi/dx = (0 - Phi[j, nx-1]) / (dx/2) = -2*Phi[j, nx-1]/dx
        // U[j, nx] = U*[j, nx] - (dt/rho) * (-2*Phi[j, nx-1]/dx)
        U(j, nx) = U_STAR(j, nx) + dt_over_rho * 2.0 * PHI(j, nx - 1) / dx;
    }

    // 2. Correct V on horizontal faces: j in [0, ny], i in [0, nx-1]
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nx; ++i) {
        V(0, i) = 0.0; // Bottom wall
        for (int j = 1; j < ny; ++j) {
            if (MASK(j - 1, i) || MASK(j, i)) {
                V(j, i) = 0.0;
            } else {
                V(j, i) = V_STAR(j, i) - dt_over_rho * (PHI(j, i) - PHI(j - 1, i)) / dy;
            }
        }
        V(ny, i) = 0.0; // Top wall
    }
}

std::tuple<double, double, double, double> calculate_mac_forces_cpp(
    py::array u_arr,
    py::array v_arr,
    py::array p_arr,
    py::array mask_arr,
    double dx, double dy, double mu
) {
    py::buffer_info u_info = u_arr.request();
    py::buffer_info v_info = v_arr.request();
    py::buffer_info p_info = p_arr.request();
    py::buffer_info mask_info = mask_arr.request();

    check_fields(u_info, v_info, mask_info, dx, dy);
    int ny = static_cast<int>(mask_info.shape[0]);
    int nx = static_cast<int>(mask_info.shape[1]);
    check_grid(p_info, ny, nx);
    if (!(std::isfinite(mu) && mu >= 0)) throw py::value_error("mu must be finite and nonnegative");

    const double* u = static_cast<const double*>(u_info.ptr);
    const double* v = static_cast<const double*>(v_info.ptr);
    const double* p = static_cast<const double*>(p_info.ptr);
    const bool* mask = static_cast<const bool*>(mask_info.ptr);

    double fx_p = 0.0;
    double fy_p = 0.0;
    double fx_v = 0.0;
    double fy_v = 0.0;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (mask[j * nx + i]) {
                // 1. West face of solid: adjacent fluid is at (j, i-1)
                if (i > 0 && !mask[j * nx + (i - 1)]) {
                    fx_p += p[j * nx + (i - 1)] * dy;
                    double v_fluid = 0.5 * (v[j * nx + (i - 1)] + v[(j + 1) * nx + (i - 1)]);
                    fy_v += mu * (v_fluid / (0.5 * dx)) * dy;
                }
                // 2. East face of solid: adjacent fluid is at (j, i+1)
                if (i < nx - 1 && !mask[j * nx + (i + 1)]) {
                    fx_p -= p[j * nx + (i + 1)] * dy;
                    double v_fluid = 0.5 * (v[j * nx + (i + 1)] + v[(j + 1) * nx + (i + 1)]);
                    fy_v += mu * (v_fluid / (0.5 * dx)) * dy;
                }
                // 3. South face of solid: adjacent fluid is at (j-1, i)
                if (j > 0 && !mask[(j - 1) * nx + i]) {
                    fy_p += p[(j - 1) * nx + i] * dx;
                    double u_fluid = 0.5 * (u[(j - 1) * (nx + 1) + i] + u[(j - 1) * (nx + 1) + (i + 1)]);
                    fx_v += mu * (u_fluid / (0.5 * dy)) * dx;
                }
                // 4. North face of solid: adjacent fluid is at (j+1, i)
                if (j < ny - 1 && !mask[(j + 1) * nx + i]) {
                    fy_p -= p[(j + 1) * nx + i] * dx;
                    double u_fluid = 0.5 * (u[(j + 1) * (nx + 1) + i] + u[(j + 1) * (nx + 1) + (i + 1)]);
                    fx_v += mu * (u_fluid / (0.5 * dy)) * dx;
                }
            }
        }
    }
    return std::make_tuple(fx_p, fy_p, fx_v, fy_v);
}

std::tuple<double, double, int, double, int> calculate_mac_divergence_cpp(
    py::array u_arr,
    py::array v_arr,
    py::array mask_arr,
    double dx, double dy
) {
    py::buffer_info u_info = u_arr.request();
    py::buffer_info v_info = v_arr.request();
    py::buffer_info mask_info = mask_arr.request();

    check_fields(u_info, v_info, mask_info, dx, dy);
    int ny = static_cast<int>(mask_info.shape[0]);
    int nx = static_cast<int>(mask_info.shape[1]);

    const double* u = static_cast<const double*>(u_info.ptr);
    const double* v = static_cast<const double*>(v_info.ptr);
    const bool* mask = static_cast<const bool*>(mask_info.ptr);

    double max_div = 0.0;
    double sum_div2 = 0.0;
    int count = 0;
    double core_div2 = 0.0;
    int core_count = 0;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (mask[j * nx + i]) continue;
            double div = (u[j * (nx + 1) + (i + 1)] - u[j * (nx + 1) + i]) / dx +
                         (v[(j + 1) * nx + i] - v[j * nx + i]) / dy;
            double abs_div = std::abs(div);
            if (abs_div > max_div) max_div = abs_div;
            sum_div2 += div * div;
            count++;

            bool is_core = true;
            if (i > 0 && mask[j * nx + (i - 1)]) is_core = false;
            if (i < nx - 1 && mask[j * nx + (i + 1)]) is_core = false;
            if (j > 0 && mask[(j - 1) * nx + i]) is_core = false;
            if (j < ny - 1 && mask[(j + 1) * nx + i]) is_core = false;

            if (is_core) {
                core_div2 += div * div;
                core_count++;
            }
        }
    }
    return std::make_tuple(max_div, sum_div2, count, core_div2, core_count);
}

PYBIND11_MODULE(mac_engine, m) {
    m.doc() = "MAC Grid CFD Engine in C++ with OpenMP acceleration";
    m.def("predictor_rhs_cpp", &predictor_rhs_cpp,
          py::arg("u"), py::arg("v"), py::arg("u_star"), py::arg("v_star"), py::arg("rhs"),
          py::arg("p"), py::arg("mask"), py::arg("inlet"), py::arg("dx"), py::arg("dy"),
          py::arg("dt"), py::arg("nu"), py::arg("rho"), py::arg("slip_walls"), py::arg("centered")=false);
    m.def("corrector_cpp", &corrector_cpp, "Corrector step with exact discrete Dirichlet outlet projection");
    m.def("calculate_mac_forces_cpp", &calculate_mac_forces_cpp, "Approximate pressure and shear integration on staircase cylinder");
    m.def("calculate_mac_divergence_cpp", &calculate_mac_divergence_cpp, "Fast discrete divergence evaluation over fluid cells");
}
