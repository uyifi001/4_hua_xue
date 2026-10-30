/*
 * deal.II-mirror reproduction of chemo_mechanical.py.
 *
 * This is a dependency-free reference executable so it can be smoke-tested
 * outside the deal.II image.  The arrays correspond to the 1-D vertical
 * finite-volume degrees of freedom used by the deal.II port: oxygen transport
 * (backward-Euler reaction sink), local TGO update, and stress feedback.
 *
 * Units: length um, time h, stress MPa, concentration mol/m^3.
 * The production deal.II version can replace the arrays by DoFHandler/
 * Trilinos vectors without changing the update equations.
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr double pi = 3.14159265358979323846;
constexpr double R = 8.314;

struct Parameters {
  bool tbc = true;
  double temperature_c = 1050.0;
  double t_end = 100.0;
  double wavelength = 60.0;
  double amplitude = 10.0;
  double h_tc = 300.0;
  double h_tgo0 = 1.0;
  double h_base = 200.0;
  double c_surface = 2.5;
  double xi = 1.0e-4 * 3600.0;
  double kappa = 3950.0 / 0.10196 * 1.5;
  double gamma = 3.0e-3;
  double omega = 2.0e-6;
  double d_calib = 1.5;
  int cells = 1000;

  double temperature() const { return temperature_c + 273.15; }
  double kp() const {
    const double k0 = tbc ? 1.46e6 : 6.78e5;
    const double q = tbc ? 167e3 : 146e3;
    return k0 * std::exp(-q / (R * temperature()));
  }
  double diffusivity() const { return kp() * kappa / (2.0 * c_surface) * d_calib; }
};

struct Record {
  double time, thickness, growth, stress, c_min, n_max;
};

void write_svg(const std::string &path, const std::vector<Record> &history,
               const std::vector<double> &y, const std::vector<double> &n,
               const std::vector<double> &c, const std::vector<double> &stress,
               const Parameters &p) {
  const int width = 900, height = 620;
  std::ofstream out(path);
  out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
      << "\" height=\"" << height << "\">\n";
  out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
  auto panel = [&](int x0, int y0, const std::vector<double> &v,
                   const std::string &title, double lo, double hi,
                   const std::string &color) {
    out << "<rect x=\"" << x0 << "\" y=\"" << y0
        << "\" width=\"260\" height=\"240\" fill=\"none\" stroke=\"black\"/>\n";
    out << "<text x=\"" << x0 + 8 << "\" y=\"" << y0 + 18
        << "\" font-size=\"14\">" << title << "</text>\n";
    for (size_t i = 1; i < v.size(); ++i) {
      const double xa = x0 + 260.0 * (y[i - 1] - y.front()) / (y.back() - y.front());
      const double xb = x0 + 260.0 * (y[i] - y.front()) / (y.back() - y.front());
      const double ya = y0 + 220.0 * (1.0 - (v[i - 1] - lo) / (hi - lo));
      const double yb = y0 + 220.0 * (1.0 - (v[i] - lo) / (hi - lo));
      out << "<line x1=\"" << xa << "\" y1=\"" << ya << "\" x2=\"" << xb
          << "\" y2=\"" << yb << "\" stroke=\"" << color << "\"/>\n";
    }
  };
  const auto minmax = [](const std::vector<double> &v) {
    auto mm = std::minmax_element(v.begin(), v.end());
    return std::pair<double, double>(*mm.first, *mm.second);
  };
  auto nc = minmax(n), cc = minmax(c), ss = minmax(stress);
  if (nc.second - nc.first < 1e-12) nc.second = nc.first + 1.0;
  if (cc.second - cc.first < 1e-12) cc.second = cc.first + 1.0;
  if (ss.second - ss.first < 1e-12) ss.second = ss.first + 1.0;
  panel(20, 30, n, "TGO phase n", nc.first, nc.second, "#1565c0");
  panel(315, 30, c, "oxygen concentration", cc.first, cc.second, "#2e7d32");
  panel(610, 30, stress, "sigma_yy (MPa)", ss.first, ss.second, "#c62828");
  out << "<text x=\"20\" y=\"315\" font-size=\"14\">TGO thickness history</text>\n";
  const double ymax = std::max(1.0, std::max_element(history.begin(), history.end(),
      [](const Record &a, const Record &b) { return a.thickness < b.thickness; })->thickness);
  out << "<line x1=\"60\" y1=\"550\" x2=\"840\" y2=\"550\" stroke=\"black\"/>\n";
  for (size_t i = 1; i < history.size(); ++i) {
    const double xa = 60 + 780.0 * history[i - 1].time / p.t_end;
    const double xb = 60 + 780.0 * history[i].time / p.t_end;
    const double ya = 550 - 210.0 * history[i - 1].thickness / ymax;
    const double yb = 550 - 210.0 * history[i].thickness / ymax;
    out << "<line x1=\"" << xa << "\" y1=\"" << ya << "\" x2=\"" << xb
        << "\" y2=\"" << yb << "\" stroke=\"#6a1b9a\"/>\n";
  }
  out << "<text x=\"400\" y=\"590\" font-size=\"14\">time (h)</text>\n</svg>\n";
}
} // namespace

int main(int argc, char **argv) {
  Parameters p;
  std::string outdir = "4/results";
  if (argc > 1) p.tbc = std::string(argv[1]) != "off";
  if (argc > 2) outdir = argv[2];
  const double top = p.tbc ? p.h_tc : 0.0;
  const double bottom = -(p.h_tgo0 + p.h_base);
  const double dy = (top - bottom) / p.cells;
  const double dt = 0.05;
  const int steps = static_cast<int>(std::ceil(p.t_end / dt));
  const double D0 = p.diffusivity();
  std::vector<double> y(p.cells), c(p.cells, 0.0), n(p.cells, 0.0);
  std::vector<double> cnew(p.cells), nnew(p.cells), stress(p.cells);
  for (int i = 0; i < p.cells; ++i) {
    y[i] = bottom + (i + 0.5) * dy;
    n[i] = (y[i] >= -p.h_tgo0 && y[i] <= 0.0) ? 1.0 : 0.0;
    c[i] = y[i] >= 0.0 ? p.c_surface : 0.0;
  }
  std::vector<Record> history;
  history.reserve(steps / 20 + 1);
  const auto thickness = [&]() {
    double h = 0.0;
    for (int i = 0; i < p.cells; ++i)
      if (y[i] <= 0.0 && n[i] > 0.5) h += dy;
    return h;
  };
  double t = 0.0, h0 = thickness();
  for (int step = 0; step < steps; ++step) {
    const auto oldc = c;
    const auto oldn = n;
    // Backward-Euler diffusion (Thomas solve), with the same explicit
    // reaction linearization as the source FEniCSx formulation.
    auto cell_D = [&](int i) {
      return D0 * (p.tbc && y[i] > 0.0 ? 1.0e3 : 1.0);
    };
    std::vector<double> lower(p.cells, 0.0), diag(p.cells, 1.0);
    std::vector<double> upper(p.cells, 0.0), rhs(p.cells);
    for (int i = 0; i < p.cells; ++i) {
      if (i > 0) {
        const double face = 2.0 * cell_D(i - 1) * cell_D(i)
                            / (cell_D(i - 1) + cell_D(i));
        lower[i] = -dt * face / (dy * dy);
        diag[i] -= lower[i];
      }
      if (i + 1 < p.cells) {
        const double face = 2.0 * cell_D(i) * cell_D(i + 1)
                            / (cell_D(i) + cell_D(i + 1));
        upper[i] = -dt * face / (dy * dy);
        diag[i] -= upper[i];
      }
    }
    for (int i = 0; i < p.cells; ++i) {
      const double sink = p.kappa * p.xi * (1.0 - oldn[i]) * oldc[i]
                          / (1.0 + dt * p.xi * oldc[i]);
      rhs[i] = std::max(0.0, oldc[i] - dt * sink);
    }
    rhs.front() = 0.0;
    rhs.back() = p.c_surface;
    diag.front() = diag.back() = 1.0;
    upper.front() = lower.back() = 0.0;
    for (int i = 1; i < p.cells; ++i) {
      const double m = lower[i] / diag[i - 1];
      diag[i] -= m * upper[i - 1];
      rhs[i] -= m * rhs[i - 1];
    }
    cnew.back() = rhs.back() / diag.back();
    for (int i = p.cells - 2; i >= 0; --i)
      cnew[i] = (rhs[i] - upper[i] * cnew[i + 1]) / diag[i];
    c.swap(cnew);
    for (int i = 0; i < p.cells; ++i) {
      const double a = dt * p.xi * c[i];
      nnew[i] = (y[i] <= 0.0)
                    ? std::max(oldn[i], (oldn[i] + a) / (1.0 + a))
                    : 0.0;
    }
    n.swap(nnew);
    const double h = thickness();
    for (int i = 0; i < p.cells; ++i) {
      const double initial_tgo = (y[i] >= -p.h_tgo0 && y[i] <= 0.0) ? 1.0 : 0.0;
      const double growth = p.gamma * (n[i] - initial_tgo);
      stress[i] = 320000.0 * growth + 1.0e3 * p.omega * (c[i] - p.c_surface);
    }
    t = std::min(p.t_end, (step + 1) * dt);
    if (step % 20 == 19 || step + 1 == steps) {
      const double sm = std::accumulate(stress.begin(), stress.end(), 0.0) / p.cells;
      const double cmin = *std::min_element(c.begin(), c.end());
      const double nmax = *std::max_element(n.begin(), n.end());
      history.push_back({t, h, h - h0, sm, cmin, nmax});
    }
  }
  std::string mkdir_cmd = "mkdir -p \"" + outdir + "\"";
  if (std::system(mkdir_cmd.c_str()) != 0) return 2;
  std::ofstream csv(outdir + "/history.csv");
  csv << "time_h,h_tgo_um,dh_tgo_um,sigma_mean_MPa,c_min,n_max\n";
  csv << std::setprecision(10);
  for (const auto &r : history)
    csv << r.time << ',' << r.thickness << ',' << r.growth << ',' << r.stress
        << ',' << r.c_min << ',' << r.n_max << '\n';
  std::ofstream field(outdir + "/final_fields.csv");
  field << "y_um,n_TGO,c_O2,sigma_yy_MPa\n";
  for (int i = 0; i < p.cells; ++i)
    field << y[i] << ',' << n[i] << ',' << c[i] << ',' << stress[i] << '\n';
  write_svg(outdir + "/result.svg", history, y, n, c, stress, p);
  std::ofstream params(outdir + "/params.txt");
  params << "tbc=" << (p.tbc ? "on" : "off") << "\nT_C=" << p.temperature_c
         << "\nkp_um2_per_h=" << p.kp() << "\nD_um2_per_h=" << D0
         << "\nsteps=" << steps << "\n";
  std::cout << "[dealii-mirror] tbc=" << (p.tbc ? "on" : "off")
            << " kp=" << p.kp() << " um^2/h, final TGO=" << history.back().thickness
            << " um -> " << outdir << '\n';
}
