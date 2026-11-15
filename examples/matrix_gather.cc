#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>
#include <span>
#include <vector>


// some basic matrix class
template<typename T>
class matrix : private std::vector<T> {
  using base = std::vector<T>;

public:
  using typename base::size_type;
  using typename base::reference;
  using typename base::const_reference;
  using typename base::iterator;
  using typename base::const_iterator;

private:
  size_type nx, ny;

public:
  matrix(size_type nx, size_type ny) : base(nx * ny), nx(nx), ny(ny) {
  }

  const_reference operator()(size_type ix, size_type iy) const {
    return base::operator[](ix + nx * iy);
  }

  reference operator()(size_type ix, size_type iy) {
    return base::operator[](ix + nx * iy);
  }

  using base::begin;
  using base::end;
  using base::data;
};


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  const int p{comm_world.size()};  // total numbers of processors
  const int p_l{comm_world.rank()};
  // find integer px and py such that px*py=p and px and py as close as possible
  int px{static_cast<int>(std::sqrt(static_cast<double>(p)))};
  while (p / px * px != p)
    --px;
  int py{p / px};
  int nx{31}, ny{29};                      // total size of the matrix
  matrix<int> nx_l(px, py), ny_l(px, py);  // sizes of sub matrices for both dimensions
  matrix<int> nx_0(px, py), ny_0(px, py);  // starts of sub matrices for both dimensions
  // matrix of layouts
  matrix<mp1::layout<int>> sub_matrix_l(px, py);
  // calculate all indices and sizes, generate layouts
  for (int iy{0}; iy < py; ++iy) {
    for (int ix{0}; ix < px; ++ix) {
      nx_l(ix, iy) = nx * (ix + 1) / px - nx * ix / px;
      ny_l(ix, iy) = ny * (iy + 1) / py - ny * iy / py;
      nx_0(ix, iy) = nx * ix / px;
      ny_0(ix, iy) = ny * iy / py;
      mp1::subarray_parameter parameter;
      parameter.add(ny, ny_l(ix, iy), ny_0(ix, iy));
      parameter.add(nx, nx_l(ix, iy), nx_0(ix, iy));
      sub_matrix_l(ix, iy) = mp1::layout<int>::subarray(parameter);
    }
  }
  // process local position in global data grid
  int py_l{p_l / px}, px_l{p_l - px * py_l};

  // gather via send-recv
  {
    // fill some local matrix with data
    matrix<int> m_l(nx_l(px_l, py_l), ny_l(px_l, py_l));
    std::fill(m_l.begin(), m_l.end(), p_l);
    const auto matrix_l{mp1::layout<int>::contiguous(nx_l(px_l, py_l) * ny_l(px_l, py_l))};
    // send local sub-matrix to rank 0
    auto r{mp1::isend(comm_world, mp1::region{m_l.data(), matrix_l}, 0)};
    if (p_l == 0) {
      // gather all submatrices into one large matrix
      matrix<int> m(nx, ny);
      std::fill(m.begin(), m.end(), 0 - ' ');
      for (int iy{0}; iy < py; ++iy) {
        for (int ix{0}; ix < px; ++ix)
          mp1::recv(comm_world, mp1::region{m.data(), sub_matrix_l(ix, iy)}, ix + px * iy);
      }
      for (int iy{0}; iy < ny; ++iy) {
        for (int ix{0}; ix < nx; ++ix)
          std::cout << static_cast<unsigned char>(m(ix, iy) + 'A');
        std::cout << '\n';
      }
      std::cout << '\n';
    }
    r.wait();
  }

  // gather via gatherv
  {
    // fill some local matrix with data
    matrix<int> m_l(nx_l(px_l, py_l), ny_l(px_l, py_l));
    std::fill(m_l.begin(), m_l.end(), p_l);
    // build the layouts for the gatherv operation
    const int root{0};
    const auto matrix_l{mp1::layout<int>::contiguous(nx_l(px_l, py_l) * ny_l(px_l, py_l))};
    if (p_l == root) {
      std::vector<mp1::layout<int>> recvl;
      for (int i{0}; i < p; ++i)
        recvl.push_back(sub_matrix_l(i % px, i / px));
      matrix<int> m(nx, ny);
      mp1::gatherv(
          comm_world, root, mp1::region{m_l.data(), matrix_l},
          std::span<int>{m.data(), static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)},
          recvl);
      for (int iy{0}; iy < ny; ++iy) {
        for (int ix{0}; ix < nx; ++ix)
          std::cout << static_cast<unsigned char>(m(ix, iy) + 'A');
        std::cout << '\n';
      }
      std::cout << '\n';
    } else
      mp1::gatherv(comm_world, root, mp1::region{m_l.data(), matrix_l}, std::span<int>{},
                   mp1::blocks{});
  }

  // gather via alltoallv
  {
    // fill some local matrix with data
    matrix<int> m_l(nx_l(px_l, py_l), ny_l(px_l, py_l));
    std::fill(m_l.begin(), m_l.end(), p_l);
    // build the layouts for alltoallv to implement a gather operation
    const int root{0};
    std::vector<mp1::layout<int>> sendl, recvl;
    for (int i{0}; i < p; ++i) {
      if (i == root)
        sendl.push_back(mp1::layout<int>::contiguous(nx_l(px_l, py_l) * ny_l(px_l, py_l)));
      else
        sendl.push_back(mp1::layout<int>::contiguous(0));
    }
    if (p_l == root) {
      for (int i{0}; i < p; ++i)
        recvl.push_back(sub_matrix_l(i % px, i / px));
      matrix<int> m(nx, ny);
      mp1::alltoallv(
          comm_world,
          std::span<const int>{m_l.data(), static_cast<std::size_t>(nx_l(px_l, py_l)) *
                                               static_cast<std::size_t>(ny_l(px_l, py_l))},
          sendl, std::span<const mp1::count_t>{},
          std::span<int>{m.data(), static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)},
          recvl);
      for (int iy{0}; iy < ny; ++iy) {
        for (int ix{0}; ix < nx; ++ix)
          std::cout << static_cast<unsigned char>(m(ix, iy) + 'A');
        std::cout << '\n';
      }
      std::cout << '\n';
    } else {
      for (int i{0}; i < p; ++i)
        recvl.push_back(mp1::layout<int>::contiguous(0));
      mp1::alltoallv(
          comm_world,
          std::span<const int>{m_l.data(), static_cast<std::size_t>(nx_l(px_l, py_l)) *
                                               static_cast<std::size_t>(ny_l(px_l, py_l))},
          sendl, std::span<const mp1::count_t>{}, std::span<int>{}, recvl);
    }
  }

  return EXIT_SUCCESS;
}
