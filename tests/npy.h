#ifndef VPIPE_LTX25_TEST_NPY_H
#define VPIPE_LTX25_TEST_NPY_H

// A minimal .npy reader, for comparing against the reference goldens
// `gen_goldens.py` writes. Handles exactly what that script emits:
// little-endian, C-contiguous, float32 ('<f4'). Anything else is an
// error rather than a reinterpretation -- reading a float64 golden as
// float32 would compare garbage and report a huge, meaningless error.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace npy {

struct Array {
  std::vector<int>   shape;
  std::vector<float> data;
  bool               ok = false;
  std::string        err;

  std::size_t count() const { return data.size(); }
};

inline Array
load(const std::string& path)
{
  Array a;
  auto fail = [&](const std::string& m) { a.err = m; a.ok = false; return a; };
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) { return fail("cannot open " + path); }

  unsigned char magic[8];
  if (std::fread(magic, 1, 8, f) != 8 ||
      std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    std::fclose(f);
    return fail(path + ": not a .npy file");
  }
  const int major = magic[6];
  std::uint32_t hlen = 0;
  if (major == 1) {
    unsigned char b[2];
    if (std::fread(b, 1, 2, f) != 2) { std::fclose(f); return fail("short"); }
    hlen = (std::uint32_t)b[0] | ((std::uint32_t)b[1] << 8);
  } else {
    unsigned char b[4];
    if (std::fread(b, 1, 4, f) != 4) { std::fclose(f); return fail("short"); }
    hlen = (std::uint32_t)b[0] | ((std::uint32_t)b[1] << 8) |
           ((std::uint32_t)b[2] << 16) | ((std::uint32_t)b[3] << 24);
  }
  std::vector<char> hdrbuf(hlen);
  if (std::fread(hdrbuf.data(), 1, hlen, f) != hlen) {
    std::fclose(f);
    return fail(path + ": truncated header");
  }
  const std::string hdr(hdrbuf.begin(), hdrbuf.end());
  if (hdr.find("'<f4'") == std::string::npos &&
      hdr.find("\"<f4\"") == std::string::npos) {
    std::fclose(f);
    return fail(path + ": not float32 little-endian (" + hdr + ")");
  }
  if (hdr.find("'fortran_order': True") != std::string::npos) {
    std::fclose(f);
    return fail(path + ": fortran order is not handled");
  }
  // shape tuple, e.g. "'shape': (1, 32, 24, 64)"
  const std::size_t sp = hdr.find("'shape'");
  const std::size_t lp = hdr.find('(', sp);
  const std::size_t rp = hdr.find(')', lp);
  if (sp == std::string::npos || lp == std::string::npos ||
      rp == std::string::npos) {
    std::fclose(f);
    return fail(path + ": no shape in header");
  }
  std::size_t n = 1;
  std::string cur;
  for (std::size_t i = lp + 1; i <= rp; ++i) {
    const char c = hdr[i];
    if (c == ',' || c == ')') {
      if (!cur.empty()) {
        const int d = std::atoi(cur.c_str());
        a.shape.push_back(d);
        n *= (std::size_t)d;
        cur.clear();
      }
    } else if (c >= '0' && c <= '9') {
      cur += c;
    }
  }
  a.data.resize(n);
  const std::size_t got = std::fread(a.data.data(), sizeof(float), n, f);
  std::fclose(f);
  if (got != n) { return fail(path + ": short data"); }
  a.ok = true;
  return a;
}

// Relative L2 -- the bar the vpipe tree states for encoder work. Returns
// a huge value on a size mismatch rather than comparing a prefix, so a
// shape bug cannot read as a small error.
inline double
rel_l2(const std::vector<float>& got, const std::vector<float>& want)
{
  if (got.size() != want.size() || want.empty()) { return 1e30; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = (double)got[i] - (double)want[i];
    num += d * d;
    den += (double)want[i] * (double)want[i];
  }
  if (den == 0.0) { return num == 0.0 ? 0.0 : 1e30; }
  return std::sqrt(num / den);
}

inline double
max_abs_diff(const std::vector<float>& got, const std::vector<float>& want)
{
  if (got.size() != want.size()) { return 1e30; }
  double m = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double d = std::fabs((double)got[i] - (double)want[i]);
    if (d > m) { m = d; }
  }
  return m;
}

}  // namespace npy

#endif
