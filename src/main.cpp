#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <immintrin.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <numeric>
#include <random>

namespace py = pybind11;

class vArray {
public:
    std::vector<float> data;
    vArray(size_t size) : data(size) {}
    vArray(std::vector<float> d) : data(std::move(d)) {}
    size_t size() const { return data.size(); }

    void normalize() {
        const size_t n = data.size();
        if (n == 0) return;
        float* p = data.data();
        __m256 sum_sq = _mm256_setzero_ps();
        size_t i = 0;

        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(p + i);
            sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
        }
        
        // Horizontal sum (AVX -> float)
        __m128 lo = _mm256_castps256_ps128(sum_sq);
        __m128 hi = _mm256_extractf128_ps(sum_sq, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(sums, sums);
        sums = _mm_add_ps(sums, shuf);
        float total_ss = _mm_cvtss_f32(sums);

        for (; i < n; i++) total_ss += p[i] * p[i];

        float norm = std::sqrt(total_ss);
        if (norm < 1e-10f) return;

        float inv_norm = 1.0f / norm;
        __m256 v_inv = _mm256_set1_ps(inv_norm);
        
        i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(p + i);
            _mm256_storeu_ps(p + i, _mm256_mul_ps(v, v_inv));
        }
        for (; i < n; i++) p[i] *= inv_norm;
    }

    void fill_random() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (float& val : data) val = dis(gen);
    }
    void clear() { std::fill(data.begin(), data.end(), 0.0f); }
};

class vMath {
public:
    inline float hsum_avx(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(sums, sums);
        sums = _mm_add_ps(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    float dot(const vArray& a, const vArray& b) {
        const float *pA = a.data.data(), *pB = b.data.data();
        size_t n = a.size(), i = 0;
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        for (; i + 15 < n; i += 16) {
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(pA + i), _mm256_loadu_ps(pB + i), sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(pA + i + 8), _mm256_loadu_ps(pB + i + 8), sum1);
        }
        float res = hsum_avx(_mm256_add_ps(sum0, sum1));
        for (; i < n; i++) res += pA[i] * pB[i];
        return res;
    }

    float cosine(const vArray& a, const vArray& b) {
		const float *pA = a.data.data(), *pB = b.data.data();
		size_t n = a.size(), i = 0;
		__m256 d_v = _mm256_setzero_ps();

		for (; i + 7 < n; i += 8) {
			d_v = _mm256_fmadd_ps(_mm256_loadu_ps(pA + i), _mm256_loadu_ps(pB + i), d_v);
		}
		float dot_val = hsum_avx(d_v);
		for (; i < n; i++) dot_val += pA[i] * pB[i];
		
		return dot_val;
	}

    float l2_sq(const vArray& a, const vArray& b) {
        const float *pA = a.data.data(), *pB = b.data.data();
        size_t n = a.size(), i = 0;
        __m256 s0 = _mm256_setzero_ps();
        for (; i + 7 < n; i += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(pA + i), _mm256_loadu_ps(pB + i));
            s0 = _mm256_fmadd_ps(diff, diff, s0);
        }
        float res = hsum_avx(s0);
        for (; i < n; i++) { float d = pA[i] - pB[i]; res += d * d; }
        return res;
    }

    float sum(const vArray& a) { float s = 0; for(float v : a.data) s += v; return s; }
    float mean(const vArray& a) { return sum(a) / (a.size() + 1e-10f); }

    std::vector<std::pair<float, int>> search(const vArray& query, const std::vector<vArray>& database, int k) {
        int n = database.size();
        std::vector<std::pair<float, int>> results(n);
        
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            results[i] = {cosine(query, database[i]), i};
        }

        int actual_k = std::min(k, n);
        std::partial_sort(results.begin(), results.begin() + actual_k, results.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first; 
            });

        return std::vector<std::pair<float, int>>(results.begin(), results.begin() + actual_k);
    }
};

PYBIND11_MODULE(vmath, m) {
    py::class_<vArray>(m, "vArray")
        .def(py::init<size_t>())
        .def(py::init<std::vector<float>>())
        .def("normalize", &vArray::normalize)
        .def("fill_random", &vArray::fill_random)
        .def("clear", &vArray::clear)
        .def("size", &vArray::size);

    py::class_<vMath>(m, "vMath")
        .def(py::init<>())
        .def("dot", &vMath::dot)
        .def("sum", &vMath::sum)
        .def("mean", &vMath::mean)
        .def("cosine", &vMath::cosine)
        .def("sqeuclidean", &vMath::l2_sq)
        .def("search", &vMath::search);
}
