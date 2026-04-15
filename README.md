# vMath (Vectorized Math Library) The SIMD Killer

**vMath** is a high-performance mathematical library for Python with a C++ backend. It is specifically designed to accelerate vector computations at the hardware level for RAG (Retrieval-Augmented Generation) and semantic search systems.

The project leverages AVX-256 SIMD instructions and OpenMP parallelization to fully utilize CPU capabilities.

---

## ⚡ Why vMath?

vMath stands out for its lightweight design and focus on search-oriented logic.

| Feature      | vMath                     | SimSIMD            | FAISS                | PyTorch          |
| ------------ | ------------------------- | ------------------ | -------------------- | ---------------- |
| Primary Goal | RAG & Fast Search         | Math Primitives    | Large-scale Indexing | Deep Learning    |
| Search Logic | Built-in `search()` API   | Low-level ops only | Complex indexing     | General-purpose  |
| Parallelism  | Built-in OpenMP           | No                 | Yes                  | Yes              |
| Library Size | Very lightweight (<100KB) | Lightweight        | Medium               | Very heavy (GBs) |
| Installation | Easy (`pip install .`)    | Easy               | More complex         | Moderate         |
| Accuracy     | 100% (Exact Search)       | 100%               | Approximate (ANN)    | 100%             |

---

## 🛠 Key Advantages

* Multi-threading across all CPU cores
* SIMD-accelerated computations (AVX-256)
* Eliminates Python loop overhead
* Efficient Top-K retrieval via partial sorting
* No thread conflicts with libraries like PyTorch

---

## 🚀 Installation

Requirements:

* C++17
* CPU with AVX2 support

```bash
# Clone repository
git clone https://github.com/leomaxuz/vMath.git
cd vMath

# Install dependencies
pip install pybind11

# Build and install
pip install . --no-build-isolation
```

---

## 📖 Quick Start

### 1. Create a vector

```python
import vmath

vec = vmath.vArray([0.1, 0.2, 0.3, 0.4])
vec.normalize()
```

### 2. Prepare dataset

```python
db = [vmath.vArray(384) for _ in range(100000)]

for v in db:
    v.fill_random()
    v.normalize()
```

### 3. Parallel search

```python
vm = vmath.vMath()

results = vm.search(query_vec, db, k=5)

for score, idx in results:
    print(f"Index: {idx}, Similarity: {score:.4f}")
```

---

## 🔍 Technical Details

* **SIMD AVX-256**: Processes 8 float operations per cycle
* **Loop Unrolling**: Optimized 16-step pipeline utilization
* **Partial Sort**: Uses `std::partial_sort` for efficient Top-K
* **OpenMP**: Parallel execution across CPU cores

---

## 📊 Benchmark

* Dataset: 100,000 vectors (384 dimensions)
* Speed: ~110 ms (Ryzen 5 / Core i7)
* Memory usage: ~150 MB

---

## 📄 License

MIT License

---

## ⭐ Notes

If you need fast semantic search or a lightweight RAG pipeline, vMath is a minimal and efficient solution.
