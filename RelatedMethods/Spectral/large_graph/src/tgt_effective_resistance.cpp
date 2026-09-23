#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <unordered_map>
#include <vector>

#include <Eigen/Eigenvalues>
#include <omp.h>

extern "C" {
void dsaupd_c(int* ido, const char* bmat, int n, const char* which, int nev,
              double tol, double* resid, int ncv, double* v, int ldv,
              int* iparam, int* ipntr, double* workd, double* workl,
              int lworkl, int* info);
void dseupd_c(int rvec, const char* howmny, const int* select, double* d,
              double* z, int ldz, double sigma, const char* bmat, int n,
              const char* which, int nev, double tol, double* resid, int ncv,
              double* v, int ldv, int* iparam, int* ipntr, double* workd,
              double* workl, int lworkl, int* info);
}

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kPageHeader = 4096;
constexpr std::uint32_t kBinaryVersion = 1;
constexpr std::uint32_t kEigenVersion = 2;
constexpr std::uint32_t kLoopMapping = std::numeric_limits<std::uint32_t>::max();

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&value, &tm);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buffer;
}

double elapsed(const Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::uint64_t rss_bytes() {
    std::ifstream input("/proc/self/statm");
    std::uint64_t pages = 0;
    std::uint64_t resident = 0;
    input >> pages >> resident;
    return resident * static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
}

std::string gib(const std::uint64_t bytes) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << static_cast<double>(bytes) / static_cast<double>(1ULL << 30) << " GiB";
    return output.str();
}

void log(const std::string& message) {
    std::cout << '[' << timestamp() << "] " << message << " rss=" << gib(rss_bytes())
              << std::endl;
}

std::uint64_t checked_multiply(const std::uint64_t left,
                               const std::uint64_t right,
                               const char* label) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
        throw std::runtime_error(std::string("size overflow for ") + label);
    }
    return left * right;
}

class MappedFile {
  public:
    MappedFile() = default;
    MappedFile(const std::string& path, const bool writable, const std::uint64_t create_bytes = 0)
        : path_(path), writable_(writable) {
        const int flags = writable ? O_RDWR | (create_bytes ? O_CREAT : 0) : O_RDONLY;
        fd_ = ::open(path.c_str(), flags, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
        }
        if (create_bytes != 0 && ::ftruncate(fd_, static_cast<off_t>(create_bytes)) != 0) {
            const std::string error = std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("cannot resize " + path + ": " + error);
        }
        struct stat status {};
        if (::fstat(fd_, &status) != 0 || status.st_size <= 0) {
            const std::string error = std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("cannot stat " + path + ": " + error);
        }
        size_ = static_cast<std::uint64_t>(status.st_size);
        const int protection = PROT_READ | (writable ? PROT_WRITE : 0);
        data_ = ::mmap(nullptr, size_, protection, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            const std::string error = std::strerror(errno);
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("cannot mmap " + path + ": " + error);
        }
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept { *this = std::move(other); }
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            path_ = std::move(other.path_);
            writable_ = other.writable_;
            fd_ = other.fd_;
            data_ = other.data_;
            size_ = other.size_;
            other.fd_ = -1;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    ~MappedFile() { close(); }

    void* data() { return data_; }
    const void* data() const { return data_; }
    std::uint64_t size() const { return size_; }
    int fd() const { return fd_; }

    void sync() {
        if (writable_ && ::msync(data_, size_, MS_SYNC) != 0) {
            throw std::runtime_error("msync failed for " + path_ + ": " +
                                     std::strerror(errno));
        }
    }

    void checkpoint() {
        if (!writable_) return;
        if (::msync(data_, std::min<std::uint64_t>(size_, kPageHeader), MS_SYNC) != 0 ||
            ::fdatasync(fd_) != 0) {
            throw std::runtime_error("checkpoint failed for " + path_ + ": " +
                                     std::strerror(errno));
        }
    }

  private:
    void close() noexcept {
        if (data_ != nullptr) {
            ::munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    std::string path_;
    bool writable_ = false;
    int fd_ = -1;
    void* data_ = nullptr;
    std::uint64_t size_ = 0;
};

#pragma pack(push, 1)
struct InputHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t flags;
    std::uint64_t n;
    std::uint64_t m;
    std::uint64_t directed_m;
    char graph_fingerprint[64];
    char directed_fingerprint[64];
    char dataset[64];
    std::uint64_t src_offset;
    std::uint64_t dst_offset;
    std::uint64_t mapping_offset;
    std::uint64_t file_bytes;
};

struct EigenHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t omega;
    std::uint64_t n;
    std::uint64_t m;
    char graph_fingerprint[64];
    std::uint64_t values_offset;
    std::uint64_t vectors_offset;
    std::uint64_t file_bytes;
    double tolerance;
    std::uint32_t ncv;
    std::uint32_t reserved;
};

struct OrientedHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t omega;
    std::uint64_t n;
    std::uint64_t m;
    std::uint64_t completed_sources;
    char graph_fingerprint[64];
    double epsilon;
    double delta;
    std::int32_t gamma;
    std::uint32_t threads;
    std::uint32_t edge_components;
    std::uint32_t primary_nodes;
    std::uint64_t seed;
    std::uint64_t low_offset;
    std::uint64_t high_offset;
    std::uint64_t file_bytes;
};
#pragma pack(pop)

static_assert(sizeof(InputHeader) == 264, "input header layout changed");

std::string fixed_string(const char* value, const std::size_t size) {
    const auto end = std::find(value, value + size, '\0');
    return std::string(value, end);
}

struct DSU {
    explicit DSU(const std::size_t n) : parent(n), rank(n, 0), size(n, 1) {
        std::iota(parent.begin(), parent.end(), 0U);
    }
    std::uint32_t root(std::uint32_t value) {
        while (parent[value] != value) {
            parent[value] = parent[parent[value]];
            value = parent[value];
        }
        return value;
    }
    void merge(std::uint32_t left, std::uint32_t right) {
        left = root(left);
        right = root(right);
        if (left == right) return;
        if (rank[left] < rank[right]) std::swap(left, right);
        parent[right] = left;
        size[left] += size[right];
        if (rank[left] == rank[right]) ++rank[left];
    }
    std::uint32_t component_size(const std::uint32_t root_node) {
        return size[root(root_node)];
    }
    std::vector<std::uint32_t> parent;
    std::vector<std::uint8_t> rank;
    std::vector<std::uint32_t> size;
};

class Graph {
  public:
    explicit Graph(const std::string& path) : file_(path, false) {
        if (file_.size() < sizeof(InputHeader)) {
            throw std::runtime_error("TGT input is smaller than its header");
        }
        header_ = static_cast<const InputHeader*>(file_.data());
        if (std::memcmp(header_->magic, "SSTGT001", 8) != 0 ||
            header_->version != kBinaryVersion || header_->flags != 3) {
            throw std::runtime_error("unsupported or malformed TGT input header");
        }
        if (header_->file_bytes != file_.size() || header_->n == 0 || header_->m == 0 ||
            header_->n > std::numeric_limits<std::uint32_t>::max() ||
            header_->m > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("invalid dimensions in TGT input header");
        }
        require_range(header_->src_offset, checked_multiply(header_->m, 4, "src"));
        require_range(header_->dst_offset, checked_multiply(header_->m, 4, "dst"));
        require_range(header_->mapping_offset,
                      checked_multiply(header_->directed_m, 4, "mapping"));
        edge_u_ = at<std::uint32_t>(header_->src_offset);
        edge_v_ = at<std::uint32_t>(header_->dst_offset);
        mapping_ = at<std::uint32_t>(header_->mapping_offset);
        build_csr_and_validate();
    }

    std::uint32_t n() const { return static_cast<std::uint32_t>(header_->n); }
    std::uint64_t m() const { return header_->m; }
    std::uint64_t directed_m() const { return header_->directed_m; }
    const std::string fingerprint() const {
        return fixed_string(header_->graph_fingerprint, 64);
    }
    const std::string dataset() const { return fixed_string(header_->dataset, 64); }
    std::uint32_t edge_u(const std::uint32_t edge) const { return edge_u_[edge]; }
    std::uint32_t edge_v(const std::uint32_t edge) const { return edge_v_[edge]; }
    std::uint32_t degree(const std::uint32_t node) const {
        return static_cast<std::uint32_t>(row_[node + 1] - row_[node]);
    }
    std::uint64_t begin(const std::uint32_t node) const { return row_[node]; }
    std::uint64_t end(const std::uint32_t node) const { return row_[node + 1]; }
    std::uint32_t neighbor(const std::uint64_t position) const { return col_[position]; }
    std::uint32_t edge_id(const std::uint64_t position) const { return edge_id_[position]; }
    bool primary_node(const std::uint32_t node) const { return primary_[node] != 0; }
    bool primary_edge(const std::uint32_t edge) const {
        return primary_[edge_u_[edge]] != 0;
    }
    float small_exact_resistance(const std::uint32_t edge) const {
        return small_exact_[edge];
    }
    std::size_t edge_components() const { return edge_components_; }
    std::uint32_t primary_nodes() const { return primary_nodes_; }

    void normalized_adjacency(const double* input, double* output) const {
#pragma omp parallel for schedule(static)
        for (std::int64_t node = 0; node < static_cast<std::int64_t>(n()); ++node) {
            const auto u = static_cast<std::uint32_t>(node);
            const double du = primary_node(u) ? static_cast<double>(degree(u)) : 0.0;
            double value = 0.0;
            if (du != 0.0) {
                for (std::uint64_t position = begin(u); position < end(u); ++position) {
                    const auto v = neighbor(position);
                    value += input[v] / std::sqrt(du * static_cast<double>(degree(v)));
                }
            }
            output[u] = value;
        }
    }

    void normalized_laplacian(const double* input, double* output) const {
#pragma omp parallel for schedule(static)
        for (std::int64_t node = 0; node < static_cast<std::int64_t>(n()); ++node) {
            const auto u = static_cast<std::uint32_t>(node);
            if (!primary_node(u)) {
                output[u] = 0.0;
                continue;
            }
            const double du = static_cast<double>(degree(u));
            double value = input[u];
            for (std::uint64_t position = begin(u); position < end(u); ++position) {
                const auto v = neighbor(position);
                if (primary_node(v)) {
                    value -= input[v] /
                             std::sqrt(du * static_cast<double>(degree(v)));
                }
            }
            output[u] = value;
        }
    }

  private:
    template <typename T> const T* at(const std::uint64_t offset) const {
        return reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(file_.data()) +
                                          offset);
    }
    void require_range(const std::uint64_t offset, const std::uint64_t bytes) const {
        if (offset < sizeof(InputHeader) || offset > file_.size() ||
            bytes > file_.size() - offset) {
            throw std::runtime_error("array offset is outside the TGT input file");
        }
    }

    void build_csr_and_validate() {
        const auto started = Clock::now();
        row_.assign(static_cast<std::size_t>(n()) + 1, 0);
        DSU components(n());
        std::uint64_t previous_key = 0;
        for (std::uint64_t index = 0; index < m(); ++index) {
            const auto u = edge_u_[index];
            const auto v = edge_v_[index];
            if (!(u < v && v < n())) {
                throw std::runtime_error("input endpoints are not canonical u < v pairs");
            }
            const std::uint64_t key = static_cast<std::uint64_t>(u) * n() + v;
            if (index != 0 && key <= previous_key) {
                throw std::runtime_error("input contains duplicate or unsorted canonical edges");
            }
            previous_key = key;
            ++row_[static_cast<std::size_t>(u) + 1];
            ++row_[static_cast<std::size_t>(v) + 1];
            components.merge(u, v);
        }
        std::partial_sum(row_.begin(), row_.end(), row_.begin());
        edge_components_ = 0;
        std::uint32_t primary_root = 0;
        std::uint32_t largest_nodes = 0;
        for (std::uint32_t node = 0; node < n(); ++node) {
            if (degree(node) != 0 && components.root(node) == node) {
                ++edge_components_;
                const auto component_nodes = components.component_size(node);
                if (component_nodes > largest_nodes) {
                    largest_nodes = component_nodes;
                    primary_root = node;
                }
            }
        }
        primary_.assign(n(), 0);
        primary_nodes_ = 0;
        for (std::uint32_t node = 0; node < n(); ++node) {
            if (degree(node) != 0 && components.root(node) == primary_root) {
                primary_[node] = 1;
                ++primary_nodes_;
            }
        }

        col_.resize(static_cast<std::size_t>(2 * m()));
        edge_id_.resize(static_cast<std::size_t>(2 * m()));
        std::vector<std::uint64_t> cursor(row_.begin(), row_.end() - 1);
        for (std::uint64_t index = 0; index < m(); ++index) {
            const auto edge = static_cast<std::uint32_t>(index);
            const auto u = edge_u_[index];
            const auto v = edge_v_[index];
            auto position = cursor[u]++;
            col_[position] = v;
            edge_id_[position] = edge;
            position = cursor[v]++;
            col_[position] = u;
            edge_id_[position] = edge;
        }
        for (std::uint64_t index = 0; index < directed_m(); ++index) {
            if (mapping_[index] != kLoopMapping && mapping_[index] >= m()) {
                throw std::runtime_error("directed-to-canonical mapping is out of bounds");
            }
        }
        compute_small_components(components, primary_root);
        log("CSR_READY dataset=" + dataset() + " nodes=" + std::to_string(n()) +
            " canonical_edges=" + std::to_string(m()) +
            " directed_edges=" + std::to_string(directed_m()) +
            " edge_components=" + std::to_string(edge_components_) +
            " primary_nodes=" + std::to_string(primary_nodes_) +
            " seconds=" + std::to_string(elapsed(started)));
    }

    void compute_small_components(DSU& components, const std::uint32_t primary_root) {
        small_exact_.assign(m(), 0.0F);
        if (edge_components_ <= 1) return;
        std::vector<std::int32_t> root_to_component(n(), -1);
        std::vector<std::vector<std::uint32_t>> nodes;
        for (std::uint32_t node = 0; node < n(); ++node) {
            if (degree(node) == 0) continue;
            const auto root = components.root(node);
            if (root == primary_root) continue;
            if (root_to_component[root] < 0) {
                root_to_component[root] = static_cast<std::int32_t>(nodes.size());
                nodes.emplace_back();
            }
            nodes[static_cast<std::size_t>(root_to_component[root])].push_back(node);
        }
        std::vector<std::vector<std::uint32_t>> edges(nodes.size());
        for (std::uint64_t raw = 0; raw < m(); ++raw) {
            const auto edge = static_cast<std::uint32_t>(raw);
            const auto root = components.root(edge_u(edge));
            if (root != primary_root) {
                edges[static_cast<std::size_t>(root_to_component[root])].push_back(edge);
            }
        }
        const auto started = Clock::now();
        log("SMALL_COMPONENT_EXACT_START components=" + std::to_string(nodes.size()));
#pragma omp parallel for schedule(dynamic, 1)
        for (std::int64_t raw_component = 0;
             raw_component < static_cast<std::int64_t>(nodes.size()); ++raw_component) {
            const auto component = static_cast<std::size_t>(raw_component);
            const auto count = nodes[component].size();
            if (edges[component].size() + 1 == count) {
                for (const auto edge : edges[component]) small_exact_[edge] = 1.0F;
                continue;
            }
            std::unordered_map<std::uint32_t, std::int32_t> local;
            local.reserve(count * 2);
            for (std::size_t index = 0; index < count; ++index) {
                local[nodes[component][index]] = static_cast<std::int32_t>(index);
            }
            Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(count, count);
            for (const auto edge : edges[component]) {
                const auto u = local[edge_u(edge)];
                const auto v = local[edge_v(edge)];
                laplacian(u, u) += 1.0;
                laplacian(v, v) += 1.0;
                laplacian(u, v) -= 1.0;
                laplacian(v, u) -= 1.0;
            }
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(laplacian);
            if (solver.info() != Eigen::Success) {
                throw std::runtime_error("dense eigensolver failed for a small component");
            }
            const auto& values = solver.eigenvalues();
            const auto& vectors = solver.eigenvectors();
            const double tolerance =
                std::numeric_limits<double>::epsilon() * count * values.maxCoeff();
            for (const auto edge : edges[component]) {
                const auto u = local[edge_u(edge)];
                const auto v = local[edge_v(edge)];
                double resistance = 0.0;
                for (std::size_t index = 0; index < count; ++index) {
                    if (values[index] <= tolerance) continue;
                    const double difference = vectors(u, index) - vectors(v, index);
                    resistance += difference * difference / values[index];
                }
                small_exact_[edge] = static_cast<float>(resistance);
            }
        }
        log("SMALL_COMPONENT_EXACT_DONE seconds=" +
            std::to_string(elapsed(started)));
    }

    MappedFile file_;
    const InputHeader* header_ = nullptr;
    const std::uint32_t* edge_u_ = nullptr;
    const std::uint32_t* edge_v_ = nullptr;
    const std::uint32_t* mapping_ = nullptr;
    std::vector<std::uint64_t> row_;
    std::vector<std::uint32_t> col_;
    std::vector<std::uint32_t> edge_id_;
    std::vector<std::uint8_t> primary_;
    std::vector<float> small_exact_;
    std::size_t edge_components_ = 0;
    std::uint32_t primary_nodes_ = 0;
};

std::uint64_t align_page(const std::uint64_t value) {
    return (value + kPageHeader - 1) / kPageHeader * kPageHeader;
}

class EigenCache {
  public:
    EigenCache(const Graph& graph, const std::string& path, const std::uint32_t omega,
               const double tolerance, const std::uint32_t max_iterations)
        : path_(path) {
        if (omega < 3 || omega >= graph.primary_nodes()) {
            throw std::runtime_error(
                "omega must be at least 3 and smaller than the largest component");
        }
        if (!load_if_valid(graph, omega, tolerance)) {
            compute(graph, omega, tolerance, max_iterations);
            if (!load_if_valid(graph, omega, tolerance)) {
                throw std::runtime_error("new eigen cache failed its validation");
            }
        } else {
            log("EIGEN_CACHE_HIT path=" + path_ + " omega=" + std::to_string(omega));
        }
    }

    std::uint32_t omega() const { return header_->omega; }
    double value(const std::uint32_t index) const { return values_[index]; }
    float vector_value(const std::uint32_t node, const std::uint32_t index) const {
        return vectors_[static_cast<std::uint64_t>(node) * omega() + index];
    }

  private:
    bool load_if_valid(const Graph& graph, const std::uint32_t omega,
                       const double tolerance) {
        try {
            MappedFile candidate(path_, false);
            if (candidate.size() < sizeof(EigenHeader)) return false;
            const auto* header = static_cast<const EigenHeader*>(candidate.data());
            const std::uint64_t vector_bytes = checked_multiply(
                checked_multiply(graph.n(), omega, "eigenvector count"), sizeof(float),
                "eigenvector bytes");
            const bool valid =
                std::memcmp(header->magic, "SSEIG002", 8) == 0 &&
                header->version == kEigenVersion && header->omega == omega &&
                header->n == graph.n() && header->m == graph.m() &&
                fixed_string(header->graph_fingerprint, 64) == graph.fingerprint() &&
                header->tolerance == tolerance && header->file_bytes == candidate.size() &&
                header->values_offset >= sizeof(EigenHeader) &&
                header->values_offset + sizeof(double) * omega <= candidate.size() &&
                header->vectors_offset + vector_bytes <= candidate.size();
            if (!valid) return false;
            file_ = std::move(candidate);
            header_ = static_cast<const EigenHeader*>(file_.data());
            values_ = reinterpret_cast<const double*>(
                static_cast<const std::uint8_t*>(file_.data()) + header_->values_offset);
            vectors_ = reinterpret_cast<const float*>(
                static_cast<const std::uint8_t*>(file_.data()) + header_->vectors_offset);
            for (std::uint32_t index = 0; index < omega; ++index) {
                if (!std::isfinite(values_[index])) return false;
            }
            if (std::abs(values_[0] - 1.0) > 1e-5) {
                throw std::runtime_error(
                    "eigen cache does not place the stationary +1 eigenpair first");
            }
            return true;
        } catch (const std::runtime_error&) {
            file_ = MappedFile();
            header_ = nullptr;
            values_ = nullptr;
            vectors_ = nullptr;
            return false;
        }
    }

    void compute(const Graph& graph, const std::uint32_t omega, const double tolerance,
                 const std::uint32_t max_iterations) {
        const auto started = Clock::now();
        const int n = static_cast<int>(graph.n());
        const int nev = static_cast<int>(omega);
        const int ncv = std::min(n, std::max(2 * nev + 1, 20));
        if (ncv <= nev) throw std::runtime_error("ARPACK ncv must be larger than omega");
        log("EIGEN_START omega=" + std::to_string(omega) + " ncv=" +
            std::to_string(ncv) + " tolerance=" + std::to_string(tolerance));

        std::vector<double> resid(static_cast<std::size_t>(n), 0.0);
        std::vector<double> basis(checked_multiply(n, ncv, "ARPACK basis"));
        std::vector<double> workd(static_cast<std::size_t>(3) * n);
        const int lworkl = ncv * (ncv + 8);
        std::vector<double> workl(static_cast<std::size_t>(lworkl));
        int iparam[11] = {};
        int ipntr[14] = {};
        int ido = 0;
        int info = 0;
        iparam[0] = 1;
        iparam[2] = static_cast<int>(max_iterations);
        iparam[6] = 1;
        while (true) {
            dsaupd_c(&ido, "I", n, "LM", nev, tolerance, resid.data(), ncv,
                     basis.data(), n, iparam, ipntr, workd.data(), workl.data(),
                     lworkl, &info);
            if (ido == -1 || ido == 1) {
                const double* input = workd.data() + ipntr[0] - 1;
                double* output = workd.data() + ipntr[1] - 1;
                graph.normalized_adjacency(input, output);
            } else if (ido == 2) {
                const double* input = workd.data() + ipntr[0] - 1;
                double* output = workd.data() + ipntr[1] - 1;
                std::copy(input, input + n, output);
            } else {
                break;
            }
        }
        if (info != 0) {
            throw std::runtime_error("ARPACK dsaupd failed with info=" +
                                     std::to_string(info) +
                                     ", converged=" + std::to_string(iparam[4]));
        }

        std::vector<int> select(static_cast<std::size_t>(ncv), 0);
        std::vector<double> eigenvalues(static_cast<std::size_t>(nev) + 1, 0.0);
        // For a standard symmetric problem ARPACK permits Z to alias V.  This
        // avoids another n*omega double allocation at peak memory.
        dseupd_c(1, "A", select.data(), eigenvalues.data(), basis.data(), n, 0.0,
                 "I", n, "LM", nev, tolerance, resid.data(), ncv, basis.data(), n,
                 iparam, ipntr, workd.data(), workl.data(), lworkl, &info);
        if (info != 0) {
            throw std::runtime_error("ARPACK dseupd failed with info=" +
                                     std::to_string(info));
        }

        std::vector<std::uint32_t> order(omega);
        std::iota(order.begin(), order.end(), 0U);
        const auto stationary = static_cast<std::uint32_t>(std::distance(
            eigenvalues.begin(),
            std::min_element(eigenvalues.begin(), eigenvalues.begin() + nev,
                             [](const double left, const double right) {
                                 return std::abs(left - 1.0) < std::abs(right - 1.0);
                             })));
        if (std::abs(eigenvalues[stationary] - 1.0) > 1e-5) {
            throw std::runtime_error("ARPACK did not return the stationary +1 eigenpair");
        }
        std::swap(order[0], *std::find(order.begin(), order.end(), stationary));
        std::sort(order.begin() + 1, order.end(), [&](const auto left, const auto right) {
            return std::abs(eigenvalues[left]) > std::abs(eigenvalues[right]);
        });

        const std::uint64_t values_offset = kPageHeader;
        const std::uint64_t vectors_offset =
            align_page(values_offset + sizeof(double) * omega);
        const std::uint64_t vector_count = checked_multiply(graph.n(), omega, "vectors");
        const std::uint64_t file_bytes =
            vectors_offset + checked_multiply(vector_count, sizeof(float), "vectors");
        const std::string temporary = path_ + ".tmp." + std::to_string(::getpid());
        MappedFile output(temporary, true, file_bytes);
        auto* header = static_cast<EigenHeader*>(output.data());
        std::memset(header, 0, sizeof(*header));
        std::memcpy(header->magic, "SSEIG002", 8);
        header->version = kEigenVersion;
        header->omega = omega;
        header->n = graph.n();
        header->m = graph.m();
        std::memcpy(header->graph_fingerprint, graph.fingerprint().data(), 64);
        header->values_offset = values_offset;
        header->vectors_offset = vectors_offset;
        header->file_bytes = file_bytes;
        header->tolerance = tolerance;
        header->ncv = static_cast<std::uint32_t>(ncv);
        auto* saved_values = reinterpret_cast<double*>(
            static_cast<std::uint8_t*>(output.data()) + values_offset);
        auto* saved_vectors = reinterpret_cast<float*>(
            static_cast<std::uint8_t*>(output.data()) + vectors_offset);
        for (std::uint32_t index = 0; index < omega; ++index) {
            saved_values[index] = eigenvalues[order[index]];
        }
#pragma omp parallel for schedule(static)
        for (std::int64_t node = 0; node < static_cast<std::int64_t>(graph.n()); ++node) {
            const auto u = static_cast<std::uint32_t>(node);
            const double inverse_root_degree =
                graph.degree(u) == 0 ? 0.0 : 1.0 / std::sqrt(graph.degree(u));
            for (std::uint32_t index = 0; index < omega; ++index) {
                const double transformed =
                    basis[static_cast<std::uint64_t>(order[index]) * graph.n() + u] *
                    inverse_root_degree;
                saved_vectors[static_cast<std::uint64_t>(u) * omega + index] =
                    static_cast<float>(transformed);
            }
        }
        output.sync();
        output = MappedFile();
        if (::rename(temporary.c_str(), path_.c_str()) != 0) {
            const std::string error = std::strerror(errno);
            ::unlink(temporary.c_str());
            throw std::runtime_error("cannot publish eigen cache: " + error);
        }
        log("EIGEN_DONE seconds=" + std::to_string(elapsed(started)) +
            " iterations=" + std::to_string(iparam[2]) +
            " operations=" + std::to_string(iparam[8]));
    }

    std::string path_;
    MappedFile file_;
    const EigenHeader* header_ = nullptr;
    const double* values_ = nullptr;
    const float* vectors_ = nullptr;
};

struct Config {
    std::string algorithm = "tgt";
    std::string input;
    std::string eigen_cache;
    std::string oriented_output;
    std::uint32_t threads = 8;
    std::uint32_t omega = 128;
    std::int32_t gamma = 10;
    std::uint32_t block_sources = 1024;
    std::uint32_t max_iterations = 10000;
    double epsilon = 0.05;
    double delta = 0.01;
    double eigen_tolerance = 1e-6;
    double memory_gb = 0.0;
    double max_seconds = 19'800.0;
    double pcg_tolerance = 1e-2;
    std::uint32_t pcg_iterations = 100;
    std::uint32_t min_projections = 8;
    std::uint64_t seed = 42;
    bool force = false;
    bool estimate_only = false;
};

double safe_lambda(const double value) {
    return std::min(std::abs(value), 1.0 - 1e-12);
}

std::uint32_t tau_from_bound(const Graph& graph, const std::uint32_t u,
                             const std::uint32_t v, const double epsilon,
                             const double lambda, const double delta_term,
                             const double upsilon) {
    const double du = graph.degree(u);
    const double dv = graph.degree(v);
    const double lam = safe_lambda(lambda);
    if (lam <= 1e-14) return 1;
    const double epsilon_delta = epsilon - delta_term;
    if (epsilon_delta <= 0.0) return std::numeric_limits<std::uint32_t>::max();
    const double numerator = 1.0 / du + 1.0 / dv - 2.0 / (du * dv) - upsilon;
    const double denominator = epsilon_delta * (1.0 - lam * lam);
    const double ratio = std::max(numerator / denominator, 1.0);
    const double raw = std::ceil(std::log(ratio) / std::log(1.0 / lam) - 1.0);
    if (!std::isfinite(raw) || raw > std::numeric_limits<std::uint32_t>::max() - 2.0) {
        throw std::runtime_error("edge truncation length overflowed uint32");
    }
    std::uint32_t tau = std::max<std::uint32_t>(static_cast<std::uint32_t>(
                                                   std::max(raw, 1.0)),
                                               1);
    if ((tau & 1U) == 0U) ++tau;
    return tau;
}

std::uint32_t calculate_tau(const Graph& graph, const EigenCache& eigen,
                            const std::uint32_t u, const std::uint32_t v,
                            const double epsilon) {
    const std::uint32_t conservative_tau =
        tau_from_bound(graph, u, v, epsilon, eigen.value(1), 0.0, 0.0);
    double upsilon = 0.0;
    for (std::uint32_t index = 1; index + 1 < eigen.omega(); ++index) {
        const double difference = static_cast<double>(eigen.vector_value(u, index)) -
                                  static_cast<double>(eigen.vector_value(v, index));
        upsilon += difference * difference * (1.0 + eigen.value(index));
    }
    std::uint32_t t = 1;
    while (t <= conservative_tau) {
        double delta_term = 0.0;
        for (std::uint32_t index = 1; index + 1 < eigen.omega(); ++index) {
            const double difference = static_cast<double>(eigen.vector_value(u, index)) -
                                      static_cast<double>(eigen.vector_value(v, index));
            const double lambda = eigen.value(index);
            delta_term += difference * difference * std::pow(lambda, t + 1) /
                          (1.0 - lambda);
        }
        const auto candidate = tau_from_bound(graph, u, v, epsilon,
                                              eigen.value(eigen.omega() - 1),
                                              delta_term, upsilon);
        if (t > candidate) break;
        if (t > std::numeric_limits<std::uint32_t>::max() - 2) {
            throw std::runtime_error("edge truncation iteration overflow");
        }
        t += 2;
    }
    // Algorithm 1 returns t, not the last candidate bound.  Cap it by the
    // conservative lambda_2-only bound, which is independently valid.  The
    // upstream research code returned its stale `tau` and reset a negative
    // epsilon-delta to epsilon; both choices can violate the requested bound.
    return std::min(t, conservative_tau);
}

class XorShift64 {
  public:
    explicit XorShift64(std::uint64_t seed) : state_(mix(seed)) {
        if (state_ == 0) state_ = 0x9e3779b97f4a7c15ULL;
    }
    std::uint64_t next() {
        std::uint64_t value = state_;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        state_ = value;
        return value * 2685821657736338717ULL;
    }
    std::uint32_t bounded(const std::uint32_t bound) {
        return static_cast<std::uint32_t>(next() % bound);
    }
    static std::uint64_t mix(std::uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

  private:
    std::uint64_t state_;
};

struct Workspace {
    explicit Workspace(const std::uint32_t n)
        : current_probability(n, 0.0), next_probability(n, 0.0) {
        active.reserve(1024);
        next_active.reserve(1024);
    }
    std::vector<double> current_probability;
    std::vector<double> next_probability;
    std::vector<std::uint32_t> active;
    std::vector<std::uint32_t> next_active;
    std::vector<double> neighbor_h;
    std::vector<std::uint32_t> candidates;
};

std::uint64_t sample_count(const Graph& graph, const Config& config,
                           const std::uint32_t source_degree, const double chi) {
    const double denominator =
        std::pow(static_cast<double>(source_degree) * config.epsilon, 2);
    const double value = 8.0 * chi * chi *
                         std::log(2.0 * static_cast<double>(graph.m()) / config.delta) /
                         denominator;
    if (!std::isfinite(value) || value > static_cast<double>(
                                               std::numeric_limits<std::uint64_t>::max() - 1)) {
        throw std::runtime_error("random-walk sample count overflow");
    }
    return std::max<std::uint64_t>(static_cast<std::uint64_t>(std::ceil(value)), 1);
}

void push_probability(const Graph& graph, const std::uint32_t source,
                      Workspace& workspace, const double source_degree) {
    workspace.next_active.clear();
    for (const auto node : workspace.active) {
        const double residue = workspace.current_probability[node];
        workspace.current_probability[node] = 0.0;
        for (std::uint64_t position = graph.begin(node); position < graph.end(node);
             ++position) {
            const auto target = graph.neighbor(position);
            if (workspace.next_probability[target] == 0.0) {
                workspace.next_active.push_back(target);
            }
            workspace.next_probability[target] += residue / graph.degree(target);
        }
    }
    workspace.active.swap(workspace.next_active);
    workspace.current_probability.swap(workspace.next_probability);
    const double source_value = workspace.current_probability[source];
    std::size_t neighbor_index = 0;
    for (std::uint64_t position = graph.begin(source); position < graph.end(source);
         ++position, ++neighbor_index) {
        workspace.neighbor_h[neighbor_index] +=
            (source_value - workspace.current_probability[graph.neighbor(position)]) /
            source_degree;
    }
}

double calculate_edge_max(const Graph& graph, Workspace& workspace,
                          const double global_max, const std::int32_t gamma) {
    if (gamma <= 0 || workspace.active.empty()) return 2.0 * global_max;
    const std::size_t wanted =
        std::min<std::size_t>(static_cast<std::size_t>(gamma), workspace.active.size());
    workspace.candidates.clear();
    workspace.candidates.reserve(wanted);
    for (const auto node : workspace.active) {
        if (workspace.candidates.size() < wanted) {
            workspace.candidates.push_back(node);
            if (workspace.candidates.size() == wanted) {
                std::sort(workspace.candidates.begin(), workspace.candidates.end(),
                          [&](const auto left, const auto right) {
                              return workspace.current_probability[left] >
                                     workspace.current_probability[right];
                          });
            }
        } else if (workspace.current_probability[node] >
                   workspace.current_probability[workspace.candidates.back()]) {
            workspace.candidates.back() = node;
            for (std::size_t index = wanted - 1; index != 0; --index) {
                if (workspace.current_probability[workspace.candidates[index]] <=
                    workspace.current_probability[workspace.candidates[index - 1]])
                    break;
                std::swap(workspace.candidates[index], workspace.candidates[index - 1]);
            }
        }
    }
    const double gamma_max = workspace.active.size() > wanted
                                 ? workspace.current_probability[workspace.candidates.back()]
                                 : 0.0;
    double edge_max = global_max + gamma_max;
    for (const auto u : workspace.candidates) {
        for (std::uint64_t position = graph.begin(u); position < graph.end(u); ++position) {
            const auto v = graph.neighbor(position);
            if (std::find(workspace.candidates.begin(), workspace.candidates.end(), v) !=
                workspace.candidates.end()) {
                edge_max = std::max(edge_max, workspace.current_probability[u] +
                                                  workspace.current_probability[v]);
            }
        }
    }
    return edge_max;
}

std::pair<double, double> neighbor_range(const Graph& graph, const std::uint32_t node,
                                         const std::vector<double>& probability) {
    double minimum = std::numeric_limits<double>::max();
    double maximum = 0.0;
    for (std::uint64_t position = graph.begin(node); position < graph.end(node); ++position) {
        const double value = probability[graph.neighbor(position)];
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    return {minimum, maximum};
}

double calculate_chi(const Graph& graph, const std::uint32_t source,
                     const std::uint32_t target, const Workspace& workspace,
                     const std::uint32_t remaining, const double global_minimum,
                     const double global_maximum, const double edge_max,
                     const std::int32_t gamma) {
    if (gamma <= 0) return 2.0 * remaining * global_maximum;
    const auto source_range = neighbor_range(graph, source, workspace.current_probability);
    const auto target_range = neighbor_range(graph, target, workspace.current_probability);
    return global_maximum + (source_range.second + target_range.second) / 2.0 +
           (remaining - 1.0) * edge_max - source_range.first - target_range.first -
           2.0 * (remaining - 1.0) * global_minimum;
}

double random_walk_difference(const Graph& graph, const std::uint32_t source,
                              const std::uint32_t target, const std::uint64_t walks,
                              const std::uint32_t length,
                              const std::vector<double>& probability,
                              const std::uint64_t seed) {
    XorShift64 random(seed);
    double value = 0.0;
    for (std::uint64_t walk = 0; walk < walks; ++walk) {
        std::uint32_t left = source;
        std::uint32_t right = target;
        for (std::uint32_t step = 0; step < length; ++step) {
            const auto left_position = graph.begin(left) + random.bounded(graph.degree(left));
            const auto right_position = graph.begin(right) + random.bounded(graph.degree(right));
            left = graph.neighbor(left_position);
            right = graph.neighbor(right_position);
            value += probability[left] - probability[right];
        }
    }
    return value;
}

class OrientedOutput {
  public:
    OrientedOutput(const Graph& graph, const Config& config) {
        if (config.force) ::unlink(config.oriented_output.c_str());
        bool exists = ::access(config.oriented_output.c_str(), F_OK) == 0;
        if (!exists) create(graph, config);
        file_ = MappedFile(config.oriented_output, true);
        if (file_.size() < sizeof(OrientedHeader)) {
            throw std::runtime_error("oriented output is truncated");
        }
        header_ = static_cast<OrientedHeader*>(file_.data());
        const std::uint64_t array_bytes = checked_multiply(graph.m(), sizeof(float), "ER");
        const bool valid =
            std::memcmp(header_->magic, "SSORI001", 8) == 0 &&
            header_->version == 2 && header_->omega == config.omega &&
            header_->n == graph.n() && header_->m == graph.m() &&
            fixed_string(header_->graph_fingerprint, 64) == graph.fingerprint() &&
            header_->epsilon == config.epsilon && header_->delta == config.delta &&
            ((config.gamma < 0 && header_->gamma < 0) ||
             header_->gamma == config.gamma) &&
            header_->seed == config.seed &&
            header_->edge_components == graph.edge_components() &&
            header_->primary_nodes == graph.primary_nodes() &&
            header_->completed_sources <= graph.n() &&
            header_->low_offset + array_bytes <= file_.size() &&
            header_->high_offset + array_bytes <= file_.size() &&
            header_->file_bytes == file_.size();
        if (!valid) {
            throw std::runtime_error(
                "oriented checkpoint parameters do not match; use --force to replace it");
        }
        low_ = reinterpret_cast<float*>(static_cast<std::uint8_t*>(file_.data()) +
                                        header_->low_offset);
        high_ = reinterpret_cast<float*>(static_cast<std::uint8_t*>(file_.data()) +
                                         header_->high_offset);
    }

    std::uint64_t completed_sources() const { return header_->completed_sources; }
    float* low() { return low_; }
    float* high() { return high_; }
    void set_gamma(const std::int32_t gamma) { header_->gamma = gamma; }
    void checkpoint(const std::uint64_t completed) {
        header_->completed_sources = completed;
        file_.checkpoint();
    }

  private:
    static void create(const Graph& graph, const Config& config) {
        const std::uint64_t low_offset = kPageHeader;
        const std::uint64_t high_offset =
            align_page(low_offset + checked_multiply(graph.m(), sizeof(float), "low ER"));
        const std::uint64_t file_bytes =
            high_offset + checked_multiply(graph.m(), sizeof(float), "high ER");
        MappedFile output(config.oriented_output, true, file_bytes);
        auto* header = static_cast<OrientedHeader*>(output.data());
        std::memset(header, 0, sizeof(*header));
        std::memcpy(header->magic, "SSORI001", 8);
        header->version = 2;
        header->omega = config.omega;
        header->n = graph.n();
        header->m = graph.m();
        header->completed_sources = 0;
        std::memcpy(header->graph_fingerprint, graph.fingerprint().data(), 64);
        header->epsilon = config.epsilon;
        header->delta = config.delta;
        header->gamma = config.gamma;
        header->threads = config.threads;
        header->edge_components = static_cast<std::uint32_t>(graph.edge_components());
        header->primary_nodes = graph.primary_nodes();
        header->seed = config.seed;
        header->low_offset = low_offset;
        header->high_offset = high_offset;
        header->file_bytes = file_bytes;
        output.checkpoint();
    }

    MappedFile file_;
    OrientedHeader* header_ = nullptr;
    float* low_ = nullptr;
    float* high_ = nullptr;
};

std::uint64_t process_source(const Graph& graph, const Config& config,
                             const std::vector<std::uint32_t>& taus,
                             const std::uint32_t source, Workspace& workspace,
                             float* low_result, float* high_result) {
    const auto source_degree = graph.degree(source);
    if (source_degree == 0 || !graph.primary_node(source)) return 0;
    workspace.active.clear();
    workspace.next_active.clear();
    workspace.neighbor_h.assign(source_degree, 1.0 / source_degree);
    workspace.current_probability[source] = 1.0;
    workspace.active.push_back(source);

    std::uint32_t ell = 0;
    double global_minimum = 0.0;
    double global_maximum = 1.0;
    while (true) {
        push_probability(graph, source, workspace, source_degree);
        ++ell;
        global_minimum = 1.0;
        global_maximum = 0.0;
        std::uint64_t traversal_cost = 0;
        for (const auto node : workspace.active) {
            traversal_cost += graph.degree(node);
            global_minimum =
                std::min(global_minimum, workspace.current_probability[node]);
            global_maximum =
                std::max(global_maximum, workspace.current_probability[node]);
        }
        if (workspace.active.size() < graph.n()) global_minimum = 0.0;
        std::uint64_t walk_cost = 0;
        for (std::uint64_t position = graph.begin(source); position < graph.end(source);
             ++position) {
            const auto edge = graph.edge_id(position);
            if (taus[edge] <= ell) continue;
            const auto remaining = taus[edge] - ell;
            const double chi = 2.0 * remaining * (global_maximum - global_minimum);
            const auto count = sample_count(graph, config, source_degree, chi);
            if (walk_cost > std::numeric_limits<std::uint64_t>::max() - count) {
                throw std::runtime_error("random-walk cost overflow");
            }
            walk_cost += count;
        }
        if (walk_cost == 0 ||
            static_cast<long double>(traversal_cost) >= 25.0L * walk_cost) {
            break;
        }
    }

    double edge_max = calculate_edge_max(graph, workspace, global_maximum, config.gamma);
    if (config.gamma == 1) edge_max = 2.0 * global_maximum;
    std::uint64_t total_walks = 0;
    std::size_t neighbor_index = 0;
    for (std::uint64_t position = graph.begin(source); position < graph.end(source);
         ++position, ++neighbor_index) {
        const auto target = graph.neighbor(position);
        const auto edge = graph.edge_id(position);
        if (taus[edge] > ell) {
            const auto remaining = taus[edge] - ell;
            const double chi = calculate_chi(graph, source, target, workspace, remaining,
                                             global_minimum, global_maximum, edge_max,
                                             config.gamma);
            const auto walks = sample_count(graph, config, source_degree, chi);
            const auto random_seed =
                XorShift64::mix(config.seed ^ (static_cast<std::uint64_t>(source) << 32) ^
                                static_cast<std::uint64_t>(target));
            const double estimate = random_walk_difference(
                graph, source, target, walks, remaining,
                workspace.current_probability, random_seed);
            workspace.neighbor_h[neighbor_index] +=
                estimate / (static_cast<double>(source_degree) * walks);
            if (total_walks > std::numeric_limits<std::uint64_t>::max() - walks) {
                throw std::runtime_error("per-source random-walk count overflow");
            }
            total_walks += walks;
        }
        if (source == graph.edge_u(edge)) {
            low_result[edge] = static_cast<float>(workspace.neighbor_h[neighbor_index]);
        } else {
            high_result[edge] = static_cast<float>(workspace.neighbor_h[neighbor_index]);
        }
    }
    for (const auto node : workspace.active) workspace.current_probability[node] = 0.0;
    workspace.active.clear();
    return total_walks;
}

std::vector<std::uint32_t> precompute_taus(const Graph& graph, const EigenCache& eigen,
                                           const Config& config) {
    const auto started = Clock::now();
    log("TAU_START edges=" + std::to_string(graph.m()));
    std::vector<std::uint32_t> taus(graph.m(), 1);
    constexpr std::uint64_t chunk = 1'000'000;
    for (std::uint64_t first = 0; first < graph.m(); first += chunk) {
        const std::uint64_t last = std::min(graph.m(), first + chunk);
        std::atomic<bool> failed{false};
        std::string failure;
        std::mutex failure_mutex;
#pragma omp parallel for schedule(static)
        for (std::int64_t raw = static_cast<std::int64_t>(first);
             raw < static_cast<std::int64_t>(last); ++raw) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                const auto edge = static_cast<std::uint32_t>(raw);
                taus[edge] = graph.primary_edge(edge)
                                 ? calculate_tau(graph, eigen, graph.edge_u(edge),
                                                 graph.edge_v(edge),
                                                 config.epsilon / 2.0)
                                 : 0;
            } catch (const std::exception& error) {
                failed.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> guard(failure_mutex);
                if (failure.empty()) failure = error.what();
            }
        }
        if (failed) throw std::runtime_error("tau computation failed: " + failure);
        const auto maximum = *std::max_element(taus.begin() + first, taus.begin() + last);
        log("TAU_PROGRESS edges=" + std::to_string(last) + "/" +
            std::to_string(graph.m()) + " chunk_max=" + std::to_string(maximum) +
            " elapsed_seconds=" + std::to_string(elapsed(started)));
    }
    log("TAU_DONE seconds=" + std::to_string(elapsed(started)));
    return taus;
}

void run_tgt(const Graph& graph, const EigenCache& eigen, const Config& config) {
    auto taus = precompute_taus(graph, eigen, config);
    OrientedOutput output(graph, config);
    std::uint64_t completed = output.completed_sources();
    if (completed == 0 && graph.edge_components() > 1) {
#pragma omp parallel for schedule(static)
        for (std::int64_t raw = 0; raw < static_cast<std::int64_t>(graph.m()); ++raw) {
            const auto edge = static_cast<std::uint32_t>(raw);
            if (!graph.primary_edge(edge)) {
                output.low()[edge] = graph.small_exact_resistance(edge);
                output.high()[edge] = 0.0F;
            }
        }
        output.checkpoint(0);
    }
    std::uint64_t all_walks = 0;
    std::vector<std::unique_ptr<Workspace>> workspaces(config.threads);
#pragma omp parallel
    {
        const auto thread = static_cast<std::size_t>(omp_get_thread_num());
        workspaces[thread] = std::make_unique<Workspace>(graph.n());
    }
    log("TGT_START resume_source=" + std::to_string(completed) + " threads=" +
        std::to_string(config.threads));
    const auto started = Clock::now();
    while (completed < graph.n()) {
        const std::uint64_t last =
            std::min<std::uint64_t>(graph.n(), completed + config.block_sources);
        std::atomic<bool> failed{false};
        std::atomic<std::uint64_t> block_walks{0};
        std::string failure;
        std::mutex failure_mutex;
#pragma omp parallel
        {
            Workspace& workspace = *workspaces[static_cast<std::size_t>(omp_get_thread_num())];
#pragma omp for schedule(dynamic, 1)
            for (std::int64_t raw = static_cast<std::int64_t>(completed);
                 raw < static_cast<std::int64_t>(last); ++raw) {
                if (failed.load(std::memory_order_relaxed)) continue;
                try {
                    const auto walks = process_source(
                        graph, config, taus, static_cast<std::uint32_t>(raw), workspace,
                        output.low(), output.high());
                    block_walks.fetch_add(walks, std::memory_order_relaxed);
                } catch (const std::exception& error) {
                    failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> guard(failure_mutex);
                    if (failure.empty()) failure = error.what();
                }
            }
        }
        if (failed) throw std::runtime_error("TGT source block failed: " + failure);
        all_walks += block_walks.load();
        output.checkpoint(last);
        completed = last;
        log("TGT_PROGRESS sources=" + std::to_string(completed) + "/" +
            std::to_string(graph.n()) + " block_walks=" +
            std::to_string(block_walks.load()) + " total_walks=" +
            std::to_string(all_walks) + " elapsed_seconds=" +
            std::to_string(elapsed(started)));
    }
    log("TGT_DONE seconds=" + std::to_string(elapsed(started)) +
        " total_walks=" + std::to_string(all_walks));
}

std::uint64_t projection_hash(const std::uint64_t seed,
                              const std::uint32_t projection,
                              const std::uint32_t edge) {
    std::uint64_t value = seed ^
                          (0x9e3779b97f4a7c15ULL *
                           (static_cast<std::uint64_t>(projection) + 1ULL)) ^
                          (0xbf58476d1ce4e5b9ULL *
                           (static_cast<std::uint64_t>(edge) + 1ULL));
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

double parallel_dot(const std::vector<double>& left,
                    const std::vector<double>& right) {
    double value = 0.0;
#pragma omp parallel for reduction(+ : value) schedule(static)
    for (std::int64_t raw = 0; raw < static_cast<std::int64_t>(left.size()); ++raw) {
        const auto index = static_cast<std::size_t>(raw);
        value += left[index] * right[index];
    }
    return value;
}

struct PcgResult {
    std::uint32_t iterations = 0;
    double relative_residual = 1.0;
    bool deadline_reached = false;
};

PcgResult solve_normalized_laplacian(
    const Graph& graph, const Config& config, const Clock::time_point deadline,
    const std::vector<double>& rhs, std::vector<double>& solution,
    std::vector<double>& residual, std::vector<double>& direction,
    std::vector<double>& product) {
    std::fill(solution.begin(), solution.end(), 0.0);
    residual = rhs;
    direction = residual;
    const double initial_squared = parallel_dot(residual, residual);
    PcgResult result;
    if (!(initial_squared > 0.0) || !std::isfinite(initial_squared)) {
        result.relative_residual = 0.0;
        return result;
    }
    double squared = initial_squared;
    for (std::uint32_t iteration = 0; iteration < config.pcg_iterations; ++iteration) {
        if (Clock::now() >= deadline) {
            result.deadline_reached = true;
            break;
        }
        graph.normalized_laplacian(direction.data(), product.data());
        const double denominator = parallel_dot(direction, product);
        if (!(denominator > std::numeric_limits<double>::epsilon()) ||
            !std::isfinite(denominator)) {
            break;
        }
        const double alpha = squared / denominator;
#pragma omp parallel for schedule(static)
        for (std::int64_t raw = 0; raw < static_cast<std::int64_t>(graph.n()); ++raw) {
            const auto node = static_cast<std::size_t>(raw);
            solution[node] += alpha * direction[node];
            residual[node] -= alpha * product[node];
        }
        const double next_squared = parallel_dot(residual, residual);
        result.iterations = iteration + 1;
        result.relative_residual = std::sqrt(next_squared / initial_squared);
        if (!std::isfinite(result.relative_residual) ||
            result.relative_residual <= config.pcg_tolerance) {
            squared = next_squared;
            break;
        }
        const double beta = next_squared / squared;
#pragma omp parallel for schedule(static)
        for (std::int64_t raw = 0; raw < static_cast<std::int64_t>(graph.n()); ++raw) {
            const auto node = static_cast<std::size_t>(raw);
            direction[node] = residual[node] + beta * direction[node];
        }
        squared = next_squared;
    }
    return result;
}

void run_jl_pcg(const Graph& graph, const Config& config) {
    const auto started = Clock::now();
    const auto deadline = started +
                          std::chrono::duration_cast<Clock::duration>(
                              std::chrono::duration<double>(config.max_seconds));
    OrientedOutput output(graph, config);
    std::uint64_t completed = output.completed_sources();
    if (completed == graph.n()) {
        log("JLPCG_CACHE_HIT output=" + config.oriented_output);
        return;
    }
    if (completed > config.omega) {
        throw std::runtime_error("JL-PCG checkpoint has an invalid projection count");
    }

    std::vector<double> rhs(graph.n(), 0.0);
    std::vector<double> solution(graph.n(), 0.0);
    std::vector<double> residual(graph.n(), 0.0);
    std::vector<double> direction(graph.n(), 0.0);
    std::vector<double> product(graph.n(), 0.0);
    double maximum_residual = 0.0;
    std::uint64_t total_iterations = 0;
    bool deadline_reached = false;

    log("JLPCG_START resume_projection=" + std::to_string(completed) +
        " target_projections=" + std::to_string(config.omega) +
        " pcg_tolerance=" + std::to_string(config.pcg_tolerance) +
        " pcg_iterations=" + std::to_string(config.pcg_iterations) +
        " max_seconds=" + std::to_string(config.max_seconds));

    while (completed < config.omega) {
        if (completed > 0 && Clock::now() >= deadline) {
            deadline_reached = true;
            break;
        }
        const auto projection = static_cast<std::uint32_t>(completed);
#pragma omp parallel for schedule(static)
        for (std::int64_t raw = 0; raw < static_cast<std::int64_t>(graph.n()); ++raw) {
            const auto node = static_cast<std::uint32_t>(raw);
            if (!graph.primary_node(node)) {
                rhs[node] = 0.0;
                continue;
            }
            double value = 0.0;
            for (std::uint64_t position = graph.begin(node);
                 position < graph.end(node); ++position) {
                const auto edge = graph.edge_id(position);
                const double sign =
                    (projection_hash(config.seed, projection, edge) & 1ULL) ? 1.0 : -1.0;
                value += graph.edge_u(edge) == node ? sign : -sign;
            }
            rhs[node] = value / std::sqrt(static_cast<double>(graph.degree(node)));
        }

        const auto solve = solve_normalized_laplacian(
            graph, config, deadline, rhs, solution, residual, direction, product);
        maximum_residual = std::max(maximum_residual, solve.relative_residual);
        total_iterations += solve.iterations;
        deadline_reached = deadline_reached || solve.deadline_reached;

        float* current = (completed & 1ULL) == 0ULL ? output.low() : output.high();
        float* next = (completed & 1ULL) == 0ULL ? output.high() : output.low();
#pragma omp parallel for schedule(static)
        for (std::int64_t raw = 0; raw < static_cast<std::int64_t>(graph.m()); ++raw) {
            const auto edge = static_cast<std::uint32_t>(raw);
            if (!graph.primary_edge(edge)) continue;
            const auto u = graph.edge_u(edge);
            const auto v = graph.edge_v(edge);
            const double voltage =
                solution[u] / std::sqrt(static_cast<double>(graph.degree(u))) -
                solution[v] / std::sqrt(static_cast<double>(graph.degree(v)));
            next[edge] = current[edge] + static_cast<float>(voltage * voltage);
        }
        ++completed;
        output.checkpoint(completed);
        log("JLPCG_PROGRESS projections=" + std::to_string(completed) + "/" +
            std::to_string(config.omega) + " iterations=" +
            std::to_string(solve.iterations) + " relative_residual=" +
            std::to_string(solve.relative_residual) + " elapsed_seconds=" +
            std::to_string(elapsed(started)));
        if (deadline_reached) break;
    }

    if (completed == 0) {
        throw std::runtime_error("JL-PCG deadline expired before one projection completed");
    }
    if (completed < config.min_projections) {
        log("JLPCG_QUALITY_WARNING completed=" + std::to_string(completed) +
            " requested_minimum=" + std::to_string(config.min_projections));
    }

    const bool current_is_high = (completed & 1ULL) != 0ULL;
    const float* current = current_is_high ? output.high() : output.low();
    float* finalized = current_is_high ? output.low() : output.high();
    const std::uint32_t finalized_buffer = current_is_high ? 0U : 1U;
    double primary_sum = 0.0;
#pragma omp parallel for reduction(+ : primary_sum) schedule(static)
    for (std::int64_t raw = 0; raw < static_cast<std::int64_t>(graph.m()); ++raw) {
        const auto edge = static_cast<std::uint32_t>(raw);
        if (graph.primary_edge(edge)) {
            primary_sum += static_cast<double>(current[edge]) /
                           static_cast<double>(completed);
        }
    }
    if (!(primary_sum > 0.0) || !std::isfinite(primary_sum)) {
        throw std::runtime_error("JL-PCG produced a non-positive leverage sum");
    }
    const double kirchhoff_target = static_cast<double>(graph.primary_nodes() - 1U);
    const double calibration = kirchhoff_target / primary_sum;
    std::uint64_t clipped = 0;
#pragma omp parallel for reduction(+ : clipped) schedule(static)
    for (std::int64_t raw = 0; raw < static_cast<std::int64_t>(graph.m()); ++raw) {
        const auto edge = static_cast<std::uint32_t>(raw);
        double resistance = graph.primary_edge(edge)
                                ? static_cast<double>(current[edge]) /
                                      static_cast<double>(completed) * calibration
                                : static_cast<double>(graph.small_exact_resistance(edge));
        if (resistance > 1.0) {
            resistance = 1.0;
            ++clipped;
        }
        finalized[edge] = static_cast<float>(std::max(0.0, resistance));
    }
    constexpr std::int32_t kJlFinalEncoding = 1'000'000;
    output.set_gamma(-static_cast<std::int32_t>(
        kJlFinalEncoding + 2 * completed + finalized_buffer));
    output.checkpoint(graph.n());
    log("JLPCG_DONE projections=" + std::to_string(completed) +
        " total_iterations=" + std::to_string(total_iterations) +
        " maximum_relative_residual=" + std::to_string(maximum_residual) +
        " calibration=" + std::to_string(calibration) +
        " clipped_above_one=" + std::to_string(clipped) +
        " deadline_reached=" + std::string(deadline_reached ? "true" : "false") +
        " seconds=" + std::to_string(elapsed(started)));
}

std::uint64_t estimate_jl_pcg_peak_bytes(const Graph& graph) {
    const std::uint64_t n = graph.n();
    const std::uint64_t m = graph.m();
    const std::uint64_t csr = 8 * (n + 1) + 16 * m + 4 * m + n;
    const std::uint64_t vectors = 5 * 8 * n;
    const std::uint64_t output = 8 * m;
    return csr + vectors + output + (1ULL << 30);
}

std::uint64_t estimate_peak_bytes(const Graph& graph, const Config& config) {
    const std::uint64_t n = graph.n();
    const std::uint64_t m = graph.m();
    const std::uint64_t ncv = std::min<std::uint64_t>(
        n, std::max<std::uint64_t>(2 * config.omega + 1, 20));
    const std::uint64_t csr = 8 * (n + 1) + 16 * m + 8 * m;
    const std::uint64_t eigen_phase =
        8 * n * ncv + 8 * 4 * n + 8 * ncv * (ncv + 8) + csr;
    const std::uint64_t runtime_phase =
        csr + 4 * n * config.omega + 4 * m + 8 * m +
        static_cast<std::uint64_t>(config.threads) * 24 * n;
    return std::max(eigen_phase, runtime_phase) + (1ULL << 30);
}

void usage(std::ostream& output) {
    output
        << "Usage: tgt_effective_resistance --input GRAPH.tgtbin --eigen-cache FILE "
           "--oriented-output FILE [options]\n\n"
        << "Options:\n"
        << "  --algorithm NAME         tgt or jl-pcg (default: tgt)\n"
        << "  --threads N              OpenMP threads (default: 8)\n"
        << "  --epsilon X              absolute error parameter (default: 0.05)\n"
        << "  --delta X                failure probability (default: 0.01)\n"
        << "  --omega N                leading eigenpairs (default: 128)\n"
        << "  --gamma N                CalChi candidates (default: 10)\n"
        << "  --seed N                 deterministic random-walk seed\n"
        << "  --eigen-tolerance X      ARPACK tolerance (default: 1e-6)\n"
        << "  --max-iterations N       ARPACK iteration limit (default: 10000)\n"
        << "  --block-sources N        sources per durable checkpoint (default: 1024)\n"
        << "  --pcg-tolerance X        JL-PCG relative residual target (default: 1e-2)\n"
        << "  --pcg-iterations N       JL-PCG iterations per projection (default: 100)\n"
        << "  --min-projections N      quality warning threshold (default: 8)\n"
        << "  --max-seconds X          JL-PCG compute deadline (default: 19800)\n"
        << "  --memory-gb X            refuse if estimated peak exceeds X; 0 disables\n"
        << "  --estimate-only          validate graph and report memory without computing\n"
        << "  --force                  replace an oriented checkpoint\n";
}

template <typename T> T parse_number(const std::string& value, const char* option) {
    std::istringstream input(value);
    T parsed{};
    input >> parsed;
    if (!input || !input.eof()) {
        throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
    }
    return parsed;
}

Config parse_arguments(const int argc, char** argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("missing value after " + option);
            return argv[index];
        };
        if (option == "--algorithm")
            config.algorithm = value();
        else if (option == "--input")
            config.input = value();
        else if (option == "--eigen-cache")
            config.eigen_cache = value();
        else if (option == "--oriented-output")
            config.oriented_output = value();
        else if (option == "--threads")
            config.threads = parse_number<std::uint32_t>(value(), "--threads");
        else if (option == "--epsilon")
            config.epsilon = parse_number<double>(value(), "--epsilon");
        else if (option == "--delta")
            config.delta = parse_number<double>(value(), "--delta");
        else if (option == "--omega")
            config.omega = parse_number<std::uint32_t>(value(), "--omega");
        else if (option == "--gamma")
            config.gamma = parse_number<std::int32_t>(value(), "--gamma");
        else if (option == "--seed")
            config.seed = parse_number<std::uint64_t>(value(), "--seed");
        else if (option == "--eigen-tolerance")
            config.eigen_tolerance = parse_number<double>(value(), "--eigen-tolerance");
        else if (option == "--max-iterations")
            config.max_iterations =
                parse_number<std::uint32_t>(value(), "--max-iterations");
        else if (option == "--block-sources")
            config.block_sources =
                parse_number<std::uint32_t>(value(), "--block-sources");
        else if (option == "--pcg-tolerance")
            config.pcg_tolerance = parse_number<double>(value(), "--pcg-tolerance");
        else if (option == "--pcg-iterations")
            config.pcg_iterations =
                parse_number<std::uint32_t>(value(), "--pcg-iterations");
        else if (option == "--min-projections")
            config.min_projections =
                parse_number<std::uint32_t>(value(), "--min-projections");
        else if (option == "--max-seconds")
            config.max_seconds = parse_number<double>(value(), "--max-seconds");
        else if (option == "--memory-gb")
            config.memory_gb = parse_number<double>(value(), "--memory-gb");
        else if (option == "--force")
            config.force = true;
        else if (option == "--estimate-only")
            config.estimate_only = true;
        else if (option == "--help" || option == "-h") {
            usage(std::cout);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    if (config.input.empty() || config.eigen_cache.empty() ||
        config.oriented_output.empty()) {
        throw std::runtime_error(
            "--input, --eigen-cache, and --oriented-output are required");
    }
    if (config.algorithm != "tgt" && config.algorithm != "jl-pcg")
        throw std::runtime_error("--algorithm must be tgt or jl-pcg");
    if (config.threads == 0 || config.block_sources == 0 || config.max_iterations == 0 ||
        config.pcg_iterations == 0 || config.min_projections == 0 ||
        config.min_projections > config.omega)
        throw std::runtime_error("thread, block, and iteration counts must be positive");
    if (!(config.epsilon > 0.0) || !(config.delta > 0.0 && config.delta < 1.0) ||
        !(config.eigen_tolerance > 0.0) ||
        (config.algorithm == "tgt" && config.gamma < 0) ||
        (config.algorithm == "jl-pcg" && config.gamma >= 0) ||
        !(config.pcg_tolerance > 0.0 && config.pcg_tolerance < 1.0) ||
        !(config.max_seconds > 0.0) || config.memory_gb < 0.0)
        throw std::runtime_error("invalid TGT numerical parameter");
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_arguments(argc, argv);
        omp_set_dynamic(0);
        omp_set_num_threads(static_cast<int>(config.threads));
        log("PROCESS_START backend=" + config.algorithm +
            " pid=" + std::to_string(::getpid()));
        Graph graph(config.input);
        const auto estimated = config.algorithm == "jl-pcg"
                                   ? estimate_jl_pcg_peak_bytes(graph)
                                   : estimate_peak_bytes(graph, config);
        log("MEMORY_ESTIMATE peak=" + gib(estimated) + " requested_limit=" +
            (config.memory_gb == 0.0 ? std::string("disabled")
                                     : std::to_string(config.memory_gb) + " GiB"));
        if (config.memory_gb > 0.0 &&
            static_cast<long double>(estimated) >
                static_cast<long double>(config.memory_gb) * (1ULL << 30)) {
            throw std::runtime_error("estimated peak " + gib(estimated) +
                                     " exceeds --memory-gb; request more Slurm memory, "
                                     "reduce --threads/--omega, or override deliberately");
        }
        if (config.estimate_only) {
            log("ESTIMATE_ONLY_DONE");
            return 0;
        }
        if (config.algorithm == "jl-pcg") {
            run_jl_pcg(graph, config);
        } else {
            EigenCache eigen(graph, config.eigen_cache, config.omega,
                             config.eigen_tolerance, config.max_iterations);
            run_tgt(graph, eigen, config);
        }
        log("PROCESS_DONE backend=" + config.algorithm);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << '[' << timestamp() << "] FATAL " << error.what() << std::endl;
        return 2;
    }
}
