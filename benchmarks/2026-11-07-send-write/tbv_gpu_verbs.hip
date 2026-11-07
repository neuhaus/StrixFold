// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
// Standard verbs GPU DMA-BUF test; checks every byte and short-frame guards.
// TCP is closed after QP/MR bootstrap. Operations and barriers use verbs.
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <infiniband/verbs.h>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>
#include "tbv_profile.h"

static void check(bool ok, const char *what)
{
	if (!ok)
		throw std::runtime_error(std::string(what) + ": " + strerror(errno));
}

static void hip_check(hipError_t status, const char *what)
{
	if (status != hipSuccess)
		throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
}
#define HIP(call) hip_check((call), #call)

struct Fd {
	int fd = -1;
	Fd() = default;
	explicit Fd(int value) : fd(value) {}
	Fd(const Fd &) = delete;
	Fd &operator=(const Fd &) = delete;
	~Fd() { if (fd >= 0) close(fd); }
};

struct Pool {
	void *data = nullptr;
	Fd exported;
	~Pool() {
		if (data) {
			(void)hipDeviceSynchronize();
			if (exported.fd >= 0) {
				close(exported.fd);
				exported.fd = -1;
			}
			(void)hipFree(data);
		}
	}
	void allocate(size_t bytes, bool uncached) {
		if (uncached)
			HIP(hipExtMallocWithFlags(&data, bytes, hipDeviceMallocUncached));
		else
			HIP(hipMalloc(&data, bytes));
		hipDeviceptr_t base = 0;
		size_t size = 0;
		HIP(hipMemGetAddressRange(&base, &size, reinterpret_cast<hipDeviceptr_t>(data)));
		if (reinterpret_cast<void *>(base) != data || size != bytes)
			throw std::runtime_error("pool must be a dedicated, exact-size HIP allocation");
		HIP(hipMemset(data, 0xa5, bytes));
		HIP(hipDeviceSynchronize());
		HIP(hipMemGetHandleForAddressRange(&exported.fd,
			reinterpret_cast<hipDeviceptr_t>(data), bytes,
			hipMemRangeHandleTypeDmaBufFd, 0));
	}
};

struct Gpu {
	hipStream_t stream = nullptr;
	hipEvent_t fence = nullptr;
	unsigned long long *result = nullptr;
	~Gpu() {
		if (stream) (void)hipStreamSynchronize(stream);
		if (result) (void)hipFree(result);
		if (fence) (void)hipEventDestroy(fence);
		if (stream) (void)hipStreamDestroy(stream);
	}
	void init() {
		HIP(hipStreamCreate(&stream));
		// Timing enabled: force a real marker, including when the stream is idle.
		HIP(hipEventCreateWithFlags(&fence, hipEventReleaseToSystem));
		HIP(hipMalloc(&result, 2 * sizeof(*result)));
	}
	void release() {
		HIP(hipEventRecord(fence, stream));
		HIP(hipEventSynchronize(fence));
	}
};

struct GpuTiming {
#ifdef TBV_DISABLE_PROFILING
	explicit GpuTiming(bool) {}
	void begin(unsigned, const char *, unsigned, hipStream_t) {}
	void end(unsigned, hipStream_t) {}
	void collect(TbvProfile &) {}
#else
	struct Pair {
		hipEvent_t first = nullptr, last = nullptr;
		const char *phase = nullptr;
		unsigned slot = 0;
		~Pair() {
			if (first) (void)hipEventDestroy(first);
			if (last) (void)hipEventDestroy(last);
		}
	};
	std::array<Pair, 14> pairs{};
	bool active = false;
	explicit GpuTiming(bool enabled) : active(enabled) {
		if (!active) return;
		for (auto &p : pairs) {
			HIP(hipEventCreate(&p.first));
			HIP(hipEventCreate(&p.last));
		}
	}
	void begin(unsigned index, const char *phase, unsigned slot, hipStream_t stream) {
		if (!active) return;
		auto &p = pairs.at(index);
		p.phase = phase; p.slot = slot;
		HIP(hipEventRecord(p.first, stream));
	}
	void end(unsigned index, hipStream_t stream) {
		if (active) HIP(hipEventRecord(pairs.at(index).last, stream));
	}
	void collect(TbvProfile &profile) {
		if (!active) return;
		// Called after the existing verification stream synchronization.
		for (auto &p : pairs) {
			if (!p.phase) continue;
			float ms = 0;
			HIP(hipEventElapsedTime(&ms, p.first, p.last));
			profile.gpu(p.phase, uint64_t(ms * 1000000.0), p.slot);
			p.phase = nullptr;
		}
	}
#endif
};

static uint64_t process_cpu_us()
{
	rusage usage{};
	check(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage");
	return uint64_t(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1000000ull +
		uint64_t(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec);
}

__device__ static uint32_t pattern(uint32_t word, uint32_t generation, uint32_t rank)
{
	uint32_t x = word ^ (generation * 0x9e3779b9u) ^ (rank * 0x85ebca6bu);
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	return x ^ (x >> 16);
}

__device__ static uint8_t byte_pattern(uint32_t byte, uint32_t generation, uint32_t rank)
{
	return static_cast<uint8_t>(pattern(byte / 4, generation, rank) >> (8 * (byte % 4)));
}

__global__ static void produce(uint8_t *dst, uint32_t bytes, uint32_t generation,
			      uint32_t rank)
{
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
	     i += blockDim.x * gridDim.x)
		dst[i] = byte_pattern(i, generation, rank);
}

__global__ static void consume(const uint8_t *src, uint32_t bytes,
			      uint32_t generation, uint32_t rank,
			      unsigned long long *result, bool verify)
{
	unsigned long long bad = 0, sum = 0;
	for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes + 64;
	     i += blockDim.x * gridDim.x) {
		uint8_t value = src[i];
		bad += verify && value != (i < bytes ? byte_pattern(i, generation, rank) : 0xa5);
		sum += value;
	}
	atomicAdd(result, bad);
	atomicAdd(result + 1, sum);
}

static void socket_options(int fd)
{
	int yes = 1;
	timeval timeout{30, 0};
	check(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) == 0,
	      "TCP_NODELAY");
	check(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0,
	      "SO_RCVTIMEO");
	check(setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0,
	      "SO_SNDTIMEO");
}

static int connect_peer(const char *host, unsigned port)
{
	addrinfo hints{}, *addresses = nullptr;
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	std::string service = std::to_string(port);
	int ret = getaddrinfo(host, service.c_str(), &hints, &addresses);
	if (ret) throw std::runtime_error(gai_strerror(ret));
	int fd = -1;
	for (unsigned attempt = 0; attempt < 100 && fd < 0; attempt++) {
		for (auto *a = addresses; a; a = a->ai_next) {
			fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
			if (fd >= 0 && connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
			if (fd >= 0) close(fd);
			fd = -1;
		}
		if (fd < 0) usleep(100000);
	}
	freeaddrinfo(addresses);
	check(fd >= 0, "connect");
	return fd;
}

static int accept_peer(unsigned port)
{
	Fd listener(socket(AF_INET, SOCK_STREAM, 0));
	check(listener.fd >= 0, "socket");
	int yes = 1;
	check(setsockopt(listener.fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == 0,
	      "SO_REUSEADDR");
	sockaddr_in address{};
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl(INADDR_ANY);
	address.sin_port = htons(port);
	check(bind(listener.fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0,
	      "bind");
	check(listen(listener.fd, 1) == 0, "listen");
	printf("LISTEN port=%u\n", port);
	fflush(stdout);
	int fd = accept(listener.fd, nullptr, nullptr);
	check(fd >= 0, "accept");
	return fd;
}

static void write_all(int fd, const void *buf, size_t bytes)
{
	auto *p = static_cast<const char *>(buf);
	while (bytes) {
		ssize_t n = send(fd, p, bytes, MSG_NOSIGNAL);
		if (n < 0 && errno == EINTR) continue;
		check(n > 0, "control send");
		bytes -= n;
		p += n;
	}
}

static void read_all(int fd, void *buf, size_t bytes)
{
	auto *p = static_cast<char *>(buf);
	while (bytes) {
		ssize_t n = recv(fd, p, bytes, 0);
		if (n < 0 && errno == EINTR) continue;
		if (n == 0) throw std::runtime_error("control peer disconnected");
		check(n > 0, "control receive");
		bytes -= n;
		p += n;
	}
}

static constexpr size_t stride = 32ul << 20;
static constexpr size_t region_bytes = 3 * stride;
static constexpr uint64_t recv_bit = 1ull << 63;
static constexpr uint64_t barrier_id = 1ull << 62;
static constexpr uint64_t error_id = 1ull << 61;
static constexpr uint64_t persistent_id = 1ull << 60;
static constexpr uint32_t magic = 0x56425036;

struct Options {
	std::string profile;
	const char *host = nullptr;
	const char *device = nullptr;
	unsigned port = 7314;
	unsigned rank = 0;
	unsigned iterations = 8;
	unsigned rounds = 2;
	unsigned gid_index = 0;
	unsigned rnr_delay_ms = 0;
	unsigned depth = 3;
	unsigned spin_us = 0;
	unsigned hold_ms = 0;
	unsigned hold_rank = 0;
	uint32_t bytes = UINT32_MAX;
	int operation = -1;
	bool host_mr = false;
	bool remote_error = false;
	bool signal_all = false;
	bool late_receives = false;
	bool persistent_rq = false;
};

static void verbs_check(int ret, const char *what)
{
	if (ret)
		throw std::runtime_error(std::string(what) + ": " + strerror(ret > 0 ? ret : errno));
}

struct HostRegion {
	void *data = nullptr;
	~HostRegion() { free(data); }
	void allocate() {
		verbs_check(posix_memalign(&data, 4096, region_bytes), "posix_memalign");
		memset(data, 0xa5, region_bytes);
	}
};

struct Rdma {
	ibv_context *ctx = nullptr;
	ibv_pd *pd = nullptr;
	ibv_mr *tx = nullptr, *rx = nullptr;
	~Rdma() {
		if (rx) (void)ibv_dereg_mr(rx);
		if (tx) (void)ibv_dereg_mr(tx);
		if (pd) (void)ibv_dealloc_pd(pd);
		if (ctx) (void)ibv_close_device(ctx);
	}
	void open(const char *name) {
		int count = 0;
		ibv_device **devices = ibv_get_device_list(&count);
		check(devices != nullptr, "ibv_get_device_list");
		for (int i = 0; i < count; i++) {
			const char *candidate = ibv_get_device_name(devices[i]);
			if (name ? strcmp(name, candidate) != 0 : strncmp(candidate, "usb4_", 5) != 0)
				continue;
			ctx = ibv_open_device(devices[i]);
			if (ctx) break;
		}
		ibv_free_device_list(devices);
		check(ctx != nullptr, "open Donnerkeule verbs device");
		pd = ibv_alloc_pd(ctx);
		check(pd != nullptr, "ibv_alloc_pd");
	}
	void register_buffers(Pool &tx_pool, Pool &rx_pool, HostRegion &host_tx,
			      HostRegion &host_rx, bool use_host) {
		if (use_host) {
			host_tx.allocate();
			host_rx.allocate();
			tx = ibv_reg_mr(pd, host_tx.data, region_bytes, 0);
			rx = ibv_reg_mr(pd, host_rx.data, region_bytes,
				       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
		} else {
			// Exercise a nonzero exporter offset and IOVAs unrelated to GPU VAs.
			tx = ibv_reg_dmabuf_mr(pd, 4096, region_bytes, 0x100000000ull,
					      tx_pool.exported.fd, 0);
			rx = ibv_reg_dmabuf_mr(pd, 4096, region_bytes, 0x200000000ull,
					      rx_pool.exported.fd,
					      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
		}
		check(tx != nullptr && rx != nullptr, "register verbs MRs");
	}
};

static void probe_import(Rdma &rdma, Pool &pool)
{
	const auto expect_failure = [&](uint64_t offset, size_t length, uint64_t iova,
					int fd, int access) {
		errno = 0;
		ibv_mr *mr = ibv_reg_dmabuf_mr(rdma.pd, offset, length, iova, fd, access);
		if (mr) {
			(void)ibv_dereg_mr(mr);
			throw std::runtime_error("invalid DMA-BUF import unexpectedly succeeded");
		}
	};
	expect_failure(0, 4096, 0x300000000ull, -1, 0);
	expect_failure(1, 4096, 0x300000000ull, pool.exported.fd, 0);
	expect_failure(0, 0, 0x300000000ull, pool.exported.fd, 0);
	expect_failure(0, 4095, 0x300000000ull, pool.exported.fd, 0);
	expect_failure(region_bytes + (2ul << 20), 4096, 0x300000000ull, pool.exported.fd, 0);
	expect_failure(0, 4096, UINT64_MAX - 4095, pool.exported.fd, 0);
	expect_failure(0, 4096, 0x300000000ull, pool.exported.fd, IBV_ACCESS_REMOTE_READ);
	expect_failure(0, 4096, 0x300000000ull, pool.exported.fd, IBV_ACCESS_REMOTE_WRITE);
	ibv_mr *mr = ibv_reg_dmabuf_mr(rdma.pd, 4096, 4096, 0x300000000ull,
				      pool.exported.fd, 0);
	check(mr != nullptr, "valid subrange after failed imports");
	verbs_check(ibv_dereg_mr(mr), "deregister subrange");
	printf("IMPORT PASS cases=8 nonzero_fd_offset=4096\n");
}

struct Bootstrap {
	uint32_t magic_value = magic;
	uint32_t qpn = 0, psn = 0, lid = 0, rkey = 0, rank = 0;
	uint32_t iterations = 0, host_mr = 0, remote_error = 0, signal_all = 0;
	uint32_t test_bytes = UINT32_MAX;
	int32_t operation = -1;
	uint32_t depth = 3, late_receives = 0;
	uint32_t persistent_rq = 0, spin_us = 0;
	uint64_t address = 0, bytes = region_bytes;
	uint8_t gid[16]{};
};
static_assert(sizeof(Bootstrap) == 96);

struct Link {
	ibv_qp *qp = nullptr;
	ibv_cq *cq = nullptr;
	Bootstrap remote{};
	std::deque<ibv_wc> pending;
	bool persistent_rq = false;
	bool early_barrier = false;
	unsigned spin_us = 0;
	TbvProfile *profile = nullptr;
	uint64_t profile_empty = 0, profile_sleeps = 0, profile_spins = 0;
	~Link() {
		if (qp) (void)ibv_destroy_qp(qp);
		if (cq) (void)ibv_destroy_cq(cq);
	}
	void init(Rdma &rdma, const Options &o, unsigned round) {
		cq = ibv_create_cq(rdma.ctx, 64, nullptr, nullptr, 0);
		check(cq != nullptr, "ibv_create_cq");
		ibv_qp_init_attr init{};
		init.send_cq = init.recv_cq = cq;
		init.qp_type = IBV_QPT_RC;
		init.cap.max_send_wr = init.cap.max_recv_wr = 32;
		init.cap.max_send_sge = init.cap.max_recv_sge = 1;
		qp = ibv_create_qp(rdma.pd, &init);
		check(qp != nullptr, "ibv_create_qp");
		Bootstrap local{};
		ibv_port_attr port{};
		ibv_gid gid{};
		verbs_check(ibv_query_port(rdma.ctx, 1, &port), "ibv_query_port");
		verbs_check(ibv_query_gid(rdma.ctx, 1, o.gid_index, &gid), "ibv_query_gid");
		local.qpn = qp->qp_num;
		local.psn = 0x10 + o.rank;
		local.lid = port.lid;
		local.rkey = rdma.rx->rkey;
		local.address = o.host_mr ? reinterpret_cast<uintptr_t>(rdma.rx->addr) : 0x200000000ull;
		local.rank = o.rank;
		local.iterations = o.iterations;
		local.host_mr = o.host_mr;
		local.remote_error = o.remote_error;
		local.signal_all = o.signal_all;
		local.test_bytes = o.bytes;
		local.operation = o.operation;
		local.depth = o.depth;
		local.late_receives = o.late_receives;
		local.persistent_rq = o.persistent_rq;
		local.spin_us = o.spin_us;
		memcpy(local.gid, &gid, sizeof(gid));
		Fd socket(o.host ? connect_peer(o.host, o.port + round) : accept_peer(o.port + round));
		socket_options(socket.fd);
		write_all(socket.fd, &local, sizeof(local));
		read_all(socket.fd, &remote, sizeof(remote));
		if (remote.magic_value != magic || remote.rank != 1 - o.rank ||
		    remote.iterations != o.iterations || remote.host_mr != o.host_mr ||
		    remote.remote_error != o.remote_error || remote.signal_all != o.signal_all ||
		    remote.test_bytes != o.bytes || remote.operation != o.operation ||
		    remote.depth != o.depth || remote.late_receives != o.late_receives ||
		    remote.persistent_rq != o.persistent_rq || remote.spin_us != o.spin_us ||
		    remote.bytes != region_bytes || !remote.qpn)
			throw std::runtime_error("bootstrap peer/configuration mismatch");
		ibv_qp_attr a{};
		a.qp_state = IBV_QPS_INIT;
		a.port_num = 1;
		a.pkey_index = 0;
		a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
		verbs_check(ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
					IBV_QP_PORT | IBV_QP_ACCESS_FLAGS), "QP INIT");
		a = {};
		a.qp_state = IBV_QPS_RTR;
		a.path_mtu = IBV_MTU_4096;
		a.dest_qp_num = remote.qpn;
		a.rq_psn = remote.psn;
		a.max_dest_rd_atomic = 1;
		a.min_rnr_timer = 12;
		a.ah_attr.is_global = 1;
		memcpy(&a.ah_attr.grh.dgid, remote.gid, 16);
		a.ah_attr.grh.sgid_index = o.gid_index;
		a.ah_attr.grh.hop_limit = 1;
		a.ah_attr.dlid = remote.lid;
		a.ah_attr.port_num = 1;
		verbs_check(ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV |
					IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
					IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER), "QP RTR");
		a = {};
		a.qp_state = IBV_QPS_RTS;
		a.timeout = 18; // 1.07 s; metadata may wait for remote GPU release.
		a.retry_cnt = a.rnr_retry = 7;
		a.sq_psn = local.psn;
		a.max_rd_atomic = 1;
		verbs_check(ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT |
					IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
					IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC), "QP RTS");
		// SEND keeps real receive WQEs: the persistent windows are the
		// receive buffers themselves, reposted after their consumer.
		persistent_rq = o.persistent_rq && o.operation == 2;
		spin_us = o.spin_us;
		if (persistent_rq)
			for (unsigned i = 0; i < 4; i++)
				post_receive(persistent_id);
		uint32_t ready = magic, peer_ready = 0;
		write_all(socket.fd, &ready, sizeof(ready));
		read_all(socket.fd, &peer_ready, sizeof(peer_ready));
		if (peer_ready != ready) throw std::runtime_error("QP readiness mismatch");
		// The Fd destructor closes TCP before any verbs operation below.
		printf("BOOTSTRAP round=%u qpn=%u peer_qpn=%u tcp_bytes=%zu then_closed=1\n",
		       round, local.qpn, remote.qpn, 2 * (sizeof(Bootstrap) + sizeof(ready)));
	}
	void post_receive(uint64_t id, ibv_mr *mr = nullptr, uint64_t offset = 0,
			  uint32_t bytes = 0, bool host_mr = false) {
		ibv_sge sge{};
		ibv_recv_wr wr{}, *bad = nullptr;
		wr.wr_id = recv_bit | id;
		if (bytes) {
			sge.addr = (host_mr ? reinterpret_cast<uintptr_t>(mr->addr) : 0x200000000ull) + offset;
			sge.length = bytes;
			sge.lkey = mr->lkey;
			wr.sg_list = &sge;
			wr.num_sge = 1;
		}
		uint64_t started = profile ? profile->now() : 0;
		int ret = ibv_post_recv(qp, &wr, &bad);
		if (profile) profile->cpu("post_recv", started, profile->now(), wr.wr_id, ret, bytes);
		verbs_check(ret, "ibv_post_recv");
	}
	void recv(uint64_t id, ibv_mr *mr = nullptr, uint64_t offset = 0, uint32_t bytes = 0,
		  bool host_mr = false) {
		if (!persistent_rq) {
			post_receive(id, mr, offset, bytes, host_mr);
			return;
		}
		if (bytes) throw std::runtime_error("persistent RQ requires zero-SGE notifications");
	}
	void send(uint64_t id, ibv_wr_opcode opcode, ibv_mr *mr = nullptr,
		  uint64_t offset = 0, uint32_t bytes = 0, bool signaled = true,
		  bool host_mr = false, bool invalid_rkey = false) {
		ibv_sge sge{};
		ibv_send_wr wr{}, *bad = nullptr;
		wr.wr_id = id;
		wr.opcode = opcode;
		wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
		wr.imm_data = htonl(static_cast<uint32_t>(id));
		if (bytes) {
			sge.addr = (host_mr ? reinterpret_cast<uintptr_t>(mr->addr) : 0x100000000ull) + offset;
			sge.length = bytes;
			sge.lkey = mr->lkey;
			wr.sg_list = &sge;
			wr.num_sge = 1;
		}
		if (opcode == IBV_WR_RDMA_WRITE || opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
			wr.wr.rdma.remote_addr = remote.address + offset;
			wr.wr.rdma.rkey = invalid_rkey ? 0 : remote.rkey;
		}
		uint64_t started = profile ? profile->now() : 0;
		int ret = ibv_post_send(qp, &wr, &bad);
		if (profile) profile->cpu("post_send", started, profile->now(), id, ret, bytes);
		verbs_check(ret, "ibv_post_send");
	}
	ibv_wc poll_kernel(std::chrono::steady_clock::time_point deadline) {
		ibv_wc wc{};
		uint64_t started = profile ? profile->now() : 0;
		auto spin_deadline = std::chrono::steady_clock::now() +
			std::chrono::microseconds(spin_us);
		for (;;) {
			int n = ibv_poll_cq(cq, 1, &wc);
			if (n < 0) throw std::runtime_error("ibv_poll_cq failed");
			if (n) {
				if (persistent_rq && wc.wr_id == (recv_bit | persistent_id) &&
				    wc.status == IBV_WC_SUCCESS) {
					// Gufo keeps four generic notification WQEs ahead of
					// its WRITE_WITH_IMM traffic and replenishes on CQ.
					// Recover the checked message ID from the immediate.
					if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM &&
					    (wc.wc_flags & IBV_WC_WITH_IMM))
						wc.wr_id = recv_bit | ntohl(wc.imm_data);
					else if (wc.opcode == IBV_WC_RECV && !wc.byte_len &&
						 !(wc.wc_flags & IBV_WC_WITH_IMM))
						wc.wr_id = recv_bit | barrier_id;
					else
						throw std::runtime_error("persistent RQ notification mismatch");
					post_receive(persistent_id);
				}
				if (profile) {
					uint64_t seen = profile->now();
					profile->cpu("cq_poll", started, seen, wc.wr_id, wc.status, wc.opcode);
					profile->cpu("cq_seen", seen, seen, wc.wr_id, wc.status, wc.opcode);
				}
				return wc;
			}
			if (profile) profile_empty++;
			if (std::chrono::steady_clock::now() >= deadline)
				throw std::runtime_error("verbs CQ timed out");
			if (std::chrono::steady_clock::now() < spin_deadline) {
				if (profile) profile_spins++;
#if defined(__x86_64__)
				asm volatile("pause" ::: "memory");
#elif defined(__aarch64__)
				asm volatile("yield" ::: "memory");
#endif
			} else {
				if (profile) profile_sleeps++;
				usleep(50);
			}
		}
	}
	ibv_wc poll(std::chrono::steady_clock::time_point deadline) {
		for (;;) {
			ibv_wc wc;
			if (pending.empty()) wc = poll_kernel(deadline);
			else {
				wc = pending.front();
				pending.pop_front();
			}
			if (!persistent_rq || wc.wr_id != (recv_bit | barrier_id))
				return wc;
			// A persistent WQE may consume the peer's next barrier
			// while we still await this iteration's last TX CQ.
			if (early_barrier || wc.status != IBV_WC_SUCCESS ||
			    wc.opcode != IBV_WC_RECV || wc.byte_len ||
			    (wc.wc_flags & IBV_WC_WITH_IMM))
				throw std::runtime_error("early barrier order/duplicate mismatch");
			early_barrier = true;
		}
	}
	void barrier(bool receive_posted = false) {
		if (!receive_posted) recv(barrier_id);
		send(barrier_id, IBV_WR_SEND);
		auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
		bool sent = false, received = early_barrier;
		early_barrier = false;
		while (!sent || !received) {
			ibv_wc wc = poll_kernel(deadline);
			if (wc.wr_id != barrier_id && wc.wr_id != (recv_bit | barrier_id)) {
				// A peer may finish its barrier first and place its next
				// WRITE into our already posted window. Verify that CQ
				// in the case loop after our own barrier has completed.
				if (pending.size() >= 32) throw std::runtime_error("barrier CQ backlog overflow");
				pending.push_back(wc);
				continue;
			}
			if (wc.status != IBV_WC_SUCCESS)
				throw std::runtime_error("verbs barrier completion mismatch");
			if (!sent && wc.opcode == IBV_WC_SEND && wc.wr_id == barrier_id)
				sent = true;
			else if (!received && wc.opcode == IBV_WC_RECV &&
				 wc.wr_id == (recv_bit | barrier_id) && !wc.byte_len)
				received = true;
			else
				throw std::runtime_error("verbs barrier order/duplicate mismatch");
		}
	}
};

static void probe_bad_sge(Link &link, Rdma &rdma, Pool &pool)
{
	ibv_sge sge{0x100000001ull, 4, rdma.tx->lkey};
	ibv_send_wr wr{}, *bad = nullptr;
	wr.wr_id = error_id;
	wr.opcode = IBV_WR_SEND;
	wr.sg_list = &sge;
	wr.num_sge = 1;
	int ret = ibv_post_send(link.qp, &wr, &bad);
	if (!ret || bad != &wr)
		throw std::runtime_error("unaligned GPU SGE was not rejected synchronously");
	sge = {0x100000000ull, 1, rdma.tx->lkey};
	bad = nullptr;
	ret = ibv_post_send(link.qp, &wr, &bad);
	if (!ret || bad != &wr)
		throw std::runtime_error("sub-32-byte GPU length was not rejected synchronously");
	ibv_pd *other = ibv_alloc_pd(rdma.ctx);
	check(other != nullptr, "probe PD");
	ibv_mr *foreign = ibv_reg_dmabuf_mr(other, 4096, 4096, 0x300000000ull,
					   pool.exported.fd, 0);
	check(foreign != nullptr, "register foreign PD MR");
	sge = {0x300000000ull, 32, foreign->lkey};
	bad = nullptr;
	ret = ibv_post_send(link.qp, &wr, &bad);
	verbs_check(ibv_dereg_mr(foreign), "foreign MR cleanup");
	verbs_check(ibv_dealloc_pd(other), "probe PD cleanup");
	if (!ret || bad != &wr)
		throw std::runtime_error("foreign PD SGE was not rejected synchronously");
	printf("POST PASS unaligned_gpu_sge=1 sub_32_byte_length=1 foreign_PD=1\n");
}

static void remote_error(Link &link, Rdma &rdma, bool host_mr)
{
	link.barrier();
	link.recv(error_id);
	// Error completions are required even for this unsignaled send.
	link.send(error_id, IBV_WR_RDMA_WRITE_WITH_IMM, rdma.tx, 0, 32, false, host_mr, true);
	auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
	unsigned send = 0, recv = 0;
	while (!send || !recv) {
		ibv_wc wc = link.poll(deadline);
		if (wc.wr_id == error_id && wc.status == IBV_WC_REM_ACCESS_ERR) send++;
		else if (wc.wr_id == (recv_bit | error_id) && wc.status == IBV_WC_WR_FLUSH_ERR) recv++;
		else throw std::runtime_error("wrong rkey did not produce expected CQ error");
	}
	printf("REMOTE_ERROR PASS unsignaled_send=REM_ACCESS_ERR recv=WR_FLUSH_ERR\n");
}

static void run(Link &link, Rdma &rdma, Gpu &gpu, Pool &tx_pool, Pool &rx_pool,
		HostRegion &host_tx, HostRegion &host_rx, const Options &o, unsigned round,
		TbvProfile &profile, GpuTiming &timing)
{
	const std::array<uint32_t, 11> host_sizes{0, 1, 4, 4095, 4096, 4097, 65536,
					   (2u << 20) + 1, 1u << 20, 4u << 20,
					   (32u << 20) - 64};
	const std::array<uint32_t, 11> gpu_sizes{0, 32, 64, 4096 - 32, 4096, 4096 + 32,
					      65536, (2u << 20) + 32, 1u << 20, 4u << 20,
					      (32u << 20) - 64};
	const auto &matrix = o.host_mr ? host_sizes : gpu_sizes;
	std::vector<uint32_t> sizes(matrix.begin(), matrix.end());
	if (o.bytes != UINT32_MAX)
		sizes = {o.bytes};
	const std::array<ibv_wr_opcode, 4> operations{IBV_WR_SEND, IBV_WR_SEND_WITH_IMM,
						    IBV_WR_RDMA_WRITE_WITH_IMM, IBV_WR_RDMA_WRITE};
	const std::array<const char *, 4> names{"SEND", "SEND_IMM", "WRITE_IMM", "WRITE"};
	auto *tx_gpu = static_cast<uint8_t *>(tx_pool.data) + 4096;
	auto *rx_gpu = static_cast<uint8_t *>(rx_pool.data) + 4096;
	uint32_t generation = round * 100000 + 1;
	uint64_t verified = 0;
	for (size_t op = 0; op < operations.size(); op++) {
		if (o.operation >= 0 && op != static_cast<size_t>(o.operation))
			continue;
		bool write = operations[op] == IBV_WR_RDMA_WRITE;
		bool send_op = operations[op] == IBV_WR_SEND || operations[op] == IBV_WR_SEND_WITH_IMM;
		bool with_imm = operations[op] != IBV_WR_SEND;
		// Persistent SEND: each window's receive WQE is posted once its
		// consumer has verified it and the next prewarm has completed, so
		// the post itself hands the window back. No barrier per message.
		bool send_persistent = o.persistent_rq && send_op;
		const auto arm_windows = [&](uint32_t bytes, uint32_t first) {
			for (unsigned slot = 0; slot < o.depth; slot++) {
				HIP(hipMemsetAsync(rx_gpu + slot * stride, 0xa5, bytes + 64, gpu.stream));
				consume<<<128, 256, 0, gpu.stream>>>(rx_gpu + slot * stride, bytes,
					first + slot, 1 - o.rank, gpu.result, false);
			}
			HIP(hipGetLastError());
			HIP(hipStreamSynchronize(gpu.stream));
			if (o.host_mr)
				for (unsigned slot = 0; slot < o.depth; slot++)
					memset(static_cast<uint8_t *>(host_rx.data) + slot * stride, 0xa5, bytes + 64);
			for (unsigned slot = 0; slot < o.depth; slot++)
				link.recv(first + slot, rdma.rx, slot * stride,
					  (bytes + 4095u) / 4096u * 4096u, o.host_mr);
		};
		for (uint32_t bytes : sizes) {
			double microseconds = 0;
			std::array<double, 8> phases{};
			if (send_persistent) {
				// Both peers' windows are posted before either sends.
				link.recv(barrier_id);
				arm_windows(bytes, generation);
				link.barrier(true);
			}
			for (unsigned iteration = 0; iteration < o.iterations; iteration++) {
				profile.context = {o.rank, round, iteration, generation, bytes, o.depth,
					link.qp->qp_num, names[op]};
				link.profile_empty = link.profile_sleeps = link.profile_spins = 0;
				uint64_t cpu_started = profile.enabled() ? process_cpu_us() : 0;
				uint64_t profile_started = profile.now(), profile_phase = profile_started;
				const std::array<const char *, 8> phase_names{
					"rq_post", "gpu_submit", "tx_release", "ready", "wr_post",
					"cq_wait", "rx_release", "verify"};
				auto started = std::chrono::steady_clock::now();
				auto phase_started = started;
				const auto mark = [&](size_t phase) {
					auto now = std::chrono::steady_clock::now();
					phases[phase] += std::chrono::duration<double, std::micro>(
						now - phase_started).count();
					phase_started = now;
					if (profile.enabled()) {
						uint64_t end = profile.now();
						profile.cpu(phase_names[phase], profile_phase, end);
						profile_phase = end;
					}
				};
				unsigned count = o.depth;
				bool delayed = o.rank == 0 && generation == round * 100000 + 1 &&
					o.rnr_delay_ms;
				bool early = !send_op && !o.late_receives && !delayed;
				const auto post_receives = [&]() {
					for (unsigned slot = 0; slot < count; slot++)
						link.recv(generation + slot, rdma.rx, slot * stride,
							send_op ? (bytes + 4095u) / 4096u * 4096u : 0, o.host_mr);
				};
				// These zero-SGE notification WQEs do not arm payload DMA.
				// The barrier's SEND follows GPU release on both hosts;
				// payload WRs follow the completed barrier.
				if (early) {
					link.recv(barrier_id);
					post_receives();
				}
				mark(0);
				for (unsigned slot = 0; slot < count; slot++) {
					size_t offset = slot * stride;
					if (!send_persistent) {
						timing.begin(slot * 4, "rx_prewarm", slot, gpu.stream);
						HIP(hipMemsetAsync(rx_gpu + offset, 0xa5, bytes + 64, gpu.stream));
						HIP(hipMemsetAsync(gpu.result, 0, 16, gpu.stream));
						// Touch RX with a GPU read before external DMA.
						consume<<<128, 256, 0, gpu.stream>>>(rx_gpu + offset, bytes,
							generation + slot, 1 - o.rank, gpu.result, false);
						timing.end(slot * 4, gpu.stream);
					}
					timing.begin(slot * 4 + 1, "producer", slot, gpu.stream);
					produce<<<128, 256, 0, gpu.stream>>>(tx_gpu + offset, bytes,
						generation + slot, o.rank);
					timing.end(slot * 4 + 1, gpu.stream);
					HIP(hipGetLastError());
					if (o.host_mr) {
						timing.begin(slot * 4 + 2, "tx_copy", slot, gpu.stream);
						if (!send_persistent)
							memset(static_cast<uint8_t *>(host_rx.data) + offset, 0xa5, bytes + 64);
						HIP(hipMemcpyAsync(static_cast<uint8_t *>(host_tx.data) + offset,
							tx_gpu + offset, bytes, hipMemcpyDeviceToHost, gpu.stream));
						timing.end(slot * 4 + 2, gpu.stream);
					}
				}
				mark(1);
				gpu.release();
				mark(2);
				// RMA placement can precede notification/RQ readiness. A
				// verbs barrier reserves the reused target until its GPU
				// consumer has finished on both peers, in either mode.
				if (!send_op && o.late_receives) link.barrier();
				if (delayed)
					usleep(o.rnr_delay_ms * 1000u);
				if (!early && !send_persistent) {
					if (!send_op && !o.late_receives) link.recv(barrier_id);
					post_receives();
				}
				// Advertise target ownership only after notification WQEs
				// are posted, like Gufo's persistent receive queue.
				if (!send_op && !o.late_receives) link.barrier(true);
				mark(3);
				std::vector<std::pair<uint64_t, ibv_wc_opcode>> expected_send;
				for (unsigned slot = 0; slot < count; slot++) {
					bool signal = o.signal_all || (!write && slot + 1 == count);
					link.send(generation + slot, operations[op], rdma.tx, slot * stride,
						  bytes, signal, o.host_mr);
					if (signal) expected_send.emplace_back(generation + slot,
						send_op ? IBV_WC_SEND : IBV_WC_RDMA_WRITE);
					if (write) {
						link.send(generation + slot, IBV_WR_SEND_WITH_IMM, nullptr, 0, 0,
							  o.signal_all || slot + 1 == count);
						if (o.signal_all || slot + 1 == count)
							expected_send.emplace_back(generation + slot, IBV_WC_SEND);
					}
				}
				mark(4);
				auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
				size_t tx_seen = 0;
				unsigned rx_seen = 0;
				while (tx_seen < expected_send.size() || rx_seen < count) {
					ibv_wc wc = link.poll(deadline);
					if (wc.status != IBV_WC_SUCCESS)
						throw std::runtime_error(std::string("CQ error: ") +
							ibv_wc_status_str(wc.status) + " wr_id=" + std::to_string(wc.wr_id));
					if (wc.wr_id & recv_bit) {
						ibv_wc_opcode expected = !send_op && !write ?
							IBV_WC_RECV_RDMA_WITH_IMM : IBV_WC_RECV;
						if (wc.wr_id != (recv_bit | (generation + rx_seen)) ||
						    wc.opcode != expected || wc.byte_len != (write ? 0 : bytes) ||
						    (with_imm && (!(wc.wc_flags & IBV_WC_WITH_IMM) ||
								 ntohl(wc.imm_data) != generation + rx_seen)) ||
						    (!with_imm && (wc.wc_flags & IBV_WC_WITH_IMM)))
							throw std::runtime_error("RQ order/length/immediate mismatch");
						rx_seen++;
					} else {
						if (tx_seen >= expected_send.size() ||
						    wc.wr_id != expected_send[tx_seen].first ||
						    wc.opcode != expected_send[tx_seen].second)
							throw std::runtime_error("SQ completion order/opcode mismatch");
						tx_seen++;
					}
				}
				mark(5);
				if (o.host_mr) {
					for (unsigned slot = 0; slot < count; slot++) {
						timing.begin(slot * 4 + 3, "rx_copy", slot, gpu.stream);
						HIP(hipMemcpyAsync(rx_gpu + slot * stride,
							static_cast<uint8_t *>(host_rx.data) + slot * stride,
							bytes + 64, hipMemcpyHostToDevice, gpu.stream));
						timing.end(slot * 4 + 3, gpu.stream);
					}
				}
				gpu.release(); // Tested acquire prototype; RX allocation is uncached.
				mark(6);
				timing.begin(12, "consumer_verify", 0, gpu.stream);
				HIP(hipMemsetAsync(gpu.result, 0, 16, gpu.stream));
				for (unsigned slot = 0; slot < count; slot++)
					consume<<<128, 256, 0, gpu.stream>>>(rx_gpu + slot * stride, bytes,
						generation + slot, 1 - o.rank, gpu.result, true);
				HIP(hipGetLastError());
				timing.end(12, gpu.stream);
				std::array<unsigned long long, 2> result{};
				timing.begin(13, "result_copy", 0, gpu.stream);
				HIP(hipMemcpyAsync(result.data(), gpu.result, sizeof(result),
						   hipMemcpyDeviceToHost, gpu.stream));
				timing.end(13, gpu.stream);
				HIP(hipStreamSynchronize(gpu.stream));
				if (result[0])
					throw std::runtime_error("GPU payload/guard mismatch: " + std::to_string(result[0]));
				mark(7);
				if (send_persistent && iteration + 1 < o.iterations) {
					arm_windows(bytes, generation + count);
					mark(0);
				}
				profile.cpu("cycle", profile_started, profile_phase);
				if (profile.enabled()) {
					profile.counter("process_cpu_us", process_cpu_us() - cpu_started);
					profile.counter("empty_polls", link.profile_empty);
					profile.counter("poll_sleeps", link.profile_sleeps);
					profile.counter("poll_spins", link.profile_spins);
				}
				timing.collect(profile);
				microseconds += std::chrono::duration<double, std::micro>(
					phase_started - started).count();
				verified += static_cast<uint64_t>(count) * bytes;
				generation += count;
			}
			printf("RESULT round=%u rank=%u op=%s bytes=%u iterations=%u depth=%u "
			       "persistent_rq=%u late_receives=%u spin_us=%u profiled=%u us_per_message=%.3f\n",
			       round, o.rank, names[op], bytes, o.iterations, o.depth,
			       o.persistent_rq, o.late_receives, o.spin_us, profile.enabled(),
			       microseconds / (o.depth * o.iterations));
			printf("PHASE round=%u rank=%u op=%s bytes=%u rq_post_us=%.3f gpu_submit_us=%.3f "
			       "tx_release_us=%.3f ready_us=%.3f wr_post_us=%.3f cq_wait_us=%.3f "
			       "rx_release_us=%.3f verify_us=%.3f\n", round, o.rank, names[op], bytes,
			       phases[0] / (o.depth * o.iterations), phases[1] / (o.depth * o.iterations),
			       phases[2] / (o.depth * o.iterations), phases[3] / (o.depth * o.iterations),
			       phases[4] / (o.depth * o.iterations), phases[5] / (o.depth * o.iterations),
			       phases[6] / (o.depth * o.iterations), phases[7] / (o.depth * o.iterations));
			fflush(stdout);
			profile.flush(); // Outside the timed cases; no per-event output on the hot path.
		}
	}
	link.barrier();
	printf("VERIFY PASS round=%u rank=%u payload_bytes=%llu guard_bytes_per_message=64 tcp_data_calls=0\n",
	       round, o.rank, static_cast<unsigned long long>(verified));
}

int main(int argc, char **argv)
{
	try {
		Options o;
		for (int i = 1; i < argc; i++) {
			std::string arg = argv[i];
			const auto value = [&]() -> const char * {
				if (++i >= argc) throw std::runtime_error("missing option value");
				return argv[i];
			};
			if (arg == "--host") { o.host = value(); o.rank = 1; }
			else if (arg == "--device") o.device = value();
			else if (arg == "--port") o.port = std::stoul(value());
			else if (arg == "--iterations") o.iterations = std::stoul(value());
			else if (arg == "--rounds") o.rounds = std::stoul(value());
			else if (arg == "--gid-index") o.gid_index = std::stoul(value());
			else if (arg == "--rnr-delay-ms") o.rnr_delay_ms = std::stoul(value());
			else if (arg == "--depth") o.depth = std::stoul(value());
			else if (arg == "--spin-us") o.spin_us = std::stoul(value());
			else if (arg == "--hold-ms") o.hold_ms = std::stoul(value());
			else if (arg == "--hold-rank") o.hold_rank = std::stoul(value());
			else if (arg == "--profile") o.profile = value();
			else if (arg == "--bytes") {
				const auto bytes = std::stoull(value());
				if (bytes > stride - 64)
					throw std::runtime_error("test bytes exceed the guarded window");
				o.bytes = static_cast<uint32_t>(bytes);
			} else if (arg == "--op") {
				const std::string name = value();
				const std::array<const char *, 4> names{"SEND", "SEND_IMM", "WRITE_IMM", "WRITE"};
				auto found = std::find(names.begin(), names.end(), name);
				if (found == names.end()) throw std::runtime_error("unknown test operation");
				o.operation = static_cast<int>(found - names.begin());
			}
			else if (arg == "--host-mr") o.host_mr = true;
			else if (arg == "--remote-error") o.remote_error = true;
			else if (arg == "--signal-all") o.signal_all = true;
			else if (arg == "--late-receives") o.late_receives = true;
			else if (arg == "--persistent-rq") o.persistent_rq = true;
			else throw std::runtime_error("usage: tbv_gpu_verbs [--host IP] [--device DEV] "
				"[--port N] [--iterations N] [--rounds N] [--gid-index N] "
				"[--rnr-delay-ms N] [--depth 1|2|3] [--spin-us N] [--bytes N] "
				"[--hold-ms N --hold-rank 0|1] "
				"[--op SEND|SEND_IMM|WRITE_IMM|WRITE] [--profile NEW_FILE.jsonl] "
				"[--host-mr] [--remote-error] [--signal-all] [--late-receives] [--persistent-rq]");
		}
		if (!o.iterations || o.iterations > 10000 || !o.rounds || o.rounds > 32 ||
		    o.port < 1024 || o.port + o.rounds > 65535 || o.gid_index > 255 ||
		    o.rnr_delay_ms > 10000 || o.spin_us > 10000 || o.hold_ms > 600000 ||
		    o.hold_rank > 1 || (o.hold_ms && (o.bytes == UINT32_MAX || o.host_mr)) || !o.depth || o.depth > 3 ||
		    (!o.host_mr && o.bytes != UINT32_MAX && o.bytes % 32))
			throw std::runtime_error("option outside test bounds");
		if (o.persistent_rq && (o.operation < 0 || o.operation > 2 || o.remote_error ||
					o.rnr_delay_ms || o.late_receives))
			throw std::runtime_error("persistent RQ profiling requires SEND, SEND_IMM or WRITE_IMM without error/delay probes");
		setvbuf(stdout, nullptr, _IOLBF, 0);
		HIP(hipSetDevice(0));
		TbvProfile profile(o.profile);
		profile.configuration(o.rank, o.host_mr, o.depth, o.spin_us,
			o.persistent_rq, o.late_receives, o.iterations, o.rounds);
		Gpu gpu;
		gpu.init();
		GpuTiming timing(profile.enabled());
		Pool tx_pool, rx_pool;
		tx_pool.allocate(region_bytes + (2ul << 20), false);
		rx_pool.allocate(region_bytes + (2ul << 20), true);
		for (unsigned round = 0; round < o.rounds; round++) {
			HostRegion host_tx, host_rx;
			Rdma rdma;
			rdma.open(o.device);
			if (!o.host_mr) probe_import(rdma, tx_pool);
			rdma.register_buffers(tx_pool, rx_pool, host_tx, host_rx, o.host_mr);
			Link link;
			link.init(rdma, o, round);
			if (!o.host_mr) probe_bad_sge(link, rdma, tx_pool);
			link.profile = profile.enabled() ? &profile : nullptr;
			run(link, rdma, gpu, tx_pool, rx_pool, host_tx, host_rx, o, round, profile, timing);
			if (o.hold_ms) {
				// Loss qualification. One more SEND of the tested size makes
				// the holder re-arm a credit; its next receive WQE stays
				// posted (armed) while it waits, before QP and MR go away.
				const uint32_t aligned = (o.bytes + 4095u) / 4096u * 4096u;
				const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
				if (o.rank == o.hold_rank) {
					link.post_receive(1ull << 59, rdma.rx, 0, aligned);
					ibv_wc wc = link.poll(deadline);
					if (wc.status != IBV_WC_SUCCESS || !(wc.wr_id & recv_bit))
						throw std::runtime_error("hold: first SEND failed");
					link.post_receive(1ull << 58, rdma.rx, 0, aligned);
					printf("HOLD rank=%u armed_receive_bytes=%u ms=%u\n", o.rank, o.bytes, o.hold_ms);
					fflush(stdout);
					usleep(o.hold_ms * 1000u);
				} else {
					usleep(1000000);
					link.send(1ull << 59, IBV_WR_SEND, rdma.tx, 0, o.bytes);
					ibv_wc wc = link.poll(deadline);
					if (wc.status != IBV_WC_SUCCESS)
						throw std::runtime_error("hold: SEND failed");
					printf("HOLD peer sent bytes=%u\n", o.bytes);
				}
			}
			if (o.remote_error) remote_error(link, rdma, o.host_mr);
		}
		printf("PASS rounds=%u MR_and_QP_recreation=1\n", o.rounds);
		profile.complete();
		return 0;
	} catch (const std::exception &e) {
		fprintf(stderr, "FAIL: %s\n", e.what());
		return 1;
	}
}
