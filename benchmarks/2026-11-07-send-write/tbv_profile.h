// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
#ifndef TBV_BENCH_PROFILE_H
#define TBV_BENCH_PROFILE_H

#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

// Optional bounded recording. No file, allocation or clock read when disabled.
// GPU durations have their own clock and are never placed on the CPU timeline.
class TbvProfile {
public:
	struct Context {
		unsigned rank = 0, round = 0, iteration = 0, generation = 0;
		unsigned bytes = 0, depth = 0, qpn = 0;
		const char *op = "setup";
	};
	Context context;

	explicit TbvProfile(const std::string &path, size_t limit = 200000) : limit_(limit) {
		if (path.empty()) return;
#ifdef TBV_DISABLE_PROFILING
		throw std::runtime_error("profiling omitted from this build");
#else
		int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
		if (fd < 0) throw std::runtime_error("cannot create profile (path must be new)");
		file_ = fdopen(fd, "w");
		if (!file_) { close(fd); throw std::runtime_error("cannot open profile stream"); }
		std::fprintf(file_, "{\"kind\":\"metadata\",\"schema\":1,\"clock\":\"monotonic\","
			"\"pid\":%ld,\"profiled\":true,\"max_events\":%zu}\n", long(getpid()), limit_);
#endif
	}
	TbvProfile(const TbvProfile &) = delete;
	TbvProfile &operator=(const TbvProfile &) = delete;
	~TbvProfile() {
		if (file_) {
			flush_unchecked();
			std::fprintf(file_, "{\"kind\":\"footer\",\"complete\":%s,\"records\":%llu,"
				"\"dropped\":%llu}\n", complete_ ? "true" : "false",
				static_cast<unsigned long long>(records_),
				static_cast<unsigned long long>(dropped_));
			std::fclose(file_);
		}
	}
	bool enabled() const { return file_ != nullptr; }
	void configuration(unsigned rank, bool host_mr, unsigned depth, unsigned spin_us,
			   bool persistent_rq, bool late_receives, unsigned iterations, unsigned rounds) {
		if (!enabled()) return;
		std::fprintf(file_, "{\"kind\":\"configuration\",\"rank\":%u,\"host_mr\":%s,"
			"\"depth\":%u,\"spin_us\":%u,\"persistent_rq\":%s,\"late_receives\":%s,"
			"\"iterations\":%u,\"rounds\":%u}\n", rank, host_mr ? "true" : "false",
			depth, spin_us, persistent_rq ? "true" : "false", late_receives ? "true" : "false",
			iterations, rounds);
	}
	uint64_t now() const {
		if (!enabled()) return 0;
		timespec t{};
		if (clock_gettime(CLOCK_MONOTONIC, &t)) throw std::runtime_error("profile clock failed");
		return uint64_t(t.tv_sec) * 1000000000ull + uint64_t(t.tv_nsec);
	}
	void cpu(const char *phase, uint64_t start, uint64_t end, uint64_t wr_id = 0,
		 int status = 0, uint64_t aux = 0) {
		if (enabled() && start && end >= start)
			append({context, "cpu", phase, start, end, end - start, wr_id, aux, status});
	}
	void gpu(const char *phase, uint64_t ns, unsigned slot) {
		if (enabled()) append({context, "gpu", phase, 0, 0, ns, 0, slot, 0});
	}
	void counter(const char *phase, uint64_t value) {
		if (enabled()) append({context, "counter", phase, 0, 0, 0, 0, value, 0});
	}
	void flush() {
		if (!enabled()) return;
		flush_unchecked();
		if (std::fflush(file_) || std::ferror(file_))
			throw std::runtime_error("profile write failed");
	}
	void complete() { flush(); complete_ = true; }

private:
	struct Record {
		Context context;
		const char *kind, *phase;
		uint64_t start, end, duration, wr_id, aux;
		int status;
	};
	FILE *file_ = nullptr;
	size_t limit_;
	uint64_t records_ = 0, dropped_ = 0;
	bool complete_ = false;
	std::vector<Record> pending_;
	void append(Record r) {
		if (records_ + pending_.size() >= limit_) { dropped_++; return; }
		pending_.push_back(r);
	}
	void flush_unchecked() {
		for (const auto &r : pending_) {
			const auto &c = r.context;
			// All strings are internal phase/op constants, never CLI text.
			std::fprintf(file_, "{\"kind\":\"%s\",\"phase\":\"%s\",\"rank\":%u,"
				"\"round\":%u,\"iteration\":%u,\"generation\":%u,\"bytes\":%u,"
				"\"depth\":%u,\"qpn\":%u,\"op\":\"%s\",\"start_ns\":%llu,"
				"\"end_ns\":%llu,\"duration_ns\":%llu,\"wr_id\":%llu,"
				"\"status\":%d,\"aux\":%llu}\n", r.kind, r.phase,
				c.rank, c.round, c.iteration, c.generation, c.bytes, c.depth, c.qpn, c.op,
				static_cast<unsigned long long>(r.start),
				static_cast<unsigned long long>(r.end),
				static_cast<unsigned long long>(r.duration),
				static_cast<unsigned long long>(r.wr_id), r.status,
				static_cast<unsigned long long>(r.aux));
		}
		records_ += pending_.size();
		pending_.clear();
	}
};

#endif
