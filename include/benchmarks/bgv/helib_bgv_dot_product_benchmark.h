
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hebench/api_bridge/cpp/hebench.hpp"
#include "helib/helib.h"

#include "engine/helib_context.h"

namespace hbe {
namespace bgv {

class DotProductBenchmarkDescription
    : public hebench::cpp::BenchmarkDescription
{
public:
    HEBERROR_DECLARE_CLASS_NAME(bgv::DotProductBenchmarkDescription)
    static constexpr const char *AlgorithmName        = "Vector";
    static constexpr const char *AlgorithmDescription = "One vector per ciphertext";
    static constexpr std::size_t NumOpParams          = 2;

    // all params generated using HEToolkit by setting DefaultPtxtPrimeModulus to be a Fermat Prime in BGV
    static constexpr std::size_t DefaultPolyModulusDegree = 8192;
    static constexpr std::size_t DefaultCoeffModulusBits  = 174;
    static constexpr std::size_t DefaultKeySwitchColumns  = 3;
    static constexpr std::size_t DefaultPtxtPrimeModulus  = 114689; //-1 for CKKS
    static constexpr std::size_t DefaultHenselLifting     = 1;

    // other workload parameters
    static constexpr std::size_t DefaultNumThreads = 0; // 0 - use all available threads

    enum : std::uint64_t
    {
        Index_WParamsStart = 0,
        Index_n            = Index_WParamsStart,
        Index_ExtraWParamsStart,
        Index_PolyModulusDegree = Index_ExtraWParamsStart,
        Index_CoefficientModulusBits,
        Index_KeySwitchColumns,
        Index_PtxtPrimeModulus,
        Index_HenselLifting,
        Index_NumThreads,
        NumWorkloadParams // This workload requires 1 parameters, and we add 3
        // encryption params
    };

public:
    DotProductBenchmarkDescription(hebench::APIBridge::Category category);
    ~DotProductBenchmarkDescription() override;

    hebench::cpp::BaseBenchmark *
    createBenchmark(hebench::cpp::BaseEngine &engine,
                    const hebench::APIBridge::WorkloadParams *p_params) override;
    void destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench) override;
    std::string getBenchmarkDescription(
        const hebench::APIBridge::WorkloadParams *p_w_params) const override;
};

class DotProductBenchmark : public hebench::cpp::BaseBenchmark
{
public:
    HEBERROR_DECLARE_CLASS_NAME(bgv::DotProductBenchmark)

public:
    static constexpr std::int64_t tag = 0x1;

    DotProductBenchmark(hebench::cpp::BaseEngine &engine,
                        const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                        const hebench::APIBridge::WorkloadParams &bench_params);
    ~DotProductBenchmark() override;

    hebench::APIBridge::Handle encode(const hebench::APIBridge::DataPackCollection *p_parameters) override;
    void decode(hebench::APIBridge::Handle encoded_data,
                hebench::APIBridge::DataPackCollection *p_native) override;
    hebench::APIBridge::Handle encrypt(hebench::APIBridge::Handle encoded_data) override;
    hebench::APIBridge::Handle decrypt(hebench::APIBridge::Handle encrypted_data) override;

    hebench::APIBridge::Handle load(const hebench::APIBridge::Handle *p_local_data,
                                    std::uint64_t count) override;
    void store(hebench::APIBridge::Handle remote_data,
               hebench::APIBridge::Handle *p_local_data,
               std::uint64_t count) override;

    hebench::APIBridge::Handle operate(
        hebench::APIBridge::Handle h_remote_packed,
        const hebench::APIBridge::ParameterIndexer *p_param_indexers) override;

    std::int64_t classTag() const override
    {
        return BaseBenchmark::classTag() | DotProductBenchmark::tag;
    }

private:
    HELIBContextWrapper::Ptr m_p_ctx_wrapper;
    hebench::cpp::WorkloadParams::VectorSize m_w_params;
    int m_num_threads;
    int m_plaintext_prime_modulus;
};
} // namespace bgv
} // namespace hbe
