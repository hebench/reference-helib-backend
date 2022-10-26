
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "engine/helib_context.h"
#include "hebench/api_bridge/cpp/hebench.hpp"
#include "helib/helib.h"

class HELIBEngine;

namespace hbe {
namespace ckks {

class LogRegBenchmarkDescription : public hebench::cpp::BenchmarkDescription
{
public:
    HEBERROR_DECLARE_CLASS_NAME(LogRegBenchmarkDescription)

public:
    static constexpr std::uint64_t OpParamsCount =
        3; // number of operation parameters (W, b, X)
    static constexpr std::uint64_t DefaultBatchSize = 100;
    static constexpr std::int64_t LogRegOtherID     = 0x01;

    enum : std::uint64_t
    {
        Index_OpParamsStart = 0,
        Index_W             = Index_OpParamsStart,
        Index_b,
        Index_X,
        NumOpParams
    };

    static constexpr const char *AlgorithmName        = "EvalPoly";
    static constexpr const char *AlgorithmDescription = "using Horner method for polynomial evaluation";

    // HE specific parameters
    static constexpr std::size_t DefaultPolyModulusDegree = 32768;
    static constexpr std::size_t DefaultCoeffModulusBits  = 613;
    static constexpr std::size_t DefaultKeySwitchColumns  = 3;
    static constexpr std::size_t DefaultPrecision         = 50;

    // other workload parameters
    static constexpr std::size_t DefaultNumThreads = 1; // 0 - use all available threads

    enum : std::uint64_t
    {
        Index_WParamsStart = 0,
        Index_n            = Index_WParamsStart,
        Index_ExtraWParamsStart,
        Index_PolyModulusDegree = Index_ExtraWParamsStart,
        Index_CoefficientModulusBits,
        Index_KeySwitchColumns,
        Index_Precision,
        Index_NumThreads,
        NumWorkloadParams // This workload requires 1 parameters, and we add 3
        // encryption params
    };

    LogRegBenchmarkDescription(hebench::APIBridge::Category category,
                               std::size_t batch_size = 0);
    ~LogRegBenchmarkDescription() override;

    std::string getBenchmarkDescription(
        const hebench::APIBridge::WorkloadParams *p_w_params) const override;

    hebench::cpp::BaseBenchmark *
    createBenchmark(hebench::cpp::BaseEngine &engine,
                    const hebench::APIBridge::WorkloadParams *p_params) override;
    void destroyBenchmark(hebench::cpp::BaseBenchmark *p_bench) override;
};

class LogRegBenchmark : public hebench::cpp::BaseBenchmark
{
public:
    HEBERROR_DECLARE_CLASS_NAME(LogRegBenchmark)

public:
    static constexpr std::int64_t tag = 0x1;

    LogRegBenchmark(HELIBEngine &engine,
                    const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
                    const hebench::APIBridge::WorkloadParams &bench_params);
    ~LogRegBenchmark() override;

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
        return BaseBenchmark::classTag() | LogRegBenchmark::tag;
    }

private:
    typedef std::tuple<helib::PtxtArray, helib::PtxtArray,
                       std::vector<helib::PtxtArray>>
        EncodedOpParams;
    typedef std::tuple<helib::Ctxt, helib::Ctxt, std::vector<helib::Ctxt>>
        EncryptedOpParams; // check this-> currently SEAL's

    static constexpr std::int64_t EncodedOpParamsTag   = 0x10;
    static constexpr std::int64_t EncryptedOpParamsTag = 0x20;
    static constexpr std::int64_t EncryptedResultTag   = 0x40;
    static constexpr std::int64_t EncodedResultTag     = 0x80;

    // coefficients for sigmoid polynomial approx
    static constexpr const double SigmoidPolyCoeff[] = { 0.5, 0.150, 0.0,
                                                         -0.0015930078125 };

    helib::PtxtArray encodeW(const hebench::APIBridge::DataPack &data_pack);
    helib::PtxtArray encodeBias(const hebench::APIBridge::DataPack &data_pack);
    std::vector<helib::PtxtArray> encodeInputs(const hebench::APIBridge::DataPack &data_pack);

    HELIBContextWrapper::Ptr m_p_ctx_wrapper;
    hebench::cpp::WorkloadParams::LogisticRegression m_w_params;
    std::vector<helib::PtxtArray> m_plain_coeff; // encoded coefficients for sigmoid polynomial approx
    int m_num_threads;
};

} // namespace ckks
} // namespace hbe
