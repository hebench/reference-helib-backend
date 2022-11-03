// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>

#include "engine/helib_engine.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <mutex>
#include <sstream>

// include all benchmarks
#include "benchmarks/bgv/helib_bgv_dot_product_benchmark.h"
#include "benchmarks/bgv/helib_bgv_element_wise_benchmark.h"
#include "benchmarks/bgv/helib_bgv_matmultval_benchmark.h"

#include "benchmarks/ckks/helib_ckks_dot_product_benchmark.h"
#include "benchmarks/ckks/helib_ckks_element_wise_benchmark.h"
#include "benchmarks/ckks/helib_ckks_logreg_horner.h"
#include "benchmarks/ckks/helib_ckks_matmultval_benchmark.h"

#include "engine/helib_types.h"
#include "engine/helib_version.h"

//-----------------
// Engine creation
//-----------------

namespace hebench {
namespace cpp {

BaseEngine *createEngine()
{
    // It is a good idea to check here if the API Bridge version is correct for
    // our backend by checking against the constants defined in
    // `hebench/api_bridge/version.h.in` HEBENCH_API_VERSION_*

    if (HEBENCH_API_VERSION_MAJOR != HEBENCH_API_VERSION_NEEDED_MAJOR
        || HEBENCH_API_VERSION_MINOR != HEBENCH_API_VERSION_NEEDED_MINOR
        || HEBENCH_API_VERSION_REVISION < HEBENCH_API_VERSION_NEEDED_REVISION)
    {
        std::stringstream ss;
        ss << "Critical: Invalid HEBench API version detected. Required: "
           << HEBENCH_API_VERSION_NEEDED_MAJOR << "." << HEBENCH_API_VERSION_NEEDED_MINOR << "." << HEBENCH_API_VERSION_NEEDED_REVISION
           << ", but " << HEBENCH_API_VERSION_MAJOR << "." << HEBENCH_API_VERSION_MINOR << "." << HEBENCH_API_VERSION_REVISION
           << " received.";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG(ss.str()),
                                         HEBENCH_ECODE_CRITICAL_ERROR);
    }
    // end if
    return HELIBEngine::create();
}

void destroyEngine(BaseEngine *p)
{
    HELIBEngine *_p = dynamic_cast<HELIBEngine *>(p);
    HELIBEngine::destroy(_p);
}

} // namespace cpp
} // namespace hebench

//---------------------
// class ExampleEngine
//---------------------

HELIBEngine *HELIBEngine::create()
{
    HELIBEngine *p_retval = new HELIBEngine();
    p_retval->init();
    return p_retval;
}

void HELIBEngine::destroy(HELIBEngine *p)
{
    if (p)
        delete p;
}

HELIBEngine::HELIBEngine() {}

HELIBEngine::~HELIBEngine() {}

void HELIBEngine::init()
{
    // add any new error codes
    addErrorCode(HEBHELIB_ECODE_HELIB_ERROR, "HELIB error.");

    // add supported schemes
    addSchemeName(HEBENCH_HE_SCHEME_CKKS, "CKKS");
    addSchemeName(HEBENCH_HE_SCHEME_BGV, "BGV");

    // add supported security
    addSecurityName(HEBENCH_HE_SECURITY_128, "128 bits");

    // add the all benchmark descriptors

    // eltwiseadd

    addBenchmarkDescription(
        std::make_shared<hbe::bgv::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Latency,
                                                                    hebench::APIBridge::Workload::EltwiseAdd));

    addBenchmarkDescription(
        std::make_shared<hbe::ckks::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Latency,
                                                                     hebench::APIBridge::Workload::EltwiseAdd));
    addBenchmarkDescription(
        std::make_shared<hbe::bgv::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Offline,
                                                                    hebench::APIBridge::Workload::EltwiseAdd));

    addBenchmarkDescription(
        std::make_shared<hbe::ckks::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Offline,
                                                                     hebench::APIBridge::Workload::EltwiseAdd));

    // eltwisemult
    addBenchmarkDescription(
        std::make_shared<hbe::bgv::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Latency,
                                                                    hebench::APIBridge::Workload::EltwiseMultiply));

    addBenchmarkDescription(
        std::make_shared<hbe::ckks::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Latency,
                                                                     hebench::APIBridge::Workload::EltwiseMultiply));
    addBenchmarkDescription(
        std::make_shared<hbe::bgv::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Offline,
                                                                    hebench::APIBridge::Workload::EltwiseMultiply));

    addBenchmarkDescription(
        std::make_shared<hbe::ckks::ElementWiseBenchmarkDescription>(hebench::APIBridge::Category::Offline,
                                                                     hebench::APIBridge::Workload::EltwiseMultiply));

    // dot product
    addBenchmarkDescription(
        std::make_shared<hbe::bgv::DotProductBenchmarkDescription>(
            hebench::APIBridge::Category::Latency));

    addBenchmarkDescription(
        std::make_shared<hbe::ckks::DotProductBenchmarkDescription>(hebench::APIBridge::Category::Latency));
    addBenchmarkDescription(
        std::make_shared<hbe::bgv::DotProductBenchmarkDescription>(
            hebench::APIBridge::Category::Offline));

    addBenchmarkDescription(
        std::make_shared<hbe::ckks::DotProductBenchmarkDescription>(hebench::APIBridge::Category::Offline));

    // matrix mult.
    addBenchmarkDescription(
        std::make_shared<hbe::bgv::MatMultValBenchmarkDescription>());

    addBenchmarkDescription(
        std::make_shared<hbe::ckks::MatMultValBenchmarkDescription>());

    // LogReg
    addBenchmarkDescription(
        std::make_shared<hbe::ckks::LogRegBenchmarkDescription>(hebench::APIBridge::Category::Latency));
    addBenchmarkDescription(
        std::make_shared<hbe::ckks::LogRegBenchmarkDescription>(hebench::APIBridge::Category::Offline,
                                                                0));
}
