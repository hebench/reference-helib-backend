
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <vector>

#include "benchmarks/ckks/helib_ckks_logreg_horner.h"
#include "engine/helib_engine.h"
#include "engine/helib_types.h"
#include <omp.h>

namespace hbe {
namespace ckks {

//-----------------------------------
// class LogRegBenchmarkDescription
//-----------------------------------

LogRegBenchmarkDescription::LogRegBenchmarkDescription(
    hebench::APIBridge::Category category, std::size_t batch_size)
{
    // initialize the descriptor for this benchmark
    std::memset(&m_descriptor, 0,
                sizeof(hebench::APIBridge::BenchmarkDescriptor));
    m_descriptor.data_type = hebench::APIBridge::DataType::Float64;
    m_descriptor.category  = category;
    switch (category)
    {
    case hebench::APIBridge::Category::Latency:
        m_descriptor.cat_params.min_test_time_ms                = 0; // read from user
        m_descriptor.cat_params.latency.warmup_iterations_count = 1;
        break;

    case hebench::APIBridge::Category::Offline:
        if (batch_size > DefaultPolyModulusDegree / 2)
            throw hebench::cpp::HEBenchError(
                HEBERROR_MSG_CLASS("Batch size must be under " + std::to_string(DefaultPolyModulusDegree / 2) + "."),
                HEBENCH_ECODE_INVALID_ARGS);
        m_descriptor.cat_params.offline.data_count[0] = 0;
        m_descriptor.cat_params.offline.data_count[1] = 0;
        m_descriptor.cat_params.offline.data_count[2] = batch_size;
        break;

    default:
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid category received."),
            HEBENCH_ECODE_INVALID_ARGS);
    }

    m_descriptor.cipher_param_mask = HEBENCH_HE_PARAM_FLAGS_ALL_CIPHER;
    //
    m_descriptor.scheme   = HEBENCH_HE_SCHEME_CKKS;
    m_descriptor.security = HEBENCH_HE_SECURITY_128;
    m_descriptor.other    = LogRegOtherID;
    m_descriptor.workload = hebench::APIBridge::Workload::LogisticRegression_PolyD3;

    hebench::cpp::WorkloadParams::LogisticRegression default_workload_params;
    default_workload_params.n() = 16;

    default_workload_params.add<std::uint64_t>(
        LogRegBenchmarkDescription::DefaultPolyModulusDegree,
        "PolyModulusDegree");
    default_workload_params.add<std::uint64_t>(
        LogRegBenchmarkDescription::DefaultCoeffModulusBits,
        "CoefficientModulusBits");
    default_workload_params.add<std::uint64_t>(
        LogRegBenchmarkDescription::DefaultKeySwitchColumns, "KeySwitchColumns");
    default_workload_params.add<std::uint64_t>(
        LogRegBenchmarkDescription::DefaultPrecision, "Precision");

    default_workload_params.add<std::uint64_t>(
        LogRegBenchmarkDescription::DefaultNumThreads, "NumThreads");

    this->addDefaultParameters(default_workload_params);
}

LogRegBenchmarkDescription::~LogRegBenchmarkDescription()
{
    //
}

hebench::cpp::BaseBenchmark *LogRegBenchmarkDescription::createBenchmark(
    hebench::cpp::BaseEngine &engine,
    const hebench::APIBridge::WorkloadParams *p_params)
{
    if (!p_params)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid empty workload parameters. This workload "
                               "requires flexible parameters."),
            HEBENCH_ECODE_CRITICAL_ERROR);

    HELIBEngine &ex_engine = dynamic_cast<HELIBEngine &>(engine);
    return new LogRegBenchmark(ex_engine, m_descriptor, *p_params);
}

void LogRegBenchmarkDescription::destroyBenchmark(
    hebench::cpp::BaseBenchmark *p_bench)
{
    LogRegBenchmark *p = dynamic_cast<LogRegBenchmark *>(p_bench);
    if (p)
        delete p;
}

std::string LogRegBenchmarkDescription::getBenchmarkDescription(
    const hebench::APIBridge::WorkloadParams *p_w_params) const
{
    std::stringstream ss;
    std::string s_tmp = BenchmarkDescription::getBenchmarkDescription(p_w_params);

    if (!p_w_params)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid null workload parameters `p_w_params`"),
            HEBENCH_ECODE_INVALID_ARGS);

    assert(p_w_params->count >= LogRegBenchmarkDescription::NumWorkloadParams);

    std::uint64_t poly_modulus_degree =
        p_w_params->params[LogRegBenchmarkDescription::Index_PolyModulusDegree]
            .u_param;
    std::uint64_t coeff_modulus_bits =
        p_w_params
            ->params[LogRegBenchmarkDescription::Index_CoefficientModulusBits]
            .u_param;
    std::uint64_t key_switch_columns =
        p_w_params->params[LogRegBenchmarkDescription::Index_KeySwitchColumns]
            .u_param;
    std::uint64_t precision =
        p_w_params->params[LogRegBenchmarkDescription::Index_Precision].u_param;

    std::uint64_t num_threads = p_w_params->params[Index_NumThreads].u_param;
    if (num_threads <= 0)
        num_threads = omp_get_max_threads();

    ss << ", Encryption Parameters" << std::endl
       << ", , Poly modulus degree, " << poly_modulus_degree << std::endl
       << ", , Coefficient Modulus, " << coeff_modulus_bits << std::endl
       << ", , Key Switching Columns, " << key_switch_columns << std::endl
       << ", , Precision, " << precision << std::endl;

    ss << ", Algorithm, " << AlgorithmName << ", " << AlgorithmDescription
       << std::endl
       << ", Number of threads, " << num_threads;

    return ss.str();
}

//------------------------
// class LogRegBenchmark
//------------------------
LogRegBenchmark::LogRegBenchmark(
    HELIBEngine &engine,
    const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
    const hebench::APIBridge::WorkloadParams &bench_params) :
    hebench::cpp::BaseBenchmark(engine, bench_desc, bench_params),
    m_w_params(bench_params)
{
    // validate workload parameters

    const hebench::APIBridge::BenchmarkDescriptor &local_bench_desc =
        getDescriptor();

    if (local_bench_desc.workload != hebench::APIBridge::Workload::LogisticRegression_PolyD3 || local_bench_desc.data_type != hebench::APIBridge::DataType::Float64 || (local_bench_desc.category != hebench::APIBridge::Category::Latency && local_bench_desc.category != hebench::APIBridge::Category::Offline) || ((local_bench_desc.cipher_param_mask & 0x03) != 0x03) || local_bench_desc.scheme != HEBENCH_HE_SCHEME_CKKS || local_bench_desc.security != HEBENCH_HE_SECURITY_128 || local_bench_desc.other != LogRegBenchmarkDescription::LogRegOtherID)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Benchmark descriptor received is not supported."),
            HEBENCH_ECODE_INVALID_ARGS);
    if (local_bench_desc.category == hebench::APIBridge::Category::Offline && (local_bench_desc.cat_params.offline.data_count[LogRegBenchmarkDescription::Index_W] > 1 || local_bench_desc.cat_params.offline.data_count[LogRegBenchmarkDescription::Index_b] > 1))
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Benchmark descriptor received is not supported."),
            HEBENCH_ECODE_INVALID_ARGS);

    std::uint64_t poly_modulus_degree = m_w_params.get<std::uint64_t>(
        LogRegBenchmarkDescription::Index_PolyModulusDegree);
    std::uint64_t coeff_modulus_bits = m_w_params.get<std::uint64_t>(
        LogRegBenchmarkDescription::Index_CoefficientModulusBits);
    std::uint64_t key_switch_columns = m_w_params.get<std::uint64_t>(
        LogRegBenchmarkDescription::Index_KeySwitchColumns);
    std::uint64_t precision = m_w_params.get<std::uint64_t>(
        LogRegBenchmarkDescription::Index_Precision);

    m_num_threads = static_cast<int>(m_w_params.get<std::uint64_t>(
        LogRegBenchmarkDescription::Index_NumThreads));
    if (m_num_threads <= 0)
        m_num_threads = omp_get_max_threads();

    m_p_ctx_wrapper = HELIBContextWrapper::createCKKSContext(
        poly_modulus_degree, coeff_modulus_bits, key_switch_columns, precision);

    // check values of the workload parameters and make sure they are supported by
    // benchmark:
    if (m_w_params.n() > m_p_ctx_wrapper->getSlotCount())
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid workload parameter 'n'. Number of features "
                               "must be under "
                               + std::to_string(m_p_ctx_wrapper->getSlotCount()) + "."),
            HEBENCH_ECODE_INVALID_ARGS);

    if (m_p_ctx_wrapper->context().securityLevel() < HEBENCH_HE_SECURITY_128)
    {
        std::stringstream ss;
        ss << "Security is found to be " << m_p_ctx_wrapper->context().securityLevel()
           << " bits, which is less than " << HEBENCH_HE_SECURITY_128 << " bits. Choose a different parameter set to get a higher security.";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    }

    unsigned m_plain_coeff_size =
        (sizeof(SigmoidPolyCoeff) / sizeof(SigmoidPolyCoeff[0]));
    for (std::size_t coeff_i = 0; coeff_i < m_plain_coeff_size; ++coeff_i)
    {
        m_plain_coeff.push_back(helib::PtxtArray(m_p_ctx_wrapper->context()));
        helib::PtxtArray temp(m_p_ctx_wrapper->context(),
                              SigmoidPolyCoeff[coeff_i]);
        m_plain_coeff[coeff_i] = temp;
    }
}

LogRegBenchmark::~LogRegBenchmark()
{
    // nothing needed in this example
}

//--------------------------
// Provided methods - Start
//--------------------------

hebench::APIBridge::Handle LogRegBenchmark::encode(
    const hebench::APIBridge::DataPackCollection *p_parameters)
{
    if (p_parameters->pack_count != LogRegBenchmarkDescription::NumOpParams)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS(
                "Invalid number of operation parameters detected in parameter "
                "pack. Expected "
                + std::to_string(LogRegBenchmarkDescription::NumOpParams) + "."),
            HEBENCH_ECODE_INVALID_ARGS);
    // validate all op parameters are in this pack
    for (std::uint64_t param_i = 0;
         param_i < LogRegBenchmarkDescription::NumOpParams; ++param_i)
    {
        if (findDataPackIndex(*p_parameters, param_i) >= p_parameters->pack_count)
            throw hebench::cpp::HEBenchError(
                HEBERROR_MSG_CLASS("DataPack for Logistic Regression inference "
                                   "operation parameter "
                                   + std::to_string(param_i) + " expected, but not found in 'p_parameters'."),
                HEBENCH_ECODE_INVALID_ARGS);
    } // end for

    const hebench::APIBridge::DataPack &pack_W =
        findDataPack(*p_parameters, LogRegBenchmarkDescription::Index_W);
    const hebench::APIBridge::DataPack &pack_b =
        findDataPack(*p_parameters, LogRegBenchmarkDescription::Index_b);
    const hebench::APIBridge::DataPack &pack_X =
        findDataPack(*p_parameters, LogRegBenchmarkDescription::Index_X);

    return this->getEngine().createHandle<EncodedOpParams>(
        sizeof(EncodedOpParams), EncodedOpParamsTag,
        std::make_tuple(encodeW(pack_W), encodeBias(pack_b),
                        encodeInputs(pack_X)));
}

helib::PtxtArray
LogRegBenchmark::encodeW(const hebench::APIBridge::DataPack &data_pack)
{
    assert(data_pack.param_position == LogRegBenchmarkDescription::Index_W);
    if (data_pack.buffer_count < 1 || !data_pack.p_buffers || !data_pack.p_buffers[0].p)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Unexpected empty DataPack for 'W'."),
            HEBENCH_ECODE_INVALID_ARGS);

    // convert Test Harness format to our internal clear text format
    std::vector<double> weights;
    hebench::APIBridge::NativeDataBuffer &buffer = data_pack.p_buffers[0];
    const double *buffer_begin                   = reinterpret_cast<const double *>(buffer.p);
    const double *buffer_end                     = buffer_begin + buffer.size / sizeof(double);
    if (buffer_begin)
    {
        weights.assign(buffer_begin, buffer_end);
    }

    if (weights.size() < m_w_params.n())
    {
        std::stringstream ss;
        ss << "Insufficient features for 'W'. Expected " << m_w_params.n()
           << ", but " << weights.size() << " received.";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    } // end if

    // encode
    helib::PtxtArray retval(m_p_ctx_wrapper->context());
    retval = helib::PtxtArray(m_p_ctx_wrapper->context(), weights);
    return retval;
}

helib::PtxtArray
LogRegBenchmark::encodeBias(const hebench::APIBridge::DataPack &data_pack)
{
    assert(data_pack.param_position == LogRegBenchmarkDescription::Index_b);
    if (data_pack.buffer_count < 1 || !data_pack.p_buffers || !data_pack.p_buffers[0].p || data_pack.p_buffers[0].size < sizeof(double))
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Unexpected empty DataPack for 'b'."),
            HEBENCH_ECODE_INVALID_ARGS);
    // convert Test Harness format to our internal clear text format
    double bias = *reinterpret_cast<const double *>(data_pack.p_buffers[0].p);

    // encode
    helib::PtxtArray retval(m_p_ctx_wrapper->context());
    retval = helib::PtxtArray(m_p_ctx_wrapper->context(), bias);
    return retval;
}

std::vector<helib::PtxtArray>
LogRegBenchmark::encodeInputs(const hebench::APIBridge::DataPack &data_pack)
{
    assert(data_pack.param_position == LogRegBenchmarkDescription::Index_X);

    // prepare our internal representation

    std::uint64_t batch_size =
        this->getDescriptor().category == hebench::APIBridge::Category::Offline ? getDescriptor()
                                                                                      .cat_params.offline
                                                                                      .data_count[LogRegBenchmarkDescription::Index_X] :
                                                                                  1;
    if (!data_pack.p_buffers)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Unexpected empty DataPack for 'W'."),
            HEBENCH_ECODE_INVALID_ARGS);
    if (data_pack.buffer_count < batch_size)
    {
        std::stringstream ss;
        ss << "Unexpected batch size for inputs. Expected, at least, " << batch_size
           << ", but " << data_pack.buffer_count << " received.";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    } // end if

    std::vector<helib::PtxtArray> retval;
    for (unsigned int x = 0; x < data_pack.buffer_count; ++x)
    {
        retval.push_back(helib::PtxtArray(m_p_ctx_wrapper->context()));
    }

    for (std::uint64_t input_sample_i = 0;
         input_sample_i < data_pack.buffer_count; ++input_sample_i)
    {
        if (!data_pack.p_buffers[input_sample_i].p)
            throw hebench::cpp::HEBenchError(
                HEBERROR_MSG_CLASS("Unexpected empty input sample " + std::to_string(input_sample_i) + "."),
                HEBENCH_ECODE_INVALID_ARGS);

        // convert Test Harness format to our internal clear text format
        std::vector<double> inputs;
        hebench::APIBridge::NativeDataBuffer &buffer =
            data_pack.p_buffers[input_sample_i];
        const double *buffer_begin = reinterpret_cast<const double *>(buffer.p);
        const double *buffer_end   = buffer_begin + buffer.size / sizeof(double);
        if (buffer_begin)
        {
            inputs.assign(buffer_begin, buffer_end);
        }

        if (inputs.size() < m_w_params.n())
        {
            std::stringstream ss;
            ss << "Invalid input sample size in sample " << input_sample_i
               << ". Expected " << m_w_params.n() << ", but " << inputs.size()
               << " received.";
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                             HEBENCH_ECODE_INVALID_ARGS);
        } // end if

        // encode
        helib::PtxtArray temp(m_p_ctx_wrapper->context());
        temp = helib::PtxtArray(m_p_ctx_wrapper->context(), inputs);

        retval[input_sample_i] = temp;
    } // end for

    return retval;
}

void LogRegBenchmark::decode(hebench::APIBridge::Handle h_encoded_data,
                             hebench::APIBridge::DataPackCollection *p_native)
{
    // only supports decoding results from decrypt

    // get result component target
    hebench::APIBridge::DataPack &result = this->findDataPack(*p_native, 0);
    // find minimum batch size to decode
    std::uint64_t batch_size = 1; // for latency
    if (this->getDescriptor().category == hebench::APIBridge::Category::Offline)
        batch_size =
            this->getDescriptor()
                        .cat_params.offline
                        .data_count[LogRegBenchmarkDescription::Index_X]
                    > 0 ?
                this->getDescriptor()
                    .cat_params.offline
                    .data_count[LogRegBenchmarkDescription::Index_X] :
                result.buffer_count;
    std::uint64_t min_count = std::min(result.buffer_count, batch_size);
    if (min_count > 0)
    {
        // decode into local format
        const helib::PtxtArray &encoded =
            this->getEngine().retrieveFromHandle<helib::PtxtArray>(
                h_encoded_data, EncodedResultTag);

        std::vector<double> decoded;
        decoded = m_p_ctx_wrapper->decodeCKKS(encoded);
        decoded.resize(min_count);

        // convert local format to Test Harness format
        for (std::uint64_t result_sample_i = 0; result_sample_i < min_count;
             ++result_sample_i)
        {
            if (result.p_buffers[result_sample_i].p && result.p_buffers[result_sample_i].size >= sizeof(double))
            {
                double *p_result_sample =
                    reinterpret_cast<double *>(result.p_buffers[result_sample_i].p);
                if (std::abs(decoded[result_sample_i]) < 0.00005)
                    *p_result_sample = 0.0;
                else
                    *p_result_sample = decoded[result_sample_i];
            } // end if
        } // end for
    } // end if
}

hebench::APIBridge::Handle
LogRegBenchmark::encrypt(hebench::APIBridge::Handle h_encoded_data)
{
    // supports encryption of EncodedOpParams only

    const EncodedOpParams &encoded_params =
        this->getEngine().retrieveFromHandle<EncodedOpParams>(h_encoded_data,
                                                              EncodedOpParamsTag);

    helib::Ctxt ctxt1(m_p_ctx_wrapper->publicKey());
    helib::Ctxt ctxt2(m_p_ctx_wrapper->publicKey());
    std::vector<helib::Ctxt> ctxt_vec;

    EncryptedOpParams retval = std::make_tuple(ctxt1, ctxt2, ctxt_vec);
    //EncryptedOpParams retval;
    std::get<LogRegBenchmarkDescription::Index_W>(retval) =
        m_p_ctx_wrapper->encrypt(
            std::get<LogRegBenchmarkDescription::Index_W>(encoded_params));
    std::get<LogRegBenchmarkDescription::Index_b>(retval) =
        m_p_ctx_wrapper->encrypt(
            std::get<LogRegBenchmarkDescription::Index_b>(encoded_params));
    std::get<LogRegBenchmarkDescription::Index_X>(retval) =
        m_p_ctx_wrapper->encrypt(
            std::get<LogRegBenchmarkDescription::Index_X>(encoded_params));

    return this->getEngine().createHandle<decltype(retval)>(
        sizeof(EncryptedOpParams), EncryptedOpParamsTag, std::move(retval));
}

hebench::APIBridge::Handle
LogRegBenchmark::decrypt(hebench::APIBridge::Handle h_encrypted_data)
{
    // only supports decrypting results from operate

    const helib::Ctxt &cipher = this->getEngine().retrieveFromHandle<helib::Ctxt>(
        h_encrypted_data, EncryptedResultTag);

    helib::PtxtArray retval(m_p_ctx_wrapper->context());
    retval = m_p_ctx_wrapper->decrypt(cipher);

    // just return a copy
    return this->getEngine().createHandle<decltype(retval)>(
        m_w_params.n(), EncodedResultTag, std::move(retval));
}

hebench::APIBridge::Handle
LogRegBenchmark::load(const hebench::APIBridge::Handle *p_h_local_data,
                      uint64_t count)
{
    // supports only loading EncryptedOpParams

    if (count != 1)
        // we do ops in ciphertext only, so we should get 1 pack
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid number of handles. Expected 1."),
            HEBENCH_ECODE_INVALID_ARGS);
    return this->getEngine().duplicateHandle(p_h_local_data[0],
                                             EncryptedOpParamsTag);
}

void LogRegBenchmark::store(hebench::APIBridge::Handle h_remote_data,
                            hebench::APIBridge::Handle *p_h_local_data,
                            std::uint64_t count)
{
    // only supports storing results from operate

    assert(count == 0 || p_h_local_data);
    if (count > 0)
    {
        // pad with zeros any excess local handles as per specifications
        std::memset(p_h_local_data, 0, sizeof(hebench::APIBridge::Handle) * count);

        // since remote and host are the same, we just need to return a copy
        // of the remote as local data.
        p_h_local_data[0] =
            this->getEngine().duplicateHandle(h_remote_data, EncryptedResultTag);
    } // end if
}

hebench::APIBridge::Handle LogRegBenchmark::operate(
    hebench::APIBridge::Handle h_remote_packed,
    const hebench::APIBridge::ParameterIndexer *p_param_indexers)
{
    // input to operation is always EncryptedOpParams

    const EncryptedOpParams &remote =
        this->getEngine().retrieveFromHandle<EncryptedOpParams>(
            h_remote_packed, EncryptedOpParamsTag);

    // extract our internal representation from handle
    const helib::Ctxt &cipher_W =
        std::get<LogRegBenchmarkDescription::Index_W>(remote);
    const std::vector<helib::Ctxt> &cipher_inputs =
        std::get<LogRegBenchmarkDescription::Index_X>(remote);
    // make a copy of the bias to be able to operate without modifying the
    // original
    helib::Ctxt cipher_b = std::get<LogRegBenchmarkDescription::Index_b>(remote);

    // validate the indexers

    // this method does not support indexing portions of the batch
    if (p_param_indexers[LogRegBenchmarkDescription::Index_X].value_index != 0 || (this->getDescriptor().category == hebench::APIBridge::Category::Offline && p_param_indexers[LogRegBenchmarkDescription::Index_X].batch_size != cipher_inputs.size()) || (this->getDescriptor().category == hebench::APIBridge::Category::Latency && p_param_indexers[LogRegBenchmarkDescription::Index_X].batch_size != 1))
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid indexer range for parameter " + std::to_string(LogRegBenchmarkDescription::Index_X) + " detected."),
            HEBENCH_ECODE_INVALID_ARGS);

    // linear regression
    std::vector<helib::Ctxt> cipher_dots;
    for (unsigned int x = 0; x < cipher_inputs.size(); ++x)
    {
        cipher_dots.push_back(helib::Ctxt(m_p_ctx_wrapper->publicKey()));
    }

    std::mutex mtx;
    std::exception_ptr p_ex;
#pragma omp parallel for num_threads(m_num_threads)
    for (std::size_t input_i = 0; input_i < cipher_inputs.size(); ++input_i)
    {
        try
        {
            if (!p_ex)
            {
                cipher_dots[input_i] = m_p_ctx_wrapper->evalMult(cipher_W, cipher_inputs[input_i]);
                m_p_ctx_wrapper->totalSums(cipher_dots[input_i]);
            } // end if
        }
        catch (...)
        {
            std::scoped_lock<std::mutex> lock(mtx);
            if (!p_ex)
                p_ex = std::current_exception();
        }
    } // end for

    if (p_ex)
        std::rethrow_exception(p_ex);

    // TODO: check if collapsing before adding bias is better:
    // - adds bias once to collapsed results
    // - computes sigmoid only once on all collapsed results
    // vs: no collapse
    // - add bias to every result
    // - compute sigmoid on every result

    helib::Ctxt cipher_lr(m_p_ctx_wrapper->publicKey());
    cipher_lr = m_p_ctx_wrapper->collapseCKKS(cipher_dots, true);

    // add bias

    cipher_lr += cipher_b;

    // cipher_lr contains all the linear regressions

    // compute sigmoid approximation

    // make a copy of the coefficients since evaluatePolynomial will modify the
    // plaintexts during the operation Is there a more efficient way to do this?
    // (think: latency will do this copy many times; offline, only once)

    std::vector<helib::PtxtArray> plain_coeff_copy = m_plain_coeff;
    helib::Ctxt retval                             = m_p_ctx_wrapper->evalPoly(cipher_lr, plain_coeff_copy);

    return this->getEngine().createHandle<decltype(retval)>(
        sizeof(helib::Ctxt), EncryptedResultTag, std::move(retval));
}

} // namespace ckks
} // namespace hbe
