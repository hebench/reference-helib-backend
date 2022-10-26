
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <omp.h>
#include <vector>

#include "benchmarks/bgv/helib_bgv_matmultval_benchmark.h"
#include "engine/helib_engine.h"
#include "engine/helib_types.h"

namespace hbe {
namespace bgv {

//-----------------------------------
// class MatMultValBenchmarkDescription
//-----------------------------------

MatMultValBenchmarkDescription::MatMultValBenchmarkDescription()
{
    std::memset(&m_descriptor, 0,
                sizeof(hebench::APIBridge::BenchmarkDescriptor));
    m_descriptor.workload                                   = hebench::APIBridge::Workload::MatrixMultiply;
    m_descriptor.data_type                                  = hebench::APIBridge::DataType::Int64;
    m_descriptor.category                                   = hebench::APIBridge::Category::Latency;
    m_descriptor.cat_params.latency.warmup_iterations_count = 1;
    m_descriptor.cat_params.min_test_time_ms                = 0;
    m_descriptor.cipher_param_mask                          = HEBENCH_HE_PARAM_FLAGS_ALL_CIPHER;
    m_descriptor.scheme                                     = HEBENCH_HE_SCHEME_BGV;
    m_descriptor.security                                   = HEBENCH_HE_SECURITY_128;
    m_descriptor.other                                      = MatMultValOtherID;

    // specify default arguments for this workload flexible parameters:
    hebench::cpp::WorkloadParams::MatrixMultiply default_workload_params;
    default_workload_params.rows_M0() = 10;
    default_workload_params.cols_M0() = 9;
    default_workload_params.cols_M1() = 8;

    default_workload_params.add<std::uint64_t>(
        MatMultValBenchmarkDescription::DefaultCyclotomicPoly, "CyclotomicPoly");
    default_workload_params.add<std::uint64_t>(
        MatMultValBenchmarkDescription::DefaultCoeffModulusBits,
        "CoefficientModulusBits");
    default_workload_params.add<std::uint64_t>(
        MatMultValBenchmarkDescription::DefaultKeySwitchColumns,
        "KeySwitchColumns");
    default_workload_params.add<std::uint64_t>(
        MatMultValBenchmarkDescription::DefaultPtxtPrimeModulus,
        "PtxtPrimeModulus");
    default_workload_params.add<std::uint64_t>(
        MatMultValBenchmarkDescription::DefaultHenselLifting, "HenselLifting");

    default_workload_params.add<std::uint64_t>(
        MatMultValBenchmarkDescription::DefaultNumThreads, "NumThreads");
    this->addDefaultParameters(default_workload_params);
}

MatMultValBenchmarkDescription::~MatMultValBenchmarkDescription()
{
    //
}

hebench::cpp::BaseBenchmark *MatMultValBenchmarkDescription::createBenchmark(
    hebench::cpp::BaseEngine &engine,
    const hebench::APIBridge::WorkloadParams *p_params)
{
    if (!p_params)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid empty workload parameters. Matrix "
                               "Multiplication requires parameters."),
            HEBENCH_ECODE_INVALID_ARGS);

    return new MatMultValBenchmark(engine, m_descriptor, *p_params);
}

void MatMultValBenchmarkDescription::destroyBenchmark(
    hebench::cpp::BaseBenchmark *p_bench)
{
    if (p_bench)
        delete p_bench;
}

std::string MatMultValBenchmarkDescription::getBenchmarkDescription(
    const hebench::APIBridge::WorkloadParams *p_w_params) const
{
    std::stringstream ss;
    std::string s_tmp = BenchmarkDescription::getBenchmarkDescription(p_w_params);

    if (!p_w_params)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid null workload parameters `p_w_params`"),
            HEBENCH_ECODE_INVALID_ARGS);

    assert(p_w_params->count >= MatMultValBenchmarkDescription::NumWorkloadParams);

    std::uint64_t cyclotomic_poly =
        p_w_params->params[MatMultValBenchmarkDescription::Index_CyclotomicPoly]
            .u_param;
    std::uint64_t coeff_modulus_bits =
        p_w_params
            ->params[MatMultValBenchmarkDescription::Index_CoefficientModulusBits]
            .u_param;
    std::uint64_t key_switch_columns =
        p_w_params->params[MatMultValBenchmarkDescription::Index_KeySwitchColumns]
            .u_param;
    std::uint64_t ptxt_prime_modulus =
        p_w_params->params[MatMultValBenchmarkDescription::Index_PtxtPrimeModulus]
            .u_param;
    std::uint64_t helsel_lifting =
        p_w_params->params[MatMultValBenchmarkDescription::Index_HenselLifting]
            .u_param;

    std::uint64_t num_threads =
        p_w_params->params[MatMultValBenchmarkDescription::Index_NumThreads]
            .u_param;
    if (num_threads <= 0)
        num_threads = omp_get_max_threads();
    if (!s_tmp.empty())
        ss << s_tmp << std::endl;
    ss << ", Encryption Parameters" << std::endl
       << ", , Cyclotomic Polynomial - phi(m), " << cyclotomic_poly << std::endl
       << ", , Coefficient Modulus, " << coeff_modulus_bits << std::endl
       << ", , Key Switching Columns, " << key_switch_columns << std::endl
       << ", , Plaintext Prime Modulus, " << ptxt_prime_modulus << std::endl
       << ", , Hensel Lifting, " << helsel_lifting << std::endl;

    ss << ", Algorithm, " << AlgorithmName << ", " << AlgorithmDescription
       << std::endl
       << ", Number of threads, " << num_threads;

    return ss.str();
}

//------------------------
// class MatMultValBenchmark
//------------------------

MatMultValBenchmark::MatMultValBenchmark(
    hebench::cpp::BaseEngine &engine,
    const hebench::APIBridge::BenchmarkDescriptor &bench_desc,
    const hebench::APIBridge::WorkloadParams &bench_params) :
    hebench::cpp::BaseBenchmark(engine, bench_desc, bench_params),
    m_w_params(bench_params)
{
    assert(bench_params.count >= MatMultValBenchmarkDescription::NumWorkloadParams);

    if (bench_desc.workload != hebench::APIBridge::Workload::MatrixMultiply || bench_desc.data_type != hebench::APIBridge::DataType::Int64 || bench_desc.category != hebench::APIBridge::Category::Latency || ((bench_desc.cipher_param_mask & 0x03) != 0x03) || bench_desc.scheme != HEBENCH_HE_SCHEME_BGV || bench_desc.security != HEBENCH_HE_SECURITY_128 || bench_desc.other != MatMultValBenchmarkDescription::MatMultValOtherID)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Benchmark descriptor received is not supported."),
            HEBENCH_ECODE_INVALID_ARGS);

    // validate workload parameters
    std::uint64_t cyclotomic_poly = m_w_params.get<std::uint64_t>(
        MatMultValBenchmarkDescription::Index_CyclotomicPoly);
    std::uint64_t coeff_modulus_bits = m_w_params.get<std::uint64_t>(
        MatMultValBenchmarkDescription::Index_CoefficientModulusBits);
    std::uint64_t key_switch_columns = m_w_params.get<std::uint64_t>(
        MatMultValBenchmarkDescription::Index_KeySwitchColumns);
    std::uint64_t ptxt_prime_modulus = m_w_params.get<std::uint64_t>(
        MatMultValBenchmarkDescription::Index_PtxtPrimeModulus);
    std::uint64_t helsel_lifting = m_w_params.get<std::uint64_t>(
        MatMultValBenchmarkDescription::Index_HenselLifting);

    m_num_threads             = static_cast<int>(m_w_params.get<std::uint64_t>(
        MatMultValBenchmarkDescription::Index_NumThreads));
    m_plaintext_prime_modulus = static_cast<int>(m_w_params.get<std::uint64_t>(
        MatMultValBenchmarkDescription::Index_PtxtPrimeModulus));

    if (m_num_threads <= 0)
        m_num_threads = omp_get_max_threads();

    // check values of the workload parameters and make sure they are supported by
    // benchmark:

    if (m_w_params.rows_M0() <= 0 || m_w_params.cols_M0() <= 0 || m_w_params.cols_M1() <= 0)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Matrix dimensions must be greater than 0."),
            HEBENCH_ECODE_INVALID_ARGS);
    if (m_w_params.cols_M0() > cyclotomic_poly)
    {
        std::stringstream ss;
        ss << "Invalid workload parameters. This workload only supports matrices "
              "of dimensions (n x "
           << cyclotomic_poly << ") x (" << cyclotomic_poly << " x m).";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    } // end if
    if (coeff_modulus_bits < 1)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Multiplicative depth must be greater than 0."),
            HEBENCH_ECODE_INVALID_ARGS);

    m_p_ctx_wrapper = HELIBContextWrapper::createBGVContext(
        static_cast<int>(cyclotomic_poly), static_cast<int>(coeff_modulus_bits),
        static_cast<int>(key_switch_columns),
        static_cast<int>(ptxt_prime_modulus), static_cast<int>(helsel_lifting));

    if (m_p_ctx_wrapper->context().securityLevel() < HEBENCH_HE_SECURITY_128)
    {
        std::stringstream ss;
        ss << "Security is found to be " << m_p_ctx_wrapper->context().securityLevel()
           << " bits, which is less than " << HEBENCH_HE_SECURITY_128 << " bits. Choose a different parameter set to get a higher security.";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    }
}

MatMultValBenchmark::~MatMultValBenchmark()
{
    // nothing needed in this example
}

//--------------------------
// Provided methods - Start
//--------------------------

std::vector<std::vector<std::int64_t>> MatMultValBenchmark::prepareMatrix(
    const hebench::APIBridge::NativeDataBuffer &buffer, std::uint64_t rows,
    std::uint64_t cols)
{
    std::vector<std::vector<std::int64_t>> retval(
        rows, std::vector<std::int64_t>(cols));
    if (!buffer.p || buffer.size < rows * cols * sizeof(std::int64_t))
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Insufficient data for M0."),
            HEBENCH_ECODE_INVALID_ARGS);
    const std::int64_t *p_curr_row =
        reinterpret_cast<const std::int64_t *>(buffer.p);
    for (std::size_t row_i = 0; row_i < rows; ++row_i)
    {
        std::copy(p_curr_row, p_curr_row + cols, retval[row_i].begin());
        p_curr_row += cols;
    } // end for
    return retval;
}

std::vector<helib::PtxtArray> MatMultValBenchmark::encodeMatrix(
    const std::vector<std::vector<std::int64_t>> &data)
{
    std::vector<helib::PtxtArray> retval;
    for (unsigned int i = 0; i < data.size(); ++i)
    {
        retval.push_back(helib::PtxtArray(m_p_ctx_wrapper->context()));
    }

    std::size_t num_cols = data.empty() ? 0 : data.front().size();
    for (size_t i = 0; i < data.size(); ++i)
    {
        if (data[i].size() != num_cols)
            throw hebench::cpp::HEBenchError(
                HEBERROR_MSG_CLASS("Non-uniform number of columns found in matrix."),
                HEBENCH_ECODE_INVALID_ARGS);
        retval[i] = m_p_ctx_wrapper->encodeVector(data[i]);
    } // end for

    return retval;
}

std::vector<helib::PtxtArray> MatMultValBenchmark::encodeM0(
    const std::vector<std::vector<std::int64_t>> &data)
{
    assert(data.size() == m_w_params.rows_M0());
    assert(!data.empty() && data.front().size() == m_w_params.cols_M0());
    return encodeMatrix(data);
}

std::vector<helib::PtxtArray> MatMultValBenchmark::encodeM1(
    const std::vector<std::vector<std::int64_t>> &data)
{
    assert(data.size() == m_w_params.cols_M0());

    // transpose
    std::vector<std::vector<std::int64_t>> data_T(
        m_w_params.cols_M1(), std::vector<std::int64_t>(m_w_params.cols_M0()));
    for (size_t row_i = 0; row_i < data.size(); ++row_i)
    {
        assert(data[row_i].size() == m_w_params.cols_M1());
        for (size_t col_i = 0; col_i < data[row_i].size(); ++col_i)
            data_T[col_i][row_i] = data[row_i][col_i];
    } // end if
    return encodeMatrix(data_T);
}

std::vector<helib::Ctxt>
MatMultValBenchmark::encryptMatrix(const std::vector<helib::PtxtArray> &plain)
{
    std::vector<helib::Ctxt> retval;

    for (std::size_t i = 0; i < plain.size(); i++)
    {
        retval.push_back(helib::Ctxt(m_p_ctx_wrapper->publicKey()));
        plain[i].encrypt(retval[i]);
    }

    return retval;
}

std::vector<std::vector<helib::Ctxt>>
MatMultValBenchmark::doMatMultVal(const std::vector<helib::Ctxt> &M0,
                                  const std::vector<helib::Ctxt> &M1_T)
{
    std::vector<std::vector<helib::Ctxt>> retval;

    unsigned int m0_row_count    = m_w_params.rows_M0();
    unsigned int m1_column_count = m_w_params.cols_M1();

    retval.resize(m0_row_count);
    for (unsigned int i = 0; i < m0_row_count; ++i)
    {
        for (unsigned int j = 0; j < m1_column_count; ++j)
        {
            retval[i].push_back(helib::Ctxt(m_p_ctx_wrapper->publicKey()));
        }
    }

    std::exception_ptr p_ex;
    std::mutex mtx_ex;
#pragma omp parallel for collapse(2) num_threads(m_num_threads)
    for (size_t i = 0; i < m_w_params.rows_M0(); ++i)
    {
        for (size_t j = 0; j < m_w_params.cols_M1(); ++j)
        {
            try
            {
                if (!p_ex)
                {
                    retval[i][j] = m_p_ctx_wrapper->evalMult(M0[i], M1_T[j]);
                    m_p_ctx_wrapper->totalSums(retval[i][j]);

                } // end if
            }
            catch (...)
            {
                std::scoped_lock<std::mutex> lock(mtx_ex);
                if (!p_ex)
                    p_ex = std::current_exception();
            }
        } // end for
    } // end for
    if (p_ex)
        std::rethrow_exception(p_ex);
    return retval;
}

//--------------------------
// Provided methods - End
//--------------------------

hebench::APIBridge::Handle MatMultValBenchmark::encode(
    const hebench::APIBridge::DataPackCollection *p_parameters)
{
    std::pair<InternalMatrixPlain, InternalMatrixPlain> params =
        std::make_pair<InternalMatrixPlain, InternalMatrixPlain>(
            InternalMatrixPlain(0), InternalMatrixPlain(1));

    // encode M0
    // InternalMatrixPlain plain_M0(0);
    InternalMatrixPlain &plain_M0 = params.first;

    const hebench::APIBridge::DataPack &raw_M0 =
        MatMultValBenchmark::findDataPack(*p_parameters, 0);
    if (raw_M0.buffer_count <= 0 || !raw_M0.p_buffers)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid empty data for M0."),
            HEBENCH_ECODE_INVALID_ARGS);
    const hebench::APIBridge::NativeDataBuffer &buffer_M0 = *raw_M0.p_buffers;
    std::vector<std::vector<std::int64_t>> matrix_data =
        prepareMatrix(buffer_M0, m_w_params.rows_M0(), m_w_params.cols_M0());
    plain_M0.rows() = encodeM0(matrix_data);

    // encode M1
    // InternalMatrixPlain plain_M1(1);
    InternalMatrixPlain &plain_M1 = params.second;

    const hebench::APIBridge::DataPack &raw_M1 =
        MatMultValBenchmark::findDataPack(*p_parameters, 1);
    if (raw_M1.buffer_count <= 0 || !raw_M1.p_buffers)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid empty data for M1."),
            HEBENCH_ECODE_INVALID_ARGS);
    const hebench::APIBridge::NativeDataBuffer &buffer_M1 = *raw_M1.p_buffers;
    matrix_data =
        prepareMatrix(buffer_M1, m_w_params.cols_M0(), m_w_params.cols_M1());
    plain_M1.rows() = encodeM1(matrix_data);

    // wrap our internal object into a handle to cross the boundary of the API
    // Bridge
    return this->getEngine().template createHandle<decltype(params)>(
        sizeof(helib::PtxtArray) * 2, // size (arbitrary for our usage if we need to)
        0, // extra tags
        std::move(params)); // constructor parameters
}

void MatMultValBenchmark::decode(
    hebench::APIBridge::Handle h_encoded_data,
    hebench::APIBridge::DataPackCollection *p_native)
{
    // able to decode only encoded result

    assert(p_native && p_native->p_data_packs && p_native->pack_count > 0);

    const std::vector<std::vector<helib::PtxtArray>> &encoded_result =
        this->getEngine()
            .template retrieveFromHandle<
                std::vector<std::vector<helib::PtxtArray>>>(
                h_encoded_data, MatMultValBenchmark::tagEncodedResult);

    if (encoded_result.size() < m_w_params.rows_M0())
    {
        std::stringstream ss;
        ss << "Invalid number of rows in encoded result. Expected "
           << m_w_params.rows_M0() << ", but received " << encoded_result.size()
           << ".";
        throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                         HEBENCH_ECODE_INVALID_ARGS);
    } // end if
    for (std::size_t row_i = 0; row_i < m_w_params.rows_M0(); ++row_i)
        if (encoded_result[row_i].size() < m_w_params.cols_M1())
        {
            std::stringstream ss;
            ss << "Invalid number of columns for row " << row_i
               << " in encoded result. Expected " << m_w_params.cols_M1()
               << ", but received " << encoded_result[row_i].size() << ".";
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                             HEBENCH_ECODE_INVALID_ARGS);
        } // end if

    // index for result component 0
    std::uint64_t data_pack_index =
        MatMultValBenchmark::findDataPackIndex(*p_native, 0);
    if (data_pack_index < p_native->pack_count)
    {
        hebench::APIBridge::DataPack &result_component =
            p_native->p_data_packs[data_pack_index];
        if (result_component.buffer_count > 0 && result_component.p_buffers)
        {
            hebench::APIBridge::NativeDataBuffer &buffer =
                result_component.p_buffers[0];
            if (buffer.p && buffer.size > 0)
            {
                // copy as much as we can
                std::int64_t *p_data = reinterpret_cast<std::int64_t *>(buffer.p);
                for (std::size_t row_i = 0; row_i < m_w_params.rows_M0(); ++row_i)
                {
                    for (std::size_t col_i = 0; col_i < m_w_params.cols_M1(); ++col_i)
                    {
                        std::vector<std::int64_t> decoded;
                        decoded = m_p_ctx_wrapper->decodeBGV(encoded_result[row_i][col_i],
                                                             m_plaintext_prime_modulus);
                        p_data[m_w_params.cols_M1() * row_i + col_i] =
                            decoded.empty() ? 0.0 : decoded.front();
                    } // end for
                } // end for
            } // end if
        } // end if
    } // end if
}

hebench::APIBridge::Handle
MatMultValBenchmark::encrypt(hebench::APIBridge::Handle h_encoded_data)
{
    const std::pair<InternalMatrixPlain, InternalMatrixPlain> &encoded_params =
        this->getEngine()
            .template retrieveFromHandle<
                std::pair<InternalMatrixPlain, InternalMatrixPlain>>(
                h_encoded_data);

    std::pair<InternalMatrixCipher, InternalMatrixCipher> result_encrypted =
        std::make_pair<InternalMatrixCipher, InternalMatrixCipher>(
            InternalMatrixCipher(encoded_params.first.paramPosition()),
            InternalMatrixCipher(encoded_params.second.paramPosition()));

    result_encrypted.first.rows()  = encryptMatrix(encoded_params.first.rows());
    result_encrypted.second.rows() = encryptMatrix(encoded_params.second.rows());

    // wrap our internal object into a handle to cross the boundary of the API
    // Bridge
    return this->getEngine().template createHandle<decltype(result_encrypted)>(
        sizeof(helib::Ctxt) * 2, // size (arbitrary for our usage if we need to)
        0, // extra tags
        std::move(result_encrypted)); // constructor parameters
}

hebench::APIBridge::Handle
MatMultValBenchmark::decrypt(hebench::APIBridge::Handle h_encrypted_data)
{
    // supports only encrypted result
    const std::vector<std::vector<helib::Ctxt>> &encrypted_result =
        this->getEngine()
            .template retrieveFromHandle<std::vector<std::vector<helib::Ctxt>>>(
                h_encrypted_data, MatMultValBenchmark::tagEncryptedResult);

    std::vector<std::vector<helib::PtxtArray>> ptxt_result;
    ptxt_result.reserve(encrypted_result.size());

    for (size_t i = 0; i < encrypted_result.size(); ++i)
    {
        ptxt_result.push_back(m_p_ctx_wrapper->decrypt(encrypted_result[i]));
    } // end for

    // wrap our internal object into a handle to cross the boundary of the API
    // Bridge
    return this->getEngine().template createHandle<decltype(ptxt_result)>(
        ptxt_result.size(), MatMultValBenchmark::tagEncodedResult,
        std::move(ptxt_result));
}

hebench::APIBridge::Handle
MatMultValBenchmark::load(const hebench::APIBridge::Handle *p_local_data,
                          uint64_t count)
{
    if (count != 1)
        // we do ops in Ctxt only, so we should get 1 pack
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid number of handles. Expected 1."),
            HEBENCH_ECODE_INVALID_ARGS);
    // host is same as remote, so, just duplicate handle to let called be able to
    // destroy input handle
    return this->getEngine().duplicateHandle(p_local_data[0]);
}

void MatMultValBenchmark::store(hebench::APIBridge::Handle h_remote_data,
                                hebench::APIBridge::Handle *p_local_data,
                                std::uint64_t count)
{
    // supports only storing result
    if (count > 0 && !p_local_data)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Invalid null array of handles: \"p_local_data\""),
            HEBENCH_ECODE_INVALID_ARGS);

    std::memset(p_local_data, 0, sizeof(hebench::APIBridge::Handle) * count);

    if (count > 0)
    {
        // host is same as remote, so, just duplicate handle to let called be able
        // to destroy input handle
        p_local_data[0] = this->getEngine().duplicateHandle(
            h_remote_data, MatMultValBenchmark::tagEncryptedResult);
    } // end if
}

hebench::APIBridge::Handle MatMultValBenchmark::operate(
    hebench::APIBridge::Handle h_remote_packed,
    const hebench::APIBridge::ParameterIndexer *p_param_indexers)
{
    const std::pair<InternalMatrixCipher, InternalMatrixCipher> &loaded_data =
        this->getEngine()
            .template retrieveFromHandle<
                std::pair<InternalMatrixCipher, InternalMatrixCipher>>(
                h_remote_packed);

    assert(loaded_data.first.paramPosition() == 0 && loaded_data.second.paramPosition() == 1);

    // validate indexers
    for (std::size_t param_i = 0; param_i < ParametersCount; ++param_i)
    {
        if (p_param_indexers[param_i].value_index > 0 || p_param_indexers[param_i].batch_size != 1)
        {
            std::stringstream ss;
            ss << "Invalid parameter indexer for operation parameter " << param_i
               << ". Expected index in range [0, 1), but ["
               << p_param_indexers[param_i].value_index << ", "
               << p_param_indexers[param_i].batch_size << ") received.";
            throw hebench::cpp::HEBenchError(HEBERROR_MSG_CLASS(ss.str()),
                                             HEBENCH_ECODE_INVALID_ARGS);
        } // end if
    } // end for

    std::vector<std::vector<helib::Ctxt>> retval =
        doMatMultVal(loaded_data.first.rows(), loaded_data.second.rows());

    // send our internal result across the boundary of the API Bridge as a handle
    return this->getEngine().template createHandle<decltype(retval)>(
        sizeof(retval), MatMultValBenchmark::tagEncryptedResult,
        std::move(retval));
}

} // namespace bgv
} // namespace hbe
