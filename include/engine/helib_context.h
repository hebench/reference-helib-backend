// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ostream>
#include <string>

#include <helib/helib.h>

#include "hebench/api_bridge/cpp/hebench.hpp"

class HELIBContextWrapper
{
public:
    HEBERROR_DECLARE_CLASS_NAME(HELIBContextWrapper)
    HELIBContextWrapper(const HELIBContextWrapper &) = delete;
    HELIBContextWrapper &operator=(const HELIBContextWrapper &) = delete;

public:
    typedef std::shared_ptr<HELIBContextWrapper> Ptr;

    /**
   * @brief CKKS constructor
   * @param[in] poly_modulus_degree
   * @param[in] coeff_moduli_bits Bits for coefficient moduli. Usually equals
   * scale bits.
   * @param[in] key_switch_columns Number of columns in key-switching matrix.
   * @param[in] precision Number of bits of precision when data is encoded,
   * encrypted, or decrypted
   */
    static HELIBContextWrapper::Ptr createCKKSContext(std::size_t poly_modulus_degree, int coeff_moduli_bits,
                                                      int key_switch_columns, int precision);

    /**
   * @brief BFV constructor
   * @param[in] poly_modulus_degree Polynomial Modulus Degree or phi(m) and is
   * equal to Cyclotomic Order (m) / 2.
   * @param[in] Number of bits in the "ciphertext modulus".
   * @param[in] key_switch_columns Number of columns in key-switching matrix.
   * @param[in] ptxt_prime_modulus Plaintext prime modulus
   * @param[in] helsel_lifting Hensel Lifting (default = 1)
   */
    static HELIBContextWrapper::Ptr createBGVContext(std::size_t poly_modulus_degree,
                                                     int coeff_moduli_bits,
                                                     int key_switch_columns,
                                                     int ptxt_prime_modulus,
                                                     int helsel_lifting);

    helib::Ctxt encrypt(const helib::PtxtArray &plain) const;
    helib::Ctxt encrypt(const helib::PtxtArray &encoded_value);
    std::vector<helib::Ctxt> encrypt(const std::vector<helib::PtxtArray> &plain) const;

    void decrypt(const helib::Ctxt &cipher, helib::PtxtArray &plain);
    helib::PtxtArray decrypt(const helib::Ctxt &cipher);
    std::vector<helib::PtxtArray> decrypt(const std::vector<helib::Ctxt> &cipher);

    helib::Context &context() { return *m_context; }
    const helib::Context &context() const { return *m_context; }
    const helib::PubKey &publicKey() const { return *m_public_key; }
    const helib::SecKey &secretKey() const { return *m_secret_key; }

    std::size_t getSlotCount() const { return m_slot_count; }
    void printContextInfo(std::ostream &os,
                          const std::string &preamble = std::string());

    helib::Ctxt evalPoly(helib::Ctxt &cipher_input,
                         std::vector<helib::PtxtArray> &plain_coefficients);
    helib::Ctxt collapseCKKS(std::vector<helib::Ctxt> &ciphers, bool do_rotate);

    helib::Ctxt evalAdd(const helib::Ctxt &A, const helib::PtxtArray &B);
    helib::Ctxt evalAdd(const helib::Ctxt &A, const helib::Ctxt &B);
    void evalAdd(const helib::Ctxt &A, const helib::PtxtArray &B, helib::Ctxt &C);
    void evalAdd(const helib::Ctxt &A, const helib::Ctxt &B, helib::Ctxt &C);

    helib::Ctxt evalMult(const helib::Ctxt &A, const helib::PtxtArray &B);
    helib::Ctxt evalMult(const helib::Ctxt &A, const helib::Ctxt &B);
    void evalMult(const helib::Ctxt &A, const helib::PtxtArray &B,
                  helib::Ctxt &C);
    void evalMult(const helib::Ctxt &A, const helib::Ctxt &B, helib::Ctxt &C);

public:
    helib::PtxtArray encodeVector(const std::vector<std::int64_t> &values);
    std::vector<std::vector<int64_t>> decode(const std::vector<helib::PtxtArray> &encoded_result);
    std::vector<int64_t> decodeBGV(const helib::PtxtArray &encoded_result,
                                   int ptxt_prime_modulus);
    std::vector<std::vector<int64_t>> decodeBGV(const std::vector<helib::PtxtArray> &encoded_result,
                                                int ptxt_prime_modulus);

    std::vector<std::vector<double>> decodeCKKS(const std::vector<helib::PtxtArray> &encoded_result);
    std::vector<int64_t> decode(const helib::PtxtArray &encoded_result);
    std::vector<double> decodeCKKS(const helib::PtxtArray &encoded_result);

    void accumulateBGV(helib::Ctxt &cipher, int count);
    void accumulateCKKS(helib::Ctxt &cipher);

    void totalSums(helib::Ctxt &ctxt);

public:
    helib::PtxtArray encodeVector(const std::vector<double> &values);

protected:
    HELIBContextWrapper();
    virtual void initCKKS(std::size_t poly_modulus_degree, int coeff_moduli_bits,
                          int key_switch_columns, int precision);
    virtual void initBGV(std::size_t poly_modulus_degree, int coeff_moduli_bits,
                         int key_switch_columns, int ptxt_prime_modulus,
                         int helsel_lifting);

private:
    helib::Context *m_context;
    std::unique_ptr<helib::PubKey> m_public_key;
    std::unique_ptr<helib::SecKey> m_secret_key;
    std::size_t m_slot_count;
};
