// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <new>
#include <omp.h>
#include <stdexcept>

#include "engine/helib_context.h"
#include "engine/helib_types.h"

//-----------------------
// class HELIBContextWrapper
//-----------------------

HELIBContextWrapper::Ptr
HELIBContextWrapper::createCKKSContext(std::size_t poly_modulus_degree,
                                       int coeff_moduli_bits,
                                       int key_switch_columns, int precision)
{
    HELIBContextWrapper::Ptr retval =
        HELIBContextWrapper::Ptr(new HELIBContextWrapper());
    retval->initCKKS(poly_modulus_degree, coeff_moduli_bits, key_switch_columns,
                     precision);
    return retval;
}

HELIBContextWrapper::Ptr HELIBContextWrapper::createBGVContext(
    std::size_t poly_modulus_degree, int coeff_moduli_bits, int key_switch_columns,
    int ptxt_prime_modulus, int helsel_lifting)
{
    HELIBContextWrapper::Ptr retval =
        HELIBContextWrapper::Ptr(new HELIBContextWrapper());
    retval->initBGV(poly_modulus_degree, coeff_moduli_bits, key_switch_columns,
                    ptxt_prime_modulus, helsel_lifting);
    return retval;
}

HELIBContextWrapper::HELIBContextWrapper() :
    m_slot_count(0) {}

// m = cyclotomic polynomial which defines phi(m) (equal to 2 *
// poly_modulus_degree) bits = the number of bits in the "ciphertext modulus"
// precision =  the number of bits of precision when data is encoded, encrypted,
// or decrypted (only for CKKS) c = the number of columns in key-switching
// matrix p = plaintext prime modulus (only for BGV) r = Hensel Lifting (default
// = 1) (only for BGV)

void HELIBContextWrapper::initCKKS(std::size_t poly_modulus_degree,
                                   int coeff_moduli_bits,
                                   int key_switch_columns, int precision)
{
    try
    {
        m_context = helib::ContextBuilder<helib::CKKS>()
                        .m(2 * poly_modulus_degree)
                        .bits(coeff_moduli_bits)
                        .c(key_switch_columns)
                        .precision(precision)
                        .buildPtr();

        if (!m_context)
            throw std::bad_alloc();

        m_secret_key =
            std::unique_ptr<helib::SecKey>(new helib::SecKey(*m_context));
        m_secret_key->GenSecKey();
        helib::addSome1DMatrices(*m_secret_key); // for galois rotation operations

        m_public_key =
            std::unique_ptr<helib::PubKey>(new helib::PubKey(*m_secret_key));
        m_slot_count = m_context->getNSlots(); // equal to poly_modulus_degree/2
    }
    catch (std::exception &ex)
    {
        throw hebench::cpp::HEBenchError(ex.what(), HEBHELIB_ECODE_HELIB_ERROR);
    }
}

void HELIBContextWrapper::initBGV(std::size_t poly_modulus_degree,
                                  int coeff_moduli_bits, int key_switch_columns,
                                  int ptxt_prime_modulus, int helsel_lifting)
{

    try
    {
        m_context = helib::ContextBuilder<helib::BGV>()
                        .m(2 * poly_modulus_degree)
                        .bits(coeff_moduli_bits)
                        .c(key_switch_columns)
                        .p(ptxt_prime_modulus)
                        .r(helsel_lifting)
                        .buildPtr();
        if (!m_context)
            throw std::bad_alloc();

        m_secret_key =
            std::unique_ptr<helib::SecKey>(new helib::SecKey(*m_context));
        m_secret_key->GenSecKey();
        helib::addSome1DMatrices(*m_secret_key);

        m_public_key =
            std::unique_ptr<helib::PubKey>(new helib::PubKey(*m_secret_key));

        m_slot_count = m_context->getNSlots();
    }
    catch (std::exception &ex)
    {
        throw hebench::cpp::HEBenchError(ex.what(), HEBHELIB_ECODE_HELIB_ERROR);
    }
}

helib::Ctxt HELIBContextWrapper::encrypt(const helib::PtxtArray &plain) const
{
    helib::Ctxt cipher(publicKey());
    plain.encrypt(cipher);
    return cipher;
}

std::vector<helib::Ctxt>
HELIBContextWrapper::encrypt(const std::vector<helib::PtxtArray> &plain) const
{
    std::vector<helib::Ctxt> retval;

    for (std::size_t i = 0; i < plain.size(); i++)
    {
        retval.push_back(helib::Ctxt(publicKey()));
        plain[i].encrypt(retval[i]);
    }
    return retval;
}

void HELIBContextWrapper::decrypt(const helib::Ctxt &cipher,
                                  helib::PtxtArray &plain)
{
    try
    {
        if (context().isCKKS())
            plain.rawDecrypt(cipher, secretKey());
        else
            plain.decrypt(cipher, secretKey());
    }
    catch (std::exception &ex)
    {
        throw hebench::cpp::HEBenchError(ex.what(), HEBHELIB_ECODE_HELIB_ERROR);
    }
}

helib::PtxtArray HELIBContextWrapper::decrypt(const helib::Ctxt &cipher)
{
    helib::PtxtArray retval(context());
    helib::Ctxt temp(publicKey());
    temp = cipher;

    try
    {
        if (context().isCKKS())
            retval.rawDecrypt(temp, secretKey());
        else
            retval.decrypt(temp, secretKey());
    }
    catch (std::exception &ex)
    {
        throw hebench::cpp::HEBenchError(ex.what(), HEBHELIB_ECODE_HELIB_ERROR);
    }

    return retval;
}

std::vector<helib::PtxtArray>
HELIBContextWrapper::decrypt(const std::vector<helib::Ctxt> &cipher)
{
    std::vector<helib::PtxtArray> retval;

    for (std::size_t i = 0; i < cipher.size(); i++)
    {
        retval.push_back(helib::PtxtArray(context()));
        if (context().isCKKS())
            retval[i].rawDecrypt(cipher[i], secretKey());
        else
            retval[i].decrypt(cipher[i], secretKey());
    }
    return retval;
}

void HELIBContextWrapper::printContextInfo(std::ostream &os,
                                           const std::string &preamble)
{
    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf());

    context().printout();
    std::cout.rdbuf(old);

    os << buffer.str() << std::endl;
    os << std::endl
       << "Context, " << std::endl
       << preamble << "Scheme, ";

    if (context().isCKKS())
    {
        os << "CKKS";
    }
    else
    {
        os << "BFV";
    }

    os << std::endl
       << preamble << "Security level standard, ";
    os << context().securityLevel();

    os << std::endl
       << preamble << "";

    // TO DO -> Get the coeff modulus values

} // end while

// automatic modswitching in helib
// (https://homenc.github.io/HElib/documentation/Design_Document/HElib-design.pdf,
// Pg. 24-25)

helib::PtxtArray
HELIBContextWrapper::encodeVector(const std::vector<double> &values)
{
    std::size_t slot_count = context().getNSlots();

    if (values.size() > slot_count)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS(
                "Not enough slots available to create packed plaintext"),
            HEBENCH_ECODE_INVALID_ARGS);
    if (!context().isCKKS())
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Cannot encode a double using non-CKKS scheme"),
            HEBENCH_ECODE_INVALID_ARGS);

    helib::PtxtArray retval(context());
    retval = helib::PtxtArray(context(), values);
    return retval;
}

helib::PtxtArray
HELIBContextWrapper::encodeVector(const std::vector<std::int64_t> &values)
{
    std::size_t slot_count = context().getNSlots();
    if (values.size() > slot_count)
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS(
                "Not enough slots available to create packed plaintext"),
            HEBENCH_ECODE_INVALID_ARGS);
    if (context().isCKKS())
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Use BGV scheme for encoding integer values"),
            HEBENCH_ECODE_INVALID_ARGS);

    helib::PtxtArray retval(context());
    retval = helib::PtxtArray(context(), values);
    return retval;
}

void HELIBContextWrapper::evalAdd(const helib::Ctxt &A,
                                  const helib::PtxtArray &B, helib::Ctxt &C)
{
    helib::Ctxt retval(publicKey());
    retval = A;
    retval += B;
    C = retval;
}

void HELIBContextWrapper::evalAdd(const helib::Ctxt &A, const helib::Ctxt &B,
                                  helib::Ctxt &C)
{
    helib::Ctxt retval(publicKey());
    retval = A;
    retval += B;
    C = retval;
}

helib::Ctxt HELIBContextWrapper::evalAdd(const helib::Ctxt &A,
                                         const helib::PtxtArray &B)
{
    helib::Ctxt retval(publicKey());
    retval = A;
    retval += B;
    return retval;
}

helib::Ctxt HELIBContextWrapper::evalAdd(const helib::Ctxt &A,
                                         const helib::Ctxt &B)
{
    helib::Ctxt retval(publicKey());
    retval = A;
    retval += B;
    return retval;
}

helib::Ctxt HELIBContextWrapper::evalMult(const helib::Ctxt &A,
                                          const helib::PtxtArray &B)
{
    helib::Ctxt retval(publicKey());
    retval = A;
    retval *= B;
    return retval;
}

helib::Ctxt HELIBContextWrapper::evalMult(const helib::Ctxt &A,
                                          const helib::Ctxt &B)
{
    helib::Ctxt retval(publicKey());
    retval = A;
    retval *= B;
    return retval;
}

void HELIBContextWrapper::evalMult(const helib::Ctxt &A,
                                   const helib::PtxtArray &B, helib::Ctxt &C)
{
    helib::Ctxt retval(publicKey());
    retval = A;
    retval *= B;
    C = retval;
}
void HELIBContextWrapper::evalMult(const helib::Ctxt &A, const helib::Ctxt &B,
                                   helib::Ctxt &C)
{
    helib::Ctxt retval(publicKey());
    retval = A;
    retval *= B;
    C = retval;
}

// bgv

std::vector<int64_t>
HELIBContextWrapper::decodeBGV(const helib::PtxtArray &encoded_result,
                               int ptxt_prime_modulus)
{
    std::vector<int64_t> retval(encoded_result.size());
    encoded_result.store(retval);

    // since values are mod ptxt_prime_modulus, taking the first
    // ptxt_prime_modulus/2 to be +ve and rest to be -ve
    for (std::size_t i = 0; i < retval.size(); ++i)
    {
        if (retval[i] > int(ptxt_prime_modulus / 2))
        {
            retval[i] = retval[i] - ptxt_prime_modulus;
        }
    }

    return retval;
}

std::vector<std::vector<int64_t>> HELIBContextWrapper::decodeBGV(
    const std::vector<helib::PtxtArray> &encoded_result,
    int ptxt_prime_modulus)
{
    std::vector<std::vector<int64_t>> retval(encoded_result.size());
    for (std::size_t i = 0; i < encoded_result.size(); ++i)
    {
        encoded_result[i].store(retval[i]);
        for (std::size_t j = 0; j < retval[i].size(); ++j)
        {
            // since values are mod ptxt_prime_modulus, taking the first
            // ptxt_prime_modulus/2 to be +ve and rest to be -ve
            if (retval[i][j] > int(ptxt_prime_modulus / 2))
            {
                retval[i][j] = retval[i][j] - ptxt_prime_modulus;
            }
        }
    }

    return retval;
}

std::vector<std::vector<double>> HELIBContextWrapper::decodeCKKS(
    const std::vector<helib::PtxtArray> &encoded_result)
{
    std::vector<std::vector<double>> retval(encoded_result.size());
    for (std::size_t i = 0; i < encoded_result.size(); ++i)
    {
        encoded_result[i].store(retval[i]);
    }
    return retval;
}

// ckks
std::vector<int64_t>
HELIBContextWrapper::decode(const helib::PtxtArray &encoded_result)
{
    std::vector<int64_t> retval(encoded_result.size());
    encoded_result.store(retval);

    return retval;
}

std::vector<double>
HELIBContextWrapper::decodeCKKS(const helib::PtxtArray &encoded_result)
{
    std::vector<double> retval(encoded_result.size());
    encoded_result.store(retval);

    return retval;
}

helib::Ctxt
HELIBContextWrapper::encrypt(const helib::PtxtArray &encoded_value)
{
    helib::Ctxt retval(publicKey());
    encoded_value.encrypt(retval);
    return retval;
}

/*
   // Sum the slots of ciphertext using log(D) rotations
      for (size_t i = 1; i <= slot_count / 2; i <<= 1) {
        temp_ct = ct;
        helib::rotate(temp_ct, i);
        ct += temp_ct;
      }

*/

// implemented using https://eprint.iacr.org/2014/106.pdf, Pg. 10
void HELIBContextWrapper::totalSums(helib::Ctxt &ctxt)
{
    const helib::EncryptedArray &ea = context().getEA();
    long n                          = ea.size(); // slot-count
    if (n == 1)
        return;

    helib::Ctxt orig = ctxt;

    long k = NTL::NumBits(n);
    long e = 1;

    for (long i = k - 2; i >= 0; i--)
    {

        helib::Ctxt tmp1(publicKey());
        tmp1 = orig;
        helib::rotate(tmp1, e);

        orig += tmp1;
        e = 2 * e;

        if (NTL::bit(n, i))
        {
            helib::Ctxt tmp2(publicKey());
            tmp2 = orig;
            helib::rotate(tmp2, 1);

            tmp2 += ctxt;
            orig = tmp2;
            e += 1;
        }
    }
    ctxt = orig;
}

helib::Ctxt HELIBContextWrapper::collapseCKKS(std::vector<helib::Ctxt> &ciphers,
                                              bool do_rotate)
{
    // Rotates each cipher to the right by its position in the vector, then
    // multiplies it by an identity with all zeroes and a 1 on the same position
    // as cipher. All these results are added together into a single ciphertext
    // containing only the first element of each ciphertext in ciphers.

    double x = 0.0f;
    helib::PtxtArray ptx(context(), x);
    helib::Ctxt retval(publicKey());
    ptx.encrypt(retval);

    std::exception_ptr p_ex;
    for (std::size_t i = 0; i < ciphers.size(); ++i)
    {
        try
        {
            if (!p_ex)
            {
                helib::Ctxt tmp(publicKey());
                tmp = ciphers[i];
                if (do_rotate && i > 0)
                    helib::rotate(tmp, i);

                std::vector<double> identity(ciphers.size(), 0.0);
                identity[i] = 1.0;
                helib::PtxtArray plain(context(), identity);

                // multiply cipher by identity
                tmp *= plain;

                // add the result to output cipher
                retval += tmp;
            } // end if
        }
        catch (...)
        {
            if (!p_ex)
                p_ex = std::current_exception();
        }
    } // end for

    if (p_ex)
        std::rethrow_exception(p_ex);

    return retval;
}

helib::Ctxt HELIBContextWrapper::evalPoly(
    helib::Ctxt &cipher_input,
    std::vector<helib::PtxtArray> &plain_coefficients)
{
    if (plain_coefficients.empty())
    {
        throw hebench::cpp::HEBenchError(
            HEBERROR_MSG_CLASS("Polynomial must have, at least, 1 coefficient."),
            HEBENCH_ECODE_INVALID_ARGS);
    } // end if

    // Each coefficient is encoded per plaintext (it is assumed that every slot in
    // the plaintext contains the same coefficient value. This method requires, at
    // least plain_coefficients.size() - 1 coefficient moduli left in the chain.

    // perform evaluation using Horner's method:
    // f(x) = a_n * x^n + a_n-1 * x^(n-1) +... + a_1 * x + a_0
    //      = (...(((a_n * x + a_n-1) * x + a_n-2) * x ... + a_1) * x + a_0
    //
    // Adapted from:
    // https://github.com/MarwanNour/SEAL-FYP-Logistic-Regression/blob/master/logistic_regression_ckks.cpp

    helib::Ctxt retval(publicKey());
    int degree = plain_coefficients.size() - 1;

    helib::PtxtArray it(context());
    it = plain_coefficients[degree];
    it.encrypt(retval);

    for (int i = (degree - 1); i >= 0; --i)
    {
        retval.modDownToSet(cipher_input.getPrimeSet());
        // multiply current result by input
        retval *= cipher_input;
        retval.reLinearize();
        retval += plain_coefficients[i];
    } // end for

    return retval;
}

// extra functions

void HELIBContextWrapper::accumulateCKKS(helib::Ctxt &cipher)
{
    helib::Ctxt temp_ct(publicKey());
    unsigned int slot_count = getSlotCount();

    for (size_t i = 1; i <= slot_count / 2; i <<= 1) // log(slot_count) rotations
    {
        temp_ct = cipher;
        helib::rotate(temp_ct, i);
        cipher += temp_ct;
    }
}

void HELIBContextWrapper::accumulateBGV(helib::Ctxt &cipher, int count)
{

    std::vector<helib::Ctxt> rotations;

    rotations.push_back(cipher);

    for (int rotation_i = 0; rotation_i < count; ++rotation_i)
    {

        helib::Ctxt temp_ctxt(publicKey());
        temp_ctxt = cipher;
        helib::rotate(temp_ctxt, -(1 + rotation_i));
        rotations.push_back(temp_ctxt);
    }

    helib::Ctxt new_cipher(publicKey());
    for (unsigned int i = 0; i < rotations.size(); ++i)
    {
        new_cipher += rotations[i];
    }
    cipher = new_cipher;
}

std::vector<std::vector<int64_t>> HELIBContextWrapper::decode(
    const std::vector<helib::PtxtArray> &encoded_result)
{
    std::vector<std::vector<int64_t>> retval(encoded_result.size());
    for (std::size_t i = 0; i < encoded_result.size(); ++i)
    {
        encoded_result[i].store(retval[i]);
    }
    return retval;
}
