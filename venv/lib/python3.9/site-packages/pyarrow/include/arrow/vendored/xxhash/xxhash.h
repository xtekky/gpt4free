/*
 * xxHash - Extremely Fast Hash algorithm
 * Header File
 * Copyright (C) 2012-2020 Yann Collet
 *
 * BSD 2-Clause License (https://www.opensource.org/licenses/bsd-license.php)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following disclaimer
 *      in the documentation and/or other materials provided with the
 *      distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * You can contact the author at:
 *   - xxHash homepage: https://www.xxhash.com
 *   - xxHash source repository: https://github.com/Cyan4973/xxHash
 */

/* TODO: update */
/* Notice extracted from xxHash homepage:

xxHash is an extremely fast hash algorithm, running at RAM speed limits.
It also successfully passes all tests from the SMHasher suite.

Comparison (single thread, Windows Seven 32 bits, using SMHasher on a Core 2 Duo @3GHz)

Name            Speed       Q.Score   Author
xxHash          5.4 GB/s     10
CrapWow         3.2 GB/s      2       Andrew
MumurHash 3a    2.7 GB/s     10       Austin Appleby
SpookyHash      2.0 GB/s     10       Bob Jenkins
SBox            1.4 GB/s      9       Bret Mulvey
Lookup3         1.2 GB/s      9       Bob Jenkins
SuperFastHash   1.2 GB/s      1       Paul Hsieh
CityHash64      1.05 GB/s    10       Pike & Alakuijala
FNV             0.55 GB/s     5       Fowler, Noll, Vo
CRC32           0.43 GB/s     9
MD5-32          0.33 GB/s    10       Ronald L. Rivest
SHA1-32         0.28 GB/s    10

Q.Score is a measure of quality of the hash function.
It depends on successfully passing SMHasher test set.
10 is a perfect score.

Note: SMHasher's CRC32 implementation is not the fastest one.
Other speed-oriented implementations can be faster,
especially in combination with PCLMUL instruction:
https://fastcompression.blogspot.com/2019/03/presenting-xxh3.html?showComment=1552696407071#c3490092340461170735

A 64-bit version, named XXH64, is available since r35.
It offers much better speed, but for 64-bit applications only.
Name     Speed on 64 bits    Speed on 32 bits
XXH64       13.8 GB/s            1.9 GB/s
XXH32        6.8 GB/s            6.0 GB/s
*/

#if defined (__cplusplus)
extern "C" {
#endif

/* ****************************
 *  INLINE mode
 ******************************/
/*!
 * XXH_INLINE_ALL (and XXH_PRIVATE_API)
 * Use these build macros to inline xxhash into the target unit.
 * Inlining improves performance on small inputs, especially when the length is
 * expressed as a compile-time constant:
 *
 *      https://fastcompression.blogspot.com/2018/03/xxhash-for-small-keys-impressive-power.html
 *
 * It also keeps xxHash symbols private to the unit, so they are not exported.
 *
 * Usage:
 *     #define XXH_INLINE_ALL
 *     #include "xxhash.h"
 *
 * Do not compile and link xxhash.o as a separate object, as it is not useful.
 */
#if (defined(XXH_INLINE_ALL) || defined(XXH_PRIVATE_API)) \
    && !defined(XXH_INLINE_ALL_31684351384)
   /* this section should be traversed only once */
#  define XXH_INLINE_ALL_31684351384
   /* give access to the advanced API, required to compile implementations */
#  undef XXH_STATIC_LINKING_ONLY   /* avoid macro redef */
#  define XXH_STATIC_LINKING_ONLY
   /* make all functions private */
#  undef XXH_PUBLIC_API
#  if defined(__GNUC__)
#    define XXH_PUBLIC_API static __inline __attribute__((unused))
#  elif defined (__cplusplus) || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */)
#    define XXH_PUBLIC_API static inline
#  elif defined(_MSC_VER)
#    define XXH_PUBLIC_API static __inline
#  else
     /* note: this version may generate warnings for unused static functions */
#    define XXH_PUBLIC_API static
#  endif

   /*
    * This part deals with the special case where a unit wants to inline xxHash,
    * but "xxhash.h" has previously been included without XXH_INLINE_ALL, such
    * as part of some previously included *.h header file.
    * Without further action, the new include would just be ignored,
    * and functions would effectively _not_ be inlined (silent failure).
    * The following macros solve this situation by prefixing all inlined names,
    * avoiding naming collision with previous inclusions.
    */
#  ifdef XXH_NAMESPACE
#    error "XXH_INLINE_ALL with XXH_NAMESPACE is not supported"
     /*
      * Note: Alternative: #undef all symbols (it's a pretty large list).
      * Without #error: it compiles, but functions are actually not inlined.
      */
#  endif
#  define XXH_NAMESPACE XXH_INLINE_
   /*
    * Some identifiers (enums, type names) are not symbols, but they must
    * still be renamed to avoid redeclaration.
    * Alternative solution: do not redeclare them.
    * However, this requires some #ifdefs, and is a more dispersed action.
    * Meanwhile, renaming can be achieved in a single block
    */
#  define XXH_IPREF(Id)   XXH_INLINE_ ## Id
#  define XXH_OK XXH_IPREF(XXH_OK)
#  define XXH_ERROR XXH_IPREF(XXH_ERROR)
#  define XXH_errorcode XXH_IPREF(XXH_errorcode)
#  define XXH32_canonical_t  XXH_IPREF(XXH32_canonical_t)
#  define XXH64_canonical_t  XXH_IPREF(XXH64_canonical_t)
#  define XXH128_canonical_t XXH_IPREF(XXH128_canonical_t)
#  define XXH32_state_s XXH_IPREF(XXH32_state_s)
#  define XXH32_state_t XXH_IPREF(XXH32_state_t)
#  define XXH64_state_s XXH_IPREF(XXH64_state_s)
#  define XXH64_state_t XXH_IPREF(XXH64_state_t)
#  define XXH3_state_s  XXH_IPREF(XXH3_state_s)
#  define XXH3_state_t  XXH_IPREF(XXH3_state_t)
#  define XXH128_hash_t XXH_IPREF(XXH128_hash_t)
   /* Ensure the header is parsed again, even if it was previously included */
#  undef XXHASH_H_5627135585666179
#  undef XXHASH_H_STATIC_13879238742
#endif /* XXH_INLINE_ALL || XXH_PRIVATE_API */



/* ****************************************************************
 *  Stable API
 *****************************************************************/
#ifndef XXHASH_H_5627135585666179
#define XXHASH_H_5627135585666179 1

/* specific declaration modes for Windows */
#if !defined(XXH_INLINE_ALL) && !defined(XXH_PRIVATE_API)
#  if defined(WIN32) && defined(_MSC_VER) && (defined(XXH_IMPORT) || defined(XXH_EXPORT))
#    ifdef XXH_EXPORT
#      define XXH_PUBLIC_API __declspec(dllexport)
#    elif XXH_IMPORT
#      define XXH_PUBLIC_API __declspec(dllimport)
#    endif
#  else
#    define XXH_PUBLIC_API   /* do nothing */
#  endif
#endif

/*!
 * XXH_NAMESPACE, aka Namespace Emulation:
 *
 * If you want to include _and expose_ xxHash functions from within your own
 * library, but also want to avoid symbol collisions with other libraries which
 * may also include xxHash, you can use XXH_NAMESPACE to automatically prefix
 * any public symbol from xxhash library with the value of XXH_NAMESPACE
 * (therefore, avoid empty or numeric values).
 *
 * Note that no change is required within the calling program as long as it
 * includes `xxhash.h`: Regular symbol names will be automatically translated
 * by this header.
 */
#ifdef XXH_NAMESPACE
#  define XXH_CAT(A,B) A##B
#  define XXH_NAME2(A,B) XXH_CAT(A,B)
#  define XXH_versionNumber XXH_NAME2(XXH_NAMESPACE, XXH_versionNumber)
/* XXH32 */
#  define XXH32 XXH_NAME2(XXH_NAMESPACE, XXH32)
#  define XXH32_createState XXH_NAME2(XXH_NAMESPACE, XXH32_createState)
#  define XXH32_freeState XXH_NAME2(XXH_NAMESPACE, XXH32_freeState)
#  define XXH32_reset XXH_NAME2(XXH_NAMESPACE, XXH32_reset)
#  define XXH32_update XXH_NAME2(XXH_NAMESPACE, XXH32_update)
#  define XXH32_digest XXH_NAME2(XXH_NAMESPACE, XXH32_digest)
#  define XXH32_copyState XXH_NAME2(XXH_NAMESPACE, XXH32_copyState)
#  define XXH32_canonicalFromHash XXH_NAME2(XXH_NAMESPACE, XXH32_canonicalFromHash)
#  define XXH32_hashFromCanonical XXH_NAME2(XXH_NAMESPACE, XXH32_hashFromCanonical)
/* XXH64 */
#  define XXH64 XXH_NAME2(XXH_NAMESPACE, XXH64)
#  define XXH64_createState XXH_NAME2(XXH_NAMESPACE, XXH64_createState)
#  define XXH64_freeState XXH_NAME2(XXH_NAMESPACE, XXH64_freeState)
#  define XXH64_reset XXH_NAME2(XXH_NAMESPACE, XXH64_reset)
#  define XXH64_update XXH_NAME2(XXH_NAMESPACE, XXH64_update)
#  define XXH64_digest XXH_NAME2(XXH_NAMESPACE, XXH64_digest)
#  define XXH64_copyState XXH_NAME2(XXH_NAMESPACE, XXH64_copyState)
#  define XXH64_canonicalFromHash XXH_NAME2(XXH_NAMESPACE, XXH64_canonicalFromHash)
#  define XXH64_hashFromCanonical XXH_NAME2(XXH_NAMESPACE, XXH64_hashFromCanonical)
/* XXH3_64bits */
#  define XXH3_64bits XXH_NAME2(XXH_NAMESPACE, XXH3_64bits)
#  define XXH3_64bits_withSecret XXH_NAME2(XXH_NAMESPACE, XXH3_64bits_withSecret)
#  define XXH3_64bits_withSeed XXH_NAME2(XXH_NAMESPACE, XXH3_64bits_withSeed)
#  define XXH3_createState XXH_NAME2(XXH_NAMESPACE, XXH3_createState)
#  define XXH3_freeState XXH_NAME2(XXH_NAMESPACE, XXH3_freeState)
#  define XXH3_copyState XXH_NAME2(XXH_NAMESPACE, XXH3_copyState)
#  define XXH3_64bits_reset XXH_NAME2(XXH_NAMESPACE, XXH3_64bits_reset)
#  define XXH3_64bits_reset_withSeed XXH_NAME2(XXH_NAMESPACE, XXH3_64bits_reset_withSeed)
#  define XXH3_64bits_reset_withSecret XXH_NAME2(XXH_NAMESPACE, XXH3_64bits_reset_withSecret)
#  define XXH3_64bits_update XXH_NAME2(XXH_NAMESPACE, XXH3_64bits_update)
#  define XXH3_64bits_digest XXH_NAME2(XXH_NAMESPACE, XXH3_64bits_digest)
#  define XXH3_generateSecret XXH_NAME2(XXH_NAMESPACE, XXH3_generateSecret)
/* XXH3_128bits */
#  define XXH128 XXH_NAME2(XXH_NAMESPACE, XXH128)
#  define XXH3_128bits XXH_NAME2(XXH_NAMESPACE, XXH3_128bits)
#  define XXH3_128bits_withSeed XXH_NAME2(XXH_NAMESPACE, XXH3_128bits_withSeed)
#  define XXH3_128bits_withSecret XXH_NAME2(XXH_NAMESPACE, XXH3_128bits_withSecret)
#  define XXH3_128bits_reset XXH_NAME2(XXH_NAMESPACE, XXH3_128bits_reset)
#  define XXH3_128bits_reset_withSeed XXH_NAME2(XXH_NAMESPACE, XXH3_128bits_reset_withSeed)
#  define XXH3_128bits_reset_withSecret XXH_NAME2(XXH_NAMESPACE, XXH3_128bits_reset_withSecret)
#  define XXH3_128bits_update XXH_NAME2(XXH_NAMESPACE, XXH3_128bits_update)
#  define XXH3_128bits_digest XXH_NAME2(XXH_NAMESPACE, XXH3_128bits_digest)
#  define XXH128_isEqual XXH_NAME2(XXH_NAMESPACE, XXH128_isEqual)
#  define XXH128_cmp     XXH_NAME2(XXH_NAMESPACE, XXH128_cmp)
#  define XXH128_canonicalFromHash XXH_NAME2(XXH_NAMESPACE, XXH128_canonicalFromHash)
#  define XXH128_hashFromCanonical XXH_NAME2(XXH_NAMESPACE, XXH128_hashFromCanonical)
#endif


/* *************************************
*  Version
***************************************/
#define XXH_VERSION_MAJOR    0
#define XXH_VERSION_MINOR    8
#define XXH_VERSION_RELEASE  0
#define XXH_VERSION_NUMBER  (XXH_VERSION_MAJOR *100*100 + XXH_VERSION_MINOR *100 + XXH_VERSION_RELEASE)
XXH_PUBLIC_API unsigned XXH_versionNumber (void);


/* ****************************
*  Definitions
******************************/
#include <stddef.h>   /* size_t */
typedef enum { XXH_OK=0, XXH_ERROR } XXH_errorcode;


/*-**********************************************************************
*  32-bit hash
************************************************************************/
#if !defined (__VMS) \
  && (defined (__cplusplus) \
  || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */) )
#   include <stdint.h>
    typedef uint32_t XXH32_hash_t;
#else
#   include <limits.h>
#   if UINT_MAX == 0xFFFFFFFFUL
      typedef unsigned int XXH32_hash_t;
#   else
#     if ULONG_MAX == 0xFFFFFFFFUL
        typedef unsigned long XXH32_hash_t;
#     else
#       error "unsupported platform: need a 32-bit type"
#     endif
#   endif
#endif

/*!
 * XXH32():
 *  Calculate the 32-bit hash of sequence "length" bytes stored at memory address "input".
 *  The memory between input & input+length must be valid (allocated and read-accessible).
 *  "seed" can be used to alter the result predictably.
 *  Speed on Core 2 Duo @ 3 GHz (single thread, SMHasher benchmark): 5.4 GB/s
 *
 * Note: XXH3 provides competitive speed for both 32-bit and 64-bit systems,
 * and offers true 64/128 bit hash results. It provides a superior level of
 * dispersion, and greatly reduces the risks of collisions.
 */
XXH_PUBLIC_API XXH32_hash_t XXH32 (const void* input, size_t length, XXH32_hash_t seed);

/*******   Streaming   *******/

/*
 * Streaming functions generate the xxHash value from an incrememtal input.
 * This method is slower than single-call functions, due to state management.
 * For small inputs, prefer `XXH32()` and `XXH64()`, which are better optimized.
 *
 * An XXH state must first be allocated using `XXH*_createState()`.
 *
 * Start a new hash by initializing the state with a seed using `XXH*_reset()`.
 *
 * Then, feed the hash state by calling `XXH*_update()` as many times as necessary.
 *
 * The function returns an error code, with 0 meaning OK, and any other value
 * meaning there is an error.
 *
 * Finally, a hash value can be produced anytime, by using `XXH*_digest()`.
 * This function returns the nn-bits hash as an int or long long.
 *
 * It's still possible to continue inserting input into the hash state after a
 * digest, and generate new hash values later on by invoking `XXH*_digest()`.
 *
 * When done, release the state using `XXH*_freeState()`.
 */

typedef struct XXH32_state_s XXH32_state_t;   /* incomplete type */
XXH_PUBLIC_API XXH32_state_t* XXH32_createState(void);
XXH_PUBLIC_API XXH_errorcode  XXH32_freeState(XXH32_state_t* statePtr);
XXH_PUBLIC_API void XXH32_copyState(XXH32_state_t* dst_state, const XXH32_state_t* src_state);

XXH_PUBLIC_API XXH_errorcode XXH32_reset  (XXH32_state_t* statePtr, XXH32_hash_t seed);
XXH_PUBLIC_API XXH_errorcode XXH32_update (XXH32_state_t* statePtr, const void* input, size_t length);
XXH_PUBLIC_API XXH32_hash_t  XXH32_digest (const XXH32_state_t* statePtr);

/*******   Canonical representation   *******/

/*
 * The default return values from XXH functions are unsigned 32 and 64 bit
 * integers.
 * This the simplest and fastest format for further post-processing.
 *
 * However, this leaves open the question of what is the order on the byte level,
 * since little and big endian conventions will store the same number differently.
 *
 * The canonical representation settles this issue by mandating big-endian
 * convention, the same convention as human-readable numbers (large digits first).
 *
 * When writing hash values to storage, sending them over a network, or printing
 * them, it's highly recommended to use the canonical representation to ensure
 * portability across a wider range of systems, present and future.
 *
 * The following functions allow transformation of hash values to and from
 * canonical format.
 */

typedef struct { unsigned char digest[4]; } XXH32_canonical_t;
XXH_PUBLIC_API void XXH32_canonicalFromHash(XXH32_canonical_t* dst, XXH32_hash_t hash);
XXH_PUBLIC_API XXH32_hash_t XXH32_hashFromCanonical(const XXH32_canonical_t* src);


#ifndef XXH_NO_LONG_LONG
/*-**********************************************************************
*  64-bit hash
************************************************************************/
#if !defined (__VMS) \
  && (defined (__cplusplus) \
  || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */) )
#   include <stdint.h>
    typedef uint64_t XXH64_hash_t;
#else
    /* the following type must have a width of 64-bit */
    typedef unsigned long long XXH64_hash_t;
#endif

/*!
 * XXH64():
 * Returns the 64-bit hash of sequence of length @length stored at memory
 * address @input.
 * @seed can be used to alter the result predictably.
 *
 * This function usually runs faster on 64-bit systems, but slower on 32-bit
 * systems (see benchmark).
 *
 * Note: XXH3 provides competitive speed for both 32-bit and 64-bit systems,
 * and offers true 64/128 bit hash results. It provides a superior level of
 * dispersion, and greatly reduces the risks of collisions.
 */
XXH_PUBLIC_API XXH64_hash_t XXH64 (const void* input, size_t length, XXH64_hash_t seed);

/*******   Streaming   *******/
typedef struct XXH64_state_s XXH64_state_t;   /* incomplete type */
XXH_PUBLIC_API XXH64_state_t* XXH64_createState(void);
XXH_PUBLIC_API XXH_errorcode  XXH64_freeState(XXH64_state_t* statePtr);
XXH_PUBLIC_API void XXH64_copyState(XXH64_state_t* dst_state, const XXH64_state_t* src_state);

XXH_PUBLIC_API XXH_errorcode XXH64_reset  (XXH64_state_t* statePtr, XXH64_hash_t seed);
XXH_PUBLIC_API XXH_errorcode XXH64_update (XXH64_state_t* statePtr, const void* input, size_t length);
XXH_PUBLIC_API XXH64_hash_t  XXH64_digest (const XXH64_state_t* statePtr);

/*******   Canonical representation   *******/
typedef struct { unsigned char digest[sizeof(XXH64_hash_t)]; } XXH64_canonical_t;
XXH_PUBLIC_API void XXH64_canonicalFromHash(XXH64_canonical_t* dst, XXH64_hash_t hash);
XXH_PUBLIC_API XXH64_hash_t XXH64_hashFromCanonical(const XXH64_canonical_t* src);


/*-**********************************************************************
*  XXH3 64-bit variant
************************************************************************/

/* ************************************************************************
 * XXH3 is a new hash algorithm featuring:
 *  - Improved speed for both small and large inputs
 *  - True 64-bit and 128-bit outputs
 *  - SIMD acceleration
 *  - Improved 32-bit viability
 *
 * Speed analysis methodology is explained here:
 *
 *    https://fastcompression.blogspot.com/2019/03/presenting-xxh3.html
 *
 * In general, expect XXH3 to run about ~2x faster on large inputs and >3x
 * faster on small ones compared to XXH64, though exact differences depend on
 * the platform.
 *
 * The algorithm is portable: Like XXH32 and XXH64, it generates the same hash
 * on all platforms.
 *
 * It benefits greatly from SIMD and 64-bit arithmetic, but does not require it.
 *
 * Almost all 32-bit and 64-bit targets that can run XXH32 smoothly can run
 * XXH3 at competitive speeds, even if XXH64 runs slowly. Further details are
 * explained in the implementation.
 *
 * Optimized implementations are provided for AVX512, AVX2, SSE2, NEON, POWER8,
 * ZVector and scalar targets. This can be controlled with the XXH_VECTOR macro.
 *
 * XXH3 offers 2 variants, _64bits and _128bits.
 * When only 64 bits are needed, prefer calling the _64bits variant, as it
 * reduces the amount of mixing, resulting in faster speed on small inputs.
 *
 * It's also generally simpler to manipulate a scalar return type than a struct.
 *
 * The 128-bit version adds additional strength, but it is slightly slower.
 *
 * The XXH3 algorithm is still in development.
 * The results it produces may still change in future versions.
 *
 * Results produced by v0.7.x are not comparable with results from v0.7.y.
 * However, the API is completely stable, and it can safely be used for
 * ephemeral data (local sessions).
 *
 * Avoid storing values in long-term storage until the algorithm is finalized.
 * XXH3's return values will be officially finalized upon reaching v0.8.0.
 *
 * After which, return values of XXH3 and XXH128 will no longer change in
 * future versions.
 *
 * The API supports one-shot hashing, streaming mode, and custom secrets.
 */

/* XXH3_64bits():
 * default 64-bit variant, using default secret and default seed of 0.
 * It's the fastest variant. */
XXH_PUBLIC_API XXH64_hash_t XXH3_64bits(const void* data, size_t len);

/*
 * XXH3_64bits_withSeed():
 * This variant generates a custom secret on the fly
 * based on default secret altered using the `seed` value.
 * While this operation is decently fast, note that it's not completely free.
 * Note: seed==0 produces the same results as XXH3_64bits().
 */
XXH_PUBLIC_API XXH64_hash_t XXH3_64bits_withSeed(const void* data, size_t len, XXH64_hash_t seed);

/*
 * XXH3_64bits_withSecret():
 * It's possible to provide any blob of bytes as a "secret" to generate the hash.
 * This makes it more difficult for an external actor to prepare an intentional collision.
 * The main condition is that secretSize *must* be large enough (>= XXH3_SECRET_SIZE_MIN).
 * However, the quality of produced hash values depends on secret's entropy.
 * Technically, the secret must look like a bunch of random bytes.
 * Avoid "trivial" or structured data such as repeated sequences or a text document.
 * Whenever unsure about the "randomness" of the blob of bytes,
 * consider relabelling it as a "custom seed" instead,
 * and employ "XXH3_generateSecret()" (see below)
 * to generate a high entropy secret derived from the custom seed.
 */
#define XXH3_SECRET_SIZE_MIN 136
XXH_PUBLIC_API XXH64_hash_t XXH3_64bits_withSecret(const void* data, size_t len, const void* secret, size_t secretSize);


/*******   Streaming   *******/
/*
 * Streaming requires state maintenance.
 * This operation costs memory and CPU.
 * As a consequence, streaming is slower than one-shot hashing.
 * For better performance, prefer one-shot functions whenever applicable.
 */
typedef struct XXH3_state_s XXH3_state_t;
XXH_PUBLIC_API XXH3_state_t* XXH3_createState(void);
XXH_PUBLIC_API XXH_errorcode XXH3_freeState(XXH3_state_t* statePtr);
XXH_PUBLIC_API void XXH3_copyState(XXH3_state_t* dst_state, const XXH3_state_t* src_state);

/*
 * XXH3_64bits_reset():
 * Initialize with default parameters.
 * digest will be equivalent to `XXH3_64bits()`.
 */
XXH_PUBLIC_API XXH_errorcode XXH3_64bits_reset(XXH3_state_t* statePtr);
/*
 * XXH3_64bits_reset_withSeed():
 * Generate a custom secret from `seed`, and store it into `statePtr`.
 * digest will be equivalent to `XXH3_64bits_withSeed()`.
 */
XXH_PUBLIC_API XXH_errorcode XXH3_64bits_reset_withSeed(XXH3_state_t* statePtr, XXH64_hash_t seed);
/*
 * XXH3_64bits_reset_withSecret():
 * `secret` is referenced, it _must outlive_ the hash streaming session.
 * Similar to one-shot API, `secretSize` must be >= `XXH3_SECRET_SIZE_MIN`,
 * and the quality of produced hash values depends on secret's entropy
 * (secret's content should look like a bunch of random bytes).
 * When in doubt about the randomness of a candidate `secret`,
 * consider employing `XXH3_generateSecret()` instead (see below).
 */
XXH_PUBLIC_API XXH_errorcode XXH3_64bits_reset_withSecret(XXH3_state_t* statePtr, const void* secret, size_t secretSize);

XXH_PUBLIC_API XXH_errorcode XXH3_64bits_update (XXH3_state_t* statePtr, const void* input, size_t length);
XXH_PUBLIC_API XXH64_hash_t  XXH3_64bits_digest (const XXH3_state_t* statePtr);

/* note : canonical representation of XXH3 is the same as XXH64
 * since they both produce XXH64_hash_t values */


/*-**********************************************************************
*  XXH3 128-bit variant
************************************************************************/

typedef struct {
 XXH64_hash_t low64;
 XXH64_hash_t high64;
} XXH128_hash_t;

XXH_PUBLIC_API XXH128_hash_t XXH3_128bits(const void* data, size_t len);
XXH_PUBLIC_API XXH128_hash_t XXH3_128bits_withSeed(const void* data, size_t len, XXH64_hash_t seed);
XXH_PUBLIC_API XXH128_hash_t XXH3_128bits_withSecret(const void* data, size_t len, const void* secret, size_t secretSize);

/*******   Streaming   *******/
/*
 * Streaming requires state maintenance.
 * This operation costs memory and CPU.
 * As a consequence, streaming is slower than one-shot hashing.
 * For better performance, prefer one-shot functions whenever applicable.
 *
 * XXH3_128bits uses the same XXH3_state_t as XXH3_64bits().
 * Use already declared XXH3_createState() and XXH3_freeState().
 *
 * All reset and streaming functions have same meaning as their 64-bit counterpart.
 */

XXH_PUBLIC_API XXH_errorcode XXH3_128bits_reset(XXH3_state_t* statePtr);
XXH_PUBLIC_API XXH_errorcode XXH3_128bits_reset_withSeed(XXH3_state_t* statePtr, XXH64_hash_t seed);
XXH_PUBLIC_API XXH_errorcode XXH3_128bits_reset_withSecret(XXH3_state_t* statePtr, const void* secret, size_t secretSize);

XXH_PUBLIC_API XXH_errorcode XXH3_128bits_update (XXH3_state_t* statePtr, const void* input, size_t length);
XXH_PUBLIC_API XXH128_hash_t XXH3_128bits_digest (const XXH3_state_t* statePtr);

/* Following helper functions make it possible to compare XXH128_hast_t values.
 * Since XXH128_hash_t is a structure, this capability is not offered by the language.
 * Note: For better performance, these functions can be inlined using XXH_INLINE_ALL */

/*!
 * XXH128_isEqual():
 * Return: 1 if `h1` and `h2` are equal, 0 if they are not.
 */
XXH_PUBLIC_API int XXH128_isEqual(XXH128_hash_t h1, XXH128_hash_t h2);

/*!
 * XXH128_cmp():
 *
 * This comparator is compatible with stdlib's `qsort()`/`bsearch()`.
 *
 * return: >0 if *h128_1  > *h128_2
 *         =0 if *h128_1 == *h128_2
 *         <0 if *h128_1  < *h128_2
 */
XXH_PUBLIC_API int XXH128_cmp(const void* h128_1, const void* h128_2);


/*******   Canonical representation   *******/
typedef struct { unsigned char digest[sizeof(XXH128_hash_t)]; } XXH128_canonical_t;
XXH_PUBLIC_API void XXH128_canonicalFromHash(XXH128_canonical_t* dst, XXH128_hash_t hash);
XXH_PUBLIC_API XXH128_hash_t XXH128_hashFromCanonical(const XXH128_canonical_t* src);


#endif  /* XXH_NO_LONG_LONG */

#endif /* XXHASH_H_5627135585666179 */



#if defined(XXH_STATIC_LINKING_ONLY) && !defined(XXHASH_H_STATIC_13879238742)
#define XXHASH_H_STATIC_13879238742
/* ****************************************************************************
 * This section contains declarations which are not guaranteed to remain stable.
 * They may change in future versions, becoming incompatible with a different
 * version of the library.
 * These declarations should only be used with static linking.
 * Never use them in association with dynamic linking!
 ***************************************************************************** */

/*
 * These definitions are only present to allow static allocation
 * of XXH states, on stack or in a struct, for example.
 * Never **ever** access their members directly.
 */

struct XXH32_state_s {
   XXH32_hash_t total_len_32;
   XXH32_hash_t large_len;
   XXH32_hash_t v1;
   XXH32_hash_t v2;
   XXH32_hash_t v3;
   XXH32_hash_t v4;
   XXH32_hash_t mem32[4];
   XXH32_hash_t memsize;
   XXH32_hash_t reserved;   /* never read nor write, might be removed in a future version */
};   /* typedef'd to XXH32_state_t */


#ifndef XXH_NO_LONG_LONG  /* defined when there is no 64-bit support */

struct XXH64_state_s {
   XXH64_hash_t total_len;
   XXH64_hash_t v1;
   XXH64_hash_t v2;
   XXH64_hash_t v3;
   XXH64_hash_t v4;
   XXH64_hash_t mem64[4];
   XXH32_hash_t memsize;
   XXH32_hash_t reserved32;  /* required for padding anyway */
   XXH64_hash_t reserved64;  /* never read nor write, might be removed in a future version */
};   /* typedef'd to XXH64_state_t */

#if defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)   /* C11+ */
#  include <stdalign.h>
#  define XXH_ALIGN(n)      alignas(n)
#elif defined(__GNUC__)
#  define XXH_ALIGN(n)      __attribute__ ((aligned(n)))
#elif defined(_MSC_VER)
#  define XXH_ALIGN(n)      __declspec(align(n))
#else
#  define XXH_ALIGN(n)   /* disabled */
#endif

/* Old GCC versions only accept the attribute after the type in structures. */
#if !(defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L))   /* C11+ */ \
    && defined(__GNUC__)
#   define XXH_ALIGN_MEMBER(align, type) type XXH_ALIGN(align)
#else
#   define XXH_ALIGN_MEMBER(align, type) XXH_ALIGN(align) type
#endif

#define XXH3_INTERNALBUFFER_SIZE 256
#define XXH3_SECRET_DEFAULT_SIZE 192
struct XXH3_state_s {
   XXH_ALIGN_MEMBER(64, XXH64_hash_t acc[8]);
   /* used to store a custom secret generated from a seed */
   XXH_ALIGN_MEMBER(64, unsigned char customSecret[XXH3_SECRET_DEFAULT_SIZE]);
   XXH_ALIGN_MEMBER(64, unsigned char buffer[XXH3_INTERNALBUFFER_SIZE]);
   XXH32_hash_t bufferedSize;
   XXH32_hash_t reserved32;
   size_t nbStripesSoFar;
   XXH64_hash_t totalLen;
   size_t nbStripesPerBlock;
   size_t secretLimit;
   XXH64_hash_t seed;
   XXH64_hash_t reserved64;
   const unsigned char* extSecret;  /* reference to external secret;
                                     * if == NULL, use .customSecret instead */
   /* note: there may be some padding at the end due to alignment on 64 bytes */
}; /* typedef'd to XXH3_state_t */

#undef XXH_ALIGN_MEMBER

/* When the XXH3_state_t structure is merely emplaced on stack,
 * it should be initialized with XXH3_INITSTATE() or a memset()
 * in case its first reset uses XXH3_NNbits_reset_withSeed().
 * This init can be omitted if the first reset uses default or _withSecret mode.
 * This operation isn't necessary when the state is created with XXH3_createState().
 * Note that this doesn't prepare the state for a streaming operation,
 * it's still necessary to use XXH3_NNbits_reset*() afterwards.
 */
#define XXH3_INITSTATE(XXH3_state_ptr)   { (XXH3_state_ptr)->seed = 0; }


/* ===   Experimental API   === */
/* Symbols defined below must be considered tied to a specific library version. */

/*
 * XXH3_generateSecret():
 *
 * Derive a high-entropy secret from any user-defined content, named customSeed.
 * The generated secret can be used in combination with `*_withSecret()` functions.
 * The `_withSecret()` variants are useful to provide a higher level of protection than 64-bit seed,
 * as it becomes much more difficult for an external actor to guess how to impact the calculation logic.
 *
 * The function accepts as input a custom seed of any length and any content,
 * and derives from it a high-entropy secret of length XXH3_SECRET_DEFAULT_SIZE
 * into an already allocated buffer secretBuffer.
 * The generated secret is _always_ XXH_SECRET_DEFAULT_SIZE bytes long.
 *
 * The generated secret can then be used with any `*_withSecret()` variant.
 * Functions `XXH3_128bits_withSecret()`, `XXH3_64bits_withSecret()`,
 * `XXH3_128bits_reset_withSecret()` and `XXH3_64bits_reset_withSecret()`
 * are part of this list. They all accept a `secret` parameter
 * which must be very long for implementation reasons (>= XXH3_SECRET_SIZE_MIN)
 * _and_ feature very high entropy (consist of random-looking bytes).
 * These conditions can be a high bar to meet, so
 * this function can be used to generate a secret of proper quality.
 *
 * customSeed can be anything. It can have any size, even small ones,
 * and its content can be anything, even stupidly "low entropy" source such as a bunch of zeroes.
 * The resulting `secret` will nonetheless provide all expected qualities.
 *
 * Supplying NULL as the customSeed copies the default secret into `secretBuffer`.
 * When customSeedSize > 0, supplying NULL as customSeed is undefined behavior.
 */
XXH_PUBLIC_API void XXH3_generateSecret(void* secretBuffer, const void* customSeed, size_t customSeedSize);


/* simple short-cut to pre-selected XXH3_128bits variant */
XXH_PUBLIC_API XXH128_hash_t XXH128(const void* data, size_t len, XXH64_hash_t seed);


#endif  /* XXH_NO_LONG_LONG */


#if defined(XXH_INLINE_ALL) || defined(XXH_PRIVATE_API)
#  define XXH_IMPLEMENTATION
#endif

#endif  /* defined(XXH_STATIC_LINKING_ONLY) && !defined(XXHASH_H_STATIC_13879238742) */


/* ======================================================================== */
/* ======================================================================== */
/* ======================================================================== */


/*-**********************************************************************
 * xxHash implementation
 *-**********************************************************************
 * xxHash's implementation used to be hosted inside xxhash.c.
 *
 * However, inlining requires implementation to be visible to the compiler,
 * hence be included alongside the header.
 * Previously, implementation was hosted inside xxhash.c,
 * which was then #included when inlining was activated.
 * This construction created issues with a few build and install systems,
 * as it required xxhash.c to be stored in /include directory.
 *
 * xxHash implementation is now directly integrated within xxhash.h.
 * As a consequence, xxhash.c is no longer needed in /include.
 *
 * xxhash.c is still available and is still useful.
 * In a "normal" setup, when xxhash is not inlined,
 * xxhash.h only exposes the prototypes and public symbols,
 * while xxhash.c can be built into an object file xxhash.o
 * which can then be linked into the final binary.
 ************************************************************************/

#if ( defined(XXH_INLINE_ALL) || defined(XXH_PRIVATE_API) \
   || defined(XXH_IMPLEMENTATION) ) && !defined(XXH_IMPLEM_13a8737387)
#  define XXH_IMPLEM_13a8737387

/* *************************************
*  Tuning parameters
***************************************/
/*!
 * XXH_FORCE_MEMORY_ACCESS:
 * By default, access to unaligned memory is controlled by `memcpy()`, which is
 * safe and portable.
 *
 * Unfortunately, on some target/compiler combinations, the generated assembly
 * is sub-optimal.
 *
 * The below switch allow selection of a different access method
 * in the search for improved performance.
 * Method 0 (default):
 *     Use `memcpy()`. Safe and portable. Default.
 * Method 1:
 *     `__attribute__((packed))` statement. It depends on compiler extensions
 *     and is therefore not portable.
 *     This method is safe if your compiler supports it, and *generally* as
 *     fast or faster than `memcpy`.
 * Method 2:
 *     Direct access via cast. This method doesn't depend on the compiler but
 *     violates the C standard.
 *     It can generate buggy code on targets which do not support unaligned
 *     memory accesses.
 *     But in some circumstances, it's the only known way to get the most
 *     performance (example: GCC + ARMv6)
 * Method 3:
 *     Byteshift. This can generate the best code on old compilers which don't
 *     inline small `memcpy()` calls, and it might also be faster on big-endian
 *     systems which lack a native byteswap instruction.
 * See https://stackoverflow.com/a/32095106/646947 for details.
 * Prefer these methods in priority order (0 > 1 > 2 > 3)
 */
#ifndef XXH_FORCE_MEMORY_ACCESS   /* can be defined externally, on command line for example */
#  if !defined(__clang__) && defined(__GNUC__) && defined(__ARM_FEATURE_UNALIGNED) && defined(__ARM_ARCH) && (__ARM_ARCH == 6)
#    define XXH_FORCE_MEMORY_ACCESS 2
#  elif !defined(__clang__) && ((defined(__INTEL_COMPILER) && !defined(_WIN32)) || \
  (defined(__GNUC__) && (defined(__ARM_ARCH) && __ARM_ARCH >= 7)))
#    define XXH_FORCE_MEMORY_ACCESS 1
#  endif
#endif

/*!
 * XXH_ACCEPT_NULL_INPUT_POINTER:
 * If the input pointer is NULL, xxHash's default behavior is to dereference it,
 * triggering a segfault.
 * When this macro is enabled, xxHash actively checks the input for a null pointer.
 * If it is, the result for null input pointers is the same as a zero-length input.
 */
#ifndef XXH_ACCEPT_NULL_INPUT_POINTER   /* can be defined externally */
#  define XXH_ACCEPT_NULL_INPUT_POINTER 0
#endif

/*!
 * XXH_FORCE_ALIGN_CHECK:
 * This is an important performance trick
 * for architectures without decent unaligned memory access performance.
 * It checks for input alignment, and when conditions are met,
 * uses a "fast path" employing direct 32-bit/64-bit read,
 * resulting in _dramatically faster_ read speed.
 *
 * The check costs one initial branch per hash, which is generally negligible, but not zero.
 * Moreover, it's not useful to generate binary for an additional code path
 * if memory access uses same instruction for both aligned and unaligned adresses.
 *
 * In these cases, the alignment check can be removed by setting this macro to 0.
 * Then the code will always use unaligned memory access.
 * Align check is automatically disabled on x86, x64 & arm64,
 * which are platforms known to offer good unaligned memory accesses performance.
 *
 * This option does not affect XXH3 (only XXH32 and XXH64).
 */
#ifndef XXH_FORCE_ALIGN_CHECK  /* can be defined externally */
#  if defined(__i386)  || defined(__x86_64__) || defined(__aarch64__) \
   || defined(_M_IX86) || defined(_M_X64)     || defined(_M_ARM64) /* visual */
#    define XXH_FORCE_ALIGN_CHECK 0
#  else
#    define XXH_FORCE_ALIGN_CHECK 1
#  endif
#endif

/*!
 * XXH_NO_INLINE_HINTS:
 *
 * By default, xxHash tries to force the compiler to inline almost all internal
 * functions.
 *
 * This can usually improve performance due to reduced jumping and improved
 * constant folding, but significantly increases the size of the binary which
 * might not be favorable.
 *
 * Additionally, sometimes the forced inlining can be detrimental to performance,
 * depending on the architecture.
 *
 * XXH_NO_INLINE_HINTS marks all internal functions as static, giving the
 * compiler full control on whether to inline or not.
 *
 * When not optimizing (-O0), optimizing for size (-Os, -Oz), or using
 * -fno-inline with GCC or Clang, this will automatically be defined.
 */
#ifndef XXH_NO_INLINE_HINTS
#  if defined(__OPTIMIZE_SIZE__) /* -Os, -Oz */ \
   || defined(__NO_INLINE__)     /* -O0, -fno-inline */
#    define XXH_NO_INLINE_HINTS 1
#  else
#    define XXH_NO_INLINE_HINTS 0
#  endif
#endif

/*!
 * XXH_REROLL:
 * Whether to reroll XXH32_finalize, and XXH64_finalize,
 * instead of using an unrolled jump table/if statement loop.
 *
 * This is automatically defined on -Os/-Oz on GCC and Clang.
 */
#ifndef XXH_REROLL
#  if defined(__OPTIMIZE_SIZE__)
#    define XXH_REROLL 1
#  else
#    define XXH_REROLL 0
#  endif
#endif


/* *************************************
*  Includes & Memory related functions
***************************************/
/*!
 * Modify the local functions below should you wish to use
 * different memory routines for malloc() and free()
 */
#include <stdlib.h>

static void* XXH_malloc(size_t s) { return malloc(s); }
static void XXH_free(void* p) { free(p); }

/*! and for memcpy() */
#include <string.h>
static void* XXH_memcpy(void* dest, const void* src, size_t size)
{
    return memcpy(dest,src,size);
}

#include <limits.h>   /* ULLONG_MAX */


/* *************************************
*  Compiler Specific Options
***************************************/
#ifdef _MSC_VER /* Visual Studio warning fix */
#  pragma warning(disable : 4127) /* disable: C4127: conditional expression is constant */
#endif

#if XXH_NO_INLINE_HINTS  /* disable inlining hints */
#  if defined(__GNUC__)
#    define XXH_FORCE_INLINE static __attribute__((unused))
#  else
#    define XXH_FORCE_INLINE static
#  endif
#  define XXH_NO_INLINE static
/* enable inlining hints */
#elif defined(_MSC_VER)  /* Visual Studio */
#  define XXH_FORCE_INLINE static __forceinline
#  define XXH_NO_INLINE static __declspec(noinline)
#elif defined(__GNUC__)
#  define XXH_FORCE_INLINE static __inline__ __attribute__((always_inline, unused))
#  define XXH_NO_INLINE static __attribute__((noinline))
#elif defined (__cplusplus) \
  || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L))   /* C99 */
#  define XXH_FORCE_INLINE static inline
#  define XXH_NO_INLINE static
#else
#  define XXH_FORCE_INLINE static
#  define XXH_NO_INLINE static
#endif



/* *************************************
*  Debug
***************************************/
/*
 * XXH_DEBUGLEVEL is expected to be defined externally, typically via the
 * compiler's command line options. The value must be a number.
 */
#ifndef XXH_DEBUGLEVEL
#  ifdef DEBUGLEVEL /* backwards compat */
#    define XXH_DEBUGLEVEL DEBUGLEVEL
#  else
#    define XXH_DEBUGLEVEL 0
#  endif
#endif

#if (XXH_DEBUGLEVEL>=1)
#  include <assert.h>   /* note: can still be disabled with NDEBUG */
#  define XXH_ASSERT(c)   assert(c)
#else
#  define XXH_ASSERT(c)   ((void)0)
#endif

/* note: use after variable declarations */
#define XXH_STATIC_ASSERT(c)  do { enum { XXH_sa = 1/(int)(!!(c)) }; } while (0)


/* *************************************
*  Basic Types
***************************************/
#if !defined (__VMS) \
 && (defined (__cplusplus) \
 || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */) )
# include <stdint.h>
  typedef uint8_t xxh_u8;
#else
  typedef unsigned char xxh_u8;
#endif
typedef XXH32_hash_t xxh_u32;

#ifdef XXH_OLD_NAMES
#  define BYTE xxh_u8
#  define U8   xxh_u8
#  define U32  xxh_u32
#endif

/* ***   Memory access   *** */

#if (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS==3))
/*
 * Manual byteshift. Best for old compilers which don't inline memcpy.
 * We actually directly use XXH_readLE32 and XXH_readBE32.
 */
#elif (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS==2))

/*
 * Force direct memory access. Only works on CPU which support unaligned memory
 * access in hardware.
 */
static xxh_u32 XXH_read32(const void* memPtr) { return *(const xxh_u32*) memPtr; }

#elif (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS==1))

/*
 * __pack instructions are safer but compiler specific, hence potentially
 * problematic for some compilers.
 *
 * Currently only defined for GCC and ICC.
 */
#ifdef XXH_OLD_NAMES
typedef union { xxh_u32 u32; } __attribute__((packed)) unalign;
#endif
static xxh_u32 XXH_read32(const void* ptr)
{
    typedef union { xxh_u32 u32; } __attribute__((packed)) xxh_unalign;
    return ((const xxh_unalign*)ptr)->u32;
}

#else

/*
 * Portable and safe solution. Generally efficient.
 * see: https://stackoverflow.com/a/32095106/646947
 */
static xxh_u32 XXH_read32(const void* memPtr)
{
    xxh_u32 val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}

#endif   /* XXH_FORCE_DIRECT_MEMORY_ACCESS */


/* ***   Endianess   *** */
typedef enum { XXH_bigEndian=0, XXH_littleEndian=1 } XXH_endianess;

/*!
 * XXH_CPU_LITTLE_ENDIAN:
 * Defined to 1 if the target is little endian, or 0 if it is big endian.
 * It can be defined externally, for example on the compiler command line.
 *
 * If it is not defined, a runtime check (which is usually constant folded)
 * is used instead.
 */
#ifndef XXH_CPU_LITTLE_ENDIAN
/*
 * Try to detect endianness automatically, to avoid the nonstandard behavior
 * in `XXH_isLittleEndian()`
 */
#  if defined(_WIN32) /* Windows is always little endian */ \
     || defined(__LITTLE_ENDIAN__) \
     || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#    define XXH_CPU_LITTLE_ENDIAN 1
#  elif defined(__BIG_ENDIAN__) \
     || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#    define XXH_CPU_LITTLE_ENDIAN 0
#  else
/*
 * runtime test, presumed to simplify to a constant by compiler
 */
static int XXH_isLittleEndian(void)
{
    /*
     * Portable and well-defined behavior.
     * Don't use static: it is detrimental to performance.
     */
    const union { xxh_u32 u; xxh_u8 c[4]; } one = { 1 };
    return one.c[0];
}
#   define XXH_CPU_LITTLE_ENDIAN   XXH_isLittleEndian()
#  endif
#endif




/* ****************************************
*  Compiler-specific Functions and Macros
******************************************/
#define XXH_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)

#ifdef __has_builtin
#  define XXH_HAS_BUILTIN(x) __has_builtin(x)
#else
#  define XXH_HAS_BUILTIN(x) 0
#endif

#if !defined(NO_CLANG_BUILTIN) && XXH_HAS_BUILTIN(__builtin_rotateleft32) \
                               && XXH_HAS_BUILTIN(__builtin_rotateleft64)
#  define XXH_rotl32 __builtin_rotateleft32
#  define XXH_rotl64 __builtin_rotateleft64
/* Note: although _rotl exists for minGW (GCC under windows), performance seems poor */
#elif defined(_MSC_VER)
#  define XXH_rotl32(x,r) _rotl(x,r)
#  define XXH_rotl64(x,r) _rotl64(x,r)
#else
#  define XXH_rotl32(x,r) (((x) << (r)) | ((x) >> (32 - (r))))
#  define XXH_rotl64(x,r) (((x) << (r)) | ((x) >> (64 - (r))))
#endif

#if defined(_MSC_VER)     /* Visual Studio */
#  define XXH_swap32 _byteswap_ulong
#elif XXH_GCC_VERSION >= 403
#  define XXH_swap32 __builtin_bswap32
#else
static xxh_u32 XXH_swap32 (xxh_u32 x)
{
    return  ((x << 24) & 0xff000000 ) |
            ((x <<  8) & 0x00ff0000 ) |
            ((x >>  8) & 0x0000ff00 ) |
            ((x >> 24) & 0x000000ff );
}
#endif


/* ***************************
*  Memory reads
*****************************/
typedef enum { XXH_aligned, XXH_unaligned } XXH_alignment;

/*
 * XXH_FORCE_MEMORY_ACCESS==3 is an endian-independent byteshift load.
 *
 * This is ideal for older compilers which don't inline memcpy.
 */
#if (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS==3))

XXH_FORCE_INLINE xxh_u32 XXH_readLE32(const void* memPtr)
{
    const xxh_u8* bytePtr = (const xxh_u8 *)memPtr;
    return bytePtr[0]
         | ((xxh_u32)bytePtr[1] << 8)
         | ((xxh_u32)bytePtr[2] << 16)
         | ((xxh_u32)bytePtr[3] << 24);
}

XXH_FORCE_INLINE xxh_u32 XXH_readBE32(const void* memPtr)
{
    const xxh_u8* bytePtr = (const xxh_u8 *)memPtr;
    return bytePtr[3]
         | ((xxh_u32)bytePtr[2] << 8)
         | ((xxh_u32)bytePtr[1] << 16)
         | ((xxh_u32)bytePtr[0] << 24);
}

#else
XXH_FORCE_INLINE xxh_u32 XXH_readLE32(const void* ptr)
{
    return XXH_CPU_LITTLE_ENDIAN ? XXH_read32(ptr) : XXH_swap32(XXH_read32(ptr));
}

static xxh_u32 XXH_readBE32(const void* ptr)
{
    return XXH_CPU_LITTLE_ENDIAN ? XXH_swap32(XXH_read32(ptr)) : XXH_read32(ptr);
}
#endif

XXH_FORCE_INLINE xxh_u32
XXH_readLE32_align(const void* ptr, XXH_alignment align)
{
    if (align==XXH_unaligned) {
        return XXH_readLE32(ptr);
    } else {
        return XXH_CPU_LITTLE_ENDIAN ? *(const xxh_u32*)ptr : XXH_swap32(*(const xxh_u32*)ptr);
    }
}


/* *************************************
*  Misc
***************************************/
XXH_PUBLIC_API unsigned XXH_versionNumber (void) { return XXH_VERSION_NUMBER; }


/* *******************************************************************
*  32-bit hash functions
*********************************************************************/
static const xxh_u32 XXH_PRIME32_1 = 0x9E3779B1U;   /* 0b10011110001101110111100110110001 */
static const xxh_u32 XXH_PRIME32_2 = 0x85EBCA77U;   /* 0b10000101111010111100101001110111 */
static const xxh_u32 XXH_PRIME32_3 = 0xC2B2AE3DU;   /* 0b11000010101100101010111000111101 */
static const xxh_u32 XXH_PRIME32_4 = 0x27D4EB2FU;   /* 0b00100111110101001110101100101111 */
static const xxh_u32 XXH_PRIME32_5 = 0x165667B1U;   /* 0b00010110010101100110011110110001 */

#ifdef XXH_OLD_NAMES
#  define PRIME32_1 XXH_PRIME32_1
#  define PRIME32_2 XXH_PRIME32_2
#  define PRIME32_3 XXH_PRIME32_3
#  define PRIME32_4 XXH_PRIME32_4
#  define PRIME32_5 XXH_PRIME32_5
#endif

static xxh_u32 XXH32_round(xxh_u32 acc, xxh_u32 input)
{
    acc += input * XXH_PRIME32_2;
    acc  = XXH_rotl32(acc, 13);
    acc *= XXH_PRIME32_1;
#if defined(__GNUC__) && defined(__SSE4_1__) && !defined(XXH_ENABLE_AUTOVECTORIZE)
    /*
     * UGLY HACK:
     * This inline assembly hack forces acc into a normal register. This is the
     * only thing that prevents GCC and Clang from autovectorizing the XXH32
     * loop (pragmas and attributes don't work for some resason) without globally
     * disabling SSE4.1.
     *
     * The reason we want to avoid vectorization is because despite working on
     * 4 integers at a time, there are multiple factors slowing XXH32 down on
     * SSE4:
     * - There's a ridiculous amount of lag from pmulld (10 cycles of latency on
     *   newer chips!) making it slightly slower to multiply four integers at
     *   once compared to four integers independently. Even when pmulld was
     *   fastest, Sandy/Ivy Bridge, it is still not worth it to go into SSE
     *   just to multiply unless doing a long operation.
     *
     * - Four instructions are required to rotate,
     *      movqda tmp,  v // not required with VEX encoding
     *      pslld  tmp, 13 // tmp <<= 13
     *      psrld  v,   19 // x >>= 19
     *      por    v,  tmp // x |= tmp
     *   compared to one for scalar:
     *      roll   v, 13    // reliably fast across the board
     *      shldl  v, v, 13 // Sandy Bridge and later prefer this for some reason
     *
     * - Instruction level parallelism is actually more beneficial here because
     *   the SIMD actually serializes this operation: While v1 is rotating, v2
     *   can load data, while v3 can multiply. SSE forces them to operate
     *   together.
     *
     * How this hack works:
     * __asm__(""       // Declare an assembly block but don't declare any instructions
     *          :       // However, as an Input/Output Operand,
     *          "+r"    // constrain a read/write operand (+) as a general purpose register (r).
     *          (acc)   // and set acc as the operand
     * );
     *
     * Because of the 'r', the compiler has promised that seed will be in a
     * general purpose register and the '+' says that it will be 'read/write',
     * so it has to assume it has changed. It is like volatile without all the
     * loads and stores.
     *
     * Since the argument has to be in a normal register (not an SSE register),
     * each time XXH32_round is called, it is impossible to vectorize.
     */
    __asm__("" : "+r" (acc));
#endif
    return acc;
}

/* mix all bits */
static xxh_u32 XXH32_avalanche(xxh_u32 h32)
{
    h32 ^= h32 >> 15;
    h32 *= XXH_PRIME32_2;
    h32 ^= h32 >> 13;
    h32 *= XXH_PRIME32_3;
    h32 ^= h32 >> 16;
    return(h32);
}

#define XXH_get32bits(p) XXH_readLE32_align(p, align)

static xxh_u32
XXH32_finalize(xxh_u32 h32, const xxh_u8* ptr, size_t len, XXH_alignment align)
{
#define XXH_PROCESS1 do {                           \
    h32 += (*ptr++) * XXH_PRIME32_5;                \
    h32 = XXH_rotl32(h32, 11) * XXH_PRIME32_1;      \
} while (0)

#define XXH_PROCESS4 do {                           \
    h32 += XXH_get32bits(ptr) * XXH_PRIME32_3;      \
    ptr += 4;                                   \
    h32  = XXH_rotl32(h32, 17) * XXH_PRIME32_4;     \
} while (0)

    /* Compact rerolled version */
    if (XXH_REROLL) {
        len &= 15;
        while (len >= 4) {
            XXH_PROCESS4;
            len -= 4;
        }
        while (len > 0) {
            XXH_PROCESS1;
            --len;
        }
        return XXH32_avalanche(h32);
    } else {
         switch(len&15) /* or switch(bEnd - p) */ {
           case 12:      XXH_PROCESS4;
                         /* fallthrough */
           case 8:       XXH_PROCESS4;
                         /* fallthrough */
           case 4:       XXH_PROCESS4;
                         return XXH32_avalanche(h32);

           case 13:      XXH_PROCESS4;
                         /* fallthrough */
           case 9:       XXH_PROCESS4;
                         /* fallthrough */
           case 5:       XXH_PROCESS4;
                         XXH_PROCESS1;
                         return XXH32_avalanche(h32);

           case 14:      XXH_PROCESS4;
                         /* fallthrough */
           case 10:      XXH_PROCESS4;
                         /* fallthrough */
           case 6:       XXH_PROCESS4;
                         XXH_PROCESS1;
                         XXH_PROCESS1;
                         return XXH32_avalanche(h32);

           case 15:      XXH_PROCESS4;
                         /* fallthrough */
           case 11:      XXH_PROCESS4;
                         /* fallthrough */
           case 7:       XXH_PROCESS4;
                         /* fallthrough */
           case 3:       XXH_PROCESS1;
                         /* fallthrough */
           case 2:       XXH_PROCESS1;
                         /* fallthrough */
           case 1:       XXH_PROCESS1;
                         /* fallthrough */
           case 0:       return XXH32_avalanche(h32);
        }
        XXH_ASSERT(0);
        return h32;   /* reaching this point is deemed impossible */
    }
}

#ifdef XXH_OLD_NAMES
#  define PROCESS1 XXH_PROCESS1
#  define PROCESS4 XXH_PROCESS4
#else
#  undef XXH_PROCESS1
#  undef XXH_PROCESS4
#endif

XXH_FORCE_INLINE xxh_u32
XXH32_endian_align(const xxh_u8* input, size_t len, xxh_u32 seed, XXH_alignment align)
{
    const xxh_u8* bEnd = input + len;
    xxh_u32 h32;

#if defined(XXH_ACCEPT_NULL_INPUT_POINTER) && (XXH_ACCEPT_NULL_INPUT_POINTER>=1)
    if (input==NULL) {
        len=0;
        bEnd=input=(const xxh_u8*)(size_t)16;
    }
#endif

    if (len>=16) {
        const xxh_u8* const limit = bEnd - 15;
        xxh_u32 v1 = seed + XXH_PRIME32_1 + XXH_PRIME32_2;
        xxh_u32 v2 = seed + XXH_PRIME32_2;
        xxh_u32 v3 = seed + 0;
        xxh_u32 v4 = seed - XXH_PRIME32_1;

        do {
            v1 = XXH32_round(v1, XXH_get32bits(input)); input += 4;
            v2 = XXH32_round(v2, XXH_get32bits(input)); input += 4;
            v3 = XXH32_round(v3, XXH_get32bits(input)); input += 4;
            v4 = XXH32_round(v4, XXH_get32bits(input)); input += 4;
        } while (input < limit);

        h32 = XXH_rotl32(v1, 1)  + XXH_rotl32(v2, 7)
            + XXH_rotl32(v3, 12) + XXH_rotl32(v4, 18);
    } else {
        h32  = seed + XXH_PRIME32_5;
    }

    h32 += (xxh_u32)len;

    return XXH32_finalize(h32, input, len&15, align);
}


XXH_PUBLIC_API XXH32_hash_t XXH32 (const void* input, size_t len, XXH32_hash_t seed)
{
#if 0
    /* Simple version, good for code maintenance, but unfortunately slow for small inputs */
    XXH32_state_t state;
    XXH32_reset(&state, seed);
    XXH32_update(&state, (const xxh_u8*)input, len);
    return XXH32_digest(&state);

#else

    if (XXH_FORCE_ALIGN_CHECK) {
        if ((((size_t)input) & 3) == 0) {   /* Input is 4-bytes aligned, leverage the speed benefit */
            return XXH32_endian_align((const xxh_u8*)input, len, seed, XXH_aligned);
    }   }

    return XXH32_endian_align((const xxh_u8*)input, len, seed, XXH_unaligned);
#endif
}



/*******   Hash streaming   *******/

XXH_PUBLIC_API XXH32_state_t* XXH32_createState(void)
{
    return (XXH32_state_t*)XXH_malloc(sizeof(XXH32_state_t));
}
XXH_PUBLIC_API XXH_errorcode XXH32_freeState(XXH32_state_t* statePtr)
{
    XXH_free(statePtr);
    return XXH_OK;
}

XXH_PUBLIC_API void XXH32_copyState(XXH32_state_t* dstState, const XXH32_state_t* srcState)
{
    memcpy(dstState, srcState, sizeof(*dstState));
}

XXH_PUBLIC_API XXH_errorcode XXH32_reset(XXH32_state_t* statePtr, XXH32_hash_t seed)
{
    XXH32_state_t state;   /* using a local state to memcpy() in order to avoid strict-aliasing warnings */
    memset(&state, 0, sizeof(state));
    state.v1 = seed + XXH_PRIME32_1 + XXH_PRIME32_2;
    state.v2 = seed + XXH_PRIME32_2;
    state.v3 = seed + 0;
    state.v4 = seed - XXH_PRIME32_1;
    /* do not write into reserved, planned to be removed in a future version */
    memcpy(statePtr, &state, sizeof(state) - sizeof(state.reserved));
    return XXH_OK;
}


XXH_PUBLIC_API XXH_errorcode
XXH32_update(XXH32_state_t* state, const void* input, size_t len)
{
    if (input==NULL)
#if defined(XXH_ACCEPT_NULL_INPUT_POINTER) && (XXH_ACCEPT_NULL_INPUT_POINTER>=1)
        return XXH_OK;
#else
        return XXH_ERROR;
#endif

    {   const xxh_u8* p = (const xxh_u8*)input;
        const xxh_u8* const bEnd = p + len;

        state->total_len_32 += (XXH32_hash_t)len;
        state->large_len |= (XXH32_hash_t)((len>=16) | (state->total_len_32>=16));

        if (state->memsize + len < 16)  {   /* fill in tmp buffer */
            XXH_memcpy((xxh_u8*)(state->mem32) + state->memsize, input, len);
            state->memsize += (XXH32_hash_t)len;
            return XXH_OK;
        }

        if (state->memsize) {   /* some data left from previous update */
            XXH_memcpy((xxh_u8*)(state->mem32) + state->memsize, input, 16-state->memsize);
            {   const xxh_u32* p32 = state->mem32;
                state->v1 = XXH32_round(state->v1, XXH_readLE32(p32)); p32++;
                state->v2 = XXH32_round(state->v2, XXH_readLE32(p32)); p32++;
                state->v3 = XXH32_round(state->v3, XXH_readLE32(p32)); p32++;
                state->v4 = XXH32_round(state->v4, XXH_readLE32(p32));
            }
            p += 16-state->memsize;
            state->memsize = 0;
        }

        if (p <= bEnd-16) {
            const xxh_u8* const limit = bEnd - 16;
            xxh_u32 v1 = state->v1;
            xxh_u32 v2 = state->v2;
            xxh_u32 v3 = state->v3;
            xxh_u32 v4 = state->v4;

            do {
                v1 = XXH32_round(v1, XXH_readLE32(p)); p+=4;
                v2 = XXH32_round(v2, XXH_readLE32(p)); p+=4;
                v3 = XXH32_round(v3, XXH_readLE32(p)); p+=4;
                v4 = XXH32_round(v4, XXH_readLE32(p)); p+=4;
            } while (p<=limit);

            state->v1 = v1;
            state->v2 = v2;
            state->v3 = v3;
            state->v4 = v4;
        }

        if (p < bEnd) {
            XXH_memcpy(state->mem32, p, (size_t)(bEnd-p));
            state->memsize = (unsigned)(bEnd-p);
        }
    }

    return XXH_OK;
}


XXH_PUBLIC_API XXH32_hash_t XXH32_digest (const XXH32_state_t* state)
{
    xxh_u32 h32;

    if (state->large_len) {
        h32 = XXH_rotl32(state->v1, 1)
            + XXH_rotl32(state->v2, 7)
            + XXH_rotl32(state->v3, 12)
            + XXH_rotl32(state->v4, 18);
    } else {
        h32 = state->v3 /* == seed */ + XXH_PRIME32_5;
    }

    h32 += state->total_len_32;

    return XXH32_finalize(h32, (const xxh_u8*)state->mem32, state->memsize, XXH_aligned);
}


/*******   Canonical representation   *******/

/*
 * The default return values from XXH functions are unsigned 32 and 64 bit
 * integers.
 *
 * The canonical representation uses big endian convention, the same convention
 * as human-readable numbers (large digits first).
 *
 * This way, hash values can be written into a file or buffer, remaining
 * comparable across different systems.
 *
 * The following functions allow transformation of hash values to and from their
 * canonical format.
 */
XXH_PUBLIC_API void XXH32_canonicalFromHash(XXH32_canonical_t* dst, XXH32_hash_t hash)
{
    XXH_STATIC_ASSERT(sizeof(XXH32_canonical_t) == sizeof(XXH32_hash_t));
    if (XXH_CPU_LITTLE_ENDIAN) hash = XXH_swap32(hash);
    memcpy(dst, &hash, sizeof(*dst));
}

XXH_PUBLIC_API XXH32_hash_t XXH32_hashFromCanonical(const XXH32_canonical_t* src)
{
    return XXH_readBE32(src);
}


#ifndef XXH_NO_LONG_LONG

/* *******************************************************************
*  64-bit hash functions
*********************************************************************/

/*******   Memory access   *******/

typedef XXH64_hash_t xxh_u64;

#ifdef XXH_OLD_NAMES
#  define U64 xxh_u64
#endif

/*!
 * XXH_REROLL_XXH64:
 * Whether to reroll the XXH64_finalize() loop.
 *
 * Just like XXH32, we can unroll the XXH64_finalize() loop. This can be a
 * performance gain on 64-bit hosts, as only one jump is required.
 *
 * However, on 32-bit hosts, because arithmetic needs to be done with two 32-bit
 * registers, and 64-bit arithmetic needs to be simulated, it isn't beneficial
 * to unroll. The code becomes ridiculously large (the largest function in the
 * binary on i386!), and rerolling it saves anywhere from 3kB to 20kB. It is
 * also slightly faster because it fits into cache better and is more likely
 * to be inlined by the compiler.
 *
 * If XXH_REROLL is defined, this is ignored and the loop is always rerolled.
 */
#ifndef XXH_REROLL_XXH64
#  if (defined(__ILP32__) || defined(_ILP32)) /* ILP32 is often defined on 32-bit GCC family */ \
   || !(defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64) /* x86-64 */ \
     || defined(_M_ARM64) || defined(__aarch64__) || defined(__arm64__) /* aarch64 */ \
     || defined(__PPC64__) || defined(__PPC64LE__) || defined(__ppc64__) || defined(__powerpc64__) /* ppc64 */ \
     || defined(__mips64__) || defined(__mips64)) /* mips64 */ \
   || (!defined(SIZE_MAX) || SIZE_MAX < ULLONG_MAX) /* check limits */
#    define XXH_REROLL_XXH64 1
#  else
#    define XXH_REROLL_XXH64 0
#  endif
#endif /* !defined(XXH_REROLL_XXH64) */

#if (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS==3))
/*
 * Manual byteshift. Best for old compilers which don't inline memcpy.
 * We actually directly use XXH_readLE64 and XXH_readBE64.
 */
#elif (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS==2))

/* Force direct memory access. Only works on CPU which support unaligned memory access in hardware */
static xxh_u64 XXH_read64(const void* memPtr) { return *(const xxh_u64*) memPtr; }

#elif (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS==1))

/*
 * __pack instructions are safer, but compiler specific, hence potentially
 * problematic for some compilers.
 *
 * Currently only defined for GCC and ICC.
 */
#ifdef XXH_OLD_NAMES
typedef union { xxh_u32 u32; xxh_u64 u64; } __attribute__((packed)) unalign64;
#endif
static xxh_u64 XXH_read64(const void* ptr)
{
    typedef union { xxh_u32 u32; xxh_u64 u64; } __attribute__((packed)) xxh_unalign64;
    return ((const xxh_unalign64*)ptr)->u64;
}

#else

/*
 * Portable and safe solution. Generally efficient.
 * see: https://stackoverflow.com/a/32095106/646947
 */
static xxh_u64 XXH_read64(const void* memPtr)
{
    xxh_u64 val;
    memcpy(&val, memPtr, sizeof(val));
    return val;
}

#endif   /* XXH_FORCE_DIRECT_MEMORY_ACCESS */

#if defined(_MSC_VER)     /* Visual Studio */
#  define XXH_swap64 _byteswap_uint64
#elif XXH_GCC_VERSION >= 403
#  define XXH_swap64 __builtin_bswap64
#else
static xxh_u64 XXH_swap64 (xxh_u64 x)
{
    return  ((x << 56) & 0xff00000000000000ULL) |
            ((x << 40) & 0x00ff000000000000ULL) |
            ((x << 24) & 0x0000ff0000000000ULL) |
            ((x << 8)  & 0x000000ff00000000ULL) |
            ((x >> 8)  & 0x00000000ff000000ULL) |
            ((x >> 24) & 0x0000000000ff0000ULL) |
            ((x >> 40) & 0x000000000000ff00ULL) |
            ((x >> 56) & 0x00000000000000ffULL);
}
#endif


/* XXH_FORCE_MEMORY_ACCESS==3 is an endian-independent byteshift load. */
#if (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS==3))

XXH_FORCE_INLINE xxh_u64 XXH_readLE64(const void* memPtr)
{
    const xxh_u8* bytePtr = (const xxh_u8 *)memPtr;
    return bytePtr[0]
         | ((xxh_u64)bytePtr[1] << 8)
         | ((xxh_u64)bytePtr[2] << 16)
         | ((xxh_u64)bytePtr[3] << 24)
         | ((xxh_u64)bytePtr[4] << 32)
         | ((xxh_u64)bytePtr[5] << 40)
         | ((xxh_u64)bytePtr[6] << 48)
         | ((xxh_u64)bytePtr[7] << 56);
}

XXH_FORCE_INLINE xxh_u64 XXH_readBE64(const void* memPtr)
{
    const xxh_u8* bytePtr = (const xxh_u8 *)memPtr;
    return bytePtr[7]
         | ((xxh_u64)bytePtr[6] << 8)
         | ((xxh_u64)bytePtr[5] << 16)
         | ((xxh_u64)bytePtr[4] << 24)
         | ((xxh_u64)bytePtr[3] << 32)
         | ((xxh_u64)bytePtr[2] << 40)
         | ((xxh_u64)bytePtr[1] << 48)
         | ((xxh_u64)bytePtr[0] << 56);
}

#else
XXH_FORCE_INLINE xxh_u64 XXH_readLE64(const void* ptr)
{
    return XXH_CPU_LITTLE_ENDIAN ? XXH_read64(ptr) : XXH_swap64(XXH_read64(ptr));
}

static xxh_u64 XXH_readBE64(const void* ptr)
{
    return XXH_CPU_LITTLE_ENDIAN ? XXH_swap64(XXH_read64(ptr)) : XXH_read64(ptr);
}
#endif

XXH_FORCE_INLINE xxh_u64
XXH_readLE64_align(const void* ptr, XXH_alignment align)
{
    if (align==XXH_unaligned)
        return XXH_readLE64(ptr);
    else
        return XXH_CPU_LITTLE_ENDIAN ? *(const xxh_u64*)ptr : XXH_swap64(*(const xxh_u64*)ptr);
}


/*******   xxh64   *******/

static const xxh_u64 XXH_PRIME64_1 = 0x9E3779B185EBCA87ULL;   /* 0b1001111000110111011110011011000110000101111010111100101010000111 */
static const xxh_u64 XXH_PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;   /* 0b1100001010110010101011100011110100100111110101001110101101001111 */
static const xxh_u64 XXH_PRIME64_3 = 0x165667B19E3779F9ULL;   /* 0b0001011001010110011001111011000110011110001101110111100111111001 */
static const xxh_u64 XXH_PRIME64_4 = 0x85EBCA77C2B2AE63ULL;   /* 0b1000010111101011110010100111011111000010101100101010111001100011 */
static const xxh_u64 XXH_PRIME64_5 = 0x27D4EB2F165667C5ULL;   /* 0b0010011111010100111010110010111100010110010101100110011111000101 */

#ifdef XXH_OLD_NAMES
#  define PRIME64_1 XXH_PRIME64_1
#  define PRIME64_2 XXH_PRIME64_2
#  define PRIME64_3 XXH_PRIME64_3
#  define PRIME64_4 XXH_PRIME64_4
#  define PRIME64_5 XXH_PRIME64_5
#endif

static xxh_u64 XXH64_round(xxh_u64 acc, xxh_u64 input)
{
    acc += input * XXH_PRIME64_2;
    acc  = XXH_rotl64(acc, 31);
    acc *= XXH_PRIME64_1;
    return acc;
}

static xxh_u64 XXH64_mergeRound(xxh_u64 acc, xxh_u64 val)
{
    val  = XXH64_round(0, val);
    acc ^= val;
    acc  = acc * XXH_PRIME64_1 + XXH_PRIME64_4;
    return acc;
}

static xxh_u64 XXH64_avalanche(xxh_u64 h64)
{
    h64 ^= h64 >> 33;
    h64 *= XXH_PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= XXH_PRIME64_3;
    h64 ^= h64 >> 32;
    return h64;
}


#define XXH_get64bits(p) XXH_readLE64_align(p, align)

static xxh_u64
XXH64_finalize(xxh_u64 h64, const xxh_u8* ptr, size_t len, XXH_alignment align)
{
#define XXH_PROCESS1_64 do {                                   \
    h64 ^= (*ptr++) * XXH_PRIME64_5;                           \
    h64 = XXH_rotl64(h64, 11) * XXH_PRIME64_1;                 \
} while (0)

#define XXH_PROCESS4_64 do {                                   \
    h64 ^= (xxh_u64)(XXH_get32bits(ptr)) * XXH_PRIME64_1;      \
    ptr += 4;                                              \
    h64 = XXH_rotl64(h64, 23) * XXH_PRIME64_2 + XXH_PRIME64_3;     \
} while (0)

#define XXH_PROCESS8_64 do {                                   \
    xxh_u64 const k1 = XXH64_round(0, XXH_get64bits(ptr)); \
    ptr += 8;                                              \
    h64 ^= k1;                                             \
    h64  = XXH_rotl64(h64,27) * XXH_PRIME64_1 + XXH_PRIME64_4;     \
} while (0)

    /* Rerolled version for 32-bit targets is faster and much smaller. */
    if (XXH_REROLL || XXH_REROLL_XXH64) {
        len &= 31;
        while (len >= 8) {
            XXH_PROCESS8_64;
            len -= 8;
        }
        if (len >= 4) {
            XXH_PROCESS4_64;
            len -= 4;
        }
        while (len > 0) {
            XXH_PROCESS1_64;
            --len;
        }
         return  XXH64_avalanche(h64);
    } else {
        switch(len & 31) {
           case 24: XXH_PROCESS8_64;
                         /* fallthrough */
           case 16: XXH_PROCESS8_64;
                         /* fallthrough */
           case  8: XXH_PROCESS8_64;
                    return XXH64_avalanche(h64);

           case 28: XXH_PROCESS8_64;
                         /* fallthrough */
           case 20: XXH_PROCESS8_64;
                         /* fallthrough */
           case 12: XXH_PROCESS8_64;
                         /* fallthrough */
           case  4: XXH_PROCESS4_64;
                    return XXH64_avalanche(h64);

           case 25: XXH_PROCESS8_64;
                         /* fallthrough */
           case 17: XXH_PROCESS8_64;
                         /* fallthrough */
           case  9: XXH_PROCESS8_64;
                    XXH_PROCESS1_64;
                    return XXH64_avalanche(h64);

           case 29: XXH_PROCESS8_64;
                         /* fallthrough */
           case 21: XXH_PROCESS8_64;
                         /* fallthrough */
           case 13: XXH_PROCESS8_64;
                         /* fallthrough */
           case  5: XXH_PROCESS4_64;
                    XXH_PROCESS1_64;
                    return XXH64_avalanche(h64);

           case 26: XXH_PROCESS8_64;
                         /* fallthrough */
           case 18: XXH_PROCESS8_64;
                         /* fallthrough */
           case 10: XXH_PROCESS8_64;
                    XXH_PROCESS1_64;
                    XXH_PROCESS1_64;
                    return XXH64_avalanche(h64);

           case 30: XXH_PROCESS8_64;
                         /* fallthrough */
           case 22: XXH_PROCESS8_64;
                         /* fallthrough */
           case 14: XXH_PROCESS8_64;
                         /* fallthrough */
           case  6: XXH_PROCESS4_64;
                    XXH_PROCESS1_64;
                    XXH_PROCESS1_64;
                    return XXH64_avalanche(h64);

           case 27: XXH_PROCESS8_64;
                         /* fallthrough */
           case 19: XXH_PROCESS8_64;
                         /* fallthrough */
           case 11: XXH_PROCESS8_64;
                    XXH_PROCESS1_64;
                    XXH_PROCESS1_64;
                    XXH_PROCESS1_64;
                    return XXH64_avalanche(h64);

           case 31: XXH_PROCESS8_64;
                         /* fallthrough */
           case 23: XXH_PROCESS8_64;
                         /* fallthrough */
           case 15: XXH_PROCESS8_64;
                         /* fallthrough */
           case  7: XXH_PROCESS4_64;
                         /* fallthrough */
           case  3: XXH_PROCESS1_64;
                         /* fallthrough */
           case  2: XXH_PROCESS1_64;
                         /* fallthrough */
           case  1: XXH_PROCESS1_64;
                         /* fallthrough */
           case  0: return XXH64_avalanche(h64);
        }
    }
    /* impossible to reach */
    XXH_ASSERT(0);
    return 0;  /* unreachable, but some compilers complain without it */
}

#ifdef XXH_OLD_NAMES
#  define PROCESS1_64 XXH_PROCESS1_64
#  define PROCESS4_64 XXH_PROCESS4_64
#  define PROCESS8_64 XXH_PROCESS8_64
#else
#  undef XXH_PROCESS1_64
#  undef XXH_PROCESS4_64
#  undef XXH_PROCESS8_64
#endif

XXH_FORCE_INLINE xxh_u64
XXH64_endian_align(const xxh_u8* input, size_t len, xxh_u64 seed, XXH_alignment align)
{
    const xxh_u8* bEnd = input + len;
    xxh_u64 h64;

#if defined(XXH_ACCEPT_NULL_INPUT_POINTER) && (XXH_ACCEPT_NULL_INPUT_POINTER>=1)
    if (input==NULL) {
        len=0;
        bEnd=input=(const xxh_u8*)(size_t)32;
    }
#endif

    if (len>=32) {
        const xxh_u8* const limit = bEnd - 32;
        xxh_u64 v1 = seed + XXH_PRIME64_1 + XXH_PRIME64_2;
        xxh_u64 v2 = seed + XXH_PRIME64_2;
        xxh_u64 v3 = seed + 0;
        xxh_u64 v4 = seed - XXH_PRIME64_1;

        do {
            v1 = XXH64_round(v1, XXH_get64bits(input)); input+=8;
            v2 = XXH64_round(v2, XXH_get64bits(input)); input+=8;
            v3 = XXH64_round(v3, XXH_get64bits(input)); input+=8;
            v4 = XXH64_round(v4, XXH_get64bits(input)); input+=8;
        } while (input<=limit);

        h64 = XXH_rotl64(v1, 1) + XXH_rotl64(v2, 7) + XXH_rotl64(v3, 12) + XXH_rotl64(v4, 18);
        h64 = XXH64_mergeRound(h64, v1);
        h64 = XXH64_mergeRound(h64, v2);
        h64 = XXH64_mergeRound(h64, v3);
        h64 = XXH64_mergeRound(h64, v4);

    } else {
        h64  = seed + XXH_PRIME64_5;
    }

    h64 += (xxh_u64) len;

    return XXH64_finalize(h64, input, len, align);
}


XXH_PUBLIC_API XXH64_hash_t XXH64 (const void* input, size_t len, XXH64_hash_t seed)
{
#if 0
    /* Simple version, good for code maintenance, but unfortunately slow for small inputs */
    XXH64_state_t state;
    XXH64_reset(&state, seed);
    XXH64_update(&state, (const xxh_u8*)input, len);
    return XXH64_digest(&state);

#else

    if (XXH_FORCE_ALIGN_CHECK) {
        if ((((size_t)input) & 7)==0) {  /* Input is aligned, let's leverage the speed advantage */
            return XXH64_endian_align((const xxh_u8*)input, len, seed, XXH_aligned);
    }   }

    return XXH64_endian_align((const xxh_u8*)input, len, seed, XXH_unaligned);

#endif
}

/*******   Hash Streaming   *******/

XXH_PUBLIC_API XXH64_state_t* XXH64_createState(void)
{
    return (XXH64_state_t*)XXH_malloc(sizeof(XXH64_state_t));
}
XXH_PUBLIC_API XXH_errorcode XXH64_freeState(XXH64_state_t* statePtr)
{
    XXH_free(statePtr);
    return XXH_OK;
}

XXH_PUBLIC_API void XXH64_copyState(XXH64_state_t* dstState, const XXH64_state_t* srcState)
{
    memcpy(dstState, srcState, sizeof(*dstState));
}

XXH_PUBLIC_API XXH_errorcode XXH64_reset(XXH64_state_t* statePtr, XXH64_hash_t seed)
{
    XXH64_state_t state;   /* use a local state to memcpy() in order to avoid strict-aliasing warnings */
    memset(&state, 0, sizeof(state));
    state.v1 = seed + XXH_PRIME64_1 + XXH_PRIME64_2;
    state.v2 = seed + XXH_PRIME64_2;
    state.v3 = seed + 0;
    state.v4 = seed - XXH_PRIME64_1;
     /* do not write into reserved64, might be removed in a future version */
    memcpy(statePtr, &state, sizeof(state) - sizeof(state.reserved64));
    return XXH_OK;
}

XXH_PUBLIC_API XXH_errorcode
XXH64_update (XXH64_state_t* state, const void* input, size_t len)
{
    if (input==NULL)
#if defined(XXH_ACCEPT_NULL_INPUT_POINTER) && (XXH_ACCEPT_NULL_INPUT_POINTER>=1)
        return XXH_OK;
#else
        return XXH_ERROR;
#endif

    {   const xxh_u8* p = (const xxh_u8*)input;
        const xxh_u8* const bEnd = p + len;

        state->total_len += len;

        if (state->memsize + len < 32) {  /* fill in tmp buffer */
            XXH_memcpy(((xxh_u8*)state->mem64) + state->memsize, input, len);
            state->memsize += (xxh_u32)len;
            return XXH_OK;
        }

        if (state->memsize) {   /* tmp buffer is full */
            XXH_memcpy(((xxh_u8*)state->mem64) + state->memsize, input, 32-state->memsize);
            state->v1 = XXH64_round(state->v1, XXH_readLE64(state->mem64+0));
            state->v2 = XXH64_round(state->v2, XXH_readLE64(state->mem64+1));
            state->v3 = XXH64_round(state->v3, XXH_readLE64(state->mem64+2));
            state->v4 = XXH64_round(state->v4, XXH_readLE64(state->mem64+3));
            p += 32-state->memsize;
            state->memsize = 0;
        }

        if (p+32 <= bEnd) {
            const xxh_u8* const limit = bEnd - 32;
            xxh_u64 v1 = state->v1;
            xxh_u64 v2 = state->v2;
            xxh_u64 v3 = state->v3;
            xxh_u64 v4 = state->v4;

            do {
                v1 = XXH64_round(v1, XXH_readLE64(p)); p+=8;
                v2 = XXH64_round(v2, XXH_readLE64(p)); p+=8;
                v3 = XXH64_round(v3, XXH_readLE64(p)); p+=8;
                v4 = XXH64_round(v4, XXH_readLE64(p)); p+=8;
            } while (p<=limit);

            state->v1 = v1;
            state->v2 = v2;
            state->v3 = v3;
            state->v4 = v4;
        }

        if (p < bEnd) {
            XXH_memcpy(state->mem64, p, (size_t)(bEnd-p));
            state->memsize = (unsigned)(bEnd-p);
        }
    }

    return XXH_OK;
}


XXH_PUBLIC_API XXH64_hash_t XXH64_digest (const XXH64_state_t* state)
{
    xxh_u64 h64;

    if (state->total_len >= 32) {
        xxh_u64 const v1 = state->v1;
        xxh_u64 const v2 = state->v2;
        xxh_u64 const v3 = state->v3;
        xxh_u64 const v4 = state->v4;

        h64 = XXH_rotl64(v1, 1) + XXH_rotl64(v2, 7) + XXH_rotl64(v3, 12) + XXH_rotl64(v4, 18);
        h64 = XXH64_mergeRound(h64, v1);
        h64 = XXH64_mergeRound(h64, v2);
        h64 = XXH64_mergeRound(h64, v3);
        h64 = XXH64_mergeRound(h64, v4);
    } else {
        h64  = state->v3 /*seed*/ + XXH_PRIME64_5;
    }

    h64 += (xxh_u64) state->total_len;

    return XXH64_finalize(h64, (const xxh_u8*)state->mem64, (size_t)state->total_len, XXH_aligned);
}


/******* Canonical representation   *******/

XXH_PUBLIC_API void XXH64_canonicalFromHash(XXH64_canonical_t* dst, XXH64_hash_t hash)
{
    XXH_STATIC_ASSERT(sizeof(XXH64_canonical_t) == sizeof(XXH64_hash_t));
    if (XXH_CPU_LITTLE_ENDIAN) hash = XXH_swap64(hash);
    memcpy(dst, &hash, sizeof(*dst));
}

XXH_PUBLIC_API XXH64_hash_t XXH64_hashFromCanonical(const XXH64_canonical_t* src)
{
    return XXH_readBE64(src);
}



/* *********************************************************************
*  XXH3
*  New generation hash designed for speed on small keys and vectorization
************************************************************************ */

/* ===   Compiler specifics   === */

/* Patch from https://github.com/Cyan4973/xxHash/pull/498 */
#if ((defined(sun) || defined(__sun)) && __cplusplus) /* Solaris includes __STDC_VERSION__ with C++. Tested with GCC 5.5 */
#  define XXH_RESTRICT /* disable */
#elif defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199901L   /* >= C99 */
#  define XXH_RESTRICT   restrict
#else
/* Note: it might be useful to define __restrict or __restrict__ for some C++ compilers */
#  define XXH_RESTRICT   /* disable */
#endif

#if (defined(__GNUC__) && (__GNUC__ >= 3))  \
  || (defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 800)) \
  || defined(__clang__)
#    define XXH_likely(x) __builtin_expect(x, 1)
#    define XXH_unlikely(x) __builtin_expect(x, 0)
#else
#    define XXH_likely(x) (x)
#    define XXH_unlikely(x) (x)
#endif

#if defined(__GNUC__)
#  if defined(__AVX2__)
#    include <immintrin.h>
#  elif defined(__SSE2__)
#    include <emmintrin.h>
#  elif defined(__ARM_NEON__) || defined(__ARM_NEON)
#    define inline __inline__  /* circumvent a clang bug */
#    include <arm_neon.h>
#    undef inline
#  endif
#elif defined(_MSC_VER)
#  include <intrin.h>
#endif

/*
 * One goal of XXH3 is to make it fast on both 32-bit and 64-bit, while
 * remaining a true 64-bit/128-bit hash function.
 *
 * This is done by prioritizing a subset of 64-bit operations that can be
 * emulated without too many steps on the average 32-bit machine.
 *
 * For example, these two lines seem similar, and run equally fast on 64-bit:
 *
 *   xxh_u64 x;
 *   x ^= (x >> 47); // good
 *   x ^= (x >> 13); // bad
 *
 * However, to a 32-bit machine, there is a major difference.
 *
 * x ^= (x >> 47) looks like this:
 *
 *   x.lo ^= (x.hi >> (47 - 32));
 *
 * while x ^= (x >> 13) looks like this:
 *
 *   // note: funnel shifts are not usually cheap.
 *   x.lo ^= (x.lo >> 13) | (x.hi << (32 - 13));
 *   x.hi ^= (x.hi >> 13);
 *
 * The first one is significantly faster than the second, simply because the
 * shift is larger than 32. This means:
 *  - All the bits we need are in the upper 32 bits, so we can ignore the lower
 *    32 bits in the shift.
 *  - The shift result will always fit in the lower 32 bits, and therefore,
 *    we can ignore the upper 32 bits in the xor.
 *
 * Thanks to this optimization, XXH3 only requires these features to be efficient:
 *
 *  - Usable unaligned access
 *  - A 32-bit or 64-bit ALU
 *      - If 32-bit, a decent ADC instruction
 *  - A 32 or 64-bit multiply with a 64-bit result
 *  - For the 128-bit variant, a decent byteswap helps short inputs.
 *
 * The first two are already required by XXH32, and almost all 32-bit and 64-bit
 * platforms which can run XXH32 can run XXH3 efficiently.
 *
 * Thumb-1, the classic 16-bit only subset of ARM's instruction set, is one
 * notable exception.
 *
 * First of all, Thumb-1 lacks support for the UMULL instruction which
 * performs the important long multiply. This means numerous __aeabi_lmul
 * calls.
 *
 * Second of all, the 8 functional registers are just not enough.
 * Setup for __aeabi_lmul, byteshift loads, pointers, and all arithmetic need
 * Lo registers, and this shuffling results in thousands more MOVs than A32.
 *
 * A32 and T32 don't have this limitation. They can access all 14 registers,
 * do a 32->64 multiply with UMULL, and the flexible operand allowing free
 * shifts is helpful, too.
 *
 * Therefore, we do a quick sanity check.
 *
 * If compiling Thumb-1 for a target which supports ARM instructions, we will
 * emit a warning, as it is not a "sane" platform to compile for.
 *
 * Usually, if this happens, it is because of an accident and you probably need
 * to specify -march, as you likely meant to compile for a newer architecture.
 *
 * Credit: large sections of the vectorial and asm source code paths
 *         have been contributed by @easyaspi314
 */
#if defined(__thumb__) && !defined(__thumb2__) && defined(__ARM_ARCH_ISA_ARM)
#   warning "XXH3 is highly inefficient without ARM or Thumb-2."
#endif

/* ==========================================
 * Vectorization detection
 * ========================================== */
#define XXH_SCALAR 0  /* Portable scalar version */
#define XXH_SSE2   1  /* SSE2 for Pentium 4 and all x86_64 */
#define XXH_AVX2   2  /* AVX2 for Haswell and Bulldozer */
#define XXH_AVX512 3  /* AVX512 for Skylake and Icelake */
#define XXH_NEON   4  /* NEON for most ARMv7-A and all AArch64 */
#define XXH_VSX    5  /* VSX and ZVector for POWER8/z13 */

#ifndef XXH_VECTOR    /* can be defined on command line */
#  if defined(__AVX512F__)
#    define XXH_VECTOR XXH_AVX512
#  elif defined(__AVX2__)
#    define XXH_VECTOR XXH_AVX2
#  elif defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64) || (defined(_M_IX86_FP) && (_M_IX86_FP == 2))
#    define XXH_VECTOR XXH_SSE2
#  elif defined(__GNUC__) /* msvc support maybe later */ \
  && (defined(__ARM_NEON__) || defined(__ARM_NEON)) \
  && (defined(__LITTLE_ENDIAN__) /* We only support little endian NEON */ \
    || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
#    define XXH_VECTOR XXH_NEON
#  elif (defined(__PPC64__) && defined(__POWER8_VECTOR__)) \
     || (defined(__s390x__) && defined(__VEC__)) \
     && defined(__GNUC__) /* TODO: IBM XL */
#    define XXH_VECTOR XXH_VSX
#  else
#    define XXH_VECTOR XXH_SCALAR
#  endif
#endif

/*
 * Controls the alignment of the accumulator,
 * for compatibility with aligned vector loads, which are usually faster.
 */
#ifndef XXH_ACC_ALIGN
#  if defined(XXH_X86DISPATCH)
#     define XXH_ACC_ALIGN 64  /* for compatibility with avx512 */
#  elif XXH_VECTOR == XXH_SCALAR  /* scalar */
#     define XXH_ACC_ALIGN 8
#  elif XXH_VECTOR == XXH_SSE2  /* sse2 */
#     define XXH_ACC_ALIGN 16
#  elif XXH_VECTOR == XXH_AVX2  /* avx2 */
#     define XXH_ACC_ALIGN 32
#  elif XXH_VECTOR == XXH_NEON  /* neon */
#     define XXH_ACC_ALIGN 16
#  elif XXH_VECTOR == XXH_VSX   /* vsx */
#     define XXH_ACC_ALIGN 16
#  elif XXH_VECTOR == XXH_AVX512  /* avx512 */
#     define XXH_ACC_ALIGN 64
#  endif
#endif

#if defined(XXH_X86DISPATCH) || XXH_VECTOR == XXH_SSE2 \
    || XXH_VECTOR == XXH_AVX2 || XXH_VECTOR == XXH_AVX512
#  define XXH_SEC_ALIGN XXH_ACC_ALIGN
#else
#  define XXH_SEC_ALIGN 8
#endif

/*
 * UGLY HACK:
 * GCC usually generates the best code with -O3 for xxHash.
 *
 * However, when targeting AVX2, it is overzealous in its unrolling resulting
 * in code roughly 3/4 the speed of Clang.
 *
 * There are other issues, such as GCC splitting _mm256_loadu_si256 into
 * _mm_loadu_si128 + _mm256_inserti128_si256. This is an optimization which
 * only applies to Sandy and Ivy Bridge... which don't even support AVX2.
 *
 * That is why when compiling the AVX2 version, it is recommended to use either
 *   -O2 -mavx2 -march=haswell
 * or
 *   -O2 -mavx2 -mno-avx256-split-unaligned-load
 * for decent performance, or to use Clang instead.
 *
 * Fortunately, we can control the first one with a pragma that forces GCC into
 * -O2, but the other one we can't control without "failed to inline always
 * inline function due to target mismatch" warnings.
 */
#if XXH_VECTOR == XXH_AVX2 /* AVX2 */ \
  && defined(__GNUC__) && !defined(__clang__) /* GCC, not Clang */ \
  && defined(__OPTIMIZE__) && !defined(__OPTIMIZE_SIZE__) /* respect -O0 and -Os */
#  pragma GCC push_options
#  pragma GCC optimize("-O2")
#endif


#if XXH_VECTOR == XXH_NEON
/*
 * NEON's setup for vmlal_u32 is a little more complicated than it is on
 * SSE2, AVX2, and VSX.
 *
 * While PMULUDQ and VMULEUW both perform a mask, VMLAL.U32 performs an upcast.
 *
 * To do the same operation, the 128-bit 'Q' register needs to be split into
 * two 64-bit 'D' registers, performing this operation::
 *
 *   [                a                 |                 b                ]
 *            |              '---------. .--------'                |
 *            |                         x                          |
 *            |              .---------' '--------.                |
 *   [ a & 0xFFFFFFFF | b & 0xFFFFFFFF ],[    a >> 32     |     b >> 32    ]
 *
 * Due to significant changes in aarch64, the fastest method for aarch64 is
 * completely different than the fastest method for ARMv7-A.
 *
 * ARMv7-A treats D registers as unions overlaying Q registers, so modifying
 * D11 will modify the high half of Q5. This is similar to how modifying AH
 * will only affect bits 8-15 of AX on x86.
 *
 * VZIP takes two registers, and puts even lanes in one register and odd lanes
 * in the other.
 *
 * On ARMv7-A, this strangely modifies both parameters in place instead of
 * taking the usual 3-operand form.
 *
 * Therefore, if we want to do this, we can simply use a D-form VZIP.32 on the
 * lower and upper halves of the Q register to end up with the high and low
 * halves where we want - all in one instruction.
 *
 *   vzip.32   d10, d11       @ d10 = { d10[0], d11[0] }; d11 = { d10[1], d11[1] }
 *
 * Unfortunately we need inline assembly for this: Instructions modifying two
 * registers at once is not possible in GCC or Clang's IR, and they have to
 * create a copy.
 *
 * aarch64 requires a different approach.
 *
 * In order to make it easier to write a decent compiler for aarch64, many
 * quirks were removed, such as conditional execution.
 *
 * NEON was also affected by this.
 *
 * aarch64 cannot access the high bits of a Q-form register, and writes to a
 * D-form register zero the high bits, similar to how writes to W-form scalar
 * registers (or DWORD registers on x86_64) work.
 *
 * The formerly free vget_high intrinsics now require a vext (with a few
 * exceptions)
 *
 * Additionally, VZIP was replaced by ZIP1 and ZIP2, which are the equivalent
 * of PUNPCKL* and PUNPCKH* in SSE, respectively, in order to only modify one
 * operand.
 *
 * The equivalent of the VZIP.32 on the lower and upper halves would be this
 * mess:
 *
 *   ext     v2.4s, v0.4s, v0.4s, #2 // v2 = { v0[2], v0[3], v0[0], v0[1] }
 *   zip1    v1.2s, v0.2s, v2.2s     // v1 = { v0[0], v2[0] }
 *   zip2    v0.2s, v0.2s, v1.2s     // v0 = { v0[1], v2[1] }
 *
 * Instead, we use a literal downcast, vmovn_u64 (XTN), and vshrn_n_u64 (SHRN):
 *
 *   shrn    v1.2s, v0.2d, #32  // v1 = (uint32x2_t)(v0 >> 32);
 *   xtn     v0.2s, v0.2d       // v0 = (uint32x2_t)(v0 & 0xFFFFFFFF);
 *
 * This is available on ARMv7-A, but is less efficient than a single VZIP.32.
 */

/*
 * Function-like macro:
 * void XXH_SPLIT_IN_PLACE(uint64x2_t &in, uint32x2_t &outLo, uint32x2_t &outHi)
 * {
 *     outLo = (uint32x2_t)(in & 0xFFFFFFFF);
 *     outHi = (uint32x2_t)(in >> 32);
 *     in = UNDEFINED;
 * }
 */
# if !defined(XXH_NO_VZIP_HACK) /* define to disable */ \
   && defined(__GNUC__) \
   && !defined(__aarch64__) && !defined(__arm64__)
#  define XXH_SPLIT_IN_PLACE(in, outLo, outHi)                                              \
    do {                                                                                    \
      /* Undocumented GCC/Clang operand modifier: %e0 = lower D half, %f0 = upper D half */ \
      /* https://github.com/gcc-mirror/gcc/blob/38cf91e5/gcc/config/arm/arm.c#L22486 */     \
      /* https://github.com/llvm-mirror/llvm/blob/2c4ca683/lib/Target/ARM/ARMAsmPrinter.cpp#L399 */ \
      __asm__("vzip.32  %e0, %f0" : "+w" (in));                                             \
      (outLo) = vget_low_u32 (vreinterpretq_u32_u64(in));                                   \
      (outHi) = vget_high_u32(vreinterpretq_u32_u64(in));                                   \
   } while (0)
# else
#  define XXH_SPLIT_IN_PLACE(in, outLo, outHi)                                            \
    do {                                                                                  \
      (outLo) = vmovn_u64    (in);                                                        \
      (outHi) = vshrn_n_u64  ((in), 32);                                                  \
    } while (0)
# endif
#endif  /* XXH_VECTOR == XXH_NEON */

/*
 * VSX and Z Vector helpers.
 *
 * This is very messy, and any pull requests to clean this up are welcome.
 *
 * There are a lot of problems with supporting VSX and s390x, due to
 * inconsistent intrinsics, spotty coverage, and multiple endiannesses.
 */
#if XXH_VECTOR == XXH_VSX
#  if defined(__s390x__)
#    include <s390intrin.h>
#  else
/* gcc's altivec.h can have the unwanted consequence to unconditionally
 * #define bool, vector, and pixel keywords,
 * with bad consequences for programs already using these keywords for other purposes.
 * The paragraph defining these macros is skipped when __APPLE_ALTIVEC__ is defined.
 * __APPLE_ALTIVEC__ is _generally_ defined automatically by the compiler,
 * but it seems that, in some cases, it isn't.
 * Force the build macro to be defined, so that keywords are not altered.
 */
#    if defined(__GNUC__) && !defined(__APPLE_ALTIVEC__)
#      define __APPLE_ALTIVEC__
#    endif
#    include <altivec.h>
#  endif

typedef __vector unsigned long long xxh_u64x2;
typedef __vector unsigned char xxh_u8x16;
typedef __vector unsigned xxh_u32x4;

# ifndef XXH_VSX_BE
#  if defined(__BIG_ENDIAN__) \
  || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#    define XXH_VSX_BE 1
#  elif defined(__VEC_ELEMENT_REG_ORDER__) && __VEC_ELEMENT_REG_ORDER__ == __ORDER_BIG_ENDIAN__
#    warning "-maltivec=be is not recommended. Please use native endianness."
#    define XXH_VSX_BE 1
#  else
#    define XXH_VSX_BE 0
#  endif
# endif /* !defined(XXH_VSX_BE) */

# if XXH_VSX_BE
/* A wrapper for POWER9's vec_revb. */
#  if defined(__POWER9_VECTOR__) || (defined(__clang__) && defined(__s390x__))
#    define XXH_vec_revb vec_revb
#  else
XXH_FORCE_INLINE xxh_u64x2 XXH_vec_revb(xxh_u64x2 val)
{
    xxh_u8x16 const vByteSwap = { 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00,
                                  0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08 };
    return vec_perm(val, val, vByteSwap);
}
#  endif
# endif /* XXH_VSX_BE */

/*
 * Performs an unaligned load and byte swaps it on big endian.
 */
XXH_FORCE_INLINE xxh_u64x2 XXH_vec_loadu(const void *ptr)
{
    xxh_u64x2 ret;
    memcpy(&ret, ptr, sizeof(xxh_u64x2));
# if XXH_VSX_BE
    ret = XXH_vec_revb(ret);
# endif
    return ret;
}

/*
 * vec_mulo and vec_mule are very problematic intrinsics on PowerPC
 *
 * These intrinsics weren't added until GCC 8, despite existing for a while,
 * and they are endian dependent. Also, their meaning swap depending on version.
 * */
# if defined(__s390x__)
 /* s390x is always big endian, no issue on this platform */
#  define XXH_vec_mulo vec_mulo
#  define XXH_vec_mule vec_mule
# elif defined(__clang__) && XXH_HAS_BUILTIN(__builtin_altivec_vmuleuw)
/* Clang has a better way to control this, we can just use the builtin which doesn't swap. */
#  define XXH_vec_mulo __builtin_altivec_vmulouw
#  define XXH_vec_mule __builtin_altivec_vmuleuw
# else
/* gcc needs inline assembly */
/* Adapted from https://github.com/google/highwayhash/blob/master/highwayhash/hh_vsx.h. */
XXH_FORCE_INLINE xxh_u64x2 XXH_vec_mulo(xxh_u32x4 a, xxh_u32x4 b)
{
    xxh_u64x2 result;
    __asm__("vmulouw %0, %1, %2" : "=v" (result) : "v" (a), "v" (b));
    return result;
}
XXH_FORCE_INLINE xxh_u64x2 XXH_vec_mule(xxh_u32x4 a, xxh_u32x4 b)
{
    xxh_u64x2 result;
    __asm__("vmuleuw %0, %1, %2" : "=v" (result) : "v" (a), "v" (b));
    return result;
}
# endif /* XXH_vec_mulo, XXH_vec_mule */
#endif /* XXH_VECTOR == XXH_VSX */


/* prefetch
 * can be disabled, by declaring XXH_NO_PREFETCH build macro */
#if defined(XXH_NO_PREFETCH)
#  define XXH_PREFETCH(ptr)  (void)(ptr)  /* disabled */
#else
#  if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_I86))  /* _mm_prefetch() is not defined outside of x86/x64 */
#    include <mmintrin.h>   /* https://msdn.microsoft.com/fr-fr/library/84szxsww(v=vs.90).aspx */
#    define XXH_PREFETCH(ptr)  _mm_prefetch((const char*)(ptr), _MM_HINT_T0)
#  elif defined(__GNUC__) && ( (__GNUC__ >= 4) || ( (__GNUC__ == 3) && (__GNUC_MINOR__ >= 1) ) )
#    define XXH_PREFETCH(ptr)  __builtin_prefetch((ptr), 0 /* rw==read */, 3 /* locality */)
#  else
#    define XXH_PREFETCH(ptr) (void)(ptr)  /* disabled */
#  endif
#endif  /* XXH_NO_PREFETCH */


/* ==========================================
 * XXH3 default settings
 * ========================================== */

#define XXH_SECRET_DEFAULT_SIZE 192   /* minimum XXH3_SECRET_SIZE_MIN */

#if (XXH_SECRET_DEFAULT_SIZE < XXH3_SECRET_SIZE_MIN)
#  error "default keyset is not large enough"
#endif

/* Pseudorandom secret taken directly from FARSH */
XXH_ALIGN(64) static const xxh_u8 XXH3_kSecret[XXH_SECRET_DEFAULT_SIZE] = {
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c,
    0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
    0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21,
    0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c,
    0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3,
    0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8,
    0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d,
    0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64,
    0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
    0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
    0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce,
    0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
};


#ifdef XXH_OLD_NAMES
#  define kSecret XXH3_kSecret
#endif

/*
 * Calculates a 32-bit to 64-bit long multiply.
 *
 * Wraps __emulu on MSVC x86 because it tends to call __allmul when it doesn't
 * need to (but it shouldn't need to anyways, it is about 7 instructions to do
 * a 64x64 multiply...). Since we know that this will _always_ emit MULL, we
 * use that instead of the normal method.
 *
 * If you are compiling for platforms like Thumb-1 and don't have a better option,
 * you may also want to write your own long multiply routine here.
 *
 * XXH_FORCE_INLINE xxh_u64 XXH_mult32to64(xxh_u64 x, xxh_u64 y)
 * {
 *    return (x & 0xFFFFFFFF) * (y & 0xFFFFFFFF);
 * }
 */
#if defined(_MSC_VER) && defined(_M_IX86)
#    include <intrin.h>
#    define XXH_mult32to64(x, y) __emulu((unsigned)(x), (unsigned)(y))
#else
/*
 * Downcast + upcast is usually better than masking on older compilers like
 * GCC 4.2 (especially 32-bit ones), all without affecting newer compilers.
 *
 * The other method, (x & 0xFFFFFFFF) * (y & 0xFFFFFFFF), will AND both operands
 * and perform a full 64x64 multiply -- entirely redundant on 32-bit.
 */
#    define XXH_mult32to64(x, y) ((xxh_u64)(xxh_u32)(x) * (xxh_u64)(xxh_u32)(y))
#endif

/*
 * Calculates a 64->128-bit long multiply.
 *
 * Uses __uint128_t and _umul128 if available, otherwise uses a scalar version.
 */
static XXH128_hash_t
XXH_mult64to128(xxh_u64 lhs, xxh_u64 rhs)
{
    /*
     * GCC/Clang __uint128_t method.
     *
     * On most 64-bit targets, GCC and Clang define a __uint128_t type.
     * This is usually the best way as it usually uses a native long 64-bit
     * multiply, such as MULQ on x86_64 or MUL + UMULH on aarch64.
     *
     * Usually.
     *
     * Despite being a 32-bit platform, Clang (and emscripten) define this type
     * despite not having the arithmetic for it. This results in a laggy
     * compiler builtin call which calculates a full 128-bit multiply.
     * In that case it is best to use the portable one.
     * https://github.com/Cyan4973/xxHash/issues/211#issuecomment-515575677
     */
#if defined(__GNUC__) && !defined(__wasm__) \
    && defined(__SIZEOF_INT128__) \
    || (defined(_INTEGRAL_MAX_BITS) && _INTEGRAL_MAX_BITS >= 128)

    __uint128_t const product = (__uint128_t)lhs * (__uint128_t)rhs;
    XXH128_hash_t r128;
    r128.low64  = (xxh_u64)(product);
    r128.high64 = (xxh_u64)(product >> 64);
    return r128;

    /*
     * MSVC for x64's _umul128 method.
     *
     * xxh_u64 _umul128(xxh_u64 Multiplier, xxh_u64 Multiplicand, xxh_u64 *HighProduct);
     *
     * This compiles to single operand MUL on x64.
     */
#elif defined(_M_X64) || defined(_M_IA64)

#ifndef _MSC_VER
#   pragma intrinsic(_umul128)
#endif
    xxh_u64 product_high;
    xxh_u64 const product_low = _umul128(lhs, rhs, &product_high);
    XXH128_hash_t r128;
    r128.low64  = product_low;
    r128.high64 = product_high;
    return r128;

#else
    /*
     * Portable scalar method. Optimized for 32-bit and 64-bit ALUs.
     *
     * This is a fast and simple grade school multiply, which is shown below
     * with base 10 arithmetic instead of base 0x100000000.
     *
     *           9 3 // D2 lhs = 93
     *         x 7 5 // D2 rhs = 75
     *     ----------
     *           1 5 // D2 lo_lo = (93 % 10) * (75 % 10) = 15
     *         4 5 | // D2 hi_lo = (93 / 10) * (75 % 10) = 45
     *         2 1 | // D2 lo_hi = (93 % 10) * (75 / 10) = 21
     *     + 6 3 | | // D2 hi_hi = (93 / 10) * (75 / 10) = 63
     *     ---------
     *         2 7 | // D2 cross = (15 / 10) + (45 % 10) + 21 = 27
     *     + 6 7 | | // D2 upper = (27 / 10) + (45 / 10) + 63 = 67
     *     ---------
     *       6 9 7 5 // D4 res = (27 * 10) + (15 % 10) + (67 * 100) = 6975
     *
     * The reasons for adding the products like this are:
     *  1. It avoids manual carry tracking. Just like how
     *     (9 * 9) + 9 + 9 = 99, the same applies with this for UINT64_MAX.
     *     This avoids a lot of complexity.
     *
     *  2. It hints for, and on Clang, compiles to, the powerful UMAAL
     *     instruction available in ARM's Digital Signal Processing extension
     *     in 32-bit ARMv6 and later, which is shown below:
     *
     *         void UMAAL(xxh_u32 *RdLo, xxh_u32 *RdHi, xxh_u32 Rn, xxh_u32 Rm)
     *         {
     *             xxh_u64 product = (xxh_u64)*RdLo * (xxh_u64)*RdHi + Rn + Rm;
     *             *RdLo = (xxh_u32)(product & 0xFFFFFFFF);
     *             *RdHi = (xxh_u32)(product >> 32);
     *         }
     *
     *     This instruction was designed for efficient long multiplication, and
     *     allows this to be calculated in only 4 instructions at speeds
     *     comparable to some 64-bit ALUs.
     *
     *  3. It isn't terrible on other platforms. Usually this will be a couple
     *     of 32-bit ADD/ADCs.
     */

    /* First calculate all of the cross products. */
    xxh_u64 const lo_lo = XXH_mult32to64(lhs & 0xFFFFFFFF, rhs & 0xFFFFFFFF);
    xxh_u64 const hi_lo = XXH_mult32to64(lhs >> 32,        rhs & 0xFFFFFFFF);
    xxh_u64 const lo_hi = XXH_mult32to64(lhs & 0xFFFFFFFF, rhs >> 32);
    xxh_u64 const hi_hi = XXH_mult32to64(lhs >> 32,        rhs >> 32);

    /* Now add the products together. These will never overflow. */
    xxh_u64 const cross = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFF) + lo_hi;
    xxh_u64 const upper = (hi_lo >> 32) + (cross >> 32)        + hi_hi;
    xxh_u64 const lower = (cross << 32) | (lo_lo & 0xFFFFFFFF);

    XXH128_hash_t r128;
    r128.low64  = lower;
    r128.high64 = upper;
    return r128;
#endif
}

/*
 * Does a 64-bit to 128-bit multiply, then XOR folds it.
 *
 * The reason for the separate function is to prevent passing too many structs
 * around by value. This will hopefully inline the multiply, but we don't force it.
 */
static xxh_u64
XXH3_mul128_fold64(xxh_u64 lhs, xxh_u64 rhs)
{
    XXH128_hash_t product = XXH_mult64to128(lhs, rhs);
    return product.low64 ^ product.high64;
}

/* Seems to produce slightly better code on GCC for some reason. */
XXH_FORCE_INLINE xxh_u64 XXH_xorshift64(xxh_u64 v64, int shift)
{
    XXH_ASSERT(0 <= shift && shift < 64);
    return v64 ^ (v64 >> shift);
}

/*
 * This is a fast avalanche stage,
 * suitable when input bits are already partially mixed
 */
static XXH64_hash_t XXH3_avalanche(xxh_u64 h64)
{
    h64 = XXH_xorshift64(h64, 37);
    h64 *= 0x165667919E3779F9ULL;
    h64 = XXH_xorshift64(h64, 32);
    return h64;
}

/*
 * This is a stronger avalanche,
 * inspired by Pelle Evensen's rrmxmx
 * preferable when input has not been previously mixed
 */
static XXH64_hash_t XXH3_rrmxmx(xxh_u64 h64, xxh_u64 len)
{
    /* this mix is inspired by Pelle Evensen's rrmxmx */
    h64 ^= XXH_rotl64(h64, 49) ^ XXH_rotl64(h64, 24);
    h64 *= 0x9FB21C651E98DF25ULL;
    h64 ^= (h64 >> 35) + len ;
    h64 *= 0x9FB21C651E98DF25ULL;
    return XXH_xorshift64(h64, 28);
}


/* ==========================================
 * Short keys
 * ==========================================
 * One of the shortcomings of XXH32 and XXH64 was that their performance was
 * sub-optimal on short lengths. It used an iterative algorithm which strongly
 * favored lengths that were a multiple of 4 or 8.
 *
 * Instead of iterating over individual inputs, we use a set of single shot
 * functions which piece together a range of lengths and operate in constant time.
 *
 * Additionally, the number of multiplies has been significantly reduced. This
 * reduces latency, especially when emulating 64-bit multiplies on 32-bit.
 *
 * Depending on the platform, this may or may not be faster than XXH32, but it
 * is almost guaranteed to be faster than XXH64.
 */

/*
 * At very short lengths, there isn't enough input to fully hide secrets, or use
 * the entire secret.
 *
 * There is also only a limited amount of mixing we can do before significantly
 * impacting performance.
 *
 * Therefore, we use different sections of the secret and always mix two secret
 * samples with an XOR. This should have no effect on performance on the
 * seedless or withSeed variants because everything _should_ be constant folded
 * by modern compilers.
 *
 * The XOR mixing hides individual parts of the secret and increases entropy.
 *
 * This adds an extra layer of strength for custom secrets.
 */
XXH_FORCE_INLINE XXH64_hash_t
XXH3_len_1to3_64b(const xxh_u8* input, size_t len, const xxh_u8* secret, XXH64_hash_t seed)
{
    XXH_ASSERT(input != NULL);
    XXH_ASSERT(1 <= len && len <= 3);
    XXH_ASSERT(secret != NULL);
    /*
     * len = 1: combined = { input[0], 0x01, input[0], input[0] }
     * len = 2: combined = { input[1], 0x02, input[0], input[1] }
     * len = 3: combined = { input[2], 0x03, input[0], input[1] }
     */
    {   xxh_u8  const c1 = input[0];
        xxh_u8  const c2 = input[len >> 1];
        xxh_u8  const c3 = input[len - 1];
        xxh_u32 const combined = ((xxh_u32)c1 << 16) | ((xxh_u32)c2  << 24)
                               | ((xxh_u32)c3 <<  0) | ((xxh_u32)len << 8);
        xxh_u64 const bitflip = (XXH_readLE32(secret) ^ XXH_readLE32(secret+4)) + seed;
        xxh_u64 const keyed = (xxh_u64)combined ^ bitflip;
        return XXH64_avalanche(keyed);
    }
}

XXH_FORCE_INLINE XXH64_hash_t
XXH3_len_4to8_64b(const xxh_u8* input, size_t len, const xxh_u8* secret, XXH64_hash_t seed)
{
    XXH_ASSERT(input != NULL);
    XXH_ASSERT(secret != NULL);
    XXH_ASSERT(4 <= len && len < 8);
    seed ^= (xxh_u64)XXH_swap32((xxh_u32)seed) << 32;
    {   xxh_u32 const input1 = XXH_readLE32(input);
        xxh_u32 const input2 = XXH_readLE32(input + len - 4);
        xxh_u64 const bitflip = (XXH_readLE64(secret+8) ^ XXH_readLE64(secret+16)) - seed;
        xxh_u64 const input64 = input2 + (((xxh_u64)input1) << 32);
        xxh_u64 const keyed = input64 ^ bitflip;
        return XXH3_rrmxmx(keyed, len);
    }
}

XXH_FORCE_INLINE XXH64_hash_t
XXH3_len_9to16_64b(const xxh_u8* input, size_t len, const xxh_u8* secret, XXH64_hash_t seed)
{
    XXH_ASSERT(input != NULL);
    XXH_ASSERT(secret != NULL);
    XXH_ASSERT(8 <= len && len <= 16);
    {   xxh_u64 const bitflip1 = (XXH_readLE64(secret+24) ^ XXH_readLE64(secret+32)) + seed;
        xxh_u64 const bitflip2 = (XXH_readLE64(secret+40) ^ XXH_readLE64(secret+48)) - seed;
        xxh_u64 const input_lo = XXH_readLE64(input)           ^ bitflip1;
        xxh_u64 const input_hi = XXH_readLE64(input + len - 8) ^ bitflip2;
        xxh_u64 const acc = len
                          + XXH_swap64(input_lo) + input_hi
                          + XXH3_mul128_fold64(input_lo, input_hi);
        return XXH3_avalanche(acc);
    }
}

XXH_FORCE_INLINE XXH64_hash_t
XXH3_len_0to16_64b(const xxh_u8* input, size_t len, const xxh_u8* secret, XXH64_hash_t seed)
{
    XXH_ASSERT(len <= 16);
    {   if (XXH_likely(len >  8)) return XXH3_len_9to16_64b(input, len, secret, seed);
        if (XXH_likely(len >= 4)) return XXH3_len_4to8_64b(input, len, secret, seed);
        if (len) return XXH3_len_1to3_64b(input, len, secret, seed);
        return XXH64_avalanche(seed ^ (XXH_readLE64(secret+56) ^ XXH_readLE64(secret+64)));
    }
}

/*
 * DISCLAIMER: There are known *seed-dependent* multicollisions here due to
 * multiplication by zero, affecting hashes of lengths 17 to 240.
 *
 * However, they are very unlikely.
 *
 * Keep this in mind when using the unseeded XXH3_64bits() variant: As with all
 * unseeded non-cryptographic hashes, it does not attempt to defend itself
 * against specially crafted inputs, only random inputs.
 *
 * Compared to classic UMAC where a 1 in 2^31 chance of 4 consecutive bytes
 * cancelling out the secret is taken an arbitrary number of times (addressed
 * in XXH3_accumulate_512), this collision is very unlikely with random inputs
 * and/or proper seeding:
 *
 * This only has a 1 in 2^63 chance of 8 consecutive bytes cancelling out, in a
 * function that is only called up to 16 times per hash with up to 240 bytes of
 * input.
 *
 * This is not too bad for a non-cryptographic hash function, especially with
 * only 64 bit outputs.
 *
 * The 128-bit variant (which trades some speed for strength) is NOT affected
 * by this, although it is always a good idea to use a proper seed if you care
 * about strength.
 */
XXH_FORCE_INLINE xxh_u64 XXH3_mix16B(const xxh_u8* XXH_RESTRICT input,
                                     const xxh_u8* XXH_RESTRICT secret, xxh_u64 seed64)
{
#if defined(__GNUC__) && !defined(__clang__) /* GCC, not Clang */ \
  && defined(__i386__) && defined(__SSE2__)  /* x86 + SSE2 */ \
  && !defined(XXH_ENABLE_AUTOVECTORIZE)      /* Define to disable like XXH32 hack */
    /*
     * UGLY HACK:
     * GCC for x86 tends to autovectorize the 128-bit multiply, resulting in
     * slower code.
     *
     * By forcing seed64 into a register, we disrupt the cost model and
     * cause it to scalarize. See `XXH32_round()`
     *
     * FIXME: Clang's output is still _much_ faster -- On an AMD Ryzen 3600,
     * XXH3_64bits @ len=240 runs at 4.6 GB/s with Clang 9, but 3.3 GB/s on
     * GCC 9.2, despite both emitting scalar code.
     *
     * GCC generates much better scalar code than Clang for the rest of XXH3,
     * which is why finding a more optimal codepath is an interest.
     */
    __asm__ ("" : "+r" (seed64));
#endif
    {   xxh_u64 const input_lo = XXH_readLE64(input);
        xxh_u64 const input_hi = XXH_readLE64(input+8);
        return XXH3_mul128_fold64(
            input_lo ^ (XXH_readLE64(secret)   + seed64),
            input_hi ^ (XXH_readLE64(secret+8) - seed64)
        );
    }
}

/* For mid range keys, XXH3 uses a Mum-hash variant. */
XXH_FORCE_INLINE XXH64_hash_t
XXH3_len_17to128_64b(const xxh_u8* XXH_RESTRICT input, size_t len,
                     const xxh_u8* XXH_RESTRICT secret, size_t secretSize,
                     XXH64_hash_t seed)
{
    XXH_ASSERT(secretSize >= XXH3_SECRET_SIZE_MIN); (void)secretSize;
    XXH_ASSERT(16 < len && len <= 128);

    {   xxh_u64 acc = len * XXH_PRIME64_1;
        if (len > 32) {
            if (len > 64) {
                if (len > 96) {
                    acc += XXH3_mix16B(input+48, secret+96, seed);
                    acc += XXH3_mix16B(input+len-64, secret+112, seed);
                }
                acc += XXH3_mix16B(input+32, secret+64, seed);
                acc += XXH3_mix16B(input+len-48, secret+80, seed);
            }
            acc += XXH3_mix16B(input+16, secret+32, seed);
            acc += XXH3_mix16B(input+len-32, secret+48, seed);
        }
        acc += XXH3_mix16B(input+0, secret+0, seed);
        acc += XXH3_mix16B(input+len-16, secret+16, seed);

        return XXH3_avalanche(acc);
    }
}

#define XXH3_MIDSIZE_MAX 240

XXH_NO_INLINE XXH64_hash_t
XXH3_len_129to240_64b(const xxh_u8* XXH_RESTRICT input, size_t len,
                      const xxh_u8* XXH_RESTRICT secret, size_t secretSize,
                      XXH64_hash_t seed)
{
    XXH_ASSERT(secretSize >= XXH3_SECRET_SIZE_MIN); (void)secretSize;
    XXH_ASSERT(128 < len && len <= XXH3_MIDSIZE_MAX);

    #define XXH3_MIDSIZE_STARTOFFSET 3
    #define XXH3_MIDSIZE_LASTOFFSET  17

    {   xxh_u64 acc = len * XXH_PRIME64_1;
        int const nbRounds = (int)len / 16;
        int i;
        for (i=0; i<8; i++) {
            acc += XXH3_mix16B(input+(16*i), secret+(16*i), seed);
        }
        acc = XXH3_avalanche(acc);
        XXH_ASSERT(nbRounds >= 8);
#if defined(__clang__)                                /* Clang */ \
    && (defined(__ARM_NEON) || defined(__ARM_NEON__)) /* NEON */ \
    && !defined(XXH_ENABLE_AUTOVECTORIZE)             /* Define to disable */
        /*
         * UGLY HACK:
         * Clang for ARMv7-A tries to vectorize this loop, similar to GCC x86.
         * In everywhere else, it uses scalar code.
         *
         * For 64->128-bit multiplies, even if the NEON was 100% optimal, it
         * would still be slower than UMAAL (see XXH_mult64to128).
         *
         * Unfortunately, Clang doesn't handle the long multiplies properly and
         * converts them to the nonexistent "vmulq_u64" intrinsic, which is then
         * scalarized into an ugly mess of VMOV.32 instructions.
         *
         * This mess is difficult to avoid without turning autovectorization
         * off completely, but they are usually relatively minor and/or not
         * worth it to fix.
         *
         * This loop is the easiest to fix, as unlike XXH32, this pragma
         * _actually works_ because it is a loop vectorization instead of an
         * SLP vectorization.
         */
        #pragma clang loop vectorize(disable)
#endif
        for (i=8 ; i < nbRounds; i++) {
            acc += XXH3_mix16B(input+(16*i), secret+(16*(i-8)) + XXH3_MIDSIZE_STARTOFFSET, seed);
        }
        /* last bytes */
        acc += XXH3_mix16B(input + len - 16, secret + XXH3_SECRET_SIZE_MIN - XXH3_MIDSIZE_LASTOFFSET, seed);
        return XXH3_avalanche(acc);
    }
}


/* =======     Long Keys     ======= */

#define XXH_STRIPE_LEN 64
#define XXH_SECRET_CONSUME_RATE 8   /* nb of secret bytes consumed at each accumulation */
#define XXH_ACC_NB (XXH_STRIPE_LEN / sizeof(xxh_u64))

#ifdef XXH_OLD_NAMES
#  define STRIPE_LEN XXH_STRIPE_LEN
#  define ACC_NB XXH_ACC_NB
#endif

XXH_FORCE_INLINE void XXH_writeLE64(void* dst, xxh_u64 v64)
{
    if (!XXH_CPU_LITTLE_ENDIAN) v64 = XXH_swap64(v64);
    memcpy(dst, &v64, sizeof(v64));
}

/* Several intrinsic functions below are supposed to accept __int64 as argument,
 * as documented in https://software.intel.com/sites/landingpage/IntrinsicsGuide/ .
 * However, several environments do not define __int64 type,
 * requiring a workaround.
 */
#if !defined (__VMS) \
  && (defined (__cplusplus) \
  || (defined (__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) /* C99 */) )
    typedef int64_t xxh_i64;
#else
    /* the following type must have a width of 64-bit */
    typedef long long xxh_i64;
#endif

/*
 * XXH3_accumulate_512 is the tightest loop for long inputs, and it is the most optimized.
 *
 * It is a hardened version of UMAC, based off of FARSH's implementation.
 *
 * This was chosen because it adapts quite well to 32-bit, 64-bit, and SIMD
 * implementations, and it is ridiculously fast.
 *
 * We harden it by mixing the original input to the accumulators as well as the product.
 *
 * This means that in the (relatively likely) case of a multiply by zero, the
 * original input is preserved.
 *
 * On 128-bit inputs, we swap 64-bit pairs when we add the input to improve
 * cross-pollination, as otherwise the upper and lower halves would be
 * essentially independent.
 *
 * This doesn't matter on 64-bit hashes since they all get merged together in
 * the end, so we skip the extra step.
 *
 * Both XXH3_64bits and XXH3_128bits use this subroutine.
 */

#if (XXH_VECTOR == XXH_AVX512) || defined(XXH_X86DISPATCH)

#ifndef XXH_TARGET_AVX512
# define XXH_TARGET_AVX512  /* disable attribute target */
#endif

XXH_FORCE_INLINE XXH_TARGET_AVX512 void
XXH3_accumulate_512_avx512(void* XXH_RESTRICT acc,
                     const void* XXH_RESTRICT input,
                     const void* XXH_RESTRICT secret)
{
    XXH_ALIGN(64) __m512i* const xacc = (__m512i *) acc;
    XXH_ASSERT((((size_t)acc) & 63) == 0);
    XXH_STATIC_ASSERT(XXH_STRIPE_LEN == sizeof(__m512i));

    {
        /* data_vec    = input[0]; */
        __m512i const data_vec    = _mm512_loadu_si512   (input);
        /* key_vec     = secret[0]; */
        __m512i const key_vec     = _mm512_loadu_si512   (secret);
        /* data_key    = data_vec ^ key_vec; */
        __m512i const data_key    = _mm512_xor_si512     (data_vec, key_vec);
        /* data_key_lo = data_key >> 32; */
        __m512i const data_key_lo = _mm512_shuffle_epi32 (data_key, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 3, 0, 1));
        /* product     = (data_key & 0xffffffff) * (data_key_lo & 0xffffffff); */
        __m512i const product     = _mm512_mul_epu32     (data_key, data_key_lo);
        /* xacc[0] += swap(data_vec); */
        __m512i const data_swap = _mm512_shuffle_epi32(data_vec, (_MM_PERM_ENUM)_MM_SHUFFLE(1, 0, 3, 2));
        __m512i const sum       = _mm512_add_epi64(*xacc, data_swap);
        /* xacc[0] += product; */
        *xacc = _mm512_add_epi64(product, sum);
    }
}

/*
 * XXH3_scrambleAcc: Scrambles the accumulators to improve mixing.
 *
 * Multiplication isn't perfect, as explained by Google in HighwayHash:
 *
 *  // Multiplication mixes/scrambles bytes 0-7 of the 64-bit result to
 *  // varying degrees. In descending order of goodness, bytes
 *  // 3 4 2 5 1 6 0 7 have quality 228 224 164 160 100 96 36 32.
 *  // As expected, the upper and lower bytes are much worse.
 *
 * Source: https://github.com/google/highwayhash/blob/0aaf66b/highwayhash/hh_avx2.h#L291
 *
 * Since our algorithm uses a pseudorandom secret to add some variance into the
 * mix, we don't need to (or want to) mix as often or as much as HighwayHash does.
 *
 * This isn't as tight as XXH3_accumulate, but still written in SIMD to avoid
 * extraction.
 *
 * Both XXH3_64bits and XXH3_128bits use this subroutine.
 */

XXH_FORCE_INLINE XXH_TARGET_AVX512 void
XXH3_scrambleAcc_avx512(void* XXH_RESTRICT acc, const void* XXH_RESTRICT secret)
{
    XXH_ASSERT((((size_t)acc) & 63) == 0);
    XXH_STATIC_ASSERT(XXH_STRIPE_LEN == sizeof(__m512i));
    {   XXH_ALIGN(64) __m512i* const xacc = (__m512i*) acc;
        const __m512i prime32 = _mm512_set1_epi32((int)XXH_PRIME32_1);

        /* xacc[0] ^= (xacc[0] >> 47) */
        __m512i const acc_vec     = *xacc;
        __m512i const shifted     = _mm512_srli_epi64    (acc_vec, 47);
        __m512i const data_vec    = _mm512_xor_si512     (acc_vec, shifted);
        /* xacc[0] ^= secret; */
        __m512i const key_vec     = _mm512_loadu_si512   (secret);
        __m512i const data_key    = _mm512_xor_si512     (data_vec, key_vec);

        /* xacc[0] *= XXH_PRIME32_1; */
        __m512i const data_key_hi = _mm512_shuffle_epi32 (data_key, (_MM_PERM_ENUM)_MM_SHUFFLE(0, 3, 0, 1));
        __m512i const prod_lo     = _mm512_mul_epu32     (data_key, prime32);
        __m512i const prod_hi     = _mm512_mul_epu32     (data_key_hi, prime32);
        *xacc = _mm512_add_epi64(prod_lo, _mm512_slli_epi64(prod_hi, 32));
    }
}

XXH_FORCE_INLINE XXH_TARGET_AVX512 void
XXH3_initCustomSecret_avx512(void* XXH_RESTRICT customSecret, xxh_u64 seed64)
{
    XXH_STATIC_ASSERT((XXH_SECRET_DEFAULT_SIZE & 63) == 0);
    XXH_STATIC_ASSERT(XXH_SEC_ALIGN == 64);
    XXH_ASSERT(((size_t)customSecret & 63) == 0);
    (void)(&XXH_writeLE64);
    {   int const nbRounds = XXH_SECRET_DEFAULT_SIZE / sizeof(__m512i);
        __m512i const seed = _mm512_mask_set1_epi64(_mm512_set1_epi64((xxh_i64)seed64), 0xAA, -(xxh_i64)seed64);

        XXH_ALIGN(64) const __m512i* const src  = (const __m512i*) XXH3_kSecret;
        XXH_ALIGN(64)       __m512i* const dest = (      __m512i*) customSecret;
        int i;
        for (i=0; i < nbRounds; ++i) {
            /* GCC has a bug, _mm512_stream_load_si512 accepts 'void*', not 'void const*',
             * this will warn "discards ‘const’ qualifier". */
            union {
                XXH_ALIGN(64) const __m512i* cp;
                XXH_ALIGN(64) void* p;
            } remote_const_void;
            remote_const_void.cp = src + i;
            dest[i] = _mm512_add_epi64(_mm512_stream_load_si512(remote_const_void.p), seed);
    }   }
}

#endif

#if (XXH_VECTOR == XXH_AVX2) || defined(XXH_X86DISPATCH)

#ifndef XXH_TARGET_AVX2
# define XXH_TARGET_AVX2  /* disable attribute target */
#endif

XXH_FORCE_INLINE XXH_TARGET_AVX2 void
XXH3_accumulate_512_avx2( void* XXH_RESTRICT acc,
                    const void* XXH_RESTRICT input,
                    const void* XXH_RESTRICT secret)
{
    XXH_ASSERT((((size_t)acc) & 31) == 0);
    {   XXH_ALIGN(32) __m256i* const xacc    =       (__m256i *) acc;
        /* Unaligned. This is mainly for pointer arithmetic, and because
         * _mm256_loadu_si256 requires  a const __m256i * pointer for some reason. */
        const         __m256i* const xinput  = (const __m256i *) input;
        /* Unaligned. This is mainly for pointer arithmetic, and because
         * _mm256_loadu_si256 requires a const __m256i * pointer for some reason. */
        const         __m256i* const xsecret = (const __m256i *) secret;

        size_t i;
        for (i=0; i < XXH_STRIPE_LEN/sizeof(__m256i); i++) {
            /* data_vec    = xinput[i]; */
            __m256i const data_vec    = _mm256_loadu_si256    (xinput+i);
            /* key_vec     = xsecret[i]; */
            __m256i const key_vec     = _mm256_loadu_si256   (xsecret+i);
            /* data_key    = data_vec ^ key_vec; */
            __m256i const data_key    = _mm256_xor_si256     (data_vec, key_vec);
            /* data_key_lo = data_key >> 32; */
            __m256i const data_key_lo = _mm256_shuffle_epi32 (data_key, _MM_SHUFFLE(0, 3, 0, 1));
            /* product     = (data_key & 0xffffffff) * (data_key_lo & 0xffffffff); */
            __m256i const product     = _mm256_mul_epu32     (data_key, data_key_lo);
            /* xacc[i] += swap(data_vec); */
            __m256i const data_swap = _mm256_shuffle_epi32(data_vec, _MM_SHUFFLE(1, 0, 3, 2));
            __m256i const sum       = _mm256_add_epi64(xacc[i], data_swap);
            /* xacc[i] += product; */
            xacc[i] = _mm256_add_epi64(product, sum);
    }   }
}

XXH_FORCE_INLINE XXH_TARGET_AVX2 void
XXH3_scrambleAcc_avx2(void* XXH_RESTRICT acc, const void* XXH_RESTRICT secret)
{
    XXH_ASSERT((((size_t)acc) & 31) == 0);
    {   XXH_ALIGN(32) __m256i* const xacc = (__m256i*) acc;
        /* Unaligned. This is mainly for pointer arithmetic, and because
         * _mm256_loadu_si256 requires a const __m256i * pointer for some reason. */
        const         __m256i* const xsecret = (const __m256i *) secret;
        const __m256i prime32 = _mm256_set1_epi32((int)XXH_PRIME32_1);

        size_t i;
        for (i=0; i < XXH_STRIPE_LEN/sizeof(__m256i); i++) {
            /* xacc[i] ^= (xacc[i] >> 47) */
            __m256i const acc_vec     = xacc[i];
            __m256i const shifted     = _mm256_srli_epi64    (acc_vec, 47);
            __m256i const data_vec    = _mm256_xor_si256     (acc_vec, shifted);
            /* xacc[i] ^= xsecret; */
            __m256i const key_vec     = _mm256_loadu_si256   (xsecret+i);
            __m256i const data_key    = _mm256_xor_si256     (data_vec, key_vec);

            /* xacc[i] *= XXH_PRIME32_1; */
            __m256i const data_key_hi = _mm256_shuffle_epi32 (data_key, _MM_SHUFFLE(0, 3, 0, 1));
            __m256i const prod_lo     = _mm256_mul_epu32     (data_key, prime32);
            __m256i const prod_hi     = _mm256_mul_epu32     (data_key_hi, prime32);
            xacc[i] = _mm256_add_epi64(prod_lo, _mm256_slli_epi64(prod_hi, 32));
        }
    }
}

XXH_FORCE_INLINE XXH_TARGET_AVX2 void XXH3_initCustomSecret_avx2(void* XXH_RESTRICT customSecret, xxh_u64 seed64)
{
    XXH_STATIC_ASSERT((XXH_SECRET_DEFAULT_SIZE & 31) == 0);
    XXH_STATIC_ASSERT((XXH_SECRET_DEFAULT_SIZE / sizeof(__m256i)) == 6);
    XXH_STATIC_ASSERT(XXH_SEC_ALIGN <= 64);
    (void)(&XXH_writeLE64);
    XXH_PREFETCH(customSecret);
    {   __m256i const seed = _mm256_set_epi64x(-(xxh_i64)seed64, (xxh_i64)seed64, -(xxh_i64)seed64, (xxh_i64)seed64);

        XXH_ALIGN(64) const __m256i* const src  = (const __m256i*) XXH3_kSecret;
        XXH_ALIGN(64)       __m256i*       dest = (      __m256i*) customSecret;

#       if defined(__GNUC__) || defined(__clang__)
        /*
         * On GCC & Clang, marking 'dest' as modified will cause the compiler:
         *   - do not extract the secret from sse registers in the internal loop
         *   - use less common registers, and avoid pushing these reg into stack
         * The asm hack causes Clang to assume that XXH3_kSecretPtr aliases with
         * customSecret, and on aarch64, this prevented LDP from merging two
         * loads together for free. Putting the loads together before the stores
         * properly generates LDP.
         */
        __asm__("" : "+r" (dest));
#       endif

        /* GCC -O2 need unroll loop manually */
        dest[0] = _mm256_add_epi64(_mm256_stream_load_si256(src+0), seed);
        dest[1] = _mm256_add_epi64(_mm256_stream_load_si256(src+1), seed);
        dest[2] = _mm256_add_epi64(_mm256_stream_load_si256(src+2), seed);
        dest[3] = _mm256_add_epi64(_mm256_stream_load_si256(src+3), seed);
        dest[4] = _mm256_add_epi64(_mm256_stream_load_si256(src+4), seed);
        dest[5] = _mm256_add_epi64(_mm256_stream_load_si256(src+5), seed);
    }
}

#endif

#if (XXH_VECTOR == XXH_SSE2) || defined(XXH_X86DISPATCH)

#ifndef XXH_TARGET_SSE2
# define XXH_TARGET_SSE2  /* disable attribute target */
#endif

XXH_FORCE_INLINE XXH_TARGET_SSE2 void
XXH3_accumulate_512_sse2( void* XXH_RESTRICT acc,
                    const void* XXH_RESTRICT input,
                    const void* XXH_RESTRICT secret)
{
    /* SSE2 is just a half-scale version of the AVX2 version. */
    XXH_ASSERT((((size_t)acc) & 15) == 0);
    {   XXH_ALIGN(16) __m128i* const xacc    =       (__m128i *) acc;
        /* Unaligned. This is mainly for pointer arithmetic, and because
         * _mm_loadu_si128 requires a const __m128i * pointer for some reason. */
        const         __m128i* const xinput  = (const __m128i *) input;
        /* Unaligned. This is mainly for pointer arithmetic, and because
         * _mm_loadu_si128 requires a const __m128i * pointer for some reason. */
        const         __m128i* const xsecret = (const __m128i *) secret;

        size_t i;
        for (i=0; i < XXH_STRIPE_LEN/sizeof(__m128i); i++) {
            /* data_vec    = xinput[i]; */
            __m128i const data_vec    = _mm_loadu_si128   (xinput+i);
            /* key_vec     = xsecret[i]; */
            __m128i const key_vec     = _mm_loadu_si128   (xsecret+i);
            /* data_key    = data_vec ^ key_vec; */
            __m128i const data_key    = _mm_xor_si128     (data_vec, key_vec);
            /* data_key_lo = data_key >> 32; */
            __m128i const data_key_lo = _mm_shuffle_epi32 (data_key, _MM_SHUFFLE(0, 3, 0, 1));
            /* product     = (data_key & 0xffffffff) * (data_key_lo & 0xffffffff); */
            __m128i const product     = _mm_mul_epu32     (data_key, data_key_lo);
            /* xacc[i] += swap(data_vec); */
            __m128i const data_swap = _mm_shuffle_epi32(data_vec, _MM_SHUFFLE(1,0,3,2));
            __m128i const sum       = _mm_add_epi64(xacc[i], data_swap);
            /* xacc[i] += product; */
            xacc[i] = _mm_add_epi64(product, sum);
    }   }
}

XXH_FORCE_INLINE XXH_TARGET_SSE2 void
XXH3_scrambleAcc_sse2(void* XXH_RESTRICT acc, const void* XXH_RESTRICT secret)
{
    XXH_ASSERT((((size_t)acc) & 15) == 0);
    {   XXH_ALIGN(16) __m128i* const xacc = (__m128i*) acc;
        /* Unaligned. This is mainly for pointer arithmetic, and because
         * _mm_loadu_si128 requires a const __m128i * pointer for some reason. */
        const         __m128i* const xsecret = (const __m128i *) secret;
        const __m128i prime32 = _mm_set1_epi32((int)XXH_PRIME32_1);

        size_t i;
        for (i=0; i < XXH_STRIPE_LEN/sizeof(__m128i); i++) {
            /* xacc[i] ^= (xacc[i] >> 47) */
            __m128i const acc_vec     = xacc[i];
            __m128i const shifted     = _mm_srli_epi64    (acc_vec, 47);
            __m128i const data_vec    = _mm_xor_si128     (acc_vec, shifted);
            /* xacc[i] ^= xsecret[i]; */
            __m128i const key_vec     = _mm_loadu_si128   (xsecret+i);
            __m128i const data_key    = _mm_xor_si128     (data_vec, key_vec);

            /* xacc[i] *= XXH_PRIME32_1; */
            __m128i const data_key_hi = _mm_shuffle_epi32 (data_key, _MM_SHUFFLE(0, 3, 0, 1));
            __m128i const prod_lo     = _mm_mul_epu32     (data_key, prime32);
            __m128i const prod_hi     = _mm_mul_epu32     (data_key_hi, prime32);
            xacc[i] = _mm_add_epi64(prod_lo, _mm_slli_epi64(prod_hi, 32));
        }
    }
}

XXH_FORCE_INLINE XXH_TARGET_SSE2 void XXH3_initCustomSecret_sse2(void* XXH_RESTRICT customSecret, xxh_u64 seed64)
{
    XXH_STATIC_ASSERT((XXH_SECRET_DEFAULT_SIZE & 15) == 0);
    (void)(&XXH_writeLE64);
    {   int const nbRounds = XXH_SECRET_DEFAULT_SIZE / sizeof(__m128i);

#       if defined(_MSC_VER) && defined(_M_IX86) && _MSC_VER < 1900
        // MSVC 32bit mode does not support _mm_set_epi64x before 2015
        XXH_ALIGN(16) const xxh_i64 seed64x2[2] = { (xxh_i64)seed64, -(xxh_i64)seed64 };
        __m128i const seed = _mm_load_si128((__m128i const*)seed64x2);
#       else
        __m128i const seed = _mm_set_epi64x(-(xxh_i64)seed64, (xxh_i64)seed64);
#       endif
        int i;

        XXH_ALIGN(64)        const float* const src  = (float const*) XXH3_kSecret;
        XXH_ALIGN(XXH_SEC_ALIGN) __m128i*       dest = (__m128i*) customSecret;
#       if defined(__GNUC__) || defined(__clang__)
        /*
         * On GCC & Clang, marking 'dest' as modified will cause the compiler:
         *   - do not extract the secret from sse registers in the internal loop
         *   - use less common registers, and avoid pushing these reg into stack
         */
        __asm__("" : "+r" (dest));
#       endif

        for (i=0; i < nbRounds; ++i) {
            dest[i] = _mm_add_epi64(_mm_castps_si128(_mm_load_ps(src+i*4)), seed);
    }   }
}

#endif

#if (XXH_VECTOR == XXH_NEON)

XXH_FORCE_INLINE void
XXH3_accumulate_512_neon( void* XXH_RESTRICT acc,
                    const void* XXH_RESTRICT input,
                    const void* XXH_RESTRICT secret)
{
    XXH_ASSERT((((size_t)acc) & 15) == 0);
    {
        XXH_ALIGN(16) uint64x2_t* const xacc = (uint64x2_t *) acc;
        /* We don't use a uint32x4_t pointer because it causes bus errors on ARMv7. */
        uint8_t const* const xinput = (const uint8_t *) input;
        uint8_t const* const xsecret  = (const uint8_t *) secret;

        size_t i;
        for (i=0; i < XXH_STRIPE_LEN / sizeof(uint64x2_t); i++) {
            /* data_vec = xinput[i]; */
            uint8x16_t data_vec    = vld1q_u8(xinput  + (i * 16));
            /* key_vec  = xsecret[i];  */
            uint8x16_t key_vec     = vld1q_u8(xsecret + (i * 16));
            uint64x2_t data_key;
            uint32x2_t data_key_lo, data_key_hi;
            /* xacc[i] += swap(data_vec); */
            uint64x2_t const data64  = vreinterpretq_u64_u8(data_vec);
            uint64x2_t const swapped = vextq_u64(data64, data64, 1);
            xacc[i] = vaddq_u64 (xacc[i], swapped);
            /* data_key = data_vec ^ key_vec; */
            data_key = vreinterpretq_u64_u8(veorq_u8(data_vec, key_vec));
            /* data_key_lo = (uint32x2_t) (data_key & 0xFFFFFFFF);
             * data_key_hi = (uint32x2_t) (data_key >> 32);
             * data_key = UNDEFINED; */
            XXH_SPLIT_IN_PLACE(data_key, data_key_lo, data_key_hi);
            /* xacc[i] += (uint64x2_t) data_key_lo * (uint64x2_t) data_key_hi; */
            xacc[i] = vmlal_u32 (xacc[i], data_key_lo, data_key_hi);

        }
    }
}

XXH_FORCE_INLINE void
XXH3_scrambleAcc_neon(void* XXH_RESTRICT acc, const void* XXH_RESTRICT secret)
{
    XXH_ASSERT((((size_t)acc) & 15) == 0);

    {   uint64x2_t* xacc       = (uint64x2_t*) acc;
        uint8_t const* xsecret = (uint8_t const*) secret;
        uint32x2_t prime       = vdup_n_u32 (XXH_PRIME32_1);

        size_t i;
        for (i=0; i < XXH_STRIPE_LEN/sizeof(uint64x2_t); i++) {
            /* xacc[i] ^= (xacc[i] >> 47); */
            uint64x2_t acc_vec  = xacc[i];
            uint64x2_t shifted  = vshrq_n_u64 (acc_vec, 47);
            uint64x2_t data_vec = veorq_u64   (acc_vec, shifted);

            /* xacc[i] ^= xsecret[i]; */
            uint8x16_t key_vec  = vld1q_u8(xsecret + (i * 16));
            uint64x2_t data_key = veorq_u64(data_vec, vreinterpretq_u64_u8(key_vec));

            /* xacc[i] *= XXH_PRIME32_1 */
            uint32x2_t data_key_lo, data_key_hi;
            /* data_key_lo = (uint32x2_t) (xacc[i] & 0xFFFFFFFF);
             * data_key_hi = (uint32x2_t) (xacc[i] >> 32);
             * xacc[i] = UNDEFINED; */
            XXH_SPLIT_IN_PLACE(data_key, data_key_lo, data_key_hi);
            {   /*
                 * prod_hi = (data_key >> 32) * XXH_PRIME32_1;
                 *
                 * Avoid vmul_u32 + vshll_n_u32 since Clang 6 and 7 will
                 * incorrectly "optimize" this:
                 *   tmp     = vmul_u32(vmovn_u64(a), vmovn_u64(b));
                 *   shifted = vshll_n_u32(tmp, 32);
                 * to this:
                 *   tmp     = "vmulq_u64"(a, b); // no such thing!
                 *   shifted = vshlq_n_u64(tmp, 32);
                 *
                 * However, unlike SSE, Clang lacks a 64-bit multiply routine
                 * for NEON, and it scalarizes two 64-bit multiplies instead.
                 *
                 * vmull_u32 has the same timing as vmul_u32, and it avoids
                 * this bug completely.
                 * See https://bugs.llvm.org/show_bug.cgi?id=39967
                 */
                uint64x2_t prod_hi = vmull_u32 (data_key_hi, prime);
                /* xacc[i] = prod_hi << 32; */
                xacc[i] = vshlq_n_u64(prod_hi, 32);
                /* xacc[i] += (prod_hi & 0xFFFFFFFF) * XXH_PRIME32_1; */
                xacc[i] = vmlal_u32(xacc[i], data_key_lo, prime);
            }
    }   }
}

#endif

#if (XXH_VECTOR == XXH_VSX)

XXH_FORCE_INLINE void
XXH3_accumulate_512_vsx(  void* XXH_RESTRICT acc,
                    const void* XXH_RESTRICT input,
                    const void* XXH_RESTRICT secret)
{
          xxh_u64x2* const xacc     =       (xxh_u64x2*) acc;    /* presumed aligned */
    xxh_u64x2 const* const xinput   = (xxh_u64x2 const*) input;   /* no alignment restriction */
    xxh_u64x2 const* const xsecret  = (xxh_u64x2 const*) secret;    /* no alignment restriction */
    xxh_u64x2 const v32 = { 32, 32 };
    size_t i;
    for (i = 0; i < XXH_STRIPE_LEN / sizeof(xxh_u64x2); i++) {
        /* data_vec = xinput[i]; */
        xxh_u64x2 const data_vec = XXH_vec_loadu(xinput + i);
        /* key_vec = xsecret[i]; */
        xxh_u64x2 const key_vec  = XXH_vec_loadu(xsecret + i);
        xxh_u64x2 const data_key = data_vec ^ key_vec;
        /* shuffled = (data_key << 32) | (data_key >> 32); */
        xxh_u32x4 const shuffled = (xxh_u32x4)vec_rl(data_key, v32);
        /* product = ((xxh_u64x2)data_key & 0xFFFFFFFF) * ((xxh_u64x2)shuffled & 0xFFFFFFFF); */
        xxh_u64x2 const product  = XXH_vec_mulo((xxh_u32x4)data_key, shuffled);
        xacc[i] += product;

        /* swap high and low halves */
#ifdef __s390x__
        xacc[i] += vec_permi(data_vec, data_vec, 2);
#else
        xacc[i] += vec_xxpermdi(data_vec, data_vec, 2);
#endif
    }
}

XXH_FORCE_INLINE void
XXH3_scrambleAcc_vsx(void* XXH_RESTRICT acc, const void* XXH_RESTRICT secret)
{
    XXH_ASSERT((((size_t)acc) & 15) == 0);

    {         xxh_u64x2* const xacc    =       (xxh_u64x2*) acc;
        const xxh_u64x2* const xsecret = (const xxh_u64x2*) secret;
        /* constants */
        xxh_u64x2 const v32  = { 32, 32 };
        xxh_u64x2 const v47 = { 47, 47 };
        xxh_u32x4 const prime = { XXH_PRIME32_1, XXH_PRIME32_1, XXH_PRIME32_1, XXH_PRIME32_1 };
        size_t i;
        for (i = 0; i < XXH_STRIPE_LEN / sizeof(xxh_u64x2); i++) {
            /* xacc[i] ^= (xacc[i] >> 47); */
            xxh_u64x2 const acc_vec  = xacc[i];
            xxh_u64x2 const data_vec = acc_vec ^ (acc_vec >> v47);

            /* xacc[i] ^= xsecret[i]; */
            xxh_u64x2 const key_vec  = XXH_vec_loadu(xsecret + i);
            xxh_u64x2 const data_key = data_vec ^ key_vec;

            /* xacc[i] *= XXH_PRIME32_1 */
            /* prod_lo = ((xxh_u64x2)data_key & 0xFFFFFFFF) * ((xxh_u64x2)prime & 0xFFFFFFFF);  */
            xxh_u64x2 const prod_even  = XXH_vec_mule((xxh_u32x4)data_key, prime);
            /* prod_hi = ((xxh_u64x2)data_key >> 32) * ((xxh_u64x2)prime >> 32);  */
            xxh_u64x2 const prod_odd  = XXH_vec_mulo((xxh_u32x4)data_key, prime);
            xacc[i] = prod_odd + (prod_even << v32);
    }   }
}

#endif

/* scalar variants - universal */

XXH_FORCE_INLINE void
XXH3_accumulate_512_scalar(void* XXH_RESTRICT acc,
                     const void* XXH_RESTRICT input,
                     const void* XXH_RESTRICT secret)
{
    XXH_ALIGN(XXH_ACC_ALIGN) xxh_u64* const xacc = (xxh_u64*) acc; /* presumed aligned */
    const xxh_u8* const xinput  = (const xxh_u8*) input;  /* no alignment restriction */
    const xxh_u8* const xsecret = (const xxh_u8*) secret;   /* no alignment restriction */
    size_t i;
    XXH_ASSERT(((size_t)acc & (XXH_ACC_ALIGN-1)) == 0);
    for (i=0; i < XXH_ACC_NB; i++) {
        xxh_u64 const data_val = XXH_readLE64(xinput + 8*i);
        xxh_u64 const data_key = data_val ^ XXH_readLE64(xsecret + i*8);
        xacc[i ^ 1] += data_val; /* swap adjacent lanes */
        xacc[i] += XXH_mult32to64(data_key & 0xFFFFFFFF, data_key >> 32);
    }
}

XXH_FORCE_INLINE void
XXH3_scrambleAcc_scalar(void* XXH_RESTRICT acc, const void* XXH_RESTRICT secret)
{
    XXH_ALIGN(XXH_ACC_ALIGN) xxh_u64* const xacc = (xxh_u64*) acc;   /* presumed aligned */
    const xxh_u8* const xsecret = (const xxh_u8*) secret;   /* no alignment restriction */
    size_t i;
    XXH_ASSERT((((size_t)acc) & (XXH_ACC_ALIGN-1)) == 0);
    for (i=0; i < XXH_ACC_NB; i++) {
        xxh_u64 const key64 = XXH_readLE64(xsecret + 8*i);
        xxh_u64 acc64 = xacc[i];
        acc64 = XXH_xorshift64(acc64, 47);
        acc64 ^= key64;
        acc64 *= XXH_PRIME32_1;
        xacc[i] = acc64;
    }
}

XXH_FORCE_INLINE void
XXH3_initCustomSecret_scalar(void* XXH_RESTRICT customSecret, xxh_u64 seed64)
{
    /*
     * We need a separate pointer for the hack below,
     * which requires a non-const pointer.
     * Any decent compiler will optimize this out otherwise.
     */
    const xxh_u8* kSecretPtr = XXH3_kSecret;
    XXH_STATIC_ASSERT((XXH_SECRET_DEFAULT_SIZE & 15) == 0);

#if defined(__clang__) && defined(__aarch64__)
    /*
     * UGLY HACK:
     * Clang generates a bunch of MOV/MOVK pairs for aarch64, and they are
     * placed sequentially, in order, at the top of the unrolled loop.
     *
     * While MOVK is great for generating constants (2 cycles for a 64-bit
     * constant compared to 4 cycles for LDR), long MOVK chains stall the
     * integer pipelines:
     *   I   L   S
     * MOVK
     * MOVK
     * MOVK
     * MOVK
     * ADD
     * SUB      STR
     *          STR
     * By forcing loads from memory (as the asm line causes Clang to assume
     * that XXH3_kSecretPtr has been changed), the pipelines are used more
     * efficiently:
     *   I   L   S
     *      LDR
     *  ADD LDR
     *  SUB     STR
     *          STR
     * XXH3_64bits_withSeed, len == 256, Snapdragon 835
     *   without hack: 2654.4 MB/s
     *   with hack:    3202.9 MB/s
     */
    __asm__("" : "+r" (kSecretPtr));
#endif
    /*
     * Note: in debug mode, this overrides the asm optimization
     * and Clang will emit MOVK chains again.
     */
    XXH_ASSERT(kSecretPtr == XXH3_kSecret);

    {   int const nbRounds = XXH_SECRET_DEFAULT_SIZE / 16;
        int i;
        for (i=0; i < nbRounds; i++) {
            /*
             * The asm hack causes Clang to assume that kSecretPtr aliases with
             * customSecret, and on aarch64, this prevented LDP from merging two
             * loads together for free. Putting the loads together before the stores
             * properly generates LDP.
             */
            xxh_u64 lo = XXH_readLE64(kSecretPtr + 16*i)     + seed64;
            xxh_u64 hi = XXH_readLE64(kSecretPtr + 16*i + 8) - seed64;
            XXH_writeLE64((xxh_u8*)customSecret + 16*i,     lo);
            XXH_writeLE64((xxh_u8*)customSecret + 16*i + 8, hi);
    }   }
}


typedef void (*XXH3_f_accumulate_512)(void* XXH_RESTRICT, const void*, const void*);
typedef void (*XXH3_f_scrambleAcc)(void* XXH_RESTRICT, const void*);
typedef void (*XXH3_f_initCustomSecret)(void* XXH_RESTRICT, xxh_u64);


#if (XXH_VECTOR == XXH_AVX512)

#define XXH3_accumulate_512 XXH3_accumulate_512_avx512
#define XXH3_scrambleAcc    XXH3_scrambleAcc_avx512
#define XXH3_initCustomSecret XXH3_initCustomSecret_avx512

#elif (XXH_VECTOR == XXH_AVX2)

#define XXH3_accumulate_512 XXH3_accumulate_512_avx2
#define XXH3_scrambleAcc    XXH3_scrambleAcc_avx2
#define XXH3_initCustomSecret XXH3_initCustomSecret_avx2

#elif (XXH_VECTOR == XXH_SSE2)

#define XXH3_accumulate_512 XXH3_accumulate_512_sse2
#define XXH3_scrambleAcc    XXH3_scrambleAcc_sse2
#define XXH3_initCustomSecret XXH3_initCustomSecret_sse2

#elif (XXH_VECTOR == XXH_NEON)

#define XXH3_accumulate_512 XXH3_accumulate_512_neon
#define XXH3_scrambleAcc    XXH3_scrambleAcc_neon
#define XXH3_initCustomSecret XXH3_initCustomSecret_scalar

#elif (XXH_VECTOR == XXH_VSX)

#define XXH3_accumulate_512 XXH3_accumulate_512_vsx
#define XXH3_scrambleAcc    XXH3_scrambleAcc_vsx
#define XXH3_initCustomSecret XXH3_initCustomSecret_scalar

#else /* scalar */

#define XXH3_accumulate_512 XXH3_accumulate_512_scalar
#define XXH3_scrambleAcc    XXH3_scrambleAcc_scalar
#define XXH3_initCustomSecret XXH3_initCustomSecret_scalar

#endif



#ifndef XXH_PREFETCH_DIST
#  ifdef __clang__
#    define XXH_PREFETCH_DIST 320
#  else
#    if (XXH_VECTOR == XXH_AVX512)
#      define XXH_PREFETCH_DIST 512
#    else
#      define XXH_PREFETCH_DIST 384
#    endif
#  endif  /* __clang__ */
#endif  /* XXH_PREFETCH_DIST */

/*
 * XXH3_accumulate()
 * Loops over XXH3_accumulate_512().
 * Assumption: nbStripes will not overflow the secret size
 */
XXH_FORCE_INLINE void
XXH3_accumulate(     xxh_u64* XXH_RESTRICT acc,
                const xxh_u8* XXH_RESTRICT input,
                const xxh_u8* XXH_RESTRICT secret,
                      size_t nbStripes,
                      XXH3_f_accumulate_512 f_acc512)
{
    size_t n;
    for (n = 0; n < nbStripes; n++ ) {
        const xxh_u8* const in = input + n*XXH_STRIPE_LEN;
        XXH_PREFETCH(in + XXH_PREFETCH_DIST);
        f_acc512(acc,
                 in,
                 secret + n*XXH_SECRET_CONSUME_RATE);
    }
}

XXH_FORCE_INLINE void
XXH3_hashLong_internal_loop(xxh_u64* XXH_RESTRICT acc,
                      const xxh_u8* XXH_RESTRICT input, size_t len,
                      const xxh_u8* XXH_RESTRICT secret, size_t secretSize,
                            XXH3_f_accumulate_512 f_acc512,
                            XXH3_f_scrambleAcc f_scramble)
{
    size_t const nbStripesPerBlock = (secretSize - XXH_STRIPE_LEN) / XXH_SECRET_CONSUME_RATE;
    size_t const block_len = XXH_STRIPE_LEN * nbStripesPerBlock;
    size_t const nb_blocks = (len - 1) / block_len;

    size_t n;

    XXH_ASSERT(secretSize >= XXH3_SECRET_SIZE_MIN);

    for (n = 0; n < nb_blocks; n++) {
        XXH3_accumulate(acc, input + n*block_len, secret, nbStripesPerBlock, f_acc512);
        f_scramble(acc, secret + secretSize - XXH_STRIPE_LEN);
    }

    /* last partial block */
    XXH_ASSERT(len > XXH_STRIPE_LEN);
    {   size_t const nbStripes = ((len - 1) - (block_len * nb_blocks)) / XXH_STRIPE_LEN;
        XXH_ASSERT(nbStripes <= (secretSize / XXH_SECRET_CONSUME_RATE));
        XXH3_accumulate(acc, input + nb_blocks*block_len, secret, nbStripes, f_acc512);

        /* last stripe */
        {   const xxh_u8* const p = input + len - XXH_STRIPE_LEN;
#define XXH_SECRET_LASTACC_START 7  /* not aligned on 8, last secret is different from acc & scrambler */
            f_acc512(acc, p, secret + secretSize - XXH_STRIPE_LEN - XXH_SECRET_LASTACC_START);
    }   }
}

XXH_FORCE_INLINE xxh_u64
XXH3_mix2Accs(const xxh_u64* XXH_RESTRICT acc, const xxh_u8* XXH_RESTRICT secret)
{
    return XXH3_mul128_fold64(
               acc[0] ^ XXH_readLE64(secret),
               acc[1] ^ XXH_readLE64(secret+8) );
}

static XXH64_hash_t
XXH3_mergeAccs(const xxh_u64* XXH_RESTRICT acc, const xxh_u8* XXH_RESTRICT secret, xxh_u64 start)
{
    xxh_u64 result64 = start;
    size_t i = 0;

    for (i = 0; i < 4; i++) {
        result64 += XXH3_mix2Accs(acc+2*i, secret + 16*i);
#if defined(__clang__)                                /* Clang */ \
    && (defined(__arm__) || defined(__thumb__))       /* ARMv7 */ \
    && (defined(__ARM_NEON) || defined(__ARM_NEON__)) /* NEON */  \
    && !defined(XXH_ENABLE_AUTOVECTORIZE)             /* Define to disable */
        /*
         * UGLY HACK:
         * Prevent autovectorization on Clang ARMv7-a. Exact same problem as
         * the one in XXH3_len_129to240_64b. Speeds up shorter keys > 240b.
         * XXH3_64bits, len == 256, Snapdragon 835:
         *   without hack: 2063.7 MB/s
         *   with hack:    2560.7 MB/s
         */
        __asm__("" : "+r" (result64));
#endif
    }

    return XXH3_avalanche(result64);
}

#define XXH3_INIT_ACC { XXH_PRIME32_3, XXH_PRIME64_1, XXH_PRIME64_2, XXH_PRIME64_3, \
                        XXH_PRIME64_4, XXH_PRIME32_2, XXH_PRIME64_5, XXH_PRIME32_1 }

XXH_FORCE_INLINE XXH64_hash_t
XXH3_hashLong_64b_internal(const void* XXH_RESTRICT input, size_t len,
                           const void* XXH_RESTRICT secret, size_t secretSize,
                           XXH3_f_accumulate_512 f_acc512,
                           XXH3_f_scrambleAcc f_scramble)
{
    XXH_ALIGN(XXH_ACC_ALIGN) xxh_u64 acc[XXH_ACC_NB] = XXH3_INIT_ACC;

    XXH3_hashLong_internal_loop(acc, (const xxh_u8*)input, len, (const xxh_u8*)secret, secretSize, f_acc512, f_scramble);

    /* converge into final hash */
    XXH_STATIC_ASSERT(sizeof(acc) == 64);
    /* do not align on 8, so that the secret is different from the accumulator */
#define XXH_SECRET_MERGEACCS_START 11
    XXH_ASSERT(secretSize >= sizeof(acc) + XXH_SECRET_MERGEACCS_START);
    return XXH3_mergeAccs(acc, (const xxh_u8*)secret + XXH_SECRET_MERGEACCS_START, (xxh_u64)len * XXH_PRIME64_1);
}

/*
 * It's important for performance that XXH3_hashLong is not inlined.
 */
XXH_NO_INLINE XXH64_hash_t
XXH3_hashLong_64b_withSecret(const void* XXH_RESTRICT input, size_t len,
                             XXH64_hash_t seed64, const xxh_u8* XXH_RESTRICT secret, size_t secretLen)
{
    (void)seed64;
    return XXH3_hashLong_64b_internal(input, len, secret, secretLen, XXH3_accumulate_512, XXH3_scrambleAcc);
}

/*
 * It's important for performance that XXH3_hashLong is not inlined.
 * Since the function is not inlined, the compiler may not be able to understand that,
 * in some scenarios, its `secret` argument is actually a compile time constant.
 * This variant enforces that the compiler can detect that,
 * and uses this opportunity to streamline the generated code for better performance.
 */
XXH_NO_INLINE XXH64_hash_t
XXH3_hashLong_64b_default(const void* XXH_RESTRICT input, size_t len,
                          XXH64_hash_t seed64, const xxh_u8* XXH_RESTRICT secret, size_t secretLen)
{
    (void)seed64; (void)secret; (void)secretLen;
    return XXH3_hashLong_64b_internal(input, len, XXH3_kSecret, sizeof(XXH3_kSecret), XXH3_accumulate_512, XXH3_scrambleAcc);
}

/*
 * XXH3_hashLong_64b_withSeed():
 * Generate a custom key based on alteration of default XXH3_kSecret with the seed,
 * and then use this key for long mode hashing.
 *
 * This operation is decently fast but nonetheless costs a little bit of time.
 * Try to avoid it whenever possible (typically when seed==0).
 *
 * It's important for performance that XXH3_hashLong is not inlined. Not sure
 * why (uop cache maybe?), but the difference is large and easily measurable.
 */
XXH_FORCE_INLINE XXH64_hash_t
XXH3_hashLong_64b_withSeed_internal(const void* input, size_t len,
                                    XXH64_hash_t seed,
                                    XXH3_f_accumulate_512 f_acc512,
                                    XXH3_f_scrambleAcc f_scramble,
                                    XXH3_f_initCustomSecret f_initSec)
{
    if (seed == 0)
        return XXH3_hashLong_64b_internal(input, len,
                                          XXH3_kSecret, sizeof(XXH3_kSecret),
                                          f_acc512, f_scramble);
    {   XXH_ALIGN(XXH_SEC_ALIGN) xxh_u8 secret[XXH_SECRET_DEFAULT_SIZE];
        f_initSec(secret, seed);
        return XXH3_hashLong_64b_internal(input, len, secret, sizeof(secret),
                                          f_acc512, f_scramble);
    }
}

/*
 * It's important for performance that XXH3_hashLong is not inlined.
 */
XXH_NO_INLINE XXH64_hash_t
XXH3_hashLong_64b_withSeed(const void* input, size_t len,
                           XXH64_hash_t seed, const xxh_u8* secret, size_t secretLen)
{
    (void)secret; (void)secretLen;
    return XXH3_hashLong_64b_withSeed_internal(input, len, seed,
                XXH3_accumulate_512, XXH3_scrambleAcc, XXH3_initCustomSecret);
}


typedef XXH64_hash_t (*XXH3_hashLong64_f)(const void* XXH_RESTRICT, size_t,
                                          XXH64_hash_t, const xxh_u8* XXH_RESTRICT, size_t);

XXH_FORCE_INLINE XXH64_hash_t
XXH3_64bits_internal(const void* XXH_RESTRICT input, size_t len,
                     XXH64_hash_t seed64, const void* XXH_RESTRICT secret, size_t secretLen,
                     XXH3_hashLong64_f f_hashLong)
{
    XXH_ASSERT(secretLen >= XXH3_SECRET_SIZE_MIN);
    /*
     * If an action is to be taken if `secretLen` condition is not respected,
     * it should be done here.
     * For now, it's a contract pre-condition.
     * Adding a check and a branch here would cost performance at every hash.
     * Also, note that function signature doesn't offer room to return an error.
     */
    if (len <= 16)
        return XXH3_len_0to16_64b((const xxh_u8*)input, len, (const xxh_u8*)secret, seed64);
    if (len <= 128)
        return XXH3_len_17to128_64b((const xxh_u8*)input, len, (const xxh_u8*)secret, secretLen, seed64);
    if (len <= XXH3_MIDSIZE_MAX)
        return XXH3_len_129to240_64b((const xxh_u8*)input, len, (const xxh_u8*)secret, secretLen, seed64);
    return f_hashLong(input, len, seed64, (const xxh_u8*)secret, secretLen);
}


/* ===   Public entry point   === */

XXH_PUBLIC_API XXH64_hash_t XXH3_64bits(const void* input, size_t len)
{
    return XXH3_64bits_internal(input, len, 0, XXH3_kSecret, sizeof(XXH3_kSecret), XXH3_hashLong_64b_default);
}

XXH_PUBLIC_API XXH64_hash_t
XXH3_64bits_withSecret(const void* input, size_t len, const void* secret, size_t secretSize)
{
    return XXH3_64bits_internal(input, len, 0, secret, secretSize, XXH3_hashLong_64b_withSecret);
}

XXH_PUBLIC_API XXH64_hash_t
XXH3_64bits_withSeed(const void* input, size_t len, XXH64_hash_t seed)
{
    return XXH3_64bits_internal(input, len, seed, XXH3_kSecret, sizeof(XXH3_kSecret), XXH3_hashLong_64b_withSeed);
}


/* ===   XXH3 streaming   === */

/*
 * Malloc's a pointer that is always aligned to align.
 *
 * This must be freed with `XXH_alignedFree()`.
 *
 * malloc typically guarantees 16 byte alignment on 64-bit systems and 8 byte
 * alignment on 32-bit. This isn't enough for the 32 byte aligned loads in AVX2
 * or on 32-bit, the 16 byte aligned loads in SSE2 and NEON.
 *
 * This underalignment previously caused a rather obvious crash which went
 * completely unnoticed due to XXH3_createState() not actually being tested.
 * Credit to RedSpah for noticing this bug.
 *
 * The alignment is done manually: Functions like posix_memalign or _mm_malloc
 * are avoided: To maintain portability, we would have to write a fallback
 * like this anyways, and besides, testing for the existence of library
 * functions without relying on external build tools is impossible.
 *
 * The method is simple: Overallocate, manually align, and store the offset
 * to the original behind the returned pointer.
 *
 * Align must be a power of 2 and 8 <= align <= 128.
 */
static void* XXH_alignedMalloc(size_t s, size_t align)
{
    XXH_ASSERT(align <= 128 && align >= 8); /* range check */
    XXH_ASSERT((align & (align-1)) == 0);   /* power of 2 */
    XXH_ASSERT(s != 0 && s < (s + align));  /* empty/overflow */
    {   /* Overallocate to make room for manual realignment and an offset byte */
        xxh_u8* base = (xxh_u8*)XXH_malloc(s + align);
        if (base != NULL) {
            /*
             * Get the offset needed to align this pointer.
             *
             * Even if the returned pointer is aligned, there will always be
             * at least one byte to store the offset to the original pointer.
             */
            size_t offset = align - ((size_t)base & (align - 1)); /* base % align */
            /* Add the offset for the now-aligned pointer */
            xxh_u8* ptr = base + offset;

            XXH_ASSERT((size_t)ptr % align == 0);

            /* Store the offset immediately before the returned pointer. */
            ptr[-1] = (xxh_u8)offset;
            return ptr;
        }
        return NULL;
    }
}
/*
 * Frees an aligned pointer allocated by XXH_alignedMalloc(). Don't pass
 * normal malloc'd pointers, XXH_alignedMalloc has a specific data layout.
 */
static void XXH_alignedFree(void* p)
{
    if (p != NULL) {
        xxh_u8* ptr = (xxh_u8*)p;
        /* Get the offset byte we added in XXH_malloc. */
        xxh_u8 offset = ptr[-1];
        /* Free the original malloc'd pointer */
        xxh_u8* base = ptr - offset;
        XXH_free(base);
    }
}
XXH_PUBLIC_API XXH3_state_t* XXH3_createState(void)
{
    XXH3_state_t* const state = (XXH3_state_t*)XXH_alignedMalloc(sizeof(XXH3_state_t), 64);
    if (state==NULL) return NULL;
    XXH3_INITSTATE(state);
    return state;
}

XXH_PUBLIC_API XXH_errorcode XXH3_freeState(XXH3_state_t* statePtr)
{
    XXH_alignedFree(statePtr);
    return XXH_OK;
}

XXH_PUBLIC_API void
XXH3_copyState(XXH3_state_t* dst_state, const XXH3_state_t* src_state)
{
    memcpy(dst_state, src_state, sizeof(*dst_state));
}

static void
XXH3_64bits_reset_internal(XXH3_state_t* statePtr,
                           XXH64_hash_t seed,
                           const void* secret, size_t secretSize)
{
    size_t const initStart = offsetof(XXH3_state_t, bufferedSize);
    size_t const initLength = offsetof(XXH3_state_t, nbStripesPerBlock) - initStart;
    XXH_ASSERT(offsetof(XXH3_state_t, nbStripesPerBlock) > initStart);
    XXH_ASSERT(statePtr != NULL);
    /* set members from bufferedSize to nbStripesPerBlock (excluded) to 0 */
    memset((char*)statePtr + initStart, 0, initLength);
    statePtr->acc[0] = XXH_PRIME32_3;
    statePtr->acc[1] = XXH_PRIME64_1;
    statePtr->acc[2] = XXH_PRIME64_2;
    statePtr->acc[3] = XXH_PRIME64_3;
    statePtr->acc[4] = XXH_PRIME64_4;
    statePtr->acc[5] = XXH_PRIME32_2;
    statePtr->acc[6] = XXH_PRIME64_5;
    statePtr->acc[7] = XXH_PRIME32_1;
    statePtr->seed = seed;
    statePtr->extSecret = (const unsigned char*)secret;
    XXH_ASSERT(secretSize >= XXH3_SECRET_SIZE_MIN);
    statePtr->secretLimit = secretSize - XXH_STRIPE_LEN;
    statePtr->nbStripesPerBlock = statePtr->secretLimit / XXH_SECRET_CONSUME_RATE;
}

XXH_PUBLIC_API XXH_errorcode
XXH3_64bits_reset(XXH3_state_t* statePtr)
{
    if (statePtr == NULL) return XXH_ERROR;
    XXH3_64bits_reset_internal(statePtr, 0, XXH3_kSecret, XXH_SECRET_DEFAULT_SIZE);
    return XXH_OK;
}

XXH_PUBLIC_API XXH_errorcode
XXH3_64bits_reset_withSecret(XXH3_state_t* statePtr, const void* secret, size_t secretSize)
{
    if (statePtr == NULL) return XXH_ERROR;
    XXH3_64bits_reset_internal(statePtr, 0, secret, secretSize);
    if (secret == NULL) return XXH_ERROR;
    if (secretSize < XXH3_SECRET_SIZE_MIN) return XXH_ERROR;
    return XXH_OK;
}

XXH_PUBLIC_API XXH_errorcode
XXH3_64bits_reset_withSeed(XXH3_state_t* statePtr, XXH64_hash_t seed)
{
    if (statePtr == NULL) return XXH_ERROR;
    if (seed==0) return XXH3_64bits_reset(statePtr);
    if (seed != statePtr->seed) XXH3_initCustomSecret(statePtr->customSecret, seed);
    XXH3_64bits_reset_internal(statePtr, seed, NULL, XXH_SECRET_DEFAULT_SIZE);
    return XXH_OK;
}

/* Note : when XXH3_consumeStripes() is invoked,
 * there must be a guarantee that at least one more byte must be consumed from input
 * so that the function can blindly consume all stripes using the "normal" secret segment */
XXH_FORCE_INLINE void
XXH3_consumeStripes(xxh_u64* XXH_RESTRICT acc,
                    size_t* XXH_RESTRICT nbStripesSoFarPtr, size_t nbStripesPerBlock,
                    const xxh_u8* XXH_RESTRICT input, size_t nbStripes,
                    const xxh_u8* XXH_RESTRICT secret, size_t secretLimit,
                    XXH3_f_accumulate_512 f_acc512,
                    XXH3_f_scrambleAcc f_scramble)
{
    XXH_ASSERT(nbStripes <= nbStripesPerBlock);  /* can handle max 1 scramble per invocation */
    XXH_ASSERT(*nbStripesSoFarPtr < nbStripesPerBlock);
    if (nbStripesPerBlock - *nbStripesSoFarPtr <= nbStripes) {
        /* need a scrambling operation */
        size_t const nbStripesToEndofBlock = nbStripesPerBlock - *nbStripesSoFarPtr;
        size_t const nbStripesAfterBlock = nbStripes - nbStripesToEndofBlock;
        XXH3_accumulate(acc, input, secret + nbStripesSoFarPtr[0] * XXH_SECRET_CONSUME_RATE, nbStripesToEndofBlock, f_acc512);
        f_scramble(acc, secret + secretLimit);
        XXH3_accumulate(acc, input + nbStripesToEndofBlock * XXH_STRIPE_LEN, secret, nbStripesAfterBlock, f_acc512);
        *nbStripesSoFarPtr = nbStripesAfterBlock;
    } else {
        XXH3_accumulate(acc, input, secret + nbStripesSoFarPtr[0] * XXH_SECRET_CONSUME_RATE, nbStripes, f_acc512);
        *nbStripesSoFarPtr += nbStripes;
    }
}

/*
 * Both XXH3_64bits_update and XXH3_128bits_update use this routine.
 */
XXH_FORCE_INLINE XXH_errorcode
XXH3_update(XXH3_state_t* state,
            const xxh_u8* input, size_t len,
            XXH3_f_accumulate_512 f_acc512,
            XXH3_f_scrambleAcc f_scramble)
{
    if (input==NULL)
#if defined(XXH_ACCEPT_NULL_INPUT_POINTER) && (XXH_ACCEPT_NULL_INPUT_POINTER>=1)
        return XXH_OK;
#else
        return XXH_ERROR;
#endif

    {   const xxh_u8* const bEnd = input + len;
        const unsigned char* const secret = (state->extSecret == NULL) ? state->customSecret : state->extSecret;

        state->totalLen += len;

        if (state->bufferedSize + len <= XXH3_INTERNALBUFFER_SIZE) {  /* fill in tmp buffer */
            XXH_memcpy(state->buffer + state->bufferedSize, input, len);
            state->bufferedSize += (XXH32_hash_t)len;
            return XXH_OK;
        }
        /* total input is now > XXH3_INTERNALBUFFER_SIZE */

        #define XXH3_INTERNALBUFFER_STRIPES (XXH3_INTERNALBUFFER_SIZE / XXH_STRIPE_LEN)
        XXH_STATIC_ASSERT(XXH3_INTERNALBUFFER_SIZE % XXH_STRIPE_LEN == 0);   /* clean multiple */

        /*
         * Internal buffer is partially filled (always, except at beginning)
         * Complete it, then consume it.
         */
        if (state->bufferedSize) {
            size_t const loadSize = XXH3_INTERNALBUFFER_SIZE - state->bufferedSize;
            XXH_memcpy(state->buffer + state->bufferedSize, input, loadSize);
            input += loadSize;
            XXH3_consumeStripes(state->acc,
                               &state->nbStripesSoFar, state->nbStripesPerBlock,
                                state->buffer, XXH3_INTERNALBUFFER_STRIPES,
                                secret, state->secretLimit,
                                f_acc512, f_scramble);
            state->bufferedSize = 0;
        }
        XXH_ASSERT(input < bEnd);

        /* Consume input by a multiple of internal buffer size */
        if (input+XXH3_INTERNALBUFFER_SIZE < bEnd) {
            const xxh_u8* const limit = bEnd - XXH3_INTERNALBUFFER_SIZE;
            do {
                XXH3_consumeStripes(state->acc,
                                   &state->nbStripesSoFar, state->nbStripesPerBlock,
                                    input, XXH3_INTERNALBUFFER_STRIPES,
                                    secret, state->secretLimit,
                                    f_acc512, f_scramble);
                input += XXH3_INTERNALBUFFER_SIZE;
            } while (input<limit);
            /* for last partial stripe */
            memcpy(state->buffer + sizeof(state->buffer) - XXH_STRIPE_LEN, input - XXH_STRIPE_LEN, XXH_STRIPE_LEN);
        }
        XXH_ASSERT(input < bEnd);

        /* Some remaining input (always) : buffer it */
        XXH_memcpy(state->buffer, input, (size_t)(bEnd-input));
        state->bufferedSize = (XXH32_hash_t)(bEnd-input);
    }

    return XXH_OK;
}

XXH_PUBLIC_API XXH_errorcode
XXH3_64bits_update(XXH3_state_t* state, const void* input, size_t len)
{
    return XXH3_update(state, (const xxh_u8*)input, len,
                       XXH3_accumulate_512, XXH3_scrambleAcc);
}


XXH_FORCE_INLINE void
XXH3_digest_long (XXH64_hash_t* acc,
                  const XXH3_state_t* state,
                  const unsigned char* secret)
{
    /*
     * Digest on a local copy. This way, the state remains unaltered, and it can
     * continue ingesting more input afterwards.
     */
    memcpy(acc, state->acc, sizeof(state->acc));
    if (state->bufferedSize >= XXH_STRIPE_LEN) {
        size_t const nbStripes = (state->bufferedSize - 1) / XXH_STRIPE_LEN;
        size_t nbStripesSoFar = state->nbStripesSoFar;
        XXH3_consumeStripes(acc,
                           &nbStripesSoFar, state->nbStripesPerBlock,
                            state->buffer, nbStripes,
                            secret, state->secretLimit,
                            XXH3_accumulate_512, XXH3_scrambleAcc);
        /* last stripe */
        XXH3_accumulate_512(acc,
                            state->buffer + state->bufferedSize - XXH_STRIPE_LEN,
                            secret + state->secretLimit - XXH_SECRET_LASTACC_START);
    } else {  /* bufferedSize < XXH_STRIPE_LEN */
        xxh_u8 lastStripe[XXH_STRIPE_LEN];
        size_t const catchupSize = XXH_STRIPE_LEN - state->bufferedSize;
        XXH_ASSERT(state->bufferedSize > 0);  /* there is always some input buffered */
        memcpy(lastStripe, state->buffer + sizeof(state->buffer) - catchupSize, catchupSize);
        memcpy(lastStripe + catchupSize, state->buffer, state->bufferedSize);
        XXH3_accumulate_512(acc,
                            lastStripe,
                            secret + state->secretLimit - XXH_SECRET_LASTACC_START);
    }
}

XXH_PUBLIC_API XXH64_hash_t XXH3_64bits_digest (const XXH3_state_t* state)
{
    const unsigned char* const secret = (state->extSecret == NULL) ? state->customSecret : state->extSecret;
    if (state->totalLen > XXH3_MIDSIZE_MAX) {
        XXH_ALIGN(XXH_ACC_ALIGN) XXH64_hash_t acc[XXH_ACC_NB];
        XXH3_digest_long(acc, state, secret);
        return XXH3_mergeAccs(acc,
                              secret + XXH_SECRET_MERGEACCS_START,
                              (xxh_u64)state->totalLen * XXH_PRIME64_1);
    }
    /* totalLen <= XXH3_MIDSIZE_MAX: digesting a short input */
    if (state->seed)
        return XXH3_64bits_withSeed(state->buffer, (size_t)state->totalLen, state->seed);
    return XXH3_64bits_withSecret(state->buffer, (size_t)(state->totalLen),
                                  secret, state->secretLimit + XXH_STRIPE_LEN);
}


#define XXH_MIN(x, y) (((x) > (y)) ? (y) : (x))

XXH_PUBLIC_API void
XXH3_generateSecret(void* secretBuffer, const void* customSeed, size_t customSeedSize)
{
    XXH_ASSERT(secretBuffer != NULL);
    if (customSeedSize == 0) {
        memcpy(secretBuffer, XXH3_kSecret, XXH_SECRET_DEFAULT_SIZE);
        return;
    }
    XXH_ASSERT(customSeed != NULL);

    {   size_t const segmentSize = sizeof(XXH128_hash_t);
        size_t const nbSegments = XXH_SECRET_DEFAULT_SIZE / segmentSize;
        XXH128_canonical_t scrambler;
        XXH64_hash_t seeds[12];
        size_t segnb;
        XXH_ASSERT(nbSegments == 12);
        XXH_ASSERT(segmentSize * nbSegments == XXH_SECRET_DEFAULT_SIZE); /* exact multiple */
        XXH128_canonicalFromHash(&scrambler, XXH128(customSeed, customSeedSize, 0));

        /*
        * Copy customSeed to seeds[], truncating or repeating as necessary.
        */
        {   size_t toFill = XXH_MIN(customSeedSize, sizeof(seeds));
            size_t filled = toFill;
            memcpy(seeds, customSeed, toFill);
            while (filled < sizeof(seeds)) {
                toFill = XXH_MIN(filled, sizeof(seeds) - filled);
                memcpy((char*)seeds + filled, seeds, toFill);
                filled += toFill;
        }   }

        /* generate secret */
        memcpy(secretBuffer, &scrambler, sizeof(scrambler));
        for (segnb=1; segnb < nbSegments; segnb++) {
            size_t const segmentStart = segnb * segmentSize;
            XXH128_canonical_t segment;
            XXH128_canonicalFromHash(&segment,
                XXH128(&scrambler, sizeof(scrambler), XXH_readLE64(seeds + segnb) + segnb) );
            memcpy((char*)secretBuffer + segmentStart, &segment, sizeof(segment));
    }   }
}


/* ==========================================
 * XXH3 128 bits (a.k.a XXH128)
 * ==========================================
 * XXH3's 128-bit variant has better mixing and strength than the 64-bit variant,
 * even without counting the significantly larger output size.
 *
 * For example, extra steps are taken to avoid the seed-dependent collisions
 * in 17-240 byte inputs (See XXH3_mix16B and XXH128_mix32B).
 *
 * This strength naturally comes at the cost of some speed, especially on short
 * lengths. Note that longer hashes are about as fast as the 64-bit version
 * due to it using only a slight modification of the 64-bit loop.
 *
 * XXH128 is also more oriented towards 64-bit machines. It is still extremely
 * fast for a _128-bit_ hash on 32-bit (it usually clears XXH64).
 */

XXH_FORCE_INLINE XXH128_hash_t
XXH3_len_1to3_128b(const xxh_u8* input, size_t len, const xxh_u8* secret, XXH64_hash_t seed)
{
    /* A doubled version of 1to3_64b with different constants. */
    XXH_ASSERT(input != NULL);
    XXH_ASSERT(1 <= len && len <= 3);
    XXH_ASSERT(secret != NULL);
    /*
     * len = 1: combinedl = { input[0], 0x01, input[0], input[0] }
     * len = 2: combinedl = { input[1], 0x02, input[0], input[1] }
     * len = 3: combinedl = { input[2], 0x03, input[0], input[1] }
     */
    {   xxh_u8 const c1 = input[0];
        xxh_u8 const c2 = input[len >> 1];
        xxh_u8 const c3 = input[len - 1];
        xxh_u32 const combinedl = ((xxh_u32)c1 <<16) | ((xxh_u32)c2 << 24)
                                | ((xxh_u32)c3 << 0) | ((xxh_u32)len << 8);
        xxh_u32 const combinedh = XXH_rotl32(XXH_swap32(combinedl), 13);
        xxh_u64 const bitflipl = (XXH_readLE32(secret) ^ XXH_readLE32(secret+4)) + seed;
        xxh_u64 const bitfliph = (XXH_readLE32(secret+8) ^ XXH_readLE32(secret+12)) - seed;
        xxh_u64 const keyed_lo = (xxh_u64)combinedl ^ bitflipl;
        xxh_u64 const keyed_hi = (xxh_u64)combinedh ^ bitfliph;
        XXH128_hash_t h128;
        h128.low64  = XXH64_avalanche(keyed_lo);
        h128.high64 = XXH64_avalanche(keyed_hi);
        return h128;
    }
}

XXH_FORCE_INLINE XXH128_hash_t
XXH3_len_4to8_128b(const xxh_u8* input, size_t len, const xxh_u8* secret, XXH64_hash_t seed)
{
    XXH_ASSERT(input != NULL);
    XXH_ASSERT(secret != NULL);
    XXH_ASSERT(4 <= len && len <= 8);
    seed ^= (xxh_u64)XXH_swap32((xxh_u32)seed) << 32;
    {   xxh_u32 const input_lo = XXH_readLE32(input);
        xxh_u32 const input_hi = XXH_readLE32(input + len - 4);
        xxh_u64 const input_64 = input_lo + ((xxh_u64)input_hi << 32);
        xxh_u64 const bitflip = (XXH_readLE64(secret+16) ^ XXH_readLE64(secret+24)) + seed;
        xxh_u64 const keyed = input_64 ^ bitflip;

        /* Shift len to the left to ensure it is even, this avoids even multiplies. */
        XXH128_hash_t m128 = XXH_mult64to128(keyed, XXH_PRIME64_1 + (len << 2));

        m128.high64 += (m128.low64 << 1);
        m128.low64  ^= (m128.high64 >> 3);

        m128.low64   = XXH_xorshift64(m128.low64, 35);
        m128.low64  *= 0x9FB21C651E98DF25ULL;
        m128.low64   = XXH_xorshift64(m128.low64, 28);
        m128.high64  = XXH3_avalanche(m128.high64);
        return m128;
    }
}

XXH_FORCE_INLINE XXH128_hash_t
XXH3_len_9to16_128b(const xxh_u8* input, size_t len, const xxh_u8* secret, XXH64_hash_t seed)
{
    XXH_ASSERT(input != NULL);
    XXH_ASSERT(secret != NULL);
    XXH_ASSERT(9 <= len && len <= 16);
    {   xxh_u64 const bitflipl = (XXH_readLE64(secret+32) ^ XXH_readLE64(secret+40)) - seed;
        xxh_u64 const bitfliph = (XXH_readLE64(secret+48) ^ XXH_readLE64(secret+56)) + seed;
        xxh_u64 const input_lo = XXH_readLE64(input);
        xxh_u64       input_hi = XXH_readLE64(input + len - 8);
        XXH128_hash_t m128 = XXH_mult64to128(input_lo ^ input_hi ^ bitflipl, XXH_PRIME64_1);
        /*
         * Put len in the middle of m128 to ensure that the length gets mixed to
         * both the low and high bits in the 128x64 multiply below.
         */
        m128.low64 += (xxh_u64)(len - 1) << 54;
        input_hi   ^= bitfliph;
        /*
         * Add the high 32 bits of input_hi to the high 32 bits of m128, then
         * add the long product of the low 32 bits of input_hi and XXH_PRIME32_2 to
         * the high 64 bits of m128.
         *
         * The best approach to this operation is different on 32-bit and 64-bit.
         */
        if (sizeof(void *) < sizeof(xxh_u64)) { /* 32-bit */
            /*
             * 32-bit optimized version, which is more readable.
             *
             * On 32-bit, it removes an ADC and delays a dependency between the two
             * halves of m128.high64, but it generates an extra mask on 64-bit.
             */
            m128.high64 += (input_hi & 0xFFFFFFFF00000000ULL) + XXH_mult32to64((xxh_u32)input_hi, XXH_PRIME32_2);
        } else {
            /*
             * 64-bit optimized (albeit more confusing) version.
             *
             * Uses some properties of addition and multiplication to remove the mask:
             *
             * Let:
             *    a = input_hi.lo = (input_hi & 0x00000000FFFFFFFF)
             *    b = input_hi.hi = (input_hi & 0xFFFFFFFF00000000)
             *    c = XXH_PRIME32_2
             *
             *    a + (b * c)
             * Inverse Property: x + y - x == y
             *    a + (b * (1 + c - 1))
             * Distributive Property: x * (y + z) == (x * y) + (x * z)
             *    a + (b * 1) + (b * (c - 1))
             * Identity Property: x * 1 == x
             *    a + b + (b * (c - 1))
             *
             * Substitute a, b, and c:
             *    input_hi.hi + input_hi.lo + ((xxh_u64)input_hi.lo * (XXH_PRIME32_2 - 1))
             *
             * Since input_hi.hi + input_hi.lo == input_hi, we get this:
             *    input_hi + ((xxh_u64)input_hi.lo * (XXH_PRIME32_2 - 1))
             */
            m128.high64 += input_hi + XXH_mult32to64((xxh_u32)input_hi, XXH_PRIME32_2 - 1);
        }
        /* m128 ^= XXH_swap64(m128 >> 64); */
        m128.low64  ^= XXH_swap64(m128.high64);

        {   /* 128x64 multiply: h128 = m128 * XXH_PRIME64_2; */
            XXH128_hash_t h128 = XXH_mult64to128(m128.low64, XXH_PRIME64_2);
            h128.high64 += m128.high64 * XXH_PRIME64_2;

            h128.low64   = XXH3_avalanche(h128.low64);
            h128.high64  = XXH3_avalanche(h128.high64);
            return h128;
    }   }
}

/*
 * Assumption: `secret` size is >= XXH3_SECRET_SIZE_MIN
 */
XXH_FORCE_INLINE XXH128_hash_t
XXH3_len_0to16_128b(const xxh_u8* input, size_t len, const xxh_u8* secret, XXH64_hash_t seed)
{
    XXH_ASSERT(len <= 16);
    {   if (len > 8) return XXH3_len_9to16_128b(input, len, secret, seed);
        if (len >= 4) return XXH3_len_4to8_128b(input, len, secret, seed);
        if (len) return XXH3_len_1to3_128b(input, len, secret, seed);
        {   XXH128_hash_t h128;
            xxh_u64 const bitflipl = XXH_readLE64(secret+64) ^ XXH_readLE64(secret+72);
            xxh_u64 const bitfliph = XXH_readLE64(secret+80) ^ XXH_readLE64(secret+88);
            h128.low64 = XXH64_avalanche(seed ^ bitflipl);
            h128.high64 = XXH64_avalanche( seed ^ bitfliph);
            return h128;
    }   }
}

/*
 * A bit slower than XXH3_mix16B, but handles multiply by zero better.
 */
XXH_FORCE_INLINE XXH128_hash_t
XXH128_mix32B(XXH128_hash_t acc, const xxh_u8* input_1, const xxh_u8* input_2,
              const xxh_u8* secret, XXH64_hash_t seed)
{
    acc.low64  += XXH3_mix16B (input_1, secret+0, seed);
    acc.low64  ^= XXH_readLE64(input_2) + XXH_readLE64(input_2 + 8);
    acc.high64 += XXH3_mix16B (input_2, secret+16, seed);
    acc.high64 ^= XXH_readLE64(input_1) + XXH_readLE64(input_1 + 8);
    return acc;
}


XXH_FORCE_INLINE XXH128_hash_t
XXH3_len_17to128_128b(const xxh_u8* XXH_RESTRICT input, size_t len,
                      const xxh_u8* XXH_RESTRICT secret, size_t secretSize,
                      XXH64_hash_t seed)
{
    XXH_ASSERT(secretSize >= XXH3_SECRET_SIZE_MIN); (void)secretSize;
    XXH_ASSERT(16 < len && len <= 128);

    {   XXH128_hash_t acc;
        acc.low64 = len * XXH_PRIME64_1;
        acc.high64 = 0;
        if (len > 32) {
            if (len > 64) {
                if (len > 96) {
                    acc = XXH128_mix32B(acc, input+48, input+len-64, secret+96, seed);
                }
                acc = XXH128_mix32B(acc, input+32, input+len-48, secret+64, seed);
            }
            acc = XXH128_mix32B(acc, input+16, input+len-32, secret+32, seed);
        }
        acc = XXH128_mix32B(acc, input, input+len-16, secret, seed);
        {   XXH128_hash_t h128;
            h128.low64  = acc.low64 + acc.high64;
            h128.high64 = (acc.low64    * XXH_PRIME64_1)
                        + (acc.high64   * XXH_PRIME64_4)
                        + ((len - seed) * XXH_PRIME64_2);
            h128.low64  = XXH3_avalanche(h128.low64);
            h128.high64 = (XXH64_hash_t)0 - XXH3_avalanche(h128.high64);
            return h128;
        }
    }
}

XXH_NO_INLINE XXH128_hash_t
XXH3_len_129to240_128b(const xxh_u8* XXH_RESTRICT input, size_t len,
                       const xxh_u8* XXH_RESTRICT secret, size_t secretSize,
                       XXH64_hash_t seed)
{
    XXH_ASSERT(secretSize >= XXH3_SECRET_SIZE_MIN); (void)secretSize;
    XXH_ASSERT(128 < len && len <= XXH3_MIDSIZE_MAX);

    {   XXH128_hash_t acc;
        int const nbRounds = (int)len / 32;
        int i;
        acc.low64 = len * XXH_PRIME64_1;
        acc.high64 = 0;
        for (i=0; i<4; i++) {
            acc = XXH128_mix32B(acc,
                                input  + (32 * i),
                                input  + (32 * i) + 16,
                                secret + (32 * i),
                                seed);
        }
        acc.low64 = XXH3_avalanche(acc.low64);
        acc.high64 = XXH3_avalanche(acc.high64);
        XXH_ASSERT(nbRounds >= 4);
        for (i=4 ; i < nbRounds; i++) {
            acc = XXH128_mix32B(acc,
                                input + (32 * i),
                                input + (32 * i) + 16,
                                secret + XXH3_MIDSIZE_STARTOFFSET + (32 * (i - 4)),
                                seed);
        }
        /* last bytes */
        acc = XXH128_mix32B(acc,
                            input + len - 16,
                            input + len - 32,
                            secret + XXH3_SECRET_SIZE_MIN - XXH3_MIDSIZE_LASTOFFSET - 16,
                            0ULL - seed);

        {   XXH128_hash_t h128;
            h128.low64  = acc.low64 + acc.high64;
            h128.high64 = (acc.low64    * XXH_PRIME64_1)
                        + (acc.high64   * XXH_PRIME64_4)
                        + ((len - seed) * XXH_PRIME64_2);
            h128.low64  = XXH3_avalanche(h128.low64);
            h128.high64 = (XXH64_hash_t)0 - XXH3_avalanche(h128.high64);
            return h128;
        }
    }
}

XXH_FORCE_INLINE XXH128_hash_t
XXH3_hashLong_128b_internal(const void* XXH_RESTRICT input, size_t len,
                            const xxh_u8* XXH_RESTRICT secret, size_t secretSize,
                            XXH3_f_accumulate_512 f_acc512,
                            XXH3_f_scrambleAcc f_scramble)
{
    XXH_ALIGN(XXH_ACC_ALIGN) xxh_u64 acc[XXH_ACC_NB] = XXH3_INIT_ACC;

    XXH3_hashLong_internal_loop(acc, (const xxh_u8*)input, len, secret, secretSize, f_acc512, f_scramble);

    /* converge into final hash */
    XXH_STATIC_ASSERT(sizeof(acc) == 64);
    XXH_ASSERT(secretSize >= sizeof(acc) + XXH_SECRET_MERGEACCS_START);
    {   XXH128_hash_t h128;
        h128.low64  = XXH3_mergeAccs(acc,
                                     secret + XXH_SECRET_MERGEACCS_START,
                                     (xxh_u64)len * XXH_PRIME64_1);
        h128.high64 = XXH3_mergeAccs(acc,
                                     secret + secretSize
                                            - sizeof(acc) - XXH_SECRET_MERGEACCS_START,
                                     ~((xxh_u64)len * XXH_PRIME64_2));
        return h128;
    }
}

/*
 * It's important for performance that XXH3_hashLong is not inlined.
 */
XXH_NO_INLINE XXH128_hash_t
XXH3_hashLong_128b_default(const void* XXH_RESTRICT input, size_t len,
                           XXH64_hash_t seed64,
                           const void* XXH_RESTRICT secret, size_t secretLen)
{
    (void)seed64; (void)secret; (void)secretLen;
    return XXH3_hashLong_128b_internal(input, len, XXH3_kSecret, sizeof(XXH3_kSecret),
                                       XXH3_accumulate_512, XXH3_scrambleAcc);
}

/*
 * It's important for performance that XXH3_hashLong is not inlined.
 */
XXH_NO_INLINE XXH128_hash_t
XXH3_hashLong_128b_withSecret(const void* XXH_RESTRICT input, size_t len,
                              XXH64_hash_t seed64,
                              const void* XXH_RESTRICT secret, size_t secretLen)
{
    (void)seed64;
    return XXH3_hashLong_128b_internal(input, len, (const xxh_u8*)secret, secretLen,
                                       XXH3_accumulate_512, XXH3_scrambleAcc);
}

XXH_FORCE_INLINE XXH128_hash_t
XXH3_hashLong_128b_withSeed_internal(const void* XXH_RESTRICT input, size_t len,
                                XXH64_hash_t seed64,
                                XXH3_f_accumulate_512 f_acc512,
                                XXH3_f_scrambleAcc f_scramble,
                                XXH3_f_initCustomSecret f_initSec)
{
    if (seed64 == 0)
        return XXH3_hashLong_128b_internal(input, len,
                                           XXH3_kSecret, sizeof(XXH3_kSecret),
                                           f_acc512, f_scramble);
    {   XXH_ALIGN(XXH_SEC_ALIGN) xxh_u8 secret[XXH_SECRET_DEFAULT_SIZE];
        f_initSec(secret, seed64);
        return XXH3_hashLong_128b_internal(input, len, (const xxh_u8*)secret, sizeof(secret),
                                           f_acc512, f_scramble);
    }
}

/*
 * It's important for performance that XXH3_hashLong is not inlined.
 */
XXH_NO_INLINE XXH128_hash_t
XXH3_hashLong_128b_withSeed(const void* input, size_t len,
                            XXH64_hash_t seed64, const void* XXH_RESTRICT secret, size_t secretLen)
{
    (void)secret; (void)secretLen;
    return XXH3_hashLong_128b_withSeed_internal(input, len, seed64,
                XXH3_accumulate_512, XXH3_scrambleAcc, XXH3_initCustomSecret);
}

typedef XXH128_hash_t (*XXH3_hashLong128_f)(const void* XXH_RESTRICT, size_t,
                                            XXH64_hash_t, const void* XXH_RESTRICT, size_t);

XXH_FORCE_INLINE XXH128_hash_t
XXH3_128bits_internal(const void* input, size_t len,
                      XXH64_hash_t seed64, const void* XXH_RESTRICT secret, size_t secretLen,
                      XXH3_hashLong128_f f_hl128)
{
    XXH_ASSERT(secretLen >= XXH3_SECRET_SIZE_MIN);
    /*
     * If an action is to be taken if `secret` conditions are not respected,
     * it should be done here.
     * For now, it's a contract pre-condition.
     * Adding a check and a branch here would cost performance at every hash.
     */
    if (len <= 16)
        return XXH3_len_0to16_128b((const xxh_u8*)input, len, (const xxh_u8*)secret, seed64);
    if (len <= 128)
        return XXH3_len_17to128_128b((const xxh_u8*)input, len, (const xxh_u8*)secret, secretLen, seed64);
    if (len <= XXH3_MIDSIZE_MAX)
        return XXH3_len_129to240_128b((const xxh_u8*)input, len, (const xxh_u8*)secret, secretLen, seed64);
    return f_hl128(input, len, seed64, secret, secretLen);
}


/* ===   Public XXH128 API   === */

XXH_PUBLIC_API XXH128_hash_t XXH3_128bits(const void* input, size_t len)
{
    return XXH3_128bits_internal(input, len, 0,
                                 XXH3_kSecret, sizeof(XXH3_kSecret),
                                 XXH3_hashLong_128b_default);
}

XXH_PUBLIC_API XXH128_hash_t
XXH3_128bits_withSecret(const void* input, size_t len, const void* secret, size_t secretSize)
{
    return XXH3_128bits_internal(input, len, 0,
                                 (const xxh_u8*)secret, secretSize,
                                 XXH3_hashLong_128b_withSecret);
}

XXH_PUBLIC_API XXH128_hash_t
XXH3_128bits_withSeed(const void* input, size_t len, XXH64_hash_t seed)
{
    return XXH3_128bits_internal(input, len, seed,
                                 XXH3_kSecret, sizeof(XXH3_kSecret),
                                 XXH3_hashLong_128b_withSeed);
}

XXH_PUBLIC_API XXH128_hash_t
XXH128(const void* input, size_t len, XXH64_hash_t seed)
{
    return XXH3_128bits_withSeed(input, len, seed);
}


/* ===   XXH3 128-bit streaming   === */

/*
 * All the functions are actually the same as for 64-bit streaming variant.
 * The only difference is the finalizatiom routine.
 */

static void
XXH3_128bits_reset_internal(XXH3_state_t* statePtr,
                            XXH64_hash_t seed,
                            const void* secret, size_t secretSize)
{
    XXH3_64bits_reset_internal(statePtr, seed, secret, secretSize);
}

XXH_PUBLIC_API XXH_errorcode
XXH3_128bits_reset(XXH3_state_t* statePtr)
{
    if (statePtr == NULL) return XXH_ERROR;
    XXH3_128bits_reset_internal(statePtr, 0, XXH3_kSecret, XXH_SECRET_DEFAULT_SIZE);
    return XXH_OK;
}

XXH_PUBLIC_API XXH_errorcode
XXH3_128bits_reset_withSecret(XXH3_state_t* statePtr, const void* secret, size_t secretSize)
{
    if (statePtr == NULL) return XXH_ERROR;
    XXH3_128bits_reset_internal(statePtr, 0, secret, secretSize);
    if (secret == NULL) return XXH_ERROR;
    if (secretSize < XXH3_SECRET_SIZE_MIN) return XXH_ERROR;
    return XXH_OK;
}

XXH_PUBLIC_API XXH_errorcode
XXH3_128bits_reset_withSeed(XXH3_state_t* statePtr, XXH64_hash_t seed)
{
    if (statePtr == NULL) return XXH_ERROR;
    if (seed==0) return XXH3_128bits_reset(statePtr);
    if (seed != statePtr->seed) XXH3_initCustomSecret(statePtr->customSecret, seed);
    XXH3_128bits_reset_internal(statePtr, seed, NULL, XXH_SECRET_DEFAULT_SIZE);
    return XXH_OK;
}

XXH_PUBLIC_API XXH_errorcode
XXH3_128bits_update(XXH3_state_t* state, const void* input, size_t len)
{
    return XXH3_update(state, (const xxh_u8*)input, len,
                       XXH3_accumulate_512, XXH3_scrambleAcc);
}

XXH_PUBLIC_API XXH128_hash_t XXH3_128bits_digest (const XXH3_state_t* state)
{
    const unsigned char* const secret = (state->extSecret == NULL) ? state->customSecret : state->extSecret;
    if (state->totalLen > XXH3_MIDSIZE_MAX) {
        XXH_ALIGN(XXH_ACC_ALIGN) XXH64_hash_t acc[XXH_ACC_NB];
        XXH3_digest_long(acc, state, secret);
        XXH_ASSERT(state->secretLimit + XXH_STRIPE_LEN >= sizeof(acc) + XXH_SECRET_MERGEACCS_START);
        {   XXH128_hash_t h128;
            h128.low64  = XXH3_mergeAccs(acc,
                                         secret + XXH_SECRET_MERGEACCS_START,
                                         (xxh_u64)state->totalLen * XXH_PRIME64_1);
            h128.high64 = XXH3_mergeAccs(acc,
                                         secret + state->secretLimit + XXH_STRIPE_LEN
                                                - sizeof(acc) - XXH_SECRET_MERGEACCS_START,
                                         ~((xxh_u64)state->totalLen * XXH_PRIME64_2));
            return h128;
        }
    }
    /* len <= XXH3_MIDSIZE_MAX : short code */
    if (state->seed)
        return XXH3_128bits_withSeed(state->buffer, (size_t)state->totalLen, state->seed);
    return XXH3_128bits_withSecret(state->buffer, (size_t)(state->totalLen),
                                   secret, state->secretLimit + XXH_STRIPE_LEN);
}

/* 128-bit utility functions */

#include <string.h>   /* memcmp, memcpy */

/* return : 1 is equal, 0 if different */
XXH_PUBLIC_API int XXH128_isEqual(XXH128_hash_t h1, XXH128_hash_t h2)
{
    /* note : XXH128_hash_t is compact, it has no padding byte */
    return !(memcmp(&h1, &h2, sizeof(h1)));
}

/* This prototype is compatible with stdlib's qsort().
 * return : >0 if *h128_1  > *h128_2
 *          <0 if *h128_1  < *h128_2
 *          =0 if *h128_1 == *h128_2  */
XXH_PUBLIC_API int XXH128_cmp(const void* h128_1, const void* h128_2)
{
    XXH128_hash_t const h1 = *(const XXH128_hash_t*)h128_1;
    XXH128_hash_t const h2 = *(const XXH128_hash_t*)h128_2;
    int const hcmp = (h1.high64 > h2.high64) - (h2.high64 > h1.high64);
    /* note : bets that, in most cases, hash values are different */
    if (hcmp) return hcmp;
    return (h1.low64 > h2.low64) - (h2.low64 > h1.low64);
}


/*======   Canonical representation   ======*/
XXH_PUBLIC_API void
XXH128_canonicalFromHash(XXH128_canonical_t* dst, XXH128_hash_t hash)
{
    XXH_STATIC_ASSERT(sizeof(XXH128_canonical_t) == sizeof(XXH128_hash_t));
    if (XXH_CPU_LITTLE_ENDIAN) {
        hash.high64 = XXH_swap64(hash.high64);
        hash.low64  = XXH_swap64(hash.low64);
    }
    memcpy(dst, &hash.high64, sizeof(hash.high64));
    memcpy((char*)dst + sizeof(hash.high64), &hash.low64, sizeof(hash.low64));
}

XXH_PUBLIC_API XXH128_hash_t
XXH128_hashFromCanonical(const XXH128_canonical_t* src)
{
    XXH128_hash_t h;
    h.high64 = XXH_readBE64(src);
    h.low64  = XXH_readBE64(src->digest + 8);
    return h;
}

/* Pop our optimization override from above */
#if XXH_VECTOR == XXH_AVX2 /* AVX2 */ \
  && defined(__GNUC__) && !defined(__clang__) /* GCC, not Clang */ \
  && defined(__OPTIMIZE__) && !defined(__OPTIMIZE_SIZE__) /* respect -O0 and -Os */
#  pragma GCC pop_options
#endif

#endif  /* XXH_NO_LONG_LONG */


#endif  /* XXH_IMPLEMENTATION */


#if defined (__cplusplus)
}
#endif
