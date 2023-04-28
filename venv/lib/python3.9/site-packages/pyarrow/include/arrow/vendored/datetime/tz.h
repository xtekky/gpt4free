#ifndef TZ_H
#define TZ_H

// The MIT License (MIT)
//
// Copyright (c) 2015, 2016, 2017 Howard Hinnant
// Copyright (c) 2017 Jiangang Zhuang
// Copyright (c) 2017 Aaron Bishop
// Copyright (c) 2017 Tomasz Kamiński
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Our apologies.  When the previous paragraph was written, lowercase had not yet
// been invented (that would involve another several millennia of evolution).
// We did not mean to shout.

// Get more recent database at http://www.iana.org/time-zones

// The notion of "current timezone" is something the operating system is expected to "just
// know". How it knows this is system specific. It's often a value set by the user at OS
// installation time and recorded by the OS somewhere. On Linux and Mac systems the current
// timezone name is obtained by looking at the name or contents of a particular file on
// disk. On Windows the current timezone name comes from the registry. In either method,
// there is no guarantee that the "native" current timezone name obtained will match any
// of the "Standard" names in this library's "database". On Linux, the names usually do
// seem to match so mapping functions to map from native to "Standard" are typically not
// required. On Windows, the names are never "Standard" so mapping is always required.
// Technically any OS may use the mapping process but currently only Windows does use it.

// NOTE(ARROW): If this is not set, then the library will attempt to
// use libcurl to obtain a timezone database, and we probably do not want this.
#ifndef _WIN32
#define USE_OS_TZDB 1
#endif

#ifndef USE_OS_TZDB
#  define USE_OS_TZDB 0
#endif

#ifndef HAS_REMOTE_API
#  if USE_OS_TZDB == 0
#    ifdef _WIN32
#      define HAS_REMOTE_API 0
#    else
#      define HAS_REMOTE_API 1
#    endif
#  else  // HAS_REMOTE_API makes no since when using the OS timezone database
#    define HAS_REMOTE_API 0
#  endif
#endif

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wconstant-logical-operand"
#endif

static_assert(!(USE_OS_TZDB && HAS_REMOTE_API),
              "USE_OS_TZDB and HAS_REMOTE_API can not be used together");

#ifdef __clang__
# pragma clang diagnostic pop
#endif

#ifndef AUTO_DOWNLOAD
#  define AUTO_DOWNLOAD HAS_REMOTE_API
#endif

static_assert(HAS_REMOTE_API == 0 ? AUTO_DOWNLOAD == 0 : true,
              "AUTO_DOWNLOAD can not be turned on without HAS_REMOTE_API");

#ifndef USE_SHELL_API
#  define USE_SHELL_API 1
#endif

#if USE_OS_TZDB
#  ifdef _WIN32
#    error "USE_OS_TZDB can not be used on Windows"
#  endif
#endif

#ifndef HAS_DEDUCTION_GUIDES
#  if __cplusplus >= 201703
#    define HAS_DEDUCTION_GUIDES 1
#  else
#    define HAS_DEDUCTION_GUIDES 0
#  endif
#endif  // HAS_DEDUCTION_GUIDES

#include "date.h"

#if defined(_MSC_VER) && (_MSC_VER < 1900)
#include "tz_private.h"
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <istream>
#include <locale>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#  ifdef DATE_BUILD_DLL
#    define DATE_API __declspec(dllexport)
#  elif defined(DATE_USE_DLL)
#    define DATE_API __declspec(dllimport)
#  else
#    define DATE_API
#  endif
#else
#  ifdef DATE_BUILD_DLL
#    define DATE_API __attribute__ ((visibility ("default")))
#  else
#    define DATE_API
#  endif
#endif

namespace arrow_vendored
{
namespace date
{

enum class choose {earliest, latest};

namespace detail
{
    struct undocumented;

    template<typename T>
    struct nodeduct
    {
       using type = T;
    };

    template<typename T>
    using nodeduct_t = typename nodeduct<T>::type;
}

struct sys_info
{
    sys_seconds          begin;
    sys_seconds          end;
    std::chrono::seconds offset;
    std::chrono::minutes save;
    std::string          abbrev;
};

template<class CharT, class Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const sys_info& r)
{
    os << r.begin << '\n';
    os << r.end << '\n';
    os << make_time(r.offset) << "\n";
    os << make_time(r.save) << "\n";
    os << r.abbrev << '\n';
    return os;
}

struct local_info
{
    enum {unique, nonexistent, ambiguous} result;
    sys_info first;
    sys_info second;
};

template<class CharT, class Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const local_info& r)
{
    if (r.result == local_info::nonexistent)
        os << "nonexistent between\n";
    else if (r.result == local_info::ambiguous)
        os << "ambiguous between\n";
    os << r.first;
    if (r.result != local_info::unique)
    {
        os << "and\n";
        os << r.second;
    }
    return os;
}

class nonexistent_local_time
    : public std::runtime_error
{
public:
    template <class Duration>
        nonexistent_local_time(local_time<Duration> tp, const local_info& i);

private:
    template <class Duration>
    static
    std::string
    make_msg(local_time<Duration> tp, const local_info& i);
};

template <class Duration>
inline
nonexistent_local_time::nonexistent_local_time(local_time<Duration> tp,
                                               const local_info& i)
    : std::runtime_error(make_msg(tp, i))
{
}

template <class Duration>
std::string
nonexistent_local_time::make_msg(local_time<Duration> tp, const local_info& i)
{
    assert(i.result == local_info::nonexistent);
    std::ostringstream os;
    os << tp << " is in a gap between\n"
       << local_seconds{i.first.end.time_since_epoch()} + i.first.offset << ' '
       << i.first.abbrev << " and\n"
       << local_seconds{i.second.begin.time_since_epoch()} + i.second.offset << ' '
       << i.second.abbrev
       << " which are both equivalent to\n"
       << i.first.end << " UTC";
    return os.str();
}

class ambiguous_local_time
    : public std::runtime_error
{
public:
    template <class Duration>
        ambiguous_local_time(local_time<Duration> tp, const local_info& i);

private:
    template <class Duration>
    static
    std::string
    make_msg(local_time<Duration> tp, const local_info& i);
};

template <class Duration>
inline
ambiguous_local_time::ambiguous_local_time(local_time<Duration> tp, const local_info& i)
    : std::runtime_error(make_msg(tp, i))
{
}

template <class Duration>
std::string
ambiguous_local_time::make_msg(local_time<Duration> tp, const local_info& i)
{
    assert(i.result == local_info::ambiguous);
    std::ostringstream os;
    os << tp << " is ambiguous.  It could be\n"
       << tp << ' ' << i.first.abbrev << " == "
       << tp - i.first.offset << " UTC or\n"
       << tp << ' ' << i.second.abbrev  << " == "
       << tp - i.second.offset  << " UTC";
    return os.str();
}

class time_zone;

#if HAS_STRING_VIEW
DATE_API const time_zone* locate_zone(std::string_view tz_name);
#else
DATE_API const time_zone* locate_zone(const std::string& tz_name);
#endif

DATE_API const time_zone* current_zone();

template <class T>
struct zoned_traits
{
};

template <>
struct zoned_traits<const time_zone*>
{
    static
    const time_zone*
    default_zone()
    {
        return date::locate_zone("Etc/UTC");
    }

#if HAS_STRING_VIEW

    static
    const time_zone*
    locate_zone(std::string_view name)
    {
        return date::locate_zone(name);
    }

#else  // !HAS_STRING_VIEW

    static
    const time_zone*
    locate_zone(const std::string& name)
    {
        return date::locate_zone(name);
    }

    static
    const time_zone*
    locate_zone(const char* name)
    {
        return date::locate_zone(name);
    }

#endif  // !HAS_STRING_VIEW
};

template <class Duration, class TimeZonePtr>
class zoned_time;

template <class Duration1, class Duration2, class TimeZonePtr>
bool
operator==(const zoned_time<Duration1, TimeZonePtr>& x,
           const zoned_time<Duration2, TimeZonePtr>& y);

template <class Duration, class TimeZonePtr = const time_zone*>
class zoned_time
{
public:
    using duration = typename std::common_type<Duration, std::chrono::seconds>::type;

private:
    TimeZonePtr        zone_;
    sys_time<duration> tp_;

public:
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = decltype(zoned_traits<T>::default_zone())>
#endif
        zoned_time();

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = decltype(zoned_traits<T>::default_zone())>
#endif
        zoned_time(const sys_time<Duration>& st);
    explicit zoned_time(TimeZonePtr z);

#if HAS_STRING_VIEW
    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string_view()))
                  >::value
              >::type>
        explicit zoned_time(std::string_view name);
#else
#  if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string()))
                  >::value
              >::type>
#  endif
        explicit zoned_time(const std::string& name);
#endif

    template <class Duration2,
              class = typename std::enable_if
                      <
                          std::is_convertible<sys_time<Duration2>,
                                              sys_time<Duration>>::value
                      >::type>
        zoned_time(const zoned_time<Duration2, TimeZonePtr>& zt) NOEXCEPT;

    zoned_time(TimeZonePtr z, const sys_time<Duration>& st);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_convertible
                  <
                      decltype(std::declval<T&>()->to_sys(local_time<Duration>{})),
                      sys_time<duration>
                  >::value
              >::type>
#endif
        zoned_time(TimeZonePtr z, const local_time<Duration>& tp);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_convertible
                  <
                      decltype(std::declval<T&>()->to_sys(local_time<Duration>{},
                                                          choose::earliest)),
                      sys_time<duration>
                  >::value
              >::type>
#endif
        zoned_time(TimeZonePtr z, const local_time<Duration>& tp, choose c);

    template <class Duration2, class TimeZonePtr2,
              class = typename std::enable_if
                      <
                          std::is_convertible<sys_time<Duration2>,
                                              sys_time<Duration>>::value
                      >::type>
        zoned_time(TimeZonePtr z, const zoned_time<Duration2, TimeZonePtr2>& zt);

    template <class Duration2, class TimeZonePtr2,
              class = typename std::enable_if
                      <
                          std::is_convertible<sys_time<Duration2>,
                                              sys_time<Duration>>::value
                      >::type>
        zoned_time(TimeZonePtr z, const zoned_time<Duration2, TimeZonePtr2>& zt, choose);

#if HAS_STRING_VIEW

    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string_view())),
                      sys_time<Duration>
                  >::value
              >::type>
        zoned_time(std::string_view name, detail::nodeduct_t<const sys_time<Duration>&> st);

    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string_view())),
                      local_time<Duration>
                  >::value
              >::type>
        zoned_time(std::string_view name, detail::nodeduct_t<const local_time<Duration>&> tp);

    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string_view())),
                      local_time<Duration>,
                      choose
                  >::value
              >::type>
        zoned_time(std::string_view name, detail::nodeduct_t<const local_time<Duration>&> tp, choose c);

    template <class Duration2, class TimeZonePtr2, class T = TimeZonePtr,
              class = typename std::enable_if
                      <
                          std::is_convertible<sys_time<Duration2>,
                                              sys_time<Duration>>::value &&
                          std::is_constructible
                          <
                              zoned_time,
                              decltype(zoned_traits<T>::locate_zone(std::string_view())),
                              zoned_time
                          >::value
                      >::type>
        zoned_time(std::string_view name, const zoned_time<Duration2, TimeZonePtr2>& zt);

    template <class Duration2, class TimeZonePtr2, class T = TimeZonePtr,
              class = typename std::enable_if
                      <
                          std::is_convertible<sys_time<Duration2>,
                                              sys_time<Duration>>::value &&
                          std::is_constructible
                          <
                              zoned_time,
                              decltype(zoned_traits<T>::locate_zone(std::string_view())),
                              zoned_time,
                              choose
                          >::value
                      >::type>
        zoned_time(std::string_view name, const zoned_time<Duration2, TimeZonePtr2>& zt, choose);

#else  // !HAS_STRING_VIEW

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string())),
                      sys_time<Duration>
                  >::value
              >::type>
#endif
        zoned_time(const std::string& name, const sys_time<Duration>& st);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string())),
                      sys_time<Duration>
                  >::value
              >::type>
#endif
        zoned_time(const char* name, const sys_time<Duration>& st);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string())),
                      local_time<Duration>
                  >::value
              >::type>
#endif
        zoned_time(const std::string& name, const local_time<Duration>& tp);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string())),
                      local_time<Duration>
                  >::value
              >::type>
#endif
        zoned_time(const char* name, const local_time<Duration>& tp);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string())),
                      local_time<Duration>,
                      choose
                  >::value
              >::type>
#endif
        zoned_time(const std::string& name, const local_time<Duration>& tp, choose c);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class T = TimeZonePtr,
              class = typename std::enable_if
              <
                  std::is_constructible
                  <
                      zoned_time,
                      decltype(zoned_traits<T>::locate_zone(std::string())),
                      local_time<Duration>,
                      choose
                  >::value
              >::type>
#endif
        zoned_time(const char* name, const local_time<Duration>& tp, choose c);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class Duration2, class TimeZonePtr2, class T = TimeZonePtr,
              class = typename std::enable_if
                      <
                          std::is_convertible<sys_time<Duration2>,
                                              sys_time<Duration>>::value &&
                          std::is_constructible
                          <
                              zoned_time,
                              decltype(zoned_traits<T>::locate_zone(std::string())),
                              zoned_time
                          >::value
                      >::type>
#else
    template <class Duration2, class TimeZonePtr2>
#endif
        zoned_time(const std::string& name, const zoned_time<Duration2, TimeZonePtr2>& zt);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class Duration2, class TimeZonePtr2, class T = TimeZonePtr,
              class = typename std::enable_if
                      <
                          std::is_convertible<sys_time<Duration2>,
                                              sys_time<Duration>>::value &&
                          std::is_constructible
                          <
                              zoned_time,
                              decltype(zoned_traits<T>::locate_zone(std::string())),
                              zoned_time
                          >::value
                      >::type>
#else
    template <class Duration2, class TimeZonePtr2>
#endif
        zoned_time(const char* name, const zoned_time<Duration2, TimeZonePtr2>& zt);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class Duration2, class TimeZonePtr2, class T = TimeZonePtr,
              class = typename std::enable_if
                      <
                          std::is_convertible<sys_time<Duration2>,
                                              sys_time<Duration>>::value &&
                          std::is_constructible
                          <
                              zoned_time,
                              decltype(zoned_traits<T>::locate_zone(std::string())),
                              zoned_time,
                              choose
                          >::value
                      >::type>
#else
    template <class Duration2, class TimeZonePtr2>
#endif
        zoned_time(const std::string& name, const zoned_time<Duration2, TimeZonePtr2>& zt,
                   choose);

#if !defined(_MSC_VER) || (_MSC_VER > 1916)
    template <class Duration2, class TimeZonePtr2, class T = TimeZonePtr,
              class = typename std::enable_if
                      <
                          std::is_convertible<sys_time<Duration2>,
                                              sys_time<Duration>>::value &&
                          std::is_constructible
                          <
                              zoned_time,
                              decltype(zoned_traits<T>::locate_zone(std::string())),
                              zoned_time,
                              choose
                          >::value
                      >::type>
#else
    template <class Duration2, class TimeZonePtr2>
#endif
        zoned_time(const char* name, const zoned_time<Duration2, TimeZonePtr2>& zt,
                   choose);

#endif  // !HAS_STRING_VIEW

    zoned_time& operator=(const sys_time<Duration>& st);
    zoned_time& operator=(const local_time<Duration>& ut);

    explicit operator sys_time<duration>() const;
    explicit operator local_time<duration>() const;

    TimeZonePtr          get_time_zone() const;
    local_time<duration> get_local_time() const;
    sys_time<duration>   get_sys_time() const;
    sys_info             get_info() const;

    template <class Duration1, class Duration2, class TimeZonePtr1>
    friend
    bool
    operator==(const zoned_time<Duration1, TimeZonePtr1>& x,
               const zoned_time<Duration2, TimeZonePtr1>& y);

    template <class CharT, class Traits, class Duration1, class TimeZonePtr1>
    friend
    std::basic_ostream<CharT, Traits>&
    operator<<(std::basic_ostream<CharT, Traits>& os,
               const zoned_time<Duration1, TimeZonePtr1>& t);

private:
    template <class D, class T> friend class zoned_time;

    template <class TimeZonePtr2>
    static
    TimeZonePtr2&&
    check(TimeZonePtr2&& p);
};

using zoned_seconds = zoned_time<std::chrono::seconds>;

#if HAS_DEDUCTION_GUIDES

namespace detail
{
   template<typename TimeZonePtrOrName>
   using time_zone_representation =
       std::conditional_t
       <
           std::is_convertible<TimeZonePtrOrName, std::string_view>::value,
           time_zone const*,
           std::remove_cv_t<std::remove_reference_t<TimeZonePtrOrName>>
       >;
}

zoned_time()
    -> zoned_time<std::chrono::seconds>;

template <class Duration>
zoned_time(sys_time<Duration>)
    -> zoned_time<std::common_type_t<Duration, std::chrono::seconds>>;

template <class TimeZonePtrOrName>
zoned_time(TimeZonePtrOrName&&)
    -> zoned_time<std::chrono::seconds, detail::time_zone_representation<TimeZonePtrOrName>>;

template <class TimeZonePtrOrName, class Duration>
zoned_time(TimeZonePtrOrName&&, sys_time<Duration>)
    -> zoned_time<std::common_type_t<Duration, std::chrono::seconds>, detail::time_zone_representation<TimeZonePtrOrName>>;

template <class TimeZonePtrOrName, class Duration>
zoned_time(TimeZonePtrOrName&&, local_time<Duration>, choose = choose::earliest)
    -> zoned_time<std::common_type_t<Duration, std::chrono::seconds>, detail::time_zone_representation<TimeZonePtrOrName>>;

template <class Duration, class TimeZonePtrOrName, class TimeZonePtr2>
zoned_time(TimeZonePtrOrName&&, zoned_time<Duration, TimeZonePtr2>, choose = choose::earliest)
    -> zoned_time<std::common_type_t<Duration, std::chrono::seconds>, detail::time_zone_representation<TimeZonePtrOrName>>;

#endif  // HAS_DEDUCTION_GUIDES

template <class Duration1, class Duration2, class TimeZonePtr>
inline
bool
operator==(const zoned_time<Duration1, TimeZonePtr>& x,
           const zoned_time<Duration2, TimeZonePtr>& y)
{
    return x.zone_ == y.zone_ && x.tp_ == y.tp_;
}

template <class Duration1, class Duration2, class TimeZonePtr>
inline
bool
operator!=(const zoned_time<Duration1, TimeZonePtr>& x,
           const zoned_time<Duration2, TimeZonePtr>& y)
{
    return !(x == y);
}

#if !defined(_MSC_VER) || (_MSC_VER >= 1900)

namespace detail
{
#  if USE_OS_TZDB
    struct transition;
    struct expanded_ttinfo;
#  else  // !USE_OS_TZDB
    struct zonelet;
    class Rule;
#  endif  // !USE_OS_TZDB
}

#endif  // !defined(_MSC_VER) || (_MSC_VER >= 1900)

class time_zone
{
private:
    std::string                          name_;
#if USE_OS_TZDB
    std::vector<detail::transition>      transitions_;
    std::vector<detail::expanded_ttinfo> ttinfos_;
#else  // !USE_OS_TZDB
    std::vector<detail::zonelet>         zonelets_;
#endif  // !USE_OS_TZDB
    std::unique_ptr<std::once_flag>      adjusted_;

public:
#if !defined(_MSC_VER) || (_MSC_VER >= 1900)
    time_zone(time_zone&&) = default;
    time_zone& operator=(time_zone&&) = default;
#else   // defined(_MSC_VER) && (_MSC_VER < 1900)
    time_zone(time_zone&& src);
    time_zone& operator=(time_zone&& src);
#endif  // defined(_MSC_VER) && (_MSC_VER < 1900)

    DATE_API explicit time_zone(const std::string& s, detail::undocumented);

    const std::string& name() const NOEXCEPT;

    template <class Duration> sys_info   get_info(sys_time<Duration> st) const;
    template <class Duration> local_info get_info(local_time<Duration> tp) const;

    template <class Duration>
        sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
        to_sys(local_time<Duration> tp) const;

    template <class Duration>
        sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
        to_sys(local_time<Duration> tp, choose z) const;

    template <class Duration>
        local_time<typename std::common_type<Duration, std::chrono::seconds>::type>
        to_local(sys_time<Duration> tp) const;

    friend bool operator==(const time_zone& x, const time_zone& y) NOEXCEPT;
    friend bool operator< (const time_zone& x, const time_zone& y) NOEXCEPT;
    friend DATE_API std::ostream& operator<<(std::ostream& os, const time_zone& z);

#if !USE_OS_TZDB
    DATE_API void add(const std::string& s);
#endif  // !USE_OS_TZDB

private:
    DATE_API sys_info   get_info_impl(sys_seconds tp) const;
    DATE_API local_info get_info_impl(local_seconds tp) const;

    template <class Duration>
        sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
        to_sys_impl(local_time<Duration> tp, choose z, std::false_type) const;
    template <class Duration>
        sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
        to_sys_impl(local_time<Duration> tp, choose, std::true_type) const;

#if USE_OS_TZDB
    DATE_API void init() const;
    DATE_API void init_impl();
    DATE_API sys_info
        load_sys_info(std::vector<detail::transition>::const_iterator i) const;

    template <class TimeType>
    DATE_API void
    load_data(std::istream& inf, std::int32_t tzh_leapcnt, std::int32_t tzh_timecnt,
                                 std::int32_t tzh_typecnt, std::int32_t tzh_charcnt);
#else  // !USE_OS_TZDB
    DATE_API sys_info   get_info_impl(sys_seconds tp, int timezone) const;
    DATE_API void adjust_infos(const std::vector<detail::Rule>& rules);
    DATE_API void parse_info(std::istream& in);
#endif  // !USE_OS_TZDB
};

#if defined(_MSC_VER) && (_MSC_VER < 1900)

inline
time_zone::time_zone(time_zone&& src)
    : name_(std::move(src.name_))
    , zonelets_(std::move(src.zonelets_))
    , adjusted_(std::move(src.adjusted_))
    {}

inline
time_zone&
time_zone::operator=(time_zone&& src)
{
    name_ = std::move(src.name_);
    zonelets_ = std::move(src.zonelets_);
    adjusted_ = std::move(src.adjusted_);
    return *this;
}

#endif  // defined(_MSC_VER) && (_MSC_VER < 1900)

inline
const std::string&
time_zone::name() const NOEXCEPT
{
    return name_;
}

template <class Duration>
inline
sys_info
time_zone::get_info(sys_time<Duration> st) const
{
    return get_info_impl(date::floor<std::chrono::seconds>(st));
}

template <class Duration>
inline
local_info
time_zone::get_info(local_time<Duration> tp) const
{
    return get_info_impl(date::floor<std::chrono::seconds>(tp));
}

template <class Duration>
inline
sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
time_zone::to_sys(local_time<Duration> tp) const
{
    return to_sys_impl(tp, choose{}, std::true_type{});
}

template <class Duration>
inline
sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
time_zone::to_sys(local_time<Duration> tp, choose z) const
{
    return to_sys_impl(tp, z, std::false_type{});
}

template <class Duration>
inline
local_time<typename std::common_type<Duration, std::chrono::seconds>::type>
time_zone::to_local(sys_time<Duration> tp) const
{
    using LT = local_time<typename std::common_type<Duration, std::chrono::seconds>::type>;
    auto i = get_info(tp);
    return LT{(tp + i.offset).time_since_epoch()};
}

inline bool operator==(const time_zone& x, const time_zone& y) NOEXCEPT {return x.name_ == y.name_;}
inline bool operator< (const time_zone& x, const time_zone& y) NOEXCEPT {return x.name_ < y.name_;}

inline bool operator!=(const time_zone& x, const time_zone& y) NOEXCEPT {return !(x == y);}
inline bool operator> (const time_zone& x, const time_zone& y) NOEXCEPT {return   y < x;}
inline bool operator<=(const time_zone& x, const time_zone& y) NOEXCEPT {return !(y < x);}
inline bool operator>=(const time_zone& x, const time_zone& y) NOEXCEPT {return !(x < y);}

template <class Duration>
sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
time_zone::to_sys_impl(local_time<Duration> tp, choose z, std::false_type) const
{
    auto i = get_info(tp);
    if (i.result == local_info::nonexistent)
    {
        return i.first.end;
    }
    else if (i.result == local_info::ambiguous)
    {
        if (z == choose::latest)
            return sys_time<Duration>{tp.time_since_epoch()} - i.second.offset;
    }
    return sys_time<Duration>{tp.time_since_epoch()} - i.first.offset;
}

template <class Duration>
sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
time_zone::to_sys_impl(local_time<Duration> tp, choose, std::true_type) const
{
    auto i = get_info(tp);
    if (i.result == local_info::nonexistent)
        throw nonexistent_local_time(tp, i);
    else if (i.result == local_info::ambiguous)
        throw ambiguous_local_time(tp, i);
    return sys_time<Duration>{tp.time_since_epoch()} - i.first.offset;
}

#if !USE_OS_TZDB

class time_zone_link
{
private:
    std::string name_;
    std::string target_;
public:
    DATE_API explicit time_zone_link(const std::string& s);

    const std::string& name() const {return name_;}
    const std::string& target() const {return target_;}

    friend bool operator==(const time_zone_link& x, const time_zone_link& y) {return x.name_ == y.name_;}
    friend bool operator< (const time_zone_link& x, const time_zone_link& y) {return x.name_ < y.name_;}

    friend DATE_API std::ostream& operator<<(std::ostream& os, const time_zone_link& x);
};

using link = time_zone_link;

inline bool operator!=(const time_zone_link& x, const time_zone_link& y) {return !(x == y);}
inline bool operator> (const time_zone_link& x, const time_zone_link& y) {return   y < x;}
inline bool operator<=(const time_zone_link& x, const time_zone_link& y) {return !(y < x);}
inline bool operator>=(const time_zone_link& x, const time_zone_link& y) {return !(x < y);}

#endif  // !USE_OS_TZDB

class leap_second
{
private:
    sys_seconds date_;

public:
#if USE_OS_TZDB
    DATE_API explicit leap_second(const sys_seconds& s, detail::undocumented);
#else
    DATE_API explicit leap_second(const std::string& s, detail::undocumented);
#endif

    sys_seconds date() const {return date_;}

    friend bool operator==(const leap_second& x, const leap_second& y) {return x.date_ == y.date_;}
    friend bool operator< (const leap_second& x, const leap_second& y) {return x.date_ < y.date_;}

    template <class Duration>
    friend
    bool
    operator==(const leap_second& x, const sys_time<Duration>& y)
    {
        return x.date_ == y;
    }

    template <class Duration>
    friend
    bool
    operator< (const leap_second& x, const sys_time<Duration>& y)
    {
        return x.date_ < y;
    }

    template <class Duration>
    friend
    bool
    operator< (const sys_time<Duration>& x, const leap_second& y)
    {
        return x < y.date_;
    }

    friend DATE_API std::ostream& operator<<(std::ostream& os, const leap_second& x);
};

inline bool operator!=(const leap_second& x, const leap_second& y) {return !(x == y);}
inline bool operator> (const leap_second& x, const leap_second& y) {return   y < x;}
inline bool operator<=(const leap_second& x, const leap_second& y) {return !(y < x);}
inline bool operator>=(const leap_second& x, const leap_second& y) {return !(x < y);}

template <class Duration>
inline
bool
operator==(const sys_time<Duration>& x, const leap_second& y)
{
    return y == x;
}

template <class Duration>
inline
bool
operator!=(const leap_second& x, const sys_time<Duration>& y)
{
    return !(x == y);
}

template <class Duration>
inline
bool
operator!=(const sys_time<Duration>& x, const leap_second& y)
{
    return !(x == y);
}

template <class Duration>
inline
bool
operator> (const leap_second& x, const sys_time<Duration>& y)
{
    return y < x;
}

template <class Duration>
inline
bool
operator> (const sys_time<Duration>& x, const leap_second& y)
{
    return y < x;
}

template <class Duration>
inline
bool
operator<=(const leap_second& x, const sys_time<Duration>& y)
{
    return !(y < x);
}

template <class Duration>
inline
bool
operator<=(const sys_time<Duration>& x, const leap_second& y)
{
    return !(y < x);
}

template <class Duration>
inline
bool
operator>=(const leap_second& x, const sys_time<Duration>& y)
{
    return !(x < y);
}

template <class Duration>
inline
bool
operator>=(const sys_time<Duration>& x, const leap_second& y)
{
    return !(x < y);
}

using leap = leap_second;

#ifdef _WIN32

namespace detail
{

// The time zone mapping is modelled after this data file:
// http://unicode.org/repos/cldr/trunk/common/supplemental/windowsZones.xml
// and the field names match the element names from the mapZone element
// of windowsZones.xml.
// The website displays this file here:
// http://www.unicode.org/cldr/charts/latest/supplemental/zone_tzid.html
// The html view is sorted before being displayed but is otherwise the same
// There is a mapping between the os centric view (in this case windows)
// the html displays uses and the generic view the xml file.
// That mapping is this:
// display column "windows" -> xml field "other".
// display column "region"  -> xml field "territory".
// display column "tzid"    -> xml field "type".
// This structure uses the generic terminology because it could be
// used to to support other os/native name conversions, not just windows,
// and using the same generic names helps retain the connection to the
// origin of the data that we are using.
struct timezone_mapping
{
    timezone_mapping(const char* other, const char* territory, const char* type)
        : other(other), territory(territory), type(type)
    {
    }
    timezone_mapping() = default;
    std::string other;
    std::string territory;
    std::string type;
};

}  // detail

#endif  // _WIN32

struct tzdb
{
    std::string                 version = "unknown";
    std::vector<time_zone>      zones;
#if !USE_OS_TZDB
    std::vector<time_zone_link> links;
#endif
    std::vector<leap_second>    leap_seconds;
#if !USE_OS_TZDB
    std::vector<detail::Rule>   rules;
#endif
#ifdef _WIN32
    std::vector<detail::timezone_mapping> mappings;
#endif
    tzdb* next = nullptr;

    tzdb() = default;
#if !defined(_MSC_VER) || (_MSC_VER >= 1900)
    tzdb(tzdb&&) = default;
    tzdb& operator=(tzdb&&) = default;
#else  // defined(_MSC_VER) && (_MSC_VER < 1900)
    tzdb(tzdb&& src)
        : version(std::move(src.version))
        , zones(std::move(src.zones))
        , links(std::move(src.links))
        , leap_seconds(std::move(src.leap_seconds))
        , rules(std::move(src.rules))
        , mappings(std::move(src.mappings))
    {}

    tzdb& operator=(tzdb&& src)
    {
        version = std::move(src.version);
        zones = std::move(src.zones);
        links = std::move(src.links);
        leap_seconds = std::move(src.leap_seconds);
        rules = std::move(src.rules);
        mappings = std::move(src.mappings);
        return *this;
    }
#endif  // defined(_MSC_VER) && (_MSC_VER < 1900)

#if HAS_STRING_VIEW
    const time_zone* locate_zone(std::string_view tz_name) const;
#else
    const time_zone* locate_zone(const std::string& tz_name) const;
#endif
    const time_zone* current_zone() const;
};

using TZ_DB = tzdb;

DATE_API std::ostream&
operator<<(std::ostream& os, const tzdb& db);

DATE_API const tzdb& get_tzdb();

class tzdb_list
{
    std::atomic<tzdb*> head_{nullptr};

public:
    ~tzdb_list();
    tzdb_list() = default;
    tzdb_list(tzdb_list&& x) NOEXCEPT;

    const tzdb& front() const NOEXCEPT {return *head_;}
          tzdb& front()       NOEXCEPT {return *head_;}

    class const_iterator;

    const_iterator begin() const NOEXCEPT;
    const_iterator end() const NOEXCEPT;

    const_iterator cbegin() const NOEXCEPT;
    const_iterator cend() const NOEXCEPT;

    const_iterator erase_after(const_iterator p) NOEXCEPT;

    struct undocumented_helper;
private:
    void push_front(tzdb* tzdb) NOEXCEPT;
};

class tzdb_list::const_iterator
{
    tzdb* p_ = nullptr;

    explicit const_iterator(tzdb* p) NOEXCEPT : p_{p} {}
public:
    const_iterator() = default;

    using iterator_category = std::forward_iterator_tag;
    using value_type        = tzdb;
    using reference         = const value_type&;
    using pointer           = const value_type*;
    using difference_type   = std::ptrdiff_t;

    reference operator*() const NOEXCEPT {return *p_;}
    pointer  operator->() const NOEXCEPT {return p_;}

    const_iterator& operator++() NOEXCEPT {p_ = p_->next; return *this;}
    const_iterator  operator++(int) NOEXCEPT {auto t = *this; ++(*this); return t;}

    friend
    bool
    operator==(const const_iterator& x, const const_iterator& y) NOEXCEPT
        {return x.p_ == y.p_;}

    friend
    bool
    operator!=(const const_iterator& x, const const_iterator& y) NOEXCEPT
        {return !(x == y);}

    friend class tzdb_list;
};

inline
tzdb_list::const_iterator
tzdb_list::begin() const NOEXCEPT
{
    return const_iterator{head_};
}

inline
tzdb_list::const_iterator
tzdb_list::end() const NOEXCEPT
{
    return const_iterator{nullptr};
}

inline
tzdb_list::const_iterator
tzdb_list::cbegin() const NOEXCEPT
{
    return begin();
}

inline
tzdb_list::const_iterator
tzdb_list::cend() const NOEXCEPT
{
    return end();
}

DATE_API tzdb_list& get_tzdb_list();

#if !USE_OS_TZDB

DATE_API const tzdb& reload_tzdb();
DATE_API void        set_install(const std::string& install);

#endif  // !USE_OS_TZDB

#if HAS_REMOTE_API

DATE_API std::string remote_version();
// if provided error_buffer size should be at least CURL_ERROR_SIZE
DATE_API bool        remote_download(const std::string& version, char* error_buffer = nullptr);
DATE_API bool        remote_install(const std::string& version);

#endif

// zoned_time

namespace detail
{

template <class T>
inline
T*
to_raw_pointer(T* p) NOEXCEPT
{
    return p;
}

template <class Pointer>
inline
auto
to_raw_pointer(Pointer p) NOEXCEPT
    -> decltype(detail::to_raw_pointer(p.operator->()))
{
    return detail::to_raw_pointer(p.operator->());
}

}  // namespace detail

template <class Duration, class TimeZonePtr>
template <class TimeZonePtr2>
inline
TimeZonePtr2&&
zoned_time<Duration, TimeZonePtr>::check(TimeZonePtr2&& p)
{
    if (detail::to_raw_pointer(p) == nullptr)
        throw std::runtime_error(
            "zoned_time constructed with a time zone pointer == nullptr");
    return std::forward<TimeZonePtr2>(p);
}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time()
    : zone_(check(zoned_traits<TimeZonePtr>::default_zone()))
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const sys_time<Duration>& st)
    : zone_(check(zoned_traits<TimeZonePtr>::default_zone()))
    , tp_(st)
    {}

template <class Duration, class TimeZonePtr>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(TimeZonePtr z)
    : zone_(check(std::move(z)))
    {}

#if HAS_STRING_VIEW

template <class Duration, class TimeZonePtr>
template <class T, class>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(std::string_view name)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name))
    {}

#else  // !HAS_STRING_VIEW

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const std::string& name)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name))
    {}

#endif  // !HAS_STRING_VIEW

template <class Duration, class TimeZonePtr>
template <class Duration2, class>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const zoned_time<Duration2, TimeZonePtr>& zt) NOEXCEPT
    : zone_(zt.zone_)
    , tp_(zt.tp_)
    {}

template <class Duration, class TimeZonePtr>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(TimeZonePtr z, const sys_time<Duration>& st)
    : zone_(check(std::move(z)))
    , tp_(st)
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(TimeZonePtr z, const local_time<Duration>& t)
    : zone_(check(std::move(z)))
    , tp_(zone_->to_sys(t))
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(TimeZonePtr z, const local_time<Duration>& t,
                                              choose c)
    : zone_(check(std::move(z)))
    , tp_(zone_->to_sys(t, c))
    {}

template <class Duration, class TimeZonePtr>
template <class Duration2, class TimeZonePtr2, class>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(TimeZonePtr z,
                                              const zoned_time<Duration2, TimeZonePtr2>& zt)
    : zone_(check(std::move(z)))
    , tp_(zt.tp_)
    {}

template <class Duration, class TimeZonePtr>
template <class Duration2, class TimeZonePtr2, class>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(TimeZonePtr z,
                                      const zoned_time<Duration2, TimeZonePtr2>& zt, choose)
    : zoned_time(std::move(z), zt)
    {}

#if HAS_STRING_VIEW

template <class Duration, class TimeZonePtr>
template <class T, class>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(std::string_view name,
                                              detail::nodeduct_t<const sys_time<Duration>&> st)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), st)
    {}

template <class Duration, class TimeZonePtr>
template <class T, class>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(std::string_view name,
                                              detail::nodeduct_t<const local_time<Duration>&> t)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), t)
    {}

template <class Duration, class TimeZonePtr>
template <class T, class>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(std::string_view name,
                                              detail::nodeduct_t<const local_time<Duration>&> t, choose c)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), t, c)
    {}

template <class Duration, class TimeZonePtr>
template <class Duration2, class TimeZonePtr2, class, class>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(std::string_view name,
                                              const zoned_time<Duration2, TimeZonePtr2>& zt)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), zt)
    {}

template <class Duration, class TimeZonePtr>
template <class Duration2, class TimeZonePtr2, class, class>
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(std::string_view name,
                                              const zoned_time<Duration2, TimeZonePtr2>& zt,
                                              choose c)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), zt, c)
    {}

#else  // !HAS_STRING_VIEW

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const std::string& name,
                                              const sys_time<Duration>& st)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), st)
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const char* name,
                                              const sys_time<Duration>& st)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), st)
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const std::string& name,
                                              const local_time<Duration>& t)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), t)
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const char* name,
                                              const local_time<Duration>& t)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), t)
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const std::string& name,
                                              const local_time<Duration>& t, choose c)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), t, c)
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class T, class>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const char* name,
                                              const local_time<Duration>& t, choose c)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), t, c)
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class Duration2, class TimeZonePtr2, class, class>
#else
template <class Duration2, class TimeZonePtr2>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const std::string& name,
                                              const zoned_time<Duration2, TimeZonePtr2>& zt)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), zt)
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class Duration2, class TimeZonePtr2, class, class>
#else
template <class Duration2, class TimeZonePtr2>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const char* name,
                                              const zoned_time<Duration2, TimeZonePtr2>& zt)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), zt)
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class Duration2, class TimeZonePtr2, class, class>
#else
template <class Duration2, class TimeZonePtr2>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const std::string& name,
                                              const zoned_time<Duration2, TimeZonePtr2>& zt,
                                              choose c)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), zt, c)
    {}

template <class Duration, class TimeZonePtr>
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
template <class Duration2, class TimeZonePtr2, class, class>
#else
template <class Duration2, class TimeZonePtr2>
#endif
inline
zoned_time<Duration, TimeZonePtr>::zoned_time(const char* name,
                                              const zoned_time<Duration2, TimeZonePtr2>& zt,
                                              choose c)
    : zoned_time(zoned_traits<TimeZonePtr>::locate_zone(name), zt, c)
    {}

#endif  // HAS_STRING_VIEW

template <class Duration, class TimeZonePtr>
inline
zoned_time<Duration, TimeZonePtr>&
zoned_time<Duration, TimeZonePtr>::operator=(const sys_time<Duration>& st)
{
    tp_ = st;
    return *this;
}

template <class Duration, class TimeZonePtr>
inline
zoned_time<Duration, TimeZonePtr>&
zoned_time<Duration, TimeZonePtr>::operator=(const local_time<Duration>& ut)
{
    tp_ = zone_->to_sys(ut);
    return *this;
}

template <class Duration, class TimeZonePtr>
inline
zoned_time<Duration, TimeZonePtr>::operator local_time<typename zoned_time<Duration, TimeZonePtr>::duration>() const
{
    return get_local_time();
}

template <class Duration, class TimeZonePtr>
inline
zoned_time<Duration, TimeZonePtr>::operator sys_time<typename zoned_time<Duration, TimeZonePtr>::duration>() const
{
    return get_sys_time();
}

template <class Duration, class TimeZonePtr>
inline
TimeZonePtr
zoned_time<Duration, TimeZonePtr>::get_time_zone() const
{
    return zone_;
}

template <class Duration, class TimeZonePtr>
inline
local_time<typename zoned_time<Duration, TimeZonePtr>::duration>
zoned_time<Duration, TimeZonePtr>::get_local_time() const
{
    return zone_->to_local(tp_);
}

template <class Duration, class TimeZonePtr>
inline
sys_time<typename zoned_time<Duration, TimeZonePtr>::duration>
zoned_time<Duration, TimeZonePtr>::get_sys_time() const
{
    return tp_;
}

template <class Duration, class TimeZonePtr>
inline
sys_info
zoned_time<Duration, TimeZonePtr>::get_info() const
{
    return zone_->get_info(tp_);
}

// make_zoned_time

inline
zoned_time<std::chrono::seconds>
make_zoned()
{
    return zoned_time<std::chrono::seconds>();
}

template <class Duration>
inline
zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type>
make_zoned(const sys_time<Duration>& tp)
{
    return zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type>(tp);
}

template <class TimeZonePtr
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
#if !defined(__INTEL_COMPILER) || (__INTEL_COMPILER > 1600)
          , class = typename std::enable_if
          <
            std::is_class
            <
                typename std::decay
                <
                    decltype(*detail::to_raw_pointer(std::declval<TimeZonePtr&>()))
                >::type
            >{}
          >::type
#endif
#endif
         >
inline
zoned_time<std::chrono::seconds, TimeZonePtr>
make_zoned(TimeZonePtr z)
{
    return zoned_time<std::chrono::seconds, TimeZonePtr>(std::move(z));
}

inline
zoned_seconds
make_zoned(const std::string& name)
{
    return zoned_seconds(name);
}

template <class Duration, class TimeZonePtr
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
#if !defined(__INTEL_COMPILER) || (__INTEL_COMPILER > 1600)
          , class = typename std::enable_if
          <
            std::is_class<typename std::decay<decltype(*std::declval<TimeZonePtr&>())>::type>{}
          >::type
#endif
#endif
         >
inline
zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type, TimeZonePtr>
make_zoned(TimeZonePtr zone, const local_time<Duration>& tp)
{
    return zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type,
                      TimeZonePtr>(std::move(zone), tp);
}

template <class Duration, class TimeZonePtr
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
#if !defined(__INTEL_COMPILER) || (__INTEL_COMPILER > 1600)
          , class = typename std::enable_if
          <
            std::is_class<typename std::decay<decltype(*std::declval<TimeZonePtr&>())>::type>{}
          >::type
#endif
#endif
         >
inline
zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type, TimeZonePtr>
make_zoned(TimeZonePtr zone, const local_time<Duration>& tp, choose c)
{
    return zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type,
                      TimeZonePtr>(std::move(zone), tp, c);
}

template <class Duration>
inline
zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type>
make_zoned(const std::string& name, const local_time<Duration>& tp)
{
    return zoned_time<typename std::common_type<Duration,
                      std::chrono::seconds>::type>(name, tp);
}

template <class Duration>
inline
zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type>
make_zoned(const std::string& name, const local_time<Duration>& tp, choose c)
{
    return zoned_time<typename std::common_type<Duration,
                      std::chrono::seconds>::type>(name, tp, c);
}

template <class Duration, class TimeZonePtr>
inline
zoned_time<Duration, TimeZonePtr>
make_zoned(TimeZonePtr zone, const zoned_time<Duration, TimeZonePtr>& zt)
{
    return zoned_time<Duration, TimeZonePtr>(std::move(zone), zt);
}

template <class Duration, class TimeZonePtr>
inline
zoned_time<Duration, TimeZonePtr>
make_zoned(const std::string& name, const zoned_time<Duration, TimeZonePtr>& zt)
{
    return zoned_time<Duration, TimeZonePtr>(name, zt);
}

template <class Duration, class TimeZonePtr>
inline
zoned_time<Duration, TimeZonePtr>
make_zoned(TimeZonePtr zone, const zoned_time<Duration, TimeZonePtr>& zt, choose c)
{
    return zoned_time<Duration, TimeZonePtr>(std::move(zone), zt, c);
}

template <class Duration, class TimeZonePtr>
inline
zoned_time<Duration, TimeZonePtr>
make_zoned(const std::string& name, const zoned_time<Duration, TimeZonePtr>& zt, choose c)
{
    return zoned_time<Duration, TimeZonePtr>(name, zt, c);
}

template <class Duration, class TimeZonePtr
#if !defined(_MSC_VER) || (_MSC_VER > 1916)
#if !defined(__INTEL_COMPILER) || (__INTEL_COMPILER > 1600)
          , class = typename std::enable_if
          <
            std::is_class<typename std::decay<decltype(*std::declval<TimeZonePtr&>())>::type>{}
          >::type
#endif
#endif
         >
inline
zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type, TimeZonePtr>
make_zoned(TimeZonePtr zone, const sys_time<Duration>& st)
{
    return zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type,
                      TimeZonePtr>(std::move(zone), st);
}

template <class Duration>
inline
zoned_time<typename std::common_type<Duration, std::chrono::seconds>::type>
make_zoned(const std::string& name, const sys_time<Duration>& st)
{
    return zoned_time<typename std::common_type<Duration,
                      std::chrono::seconds>::type>(name, st);
}

template <class CharT, class Traits, class Duration, class TimeZonePtr>
std::basic_ostream<CharT, Traits>&
to_stream(std::basic_ostream<CharT, Traits>& os, const CharT* fmt,
          const zoned_time<Duration, TimeZonePtr>& tp)
{
    using duration = typename zoned_time<Duration, TimeZonePtr>::duration;
    using LT = local_time<duration>;
    auto const st = tp.get_sys_time();
    auto const info = tp.get_time_zone()->get_info(st);
    return to_stream(os, fmt, LT{(st+info.offset).time_since_epoch()},
                     &info.abbrev, &info.offset);
}

template <class CharT, class Traits, class Duration, class TimeZonePtr>
inline
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const zoned_time<Duration, TimeZonePtr>& t)
{
    const CharT fmt[] = {'%', 'F', ' ', '%', 'T', ' ', '%', 'Z', CharT{}};
    return to_stream(os, fmt, t);
}

class utc_clock
{
public:
    using duration                  = std::chrono::system_clock::duration;
    using rep                       = duration::rep;
    using period                    = duration::period;
    using time_point                = std::chrono::time_point<utc_clock>;
    static CONSTDATA bool is_steady = false;

    static time_point now();

    template<typename Duration>
    static
    std::chrono::time_point<std::chrono::system_clock, typename std::common_type<Duration, std::chrono::seconds>::type>
    to_sys(const std::chrono::time_point<utc_clock, Duration>&);

    template<typename Duration>
    static
    std::chrono::time_point<utc_clock, typename std::common_type<Duration, std::chrono::seconds>::type>
    from_sys(const std::chrono::time_point<std::chrono::system_clock, Duration>&);

    template<typename Duration>
    static
    std::chrono::time_point<local_t, typename std::common_type<Duration, std::chrono::seconds>::type>
    to_local(const std::chrono::time_point<utc_clock, Duration>&);

    template<typename Duration>
    static
    std::chrono::time_point<utc_clock, typename std::common_type<Duration, std::chrono::seconds>::type>
    from_local(const std::chrono::time_point<local_t, Duration>&);
};

template <class Duration>
    using utc_time = std::chrono::time_point<utc_clock, Duration>;

using utc_seconds = utc_time<std::chrono::seconds>;

template <class Duration>
utc_time<typename std::common_type<Duration, std::chrono::seconds>::type>
utc_clock::from_sys(const sys_time<Duration>& st)
{
    using std::chrono::seconds;
    using CD = typename std::common_type<Duration, seconds>::type;
    auto const& leaps = get_tzdb().leap_seconds;
    auto const lt = std::upper_bound(leaps.begin(), leaps.end(), st);
    return utc_time<CD>{st.time_since_epoch() + seconds{lt-leaps.begin()}};
}

// Return pair<is_leap_second, seconds{number_of_leap_seconds_since_1970}>
// first is true if ut is during a leap second insertion, otherwise false.
// If ut is during a leap second insertion, that leap second is included in the count
template <class Duration>
std::pair<bool, std::chrono::seconds>
is_leap_second(date::utc_time<Duration> const& ut)
{
    using std::chrono::seconds;
    using duration = typename std::common_type<Duration, seconds>::type;
    auto const& leaps = get_tzdb().leap_seconds;
    auto tp = sys_time<duration>{ut.time_since_epoch()};
    auto const lt = std::upper_bound(leaps.begin(), leaps.end(), tp);
    auto ds = seconds{lt-leaps.begin()};
    tp -= ds;
    auto ls = false;
    if (lt > leaps.begin())
    {
        if (tp < lt[-1])
        {
            if (tp >= lt[-1].date() - seconds{1})
                ls = true;
            else
                --ds;
        }
    }
    return {ls, ds};
}

struct leap_second_info
{
    bool is_leap_second;
    std::chrono::seconds elapsed;
};

template <class Duration>
leap_second_info
get_leap_second_info(date::utc_time<Duration> const& ut)
{
    auto p = is_leap_second(ut);
    return {p.first, p.second};
}

template <class Duration>
sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
utc_clock::to_sys(const utc_time<Duration>& ut)
{
    using std::chrono::seconds;
    using CD = typename std::common_type<Duration, seconds>::type;
    auto ls = is_leap_second(ut);
    auto tp = sys_time<CD>{ut.time_since_epoch() - ls.second};
    if (ls.first)
        tp = floor<seconds>(tp) + seconds{1} - CD{1};
    return tp;
}

inline
utc_clock::time_point
utc_clock::now()
{
    return from_sys(std::chrono::system_clock::now());
}

template <class Duration>
utc_time<typename std::common_type<Duration, std::chrono::seconds>::type>
utc_clock::from_local(const local_time<Duration>& st)
{
    return from_sys(sys_time<Duration>{st.time_since_epoch()});
}

template <class Duration>
local_time<typename std::common_type<Duration, std::chrono::seconds>::type>
utc_clock::to_local(const utc_time<Duration>& ut)
{
    using CD = typename std::common_type<Duration, std::chrono::seconds>::type;
    return local_time<CD>{to_sys(ut).time_since_epoch()};
}

template <class CharT, class Traits, class Duration>
std::basic_ostream<CharT, Traits>&
to_stream(std::basic_ostream<CharT, Traits>& os, const CharT* fmt,
          const utc_time<Duration>& t)
{
    using std::chrono::seconds;
    using CT = typename std::common_type<Duration, seconds>::type;
    const std::string abbrev("UTC");
    CONSTDATA seconds offset{0};
    auto ls = is_leap_second(t);
    auto tp = sys_time<CT>{t.time_since_epoch() - ls.second};
    auto const sd = floor<days>(tp);
    year_month_day ymd = sd;
    auto time = make_time(tp - sys_seconds{sd});
    time.seconds(detail::undocumented{}) += seconds{ls.first};
    fields<CT> fds{ymd, time};
    return to_stream(os, fmt, fds, &abbrev, &offset);
}

template <class CharT, class Traits, class Duration>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const utc_time<Duration>& t)
{
    const CharT fmt[] = {'%', 'F', ' ', '%', 'T', CharT{}};
    return to_stream(os, fmt, t);
}

template <class Duration, class CharT, class Traits, class Alloc = std::allocator<CharT>>
std::basic_istream<CharT, Traits>&
from_stream(std::basic_istream<CharT, Traits>& is, const CharT* fmt,
            utc_time<Duration>& tp, std::basic_string<CharT, Traits, Alloc>* abbrev = nullptr,
            std::chrono::minutes* offset = nullptr)
{
    using std::chrono::seconds;
    using std::chrono::minutes;
    using CT = typename std::common_type<Duration, seconds>::type;
    minutes offset_local{};
    auto offptr = offset ? offset : &offset_local;
    fields<CT> fds{};
    fds.has_tod = true;
    from_stream(is, fmt, fds, abbrev, offptr);
    if (!fds.ymd.ok())
        is.setstate(std::ios::failbit);
    if (!is.fail())
    {
        bool is_60_sec = fds.tod.seconds() == seconds{60};
        if (is_60_sec)
            fds.tod.seconds(detail::undocumented{}) -= seconds{1};
        auto tmp = utc_clock::from_sys(sys_days(fds.ymd) - *offptr + fds.tod.to_duration());
        if (is_60_sec)
            tmp += seconds{1};
        if (is_60_sec != is_leap_second(tmp).first || !fds.tod.in_conventional_range())
        {
            is.setstate(std::ios::failbit);
            return is;
        }
        tp = std::chrono::time_point_cast<Duration>(tmp);
    }
    return is;
}

// tai_clock

class tai_clock
{
public:
    using duration                  = std::chrono::system_clock::duration;
    using rep                       = duration::rep;
    using period                    = duration::period;
    using time_point                = std::chrono::time_point<tai_clock>;
    static const bool is_steady     = false;

    static time_point now();

    template<typename Duration>
    static
    std::chrono::time_point<utc_clock, typename std::common_type<Duration, std::chrono::seconds>::type>
    to_utc(const std::chrono::time_point<tai_clock, Duration>&) NOEXCEPT;

    template<typename Duration>
    static
    std::chrono::time_point<tai_clock, typename std::common_type<Duration, std::chrono::seconds>::type>
    from_utc(const std::chrono::time_point<utc_clock, Duration>&) NOEXCEPT;

    template<typename Duration>
    static
    std::chrono::time_point<local_t, typename std::common_type<Duration, date::days>::type>
    to_local(const std::chrono::time_point<tai_clock, Duration>&) NOEXCEPT;

    template<typename Duration>
    static
    std::chrono::time_point<tai_clock, typename std::common_type<Duration, date::days>::type>
    from_local(const std::chrono::time_point<local_t, Duration>&) NOEXCEPT;
};

template <class Duration>
    using tai_time = std::chrono::time_point<tai_clock, Duration>;

using tai_seconds = tai_time<std::chrono::seconds>;

template <class Duration>
inline
utc_time<typename std::common_type<Duration, std::chrono::seconds>::type>
tai_clock::to_utc(const tai_time<Duration>& t) NOEXCEPT
{
    using std::chrono::seconds;
    using CD = typename std::common_type<Duration, seconds>::type;
    return utc_time<CD>{t.time_since_epoch()} -
            (sys_days(year{1970}/January/1) - sys_days(year{1958}/January/1) + seconds{10});
}

template <class Duration>
inline
tai_time<typename std::common_type<Duration, std::chrono::seconds>::type>
tai_clock::from_utc(const utc_time<Duration>& t) NOEXCEPT
{
    using std::chrono::seconds;
    using CD = typename std::common_type<Duration, seconds>::type;
    return tai_time<CD>{t.time_since_epoch()} +
            (sys_days(year{1970}/January/1) - sys_days(year{1958}/January/1) + seconds{10});
}

inline
tai_clock::time_point
tai_clock::now()
{
    return from_utc(utc_clock::now());
}

template <class Duration>
inline
local_time<typename std::common_type<Duration, date::days>::type>
tai_clock::to_local(const tai_time<Duration>& t) NOEXCEPT
{
    using CD = typename std::common_type<Duration, date::days>::type;
    return local_time<CD>{t.time_since_epoch()} -
           (local_days(year{1970}/January/1) - local_days(year{1958}/January/1));
}

template <class Duration>
inline
tai_time<typename std::common_type<Duration, date::days>::type>
tai_clock::from_local(const local_time<Duration>& t) NOEXCEPT
{
    using CD = typename std::common_type<Duration, date::days>::type;
    return tai_time<CD>{t.time_since_epoch()} +
            (local_days(year{1970}/January/1) - local_days(year{1958}/January/1));
}

template <class CharT, class Traits, class Duration>
std::basic_ostream<CharT, Traits>&
to_stream(std::basic_ostream<CharT, Traits>& os, const CharT* fmt,
          const tai_time<Duration>& t)
{
    const std::string abbrev("TAI");
    CONSTDATA std::chrono::seconds offset{0};
    return to_stream(os, fmt, tai_clock::to_local(t), &abbrev, &offset);
}

template <class CharT, class Traits, class Duration>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const tai_time<Duration>& t)
{
    const CharT fmt[] = {'%', 'F', ' ', '%', 'T', CharT{}};
    return to_stream(os, fmt, t);
}

template <class Duration, class CharT, class Traits, class Alloc = std::allocator<CharT>>
std::basic_istream<CharT, Traits>&
from_stream(std::basic_istream<CharT, Traits>& is, const CharT* fmt,
            tai_time<Duration>& tp,
            std::basic_string<CharT, Traits, Alloc>* abbrev = nullptr,
            std::chrono::minutes* offset = nullptr)
{
    local_time<Duration> lp;
    from_stream(is, fmt, lp, abbrev, offset);
    if (!is.fail())
        tp = tai_clock::from_local(lp);
    return is;
}

// gps_clock

class gps_clock
{
public:
    using duration                  = std::chrono::system_clock::duration;
    using rep                       = duration::rep;
    using period                    = duration::period;
    using time_point                = std::chrono::time_point<gps_clock>;
    static const bool is_steady     = false;

    static time_point now();

    template<typename Duration>
    static
    std::chrono::time_point<utc_clock, typename std::common_type<Duration, std::chrono::seconds>::type>
    to_utc(const std::chrono::time_point<gps_clock, Duration>&) NOEXCEPT;

    template<typename Duration>
    static
    std::chrono::time_point<gps_clock, typename std::common_type<Duration, std::chrono::seconds>::type>
    from_utc(const std::chrono::time_point<utc_clock, Duration>&) NOEXCEPT;

    template<typename Duration>
    static
    std::chrono::time_point<local_t, typename std::common_type<Duration, date::days>::type>
    to_local(const std::chrono::time_point<gps_clock, Duration>&) NOEXCEPT;

    template<typename Duration>
    static
    std::chrono::time_point<gps_clock, typename std::common_type<Duration, date::days>::type>
    from_local(const std::chrono::time_point<local_t, Duration>&) NOEXCEPT;
};

template <class Duration>
    using gps_time = std::chrono::time_point<gps_clock, Duration>;

using gps_seconds = gps_time<std::chrono::seconds>;

template <class Duration>
inline
utc_time<typename std::common_type<Duration, std::chrono::seconds>::type>
gps_clock::to_utc(const gps_time<Duration>& t) NOEXCEPT
{
    using std::chrono::seconds;
    using CD = typename std::common_type<Duration, seconds>::type;
    return utc_time<CD>{t.time_since_epoch()} +
            (sys_days(year{1980}/January/Sunday[1]) - sys_days(year{1970}/January/1) +
             seconds{9});
}

template <class Duration>
inline
gps_time<typename std::common_type<Duration, std::chrono::seconds>::type>
gps_clock::from_utc(const utc_time<Duration>& t) NOEXCEPT
{
    using std::chrono::seconds;
    using CD = typename std::common_type<Duration, seconds>::type;
    return gps_time<CD>{t.time_since_epoch()} -
            (sys_days(year{1980}/January/Sunday[1]) - sys_days(year{1970}/January/1) +
             seconds{9});
}

inline
gps_clock::time_point
gps_clock::now()
{
    return from_utc(utc_clock::now());
}

template <class Duration>
inline
local_time<typename std::common_type<Duration, date::days>::type>
gps_clock::to_local(const gps_time<Duration>& t) NOEXCEPT
{
    using CD = typename std::common_type<Duration, date::days>::type;
    return local_time<CD>{t.time_since_epoch()} +
            (local_days(year{1980}/January/Sunday[1]) - local_days(year{1970}/January/1));
}

template <class Duration>
inline
gps_time<typename std::common_type<Duration, date::days>::type>
gps_clock::from_local(const local_time<Duration>& t) NOEXCEPT
{
    using CD = typename std::common_type<Duration, date::days>::type;
    return gps_time<CD>{t.time_since_epoch()} -
            (local_days(year{1980}/January/Sunday[1]) - local_days(year{1970}/January/1));
}


template <class CharT, class Traits, class Duration>
std::basic_ostream<CharT, Traits>&
to_stream(std::basic_ostream<CharT, Traits>& os, const CharT* fmt,
          const gps_time<Duration>& t)
{
    const std::string abbrev("GPS");
    CONSTDATA std::chrono::seconds offset{0};
    return to_stream(os, fmt, gps_clock::to_local(t), &abbrev, &offset);
}

template <class CharT, class Traits, class Duration>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const gps_time<Duration>& t)
{
    const CharT fmt[] = {'%', 'F', ' ', '%', 'T', CharT{}};
    return to_stream(os, fmt, t);
}

template <class Duration, class CharT, class Traits, class Alloc = std::allocator<CharT>>
std::basic_istream<CharT, Traits>&
from_stream(std::basic_istream<CharT, Traits>& is, const CharT* fmt,
            gps_time<Duration>& tp,
            std::basic_string<CharT, Traits, Alloc>* abbrev = nullptr,
            std::chrono::minutes* offset = nullptr)
{
    local_time<Duration> lp;
    from_stream(is, fmt, lp, abbrev, offset);
    if (!is.fail())
        tp = gps_clock::from_local(lp);
    return is;
}

// clock_time_conversion

template <class DstClock, class SrcClock>
struct clock_time_conversion
{};

template <>
struct clock_time_conversion<std::chrono::system_clock, std::chrono::system_clock>
{
    template <class Duration>
    CONSTCD14
    sys_time<Duration>
    operator()(const sys_time<Duration>& st) const
    {
        return st;
    }
};

template <>
struct clock_time_conversion<utc_clock, utc_clock>
{
    template <class Duration>
    CONSTCD14
    utc_time<Duration>
    operator()(const utc_time<Duration>& ut) const
    {
        return ut;
    }
};

template<>
struct clock_time_conversion<local_t, local_t>
{
    template <class Duration>
    CONSTCD14
    local_time<Duration>
    operator()(const local_time<Duration>& lt) const
    {
        return lt;
    }
};

template <>
struct clock_time_conversion<utc_clock, std::chrono::system_clock>
{
    template <class Duration>
    utc_time<typename std::common_type<Duration, std::chrono::seconds>::type>
    operator()(const sys_time<Duration>& st) const
    {
        return utc_clock::from_sys(st);
    }
};

template <>
struct clock_time_conversion<std::chrono::system_clock, utc_clock>
{
    template <class Duration>
    sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
    operator()(const utc_time<Duration>& ut) const
    {
        return utc_clock::to_sys(ut);
    }
};

template<>
struct clock_time_conversion<local_t, std::chrono::system_clock>
{
    template <class Duration>
    CONSTCD14
    local_time<Duration>
    operator()(const sys_time<Duration>& st) const
    {
       return local_time<Duration>{st.time_since_epoch()};
    }
};

template<>
struct clock_time_conversion<std::chrono::system_clock, local_t>
{
    template <class Duration>
    CONSTCD14
    sys_time<Duration>
    operator()(const local_time<Duration>& lt) const
    {
        return sys_time<Duration>{lt.time_since_epoch()};
    }
};

template<>
struct clock_time_conversion<utc_clock, local_t>
{
    template <class Duration>
    utc_time<typename std::common_type<Duration, std::chrono::seconds>::type>
    operator()(const local_time<Duration>& lt) const
    {
       return utc_clock::from_local(lt);
    }
};

template<>
struct clock_time_conversion<local_t, utc_clock>
{
    template <class Duration>
    local_time<typename std::common_type<Duration, std::chrono::seconds>::type>
    operator()(const utc_time<Duration>& ut) const
    {
       return utc_clock::to_local(ut);
    }
};

template<typename Clock>
struct clock_time_conversion<Clock, Clock>
{
    template <class Duration>
    CONSTCD14
    std::chrono::time_point<Clock, Duration>
    operator()(const std::chrono::time_point<Clock, Duration>& tp) const
    {
        return tp;
    }
};

namespace ctc_detail
{

template <class Clock, class Duration>
    using time_point = std::chrono::time_point<Clock, Duration>;

using std::declval;
using std::chrono::system_clock;

//Check if TimePoint is time for given clock,
//if not emits hard error
template <class Clock, class TimePoint>
struct return_clock_time
{
    using clock_time_point = time_point<Clock, typename TimePoint::duration>;
    using type             = TimePoint;

    static_assert(std::is_same<TimePoint, clock_time_point>::value,
                  "time point with appropariate clock shall be returned");
};

// Check if Clock has to_sys method accepting TimePoint with given duration const& and
// returning sys_time. If so has nested type member equal to return type to_sys.
template <class Clock, class Duration, class = void>
struct return_to_sys
{};

template <class Clock, class Duration>
struct return_to_sys
       <
           Clock, Duration,
           decltype(Clock::to_sys(declval<time_point<Clock, Duration> const&>()), void())
       >
    : return_clock_time
      <
          system_clock,
          decltype(Clock::to_sys(declval<time_point<Clock, Duration> const&>()))
      >
{};

// Similiar to above
template <class Clock, class Duration, class = void>
struct return_from_sys
{};

template <class Clock, class Duration>
struct return_from_sys
       <
           Clock, Duration,
           decltype(Clock::from_sys(declval<time_point<system_clock, Duration> const&>()),
                    void())
       >
    : return_clock_time
      <
          Clock,
          decltype(Clock::from_sys(declval<time_point<system_clock, Duration> const&>()))
      >
{};

// Similiar to above
template <class Clock, class Duration, class = void>
struct return_to_utc
{};

template <class Clock, class Duration>
struct return_to_utc
       <
           Clock, Duration,
           decltype(Clock::to_utc(declval<time_point<Clock, Duration> const&>()), void())
       >
    : return_clock_time
      <
          utc_clock,
          decltype(Clock::to_utc(declval<time_point<Clock, Duration> const&>()))>
{};

// Similiar to above
template <class Clock, class Duration, class = void>
struct return_from_utc
{};

template <class Clock, class Duration>
struct return_from_utc
       <
           Clock, Duration,
           decltype(Clock::from_utc(declval<time_point<utc_clock, Duration> const&>()),
                    void())
       >
    : return_clock_time
      <
          Clock,
          decltype(Clock::from_utc(declval<time_point<utc_clock, Duration> const&>()))
      >
{};

// Similiar to above
template<typename Clock, typename Duration, typename = void>
struct return_to_local
{};

template<typename Clock, typename Duration>
struct return_to_local
       <
          Clock, Duration,
          decltype(Clock::to_local(declval<time_point<Clock, Duration> const&>()),
                   void())
       >
     : return_clock_time
       <
           local_t,
           decltype(Clock::to_local(declval<time_point<Clock, Duration> const&>()))
       >
{};

// Similiar to above
template<typename Clock, typename Duration, typename = void>
struct return_from_local
{};

template<typename Clock, typename Duration>
struct return_from_local
       <
           Clock, Duration,
           decltype(Clock::from_local(declval<time_point<local_t, Duration> const&>()),
                    void())
       >
     : return_clock_time
       <
           Clock,
           decltype(Clock::from_local(declval<time_point<local_t, Duration> const&>()))
       >
{};

}  // namespace ctc_detail

template <class SrcClock>
struct clock_time_conversion<std::chrono::system_clock, SrcClock>
{
    template <class Duration>
    CONSTCD14
    typename ctc_detail::return_to_sys<SrcClock, Duration>::type
    operator()(const std::chrono::time_point<SrcClock, Duration>& tp) const
    {
        return SrcClock::to_sys(tp);
    }
};

template <class DstClock>
struct clock_time_conversion<DstClock, std::chrono::system_clock>
{
    template <class Duration>
    CONSTCD14
    typename ctc_detail::return_from_sys<DstClock, Duration>::type
    operator()(const sys_time<Duration>& st) const
    {
        return DstClock::from_sys(st);
    }
};

template <class SrcClock>
struct clock_time_conversion<utc_clock, SrcClock>
{
    template <class Duration>
    CONSTCD14
    typename ctc_detail::return_to_utc<SrcClock, Duration>::type
    operator()(const std::chrono::time_point<SrcClock, Duration>& tp) const
    {
        return SrcClock::to_utc(tp);
    }
};

template <class DstClock>
struct clock_time_conversion<DstClock, utc_clock>
{
    template <class Duration>
    CONSTCD14
    typename ctc_detail::return_from_utc<DstClock, Duration>::type
    operator()(const utc_time<Duration>& ut) const
    {
        return DstClock::from_utc(ut);
    }
};

template<typename SrcClock>
struct clock_time_conversion<local_t, SrcClock>
{
    template <class Duration>
    CONSTCD14
    typename ctc_detail::return_to_local<SrcClock, Duration>::type
    operator()(const std::chrono::time_point<SrcClock, Duration>& tp) const
    {
        return SrcClock::to_local(tp);
    }
};

template<typename DstClock>
struct clock_time_conversion<DstClock, local_t>
{
    template <class Duration>
    CONSTCD14
    typename ctc_detail::return_from_local<DstClock, Duration>::type
    operator()(const local_time<Duration>& lt) const
    {
        return DstClock::from_local(lt);
    }
};

namespace clock_cast_detail
{

template <class Clock, class Duration>
    using time_point = std::chrono::time_point<Clock, Duration>;
using std::chrono::system_clock;

template <class DstClock, class SrcClock, class Duration>
CONSTCD14
auto
conv_clock(const time_point<SrcClock, Duration>& t)
    -> decltype(std::declval<clock_time_conversion<DstClock, SrcClock>>()(t))
{
    return clock_time_conversion<DstClock, SrcClock>{}(t);
}

//direct trait conversion, 1st candidate
template <class DstClock, class SrcClock, class Duration>
CONSTCD14
auto
cc_impl(const time_point<SrcClock, Duration>& t, const time_point<SrcClock, Duration>*)
    -> decltype(conv_clock<DstClock>(t))
{
    return conv_clock<DstClock>(t);
}

//conversion through sys, 2nd candidate
template <class DstClock, class SrcClock, class Duration>
CONSTCD14
auto
cc_impl(const time_point<SrcClock, Duration>& t, const void*)
    -> decltype(conv_clock<DstClock>(conv_clock<system_clock>(t)))
{
    return conv_clock<DstClock>(conv_clock<system_clock>(t));
}

//conversion through utc, 2nd candidate
template <class DstClock, class SrcClock, class Duration>
CONSTCD14
auto
cc_impl(const time_point<SrcClock, Duration>& t, const void*)
    -> decltype(0,  // MSVC_WORKAROUND
                conv_clock<DstClock>(conv_clock<utc_clock>(t)))
{
    return conv_clock<DstClock>(conv_clock<utc_clock>(t));
}

//conversion through sys and utc, 3rd candidate
template <class DstClock, class SrcClock, class Duration>
CONSTCD14
auto
cc_impl(const time_point<SrcClock, Duration>& t, ...)
    -> decltype(conv_clock<DstClock>(conv_clock<utc_clock>(conv_clock<system_clock>(t))))
{
    return conv_clock<DstClock>(conv_clock<utc_clock>(conv_clock<system_clock>(t)));
}

//conversion through utc and sys, 3rd candidate
template <class DstClock, class SrcClock, class Duration>
CONSTCD14
auto
cc_impl(const time_point<SrcClock, Duration>& t, ...)
    -> decltype(0,  // MSVC_WORKAROUND
                conv_clock<DstClock>(conv_clock<system_clock>(conv_clock<utc_clock>(t))))
{
    return conv_clock<DstClock>(conv_clock<system_clock>(conv_clock<utc_clock>(t)));
}

}  // namespace clock_cast_detail

template <class DstClock, class SrcClock, class Duration>
CONSTCD14
auto
clock_cast(const std::chrono::time_point<SrcClock, Duration>& tp)
    -> decltype(clock_cast_detail::cc_impl<DstClock>(tp, &tp))
{
    return clock_cast_detail::cc_impl<DstClock>(tp, &tp);
}

// Deprecated API

template <class Duration>
inline
sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_sys_time(const utc_time<Duration>& t)
{
    return utc_clock::to_sys(t);
}

template <class Duration>
inline
sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_sys_time(const tai_time<Duration>& t)
{
    return utc_clock::to_sys(tai_clock::to_utc(t));
}

template <class Duration>
inline
sys_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_sys_time(const gps_time<Duration>& t)
{
    return utc_clock::to_sys(gps_clock::to_utc(t));
}


template <class Duration>
inline
utc_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_utc_time(const sys_time<Duration>& t)
{
    return utc_clock::from_sys(t);
}

template <class Duration>
inline
utc_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_utc_time(const tai_time<Duration>& t)
{
    return tai_clock::to_utc(t);
}

template <class Duration>
inline
utc_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_utc_time(const gps_time<Duration>& t)
{
    return gps_clock::to_utc(t);
}


template <class Duration>
inline
tai_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_tai_time(const sys_time<Duration>& t)
{
    return tai_clock::from_utc(utc_clock::from_sys(t));
}

template <class Duration>
inline
tai_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_tai_time(const utc_time<Duration>& t)
{
    return tai_clock::from_utc(t);
}

template <class Duration>
inline
tai_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_tai_time(const gps_time<Duration>& t)
{
    return tai_clock::from_utc(gps_clock::to_utc(t));
}


template <class Duration>
inline
gps_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_gps_time(const sys_time<Duration>& t)
{
    return gps_clock::from_utc(utc_clock::from_sys(t));
}

template <class Duration>
inline
gps_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_gps_time(const utc_time<Duration>& t)
{
    return gps_clock::from_utc(t);
}

template <class Duration>
inline
gps_time<typename std::common_type<Duration, std::chrono::seconds>::type>
to_gps_time(const tai_time<Duration>& t)
{
    return gps_clock::from_utc(tai_clock::to_utc(t));
}

}  // namespace date
}  // namespace arrow_vendored

#endif  // TZ_H
