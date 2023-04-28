//
//  ios.h
//  DateTimeLib
//
// The MIT License (MIT)
//
// Copyright (c) 2016 Alexander Kormanovsky
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

#ifndef ios_hpp
#define ios_hpp

#if __APPLE__
# include <TargetConditionals.h>
# if TARGET_OS_IPHONE
#   include <string>

    namespace arrow_vendored
    {
    namespace date
    {
    namespace iOSUtils
    {

    std::string get_tzdata_path();
    std::string get_current_timezone();

    }  // namespace iOSUtils
    }  // namespace date
    }  // namespace arrow_vendored

# endif  // TARGET_OS_IPHONE
#else   // !__APPLE__
# define TARGET_OS_IPHONE 0
#endif  // !__APPLE__
#endif // ios_hpp
