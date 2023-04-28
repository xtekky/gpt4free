// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

namespace parquet {

/// \brief Feature selection when writing Parquet files
///
/// `ParquetVersion::type` governs which data types are allowed and how they
/// are represented. For example, uint32_t data will be written differently
/// depending on this value (as INT64 for PARQUET_1_0, as UINT32 for other
/// versions).
///
/// However, some features - such as compression algorithms, encryption,
/// or the improved "v2" data page format - must be enabled separately in
/// ArrowWriterProperties.
struct ParquetVersion {
  enum type : int {
    /// Enable only pre-2.2 Parquet format features when writing
    ///
    /// This setting is useful for maximum compatibility with legacy readers.
    /// Note that logical types may still be emitted, as long they have a
    /// corresponding converted type.
    PARQUET_1_0,

    /// DEPRECATED: Enable Parquet format 2.6 features
    ///
    /// This misleadingly named enum value is roughly similar to PARQUET_2_6.
    PARQUET_2_0 ARROW_DEPRECATED_ENUM_VALUE("use PARQUET_2_4 or PARQUET_2_6 "
                                            "for fine-grained feature selection"),

    /// Enable Parquet format 2.4 and earlier features when writing
    ///
    /// This enables UINT32 as well as logical types which don't have
    /// a corresponding converted type.
    ///
    /// Note: Parquet format 2.4.0 was released in October 2017.
    PARQUET_2_4,

    /// Enable Parquet format 2.6 and earlier features when writing
    ///
    /// This enables the NANOS time unit in addition to the PARQUET_2_4
    /// features.
    ///
    /// Note: Parquet format 2.6.0 was released in September 2018.
    PARQUET_2_6,

    /// Enable latest Parquet format 2.x features
    ///
    /// This value is equal to the greatest 2.x version supported by
    /// this library.
    PARQUET_2_LATEST = PARQUET_2_6
  };
};

class FileMetaData;
class SchemaDescriptor;

class ReaderProperties;
class ArrowReaderProperties;

class WriterProperties;
class WriterPropertiesBuilder;
class ArrowWriterProperties;
class ArrowWriterPropertiesBuilder;

namespace arrow {

class FileWriter;
class FileReader;

}  // namespace arrow
}  // namespace parquet
