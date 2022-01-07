// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <stdio.h>

#include <algorithm>
#include <sstream>

#include "streaming/onevpl/cfg_params_parser.hpp"
#include "streaming/onevpl/utils.hpp"
#include "logger.hpp"

#ifdef HAVE_ONEVPL
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

template <>
struct ParamCreator<CfgParam> {
    template<typename ValueType>
    CfgParam create (const std::string& name, ValueType&& value) {
        return CfgParam::create(name, std::forward<ValueType>(value), is_major_flag);
    }
    bool is_major_flag = false;
};

template <>
struct ParamCreator<mfxVariant> {
    template<typename ValueType>
    mfxVariant create (const std::string& name, ValueType&& value) {
        static_assert(std::is_same<typename std::decay<ValueType>::type, mfxU32>::value,
                      "ParamCreator<mfxVariant> supports mfxU32 at the moment. "
                      "Feel free to extend for more types");
        return create_impl(name, value);
    }
private:
    mfxVariant create_impl(const std::string&, mfxU32 value) {
        mfxVariant ret;
        ret.Type = MFX_VARIANT_TYPE_U32;
        ret.Data.U32 = value;
        return ret;
    }
};

template<typename ValueType>
std::vector<ValueType> get_params_from_string(const std::string& str) {
    std::vector<ValueType> ret;
    std::string::size_type pos = 0;
    std::string::size_type endline_pos = std::string::npos;
    do
    {
        endline_pos = str.find_first_of("\r\n", pos);
        std::string line = str.substr(pos, endline_pos == std::string::npos ? std::string::npos : endline_pos - pos);
        if (line.empty()) break;

        std::string::size_type name_endline_pos = line.find(':');
        if (name_endline_pos == std::string::npos) {
            throw std::runtime_error("Cannot parse param from string: " + line +
                                     ". Name and value should be separated by \":\"" );
        }

        std::string name = line.substr(0, name_endline_pos);
        std::string value = line.substr(name_endline_pos + 2);

        ParamCreator<ValueType> creator;
        if (name == "mfxImplDescription.Impl") {
            ret.push_back(creator.create<mfxU32>(name, cstr_to_mfx_impl(value.c_str())));
        } else if (name == "mfxImplDescription.mfxDecoderDescription.decoder.CodecID") {
            ret.push_back(creator.create<mfxU32>(name, cstr_to_mfx_codec_id(value.c_str())));
        } else if (name == "mfxImplDescription.AccelerationMode") {
            ret.push_back(creator.create<mfxU32>(name, cstr_to_mfx_accel_mode(value.c_str())));
        } else if (name == "mfxImplDescription.ApiVersion.Version") {
            ret.push_back(creator.create<mfxU32>(name, cstr_to_mfx_version(value.c_str())));
        } else {
            GAPI_LOG_DEBUG(nullptr, "Cannot parse configuration param, name: " << name <<
                                    ", value: " << value);
        }

        pos = endline_pos + 1;
    }
    while (endline_pos != std::string::npos);

    return ret;
}

template
std::vector<CfgParam> get_params_from_string(const std::string& str);
template
std::vector<mfxVariant> get_params_from_string(const std::string& str);

mfxVariant cfg_param_to_mfx_variant(const CfgParam& cfg_val) {
    const CfgParam::name_t& name = cfg_val.get_name();
    mfxVariant ret;
    cv::util::visit(cv::util::overload_lambdas(
            [&ret](uint8_t value)   { ret.Type = MFX_VARIANT_TYPE_U8;   ret.Data.U8 = value;    },
            [&ret](int8_t value)    { ret.Type = MFX_VARIANT_TYPE_I8;   ret.Data.I8 = value;    },
            [&ret](uint16_t value)  { ret.Type = MFX_VARIANT_TYPE_U16;  ret.Data.U16 = value;   },
            [&ret](int16_t value)   { ret.Type = MFX_VARIANT_TYPE_I16;  ret.Data.I16 = value;   },
            [&ret](uint32_t value)  { ret.Type = MFX_VARIANT_TYPE_U32;  ret.Data.U32 = value;   },
            [&ret](int32_t value)   { ret.Type = MFX_VARIANT_TYPE_I32;  ret.Data.I32 = value;   },
            [&ret](uint64_t value)  { ret.Type = MFX_VARIANT_TYPE_U64;  ret.Data.U64 = value;   },
            [&ret](int64_t value)   { ret.Type = MFX_VARIANT_TYPE_I64;  ret.Data.I64 = value;   },
            [&ret](float_t value)   { ret.Type = MFX_VARIANT_TYPE_F32;  ret.Data.F32 = value;   },
            [&ret](double_t value)  { ret.Type = MFX_VARIANT_TYPE_F64;  ret.Data.F64 = value;   },
            [&ret](void* value)     { ret.Type = MFX_VARIANT_TYPE_PTR;  ret.Data.Ptr = value;   },
            [&ret, &name] (const std::string& value) {
                auto parsed = get_params_from_string<mfxVariant>(name + ": " + value + "\n");
                if (parsed.empty()) {
                    throw std::logic_error("Unsupported parameter, name: " + name + ", value: " + value);
                }
                ret = *parsed.begin();
            }), cfg_val.get_value());
    return ret;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
