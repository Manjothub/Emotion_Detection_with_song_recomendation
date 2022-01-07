// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#include <cstdlib>
#include <exception>

#include "streaming/onevpl/accelerators/accel_policy_cpu.hpp"
#include "streaming/onevpl/accelerators/surface/cpu_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "logger.hpp"

#ifdef _WIN32
    #include <windows.h>
    #include <sysinfoapi.h>
#endif
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

VPLCPUAccelerationPolicy::VPLCPUAccelerationPolicy() {
    GAPI_LOG_INFO(nullptr, "created");
}

VPLCPUAccelerationPolicy::~VPLCPUAccelerationPolicy() {
    for (auto& pair : pool_table) {
        pair.second.clear();
        // do not free key here: last surface will release it
    }
    GAPI_LOG_INFO(nullptr, "destroyed");
}

void VPLCPUAccelerationPolicy::init(session_t session) {
    GAPI_LOG_INFO(nullptr, "initialize session: " << session);
}

void VPLCPUAccelerationPolicy::deinit(session_t session) {
    GAPI_LOG_INFO(nullptr, "deinitialize session: " << session);
}

VPLCPUAccelerationPolicy::pool_key_t
VPLCPUAccelerationPolicy::create_surface_pool(size_t pool_size, size_t surface_size_bytes,
                                              surface_ptr_ctr_t creator) {
    GAPI_LOG_DEBUG(nullptr, "pool size: " << pool_size << ", surface size bytes: " << surface_size_bytes);

    // NB: create empty pool with reservation
    pool_t pool(pool_size);

    // allocate workplace memory area
    size_t preallocated_raw_bytes = pool_size * surface_size_bytes;
    size_t page_size_bytes = 4 * 1024;
    void *preallocated_pool_memory_ptr = nullptr;

#ifdef _WIN32
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    page_size_bytes = sysInfo.dwPageSize;

    GAPI_LOG_DEBUG(nullptr, "page size: " << page_size_bytes << ", preallocated_raw_bytes: " << preallocated_raw_bytes);
    preallocated_pool_memory_ptr = _aligned_malloc(preallocated_raw_bytes, page_size_bytes);
#else
    GAPI_Assert(false && "Compatibility is not tested for systems differ than \"_WIN32\". "
                         "Please feel free to set it up under OPENCV contribution policy");
#endif

    if (!preallocated_pool_memory_ptr) {
        throw std::runtime_error("VPLCPUAccelerationPolicy::create_surface_pool - failed: not enough memory."
                                 "Requested surface count: " + std::to_string(pool_size) +
                                 ", surface bytes: " + std::to_string(surface_size_bytes));
    }

    // fill pool with surfaces
    std::shared_ptr<void> workspace_mem_owner (preallocated_pool_memory_ptr, [] (void *ptr){
        GAPI_LOG_INFO(nullptr, "Free workspace memory: " << ptr);
#ifdef _WIN32
        _aligned_free(ptr);
        GAPI_LOG_INFO(nullptr, "Released workspace memory: " << ptr);
        ptr = nullptr;
#else
        GAPI_Assert(false && "Not implemented for systems differ than \"_WIN32\". "
                             "Please feel free to set it up under OPENCV contribution policy");
#endif

        });
    size_t i = 0;
    try {
        for (; i < pool_size; i++) {
            size_t preallocated_mem_offset = static_cast<size_t>(i) * surface_size_bytes;

            surface_ptr_t surf_ptr = creator(workspace_mem_owner,
                                             preallocated_mem_offset,
                                             preallocated_raw_bytes);
            pool.push_back(std::move(surf_ptr));
        }
    } catch (const std::exception& ex) {
        throw std::runtime_error(std::string("VPLCPUAccelerationPolicy::create_surface_pool - ") +
                                 "cannot construct surface index: " + std::to_string(i) + ", error: " +
                                 ex.what() +
                                 "Requested surface count: " + std::to_string(pool_size) +
                                 ", surface bytes: " + std::to_string(surface_size_bytes));
    }

    // remember pool by key
    GAPI_LOG_INFO(nullptr, "New pool allocated, key: " << preallocated_pool_memory_ptr <<
                           ", surface count: " << pool.total_size() <<
                           ", surface size bytes: " << surface_size_bytes);
    if (!pool_table.emplace(preallocated_pool_memory_ptr, std::move(pool)).second) {
        GAPI_LOG_WARNING(nullptr, "Cannot insert pool, table size: " + std::to_string(pool_table.size()) <<
                                  ", key: " << preallocated_pool_memory_ptr << " exists");
        GAPI_Assert(false && "Cannot create pool in VPLCPUAccelerationPolicy");
    }

    return preallocated_pool_memory_ptr;
}

VPLCPUAccelerationPolicy::surface_weak_ptr_t VPLCPUAccelerationPolicy::get_free_surface(pool_key_t key) {
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        GAPI_LOG_WARNING(nullptr, "key is not found, table size: " <<
                                  pool_table.size());
        GAPI_Assert(false && "Invalid surface key requested in VPLCPUAccelerationPolicy");
    }

    pool_t& requested_pool = pool_it->second;
    return requested_pool.find_free();
}

size_t VPLCPUAccelerationPolicy::get_free_surface_count(pool_key_t key) const {
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        GAPI_LOG_WARNING(nullptr, "key is not found: " << key <<
                                  ", table size: " << pool_table.size());
        return 0;
    }
    const pool_t& requested_pool = pool_it->second;
    return requested_pool.available_size();
}

size_t VPLCPUAccelerationPolicy::get_surface_count(pool_key_t key) const {
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        GAPI_LOG_DEBUG(nullptr, "key is not found: " << key <<
                                ", table size: " << pool_table.size());
        return 0;
    }
    return pool_it->second.total_size();
}

cv::MediaFrame::AdapterPtr VPLCPUAccelerationPolicy::create_frame_adapter(pool_key_t key,
                                                                          mfxFrameSurface1* surface) {
    auto pool_it = pool_table.find(key);
    if (pool_it == pool_table.end()) {
        std::stringstream ss;
        ss << "key is not found: " << key << ", table size: " << pool_table.size();
        const std::string& str = ss.str();
        GAPI_LOG_WARNING(nullptr, str);
        throw std::runtime_error(std::string(__FUNCTION__) + " - " + str);
    }

    pool_t& requested_pool = pool_it->second;
    return cv::MediaFrame::AdapterPtr{new VPLMediaFrameCPUAdapter(requested_pool.find_by_handle(surface))};
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
