#pragma once
// Windows drive classification and free-space helpers (header-only).
// The bus type comes from IOCTL_STORAGE_QUERY_PROPERTY; nothing is benchmarked here.

#include <windows.h>
#include <winioctl.h>
#include <cctype>
#include <cstdint>
#include <string>

namespace asema {
namespace m8 {

enum class DriveClass { NVME, SATA, USB, UNKNOWN };

inline const char* drive_class_name(DriveClass c) {
    switch (c) {
        case DriveClass::NVME: return "NVMe";
        case DriveClass::SATA: return "SATA";
        case DriveClass::USB: return "USB";
        default: return "unknown";
    }
}

// Relative cost of re-reading data from this class of drive (NVMe = 1).
inline double drive_class_read_cost(DriveClass c) {
    switch (c) {
        case DriveClass::NVME: return 1.0;
        case DriveClass::USB: return 12.0;
        default: return 6.0;  // SATA-class or unknown
    }
}

// Drive letter of an absolute path like "D:/models/x" (upper case), or 0 if the path has none.
inline char drive_letter_of(const std::string& path) {
    if (path.size() >= 2 && path[1] == ':' && std::isalpha(static_cast<unsigned char>(path[0]))) {
        return static_cast<char>(std::toupper(static_cast<unsigned char>(path[0])));
    }
    return 0;
}

inline DriveClass classify_drive_letter(char letter) {
    if (!letter) return DriveClass::UNKNOWN;
    const std::string dev = std::string("\\\\.\\") + letter + ":";
    HANDLE h = CreateFileA(dev.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return DriveClass::UNKNOWN;
    DriveClass result = DriveClass::UNKNOWN;
    STORAGE_PROPERTY_QUERY q{};
    q.PropertyId = StorageDeviceProperty;
    q.QueryType = PropertyStandardQuery;
    alignas(8) char buf[1024] = {};
    DWORD got = 0;
    if (DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &q, sizeof(q), buf, sizeof(buf), &got, nullptr) &&
        got >= sizeof(STORAGE_DEVICE_DESCRIPTOR)) {
        const auto* d = reinterpret_cast<const STORAGE_DEVICE_DESCRIPTOR*>(buf);
        if (d->BusType == BusTypeNvme) result = DriveClass::NVME;
        else if (d->BusType == BusTypeSata || d->BusType == BusTypeAta || d->BusType == BusTypeSas) result = DriveClass::SATA;
        else if (d->BusType == BusTypeUsb) result = DriveClass::USB;
    }
    CloseHandle(h);
    return result;
}

inline DriveClass classify_drive_of_path(const std::string& path) {
    return classify_drive_letter(drive_letter_of(path));
}

// Free bytes on the volume holding `path` (0 if it cannot be queried).
inline uint64_t free_bytes_of_path(const std::string& path) {
    ULARGE_INTEGER avail{};
    if (GetDiskFreeSpaceExA(path.c_str(), &avail, nullptr, nullptr)) return avail.QuadPart;
    return 0;
}

} // namespace m8
} // namespace asema
