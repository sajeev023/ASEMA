#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <iostream>

int main() {
    std::cout << "Probing DirectCompute / D3D11 on local GPU..." << std::endl;
    
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };
    D3D_FEATURE_LEVEL featureLevel;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    
    HRESULT hr = D3D11CreateDevice(
        nullptr, // Default adapter
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        featureLevels,
        2,
        D3D11_SDK_VERSION,
        &device,
        &featureLevel,
        &context
    );
    
    if (FAILED(hr)) {
        std::cerr << "D3D11CreateDevice failed with hr=0x" << std::hex << hr << std::endl;
        return 1;
    }
    
    std::cout << "D3D11 Device created successfully! Feature level: 0x" << std::hex << featureLevel << std::dec << std::endl;
    
    IDXGIDevice* dxgiDevice = nullptr;
    if (SUCCEEDED(device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice))) {
        IDXGIAdapter* adapter = nullptr;
        if (SUCCEEDED(dxgiDevice->GetAdapter(&adapter))) {
            DXGI_ADAPTER_DESC desc;
            adapter->GetDesc(&desc);
            std::wcout << L"Adapter Name: " << desc.Description << std::endl;
            std::cout << "Dedicated VRAM: " << (desc.DedicatedVideoMemory / (1024 * 1024)) << " MB" << std::endl;
            adapter->Release();
        }
        dxgiDevice->Release();
    }
    
    context->Release();
    device->Release();
    return 0;
}
