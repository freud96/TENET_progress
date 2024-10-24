#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <cstring>  // For strncpy and memcpy
#include <atomic>
struct LoopData {
    char func_name[256];
    int i, j, k, tile_i, tile_j, tile_k;
};

std::vector<LoopData> loop_log;
std::mutex log_mutex;


extern "C" void log_loop_indices(const char* func_name, int i, int j, int k,
                                 int tile_i, int tile_j, int tile_k) {
    
        std::lock_guard<std::mutex> guard(log_mutex);
        LoopData data;
        strncpy(data.func_name, func_name, sizeof(data.func_name) - 1);
        data.func_name[sizeof(data.func_name) - 1] = '\0';
        data.i = i; data.j = j; data.k = k;
        data.tile_i = tile_i; data.tile_j = tile_j; data.tile_k = tile_k;

        loop_log.push_back(data);
        // std::cout << "Logged: " << func_name << "\n";
    

        // std::cout << "Logged: " << func_name 
        //       << " i: " << i 
        //       << " j: " << j 
        //       << " k: " << k 
        //       << " tile_i: " << tile_i 
        //       << " tile_j: " << tile_j 
        //       << " tile_k: " << tile_k 
        //       << std::endl;
}


extern "C" const LoopData* get_logged_data(size_t* size) {
    std::lock_guard<std::mutex> guard(log_mutex);
    *size = loop_log.size();
    
    if (*size == 0) return nullptr;

    //std::cout << "Returning pointer to logged data: " << *size << " elements\n";
    return loop_log.data();  // Return direct pointer to internal data
}

extern "C" void clear_log() {
    std::lock_guard<std::mutex> guard(log_mutex);
    loop_log.clear();  // Clear the log to avoid memory build-up
    //std::cout << "Log cleared.\n";
}
