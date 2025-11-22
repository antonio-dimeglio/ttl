#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include "../core/types.h"

#if defined(_WIN32)
    #include <io.h>
    #define access _access
    #define mkdir(path, mode) _mkdir(path)
#else 
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/stat.h>
#endif


bool exist(char* file) {
    return (access(file, 0)) == 0;
}

bool makeFolderRecursive(const char* path) {
    char tmp[256];
    char* p = NULL;
    size_t len;

    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);


    if (tmp[len - 1] == '/' || tmp[len - 1] == '\\') {
        tmp[len - 1] = '\0';
    }

    for (p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            *p = '\0';
            
            if (!exist(tmp)) {
                if (mkdir(tmp, S_IRWXU | S_IRWXG | S_IRWXO) != 0) {
                    return false;
                }
            }
            
            *p = '/';
        }
    }

    if (!exist(tmp)) {
        return mkdir(tmp, S_IRWXU | S_IRWXG | S_IRWXO) == 0;
    }

    return true;
}


/*
*   Downloads a file using cURL. If the file already exists, it wont be redownloaded 
*   unless force_download is set to true.
*/
void downloadFile(const char* url, const char* folder, bool force_download) {
    char filepath[512];
    
    if (!makeFolderRecursive(folder)) {
        fprintf(stderr, "Failed to create folder: %s\n", folder);
        return;
    }
    
    const char* filename = strrchr(url, '/');
    if (filename) {
        filename++;
    } else {
        filename = "download";
    }
    
    #if defined(_WIN32)
        snprintf(filepath, sizeof(filepath), "%s\\%s", folder, filename);
    #else
        snprintf(filepath, sizeof(filepath), "%s/%s", folder, filename);
    #endif
    
    if (!force_download && exist(filepath)) {
        printf("Skipping (already exists): %s\n", filename);
        return;
    }
    
    char cmd[1024];
    #if defined(_WIN32)
        snprintf(cmd, sizeof(cmd), "curl -sL -o \"%s\" \"%s\"", filepath, url);
    #else
        snprintf(cmd, sizeof(cmd), "curl -sL -o '%s' '%s'", filepath, url);
    #endif
    
    printf("Downloading: %s\n", filename);
    int result = system(cmd);
    if (result != 0) {
        fprintf(stderr, "curl failed with code %d\n", result);
    }
}

/*
*   Reads a GZip file and returns its raw bytes.
*/
unsigned char* readGZip(const char* path, usize* out_size) {
    gzFile file = gzopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open %s\n", path);
        return NULL;
    }

    const usize chunk_size = 4096;
    unsigned char buffer[chunk_size];

    usize capacity = 1024;
    usize length = 0;
    unsigned char* data = malloc(capacity);
    if (!data) {
        gzclose(file);
        return NULL;
    }

    int bytes_read;
    while ((bytes_read = gzread(file, buffer, chunk_size)) > 0) {
        if (length + bytes_read > capacity) {
            capacity = (length + bytes_read) * 2;
            unsigned char* tmp = realloc(data, capacity);

            if (!tmp) {
                free(data);
                gzclose(file);
                return NULL;
            }

            data = tmp;
        }
        
        memcpy(data + length, buffer, bytes_read);
        length += bytes_read;
    }

    gzclose(file);

    unsigned char* result = realloc(data, length);
    if (result) data = result;

    if (out_size) *out_size = length;
    return data;
}

uint32_t readBigEndian32(unsigned char* p) {
    return (p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3];
}