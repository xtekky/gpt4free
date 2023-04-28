#include "shim.h"

#define INTEGER_OPTION_MAX 10000

int _curl_easy_setopt(void* curl, int option, void* parameter) {
    // printf("****** hijack test begins: \n");
    // int val = curl_easy_setopt(instance->curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
    // printf("****** hijack test ends. opt: %d, val: %d, result is: %d\n", CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0, val);
    CURLoption opt_value = (CURLoption) option;
    // printf("option: %d, setopt parameter: %d\n", option, *(int*)parameter);
    // for integer options, we need to convert param from pointers to integers
    if (option < INTEGER_OPTION_MAX) {
        return (int)curl_easy_setopt(curl, (CURLoption)option, *(int*)parameter);
    }
    return (int)curl_easy_setopt(curl, (CURLoption)option, parameter);
}
