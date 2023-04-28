void *curl_easy_init();
int _curl_easy_setopt(void *curl, int option, void *param);
int curl_easy_getinfo(void *curl, int option, void *ret);
int curl_easy_perform(void *curl);
void curl_easy_cleanup(void *curl);
void curl_easy_reset(void *curl);
char *curl_version();
int curl_easy_impersonate(void *curl, char *target, int default_headers);
struct curl_slist *curl_slist_append(struct curl_slist *list, char *string);
void curl_slist_free_all(struct curl_slist *list);
extern "Python" size_t buffer_callback(void *ptr, size_t size, size_t nmemb, void *userdata);
extern "Python" size_t write_callback(void *ptr, size_t size, size_t nmemb, void *userdata);
extern "Python" int debug_function(void *curl, int type, char *data, size_t size, void *clientp);

// multi interfaces
struct CURLMsg {
   int msg;       /* what this message means */
   void *easy_handle; /* the handle it concerns */
   union {
     void *whatever;    /* message-specific data */
     int result;   /* return code for transfer */
   } data;
};
void *curl_multi_init();
int curl_multi_cleanup(void *curlm);
int curl_multi_add_handle(void *curlm, void *curl);
int curl_multi_remove_handle(void *curlm, void *curl);
int curl_multi_socket_action(void *curlm, int sockfd, int ev_bitmask, int *running_handle);
int curl_multi_setopt(void *curlm, int option, void* param);
int curl_multi_assign(void *curlm, int sockfd, void *sockptr);
struct CURLMsg *curl_multi_info_read(void* curlm, int *msg_in_queue);
extern "Python" void socket_function(void *curl, int sockfd, int what, void *clientp, void *socketp);
extern "Python" void timer_function(void *curlm, int timeout_ms, void *clientp);

