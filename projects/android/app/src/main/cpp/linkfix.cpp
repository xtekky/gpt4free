// Debian's dpkg creates /var/lib/dpkg/status-old with Perl's link() builtin.
// On some Android kernels PRoot cannot create hard links in app data and
// returns EACCES from link()/linkat(). Preload this shim so link() creates a
// regular copy instead: dpkg only needs a backup of the status database, not
// a second hard link to the same inode.
//
// This library intentionally has no DT_NEEDED entries. It only calls libc
// functions that are already present in the process that loads it, so the
// same .so works with bionic (shell) and glibc (Debian programs under PRoot).

extern "C" int* __errno_location(void);
extern "C" int open(const char* path, int flags, ...);
extern "C" int close(int fd);
extern "C" long read(int fd, void* buffer, unsigned long count);
extern "C" long write(int fd, const void* buffer, unsigned long count);

static const int O_RDONLY_VALUE = 0;
static const int O_WRONLY_VALUE = 1;
static const int O_CREAT_VALUE = 64;
static const int O_TRUNC_VALUE = 512;

static int error_result(int saved) {
    *__errno_location() = saved;
    return -1;
}

extern "C" int link(const char* oldpath, const char* newpath) {
    int in = open(oldpath, O_RDONLY_VALUE);
    if (in < 0) return -1;

    int out = open(newpath, O_WRONLY_VALUE | O_CREAT_VALUE | O_TRUNC_VALUE, 0644);
    if (out < 0) {
        int saved = *__errno_location();
        close(in);
        return error_result(saved);
    }

    char buffer[8192];
    for (;;) {
        long count = read(in, buffer, sizeof(buffer));
        if (count < 0) {
            int saved = *__errno_location();
            close(in);
            close(out);
            return error_result(saved);
        }
        if (count == 0) break;
        long written = 0;
        while (written < count) {
            long chunk = write(out, buffer + written,
                               static_cast<unsigned long>(count - written));
            if (chunk < 0) {
                int saved = *__errno_location();
                close(in);
                close(out);
                return error_result(saved);
            }
            written += chunk;
        }
    }

    close(in);
    close(out);
    return 0;
}
