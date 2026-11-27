#include <jni.h>
#include <android/log.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace {

const char *kTag = "g4f-pty";

struct Session {
    int master;
    pid_t pid;
    std::atomic<bool> closing;
    bool exited;
    int status;
};

Session *sessionFrom(jlong handle) {
    return reinterpret_cast<Session *>(handle);
}

char **toStringArray(JNIEnv *env, jobjectArray array) {
    if (array == nullptr) return nullptr;
    const jsize count = env->GetArrayLength(array);
    char **result = static_cast<char **>(calloc(static_cast<size_t>(count) + 1, sizeof(char *)));
    for (jsize i = 0; i < count; i++) {
        jstring value = static_cast<jstring>(env->GetObjectArrayElement(array, i));
        if (value == nullptr) {
            result[i] = nullptr;
            continue;
        }
        const char *chars = env->GetStringUTFChars(value, nullptr);
        result[i] = strdup(chars != nullptr ? chars : "");
        env->ReleaseStringUTFChars(value, chars);
        env->DeleteLocalRef(value);
    }
    result[count] = nullptr;
    return result;
}

void freeStringArray(char **array) {
    if (array == nullptr) return;
    for (size_t i = 0; array[i] != nullptr; i++) free(array[i]);
    free(array);
}

void logErrno(const char *what) {
    int saved = errno;
    __android_log_print(ANDROID_LOG_ERROR, kTag, "%s: %s", what, strerror(saved));
    errno = saved;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_dev_g4f_android_container_TerminalSession_nativeOpen(
        JNIEnv *env, jclass, jobjectArray argvArray, jobjectArray envArray,
        jint cols, jint rows) {
    char **argv = toStringArray(env, argvArray);
    char **envp = toStringArray(env, envArray);
    if (argv == nullptr || argv[0] == nullptr) {
        freeStringArray(argv);
        freeStringArray(envp);
        return 0;
    }

    int master = posix_openpt(O_RDWR | O_NOCTTY);
    if (master < 0) {
        logErrno("posix_openpt");
        freeStringArray(argv);
        freeStringArray(envp);
        return 0;
    }
    if (grantpt(master) != 0 || unlockpt(master) != 0) {
        logErrno("grantpt/unlockpt");
        close(master);
        freeStringArray(argv);
        freeStringArray(envp);
        return 0;
    }
    char *slaveName = ptsname(master);
    if (slaveName == nullptr) {
        logErrno("ptsname");
        close(master);
        freeStringArray(argv);
        freeStringArray(envp);
        return 0;
    }
    int slave = open(slaveName, O_RDWR | O_NOCTTY);
    if (slave < 0) {
        logErrno("open slave");
        close(master);
        freeStringArray(argv);
        freeStringArray(envp);
        return 0;
    }

    struct winsize ws = {};
    ws.ws_row = static_cast<unsigned short>(rows > 0 ? rows : 24);
    ws.ws_col = static_cast<unsigned short>(cols > 0 ? cols : 80);
    ioctl(slave, TIOCSWINSZ, &ws);

    pid_t pid = fork();
    if (pid < 0) {
        logErrno("fork");
        close(slave);
        close(master);
        freeStringArray(argv);
        freeStringArray(envp);
        return 0;
    }
    if (pid == 0) {
        close(master);
        setsid();
        ioctl(slave, TIOCSCTTY, 0);
        dup2(slave, STDIN_FILENO);
        dup2(slave, STDOUT_FILENO);
        dup2(slave, STDERR_FILENO);
        if (slave > STDERR_FILENO) close(slave);
        execve(argv[0], argv, envp != nullptr ? envp : environ);
        _exit(127);
    }

    close(slave);
    freeStringArray(argv);
    freeStringArray(envp);

    Session *session = new Session{master, pid, false, false, 0};
    return reinterpret_cast<jlong>(session);
}

JNIEXPORT jint JNICALL
Java_dev_g4f_android_container_TerminalSession_nativeRead(
        JNIEnv *env, jclass, jlong handle, jbyteArray buffer) {
    Session *session = sessionFrom(handle);
    if (session == nullptr || buffer == nullptr) return -1;
    const jsize capacity = env->GetArrayLength(buffer);
    if (capacity <= 0) return 0;
    jbyte *bytes = env->GetByteArrayElements(buffer, nullptr);
    // Poll in short slices so close() can stop a reader blocked on an
    // unresponsive session without closing the fd under it from another thread.
    while (!session->closing.load()) {
        struct pollfd pfd = {};
        pfd.fd = session->master;
        pfd.events = POLLIN;
        int ready = poll(&pfd, 1, 100);
        if (ready < 0) {
            if (errno == EINTR) continue;
            logErrno("poll master");
            env->ReleaseByteArrayElements(buffer, bytes, 0);
            return -1;
        }
        if (ready == 0) continue;
        ssize_t count = read(session->master, bytes, static_cast<size_t>(capacity));
        if (count > 0) {
            env->ReleaseByteArrayElements(buffer, bytes, 0);
            return static_cast<jint>(count);
        }
        if (count == 0 || errno != EINTR) {
            env->ReleaseByteArrayElements(buffer, bytes, 0);
            return -1;
        }
    }
    env->ReleaseByteArrayElements(buffer, bytes, 0);
    return -1;
}

JNIEXPORT jboolean JNICALL
Java_dev_g4f_android_container_TerminalSession_nativeWrite(
        JNIEnv *env, jclass, jlong handle, jbyteArray data) {
    Session *session = sessionFrom(handle);
    if (session == nullptr || data == nullptr) return JNI_FALSE;
    const jsize length = env->GetArrayLength(data);
    jbyte *bytes = env->GetByteArrayElements(data, nullptr);
    size_t written = 0;
    while (written < static_cast<size_t>(length)) {
        ssize_t count = write(session->master, bytes + written,
                              static_cast<size_t>(length) - written);
        if (count > 0) {
            written += static_cast<size_t>(count);
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
    env->ReleaseByteArrayElements(data, bytes, JNI_ABORT);
    return written == static_cast<size_t>(length) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_dev_g4f_android_container_TerminalSession_nativeResize(
        JNIEnv *, jclass, jlong handle, jint cols, jint rows) {
    Session *session = sessionFrom(handle);
    if (session == nullptr) return;
    struct winsize ws = {};
    ws.ws_row = static_cast<unsigned short>(rows > 0 ? rows : 24);
    ws.ws_col = static_cast<unsigned short>(cols > 0 ? cols : 80);
    ioctl(session->master, TIOCSWINSZ, &ws);
}

JNIEXPORT jboolean JNICALL
Java_dev_g4f_android_container_TerminalSession_nativeAlive(
        JNIEnv *, jclass, jlong handle) {
    Session *session = sessionFrom(handle);
    if (session == nullptr || session->exited || session->closing.load()) return JNI_FALSE;
    int status = 0;
    pid_t result = waitpid(session->pid, &status, WNOHANG);
    if (result == session->pid) {
        session->exited = true;
        session->status = status;
        return JNI_FALSE;
    }
    return result == 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL
Java_dev_g4f_android_container_TerminalSession_nativeExitCode(
        JNIEnv *, jclass, jlong handle) {
    Session *session = sessionFrom(handle);
    if (session == nullptr) return -1;
    if (!session->exited) {
        int status = 0;
        if (waitpid(session->pid, &status, WNOHANG) == session->pid) {
            session->exited = true;
            session->status = status;
        }
    }
    if (!session->exited) return session->closing.load() ? 128 + SIGTERM : -1;
    if (WIFEXITED(session->status)) return WEXITSTATUS(session->status);
    if (WIFSIGNALED(session->status)) return 128 + WTERMSIG(session->status);
    return -1;
}

JNIEXPORT void JNICALL
Java_dev_g4f_android_container_TerminalSession_nativeClose(
        JNIEnv *, jclass, jlong handle) {
    Session *session = sessionFrom(handle);
    if (session == nullptr) return;
    session->closing.store(true);
    // The child called setsid(), so its process group id is its pid. Kill the
    // whole group: PRoot may have grandchildren (apt/pip) holding the PTY open.
    if (kill(-session->pid, SIGTERM) != 0) kill(session->pid, SIGTERM);
    usleep(100000);
    if (kill(-session->pid, SIGKILL) != 0) kill(session->pid, SIGKILL);
    int status = 0;
    if (waitpid(session->pid, &status, WNOHANG) == session->pid) {
        session->exited = true;
        session->status = status;
    }
}

JNIEXPORT void JNICALL
Java_dev_g4f_android_container_TerminalSession_nativeDispose(
        JNIEnv *, jclass, jlong handle) {
    Session *session = sessionFrom(handle);
    if (session == nullptr) return;
    if (!session->exited) {
        session->closing.store(true);
        int status = 0;
        pid_t result;
        do {
            result = waitpid(session->pid, &status, 0);
        } while (result < 0 && errno == EINTR);
        if (result == session->pid) {
            session->exited = true;
            session->status = status;
        }
    }
    close(session->master);
    delete session;
}

}  // extern "C"
