//go:build !windows

package sandbox

import (
	"os"
	"os/exec"
	"syscall"
)

// configureDetached puts the server in its own session so it survives the
// launcher exiting from a Termux shell.
func configureDetached(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(syscall.Signal(0)) == nil
}

func terminateProcess(pid int) error {
	p, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return p.Signal(syscall.SIGTERM)
}

func forceKillProcess(pid int) error {
	p, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return p.Signal(syscall.SIGKILL)
}

// udocker is a Python wrapper that waits for a PRoot child, so stopping only
// the wrapper leaves the container running. configureDetached puts the wrapper
// in its own session; signal the whole group instead.
func terminateProcessGroup(pid int) error {
	return syscall.Kill(-pid, syscall.SIGTERM)
}

func forceKillProcessGroup(pid int) error {
	return syscall.Kill(-pid, syscall.SIGKILL)
}
