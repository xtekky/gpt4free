//go:build windows

package sandbox

import (
	"fmt"
	"os/exec"
)

// The Termux backend is only meaningful on Android/Linux. These stubs keep the
// Windows launcher (which uses Docker or Podman) compiling.

func configureDetached(cmd *exec.Cmd) {}

func processAlive(pid int) bool { return false }

func terminateProcess(pid int) error {
	return fmt.Errorf("the Termux backend is not supported on Windows")
}

func forceKillProcess(pid int) error {
	return fmt.Errorf("the Termux backend is not supported on Windows")
}

func terminateProcessGroup(pid int) error {
	return fmt.Errorf("the udocker backend is not supported on Windows")
}

func forceKillProcessGroup(pid int) error {
	return fmt.Errorf("the udocker backend is not supported on Windows")
}
