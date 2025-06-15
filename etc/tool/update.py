import os
from g4f import version
from subprocess import call, STDOUT

if __name__ == "__main__":
    if not os.getenv("G4F_LIVE"):
        print("Live mode is not enabled. Exiting update script.")
        exit(0)
    command = ["git", "fetch"]
    call(command, stderr=STDOUT)
    command = ["git", "reset", "--hard"]
    call(command, stderr=STDOUT)
    command = ["git" ,"pull", "origin", "main"]
    call(command, stderr=STDOUT)
    current_version = version.get_git_version()
    with open("g4f/debug.py", "a") as f:
        f.write(f"\nversion: str = '{current_version}'\n")
    #command = ["pip", "install", "-U", "-r" , "requirements-slim.txt"]
    #call(command, stderr=STDOUT)