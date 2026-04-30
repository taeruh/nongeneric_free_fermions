from archlinux:latest

run pacman -Syu --noconfirm
run pacman -S --noconfirm clang python sagemath

run sh --version
run curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
run source $HOME/.cargo/env
env PATH="/root/.cargo/bin:${PATH}"
run rustup toolchain install stable
run rustup default stable
run rustup update stable
