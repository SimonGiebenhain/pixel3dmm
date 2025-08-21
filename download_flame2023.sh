#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

cd src/pixel3dmm/preprocessing/MICA

# username and password input
echo -e "\nIf you do not have an account you can register at https://flame.is.tue.mpg.de/ following the installation instruction."
read -p "Username (FLAME):" username
read -p "Password (FLAME):" password
username=$(urle $username)
password=$(urle $password)

echo -e "\nDownloading FLAME..."
mkdir -p data/FLAME2023/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2023.zip&resume=1' -O './FLAME2023.zip' --no-check-certificate --continue
unzip FLAME2023.zip -d data/FLAME2023/
rm -rf FLAME2023.zip

echo -e "\nInstallation has finished!"
