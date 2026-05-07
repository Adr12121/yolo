#!/bin/bash
# Wrapper pour utiliser le Tesseract installé sur Windows depuis WSL
# Il convertit les chemins WSL (ex: /mnt/c/...) en chemins Windows compréhensibles par Tesseract.exe

EXE_PATH="/mnt/c/Users/Topo_4/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

args=()
for arg in "$@"; do
    if [[ "$arg" == /* ]] || [[ "$arg" == ~* ]]; then
        # Conversion du chemin WSL en chemin Windows (C:\...)
        win_path=$(wslpath -w "$arg")
        args+=("$win_path")
    else
        args+=("$arg")
    fi
done

exec "$EXE_PATH" "${args[@]}"
