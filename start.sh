#!/usr/bin/env bash

set -euo pipefail

default_venv_dir() {
  local root_dir=$1
  local root_hash

  if [[ "${root_dir}" != /mnt/* ]]; then
    printf '%s/.venv\n' "${root_dir}"
    return
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    root_hash="$(printf '%s' "${root_dir}" | sha256sum | awk '{print substr($1, 1, 12)}')"
  else
    root_hash="$(printf '%s' "${root_dir}" | shasum -a 256 | awk '{print substr($1, 1, 12)}')"
  fi

  printf '%s/.cache/stream_copier/venvs/%s\n' "${HOME}" "${root_hash}"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-2712}"
FRONTEND_PORT="${FRONTEND_PORT:-4300}"
VENV_DIR="${STREAM_COPIER_VENV_DIR:-$(default_venv_dir "${ROOT_DIR}")}"
BACKEND_DIR="${ROOT_DIR}/backend"
FRONTEND_DIR="${ROOT_DIR}/frontend"
BACKEND_DEPS_STATE_FILE="${VENV_DIR}/.backend-deps.sha256"
INTENT_RUNTIME_STATE_FILE="${VENV_DIR}/.intent-runtime.ready"
INTENT_RUNTIME_VERSION="torch==2.10.0|transformers>=4.48.0,<5|safetensors>=0.7.0"

gpu_visible() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

hash_file() {
  local file=$1

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file}" | awk '{print $1}'
    return
  fi

  shasum -a 256 "${file}" | awk '{print $1}'
}

ensure_backend_dependencies() {
  local current_hash
  local installed_hash=""

  echo "Ensuring backend dependencies are installed..."
  current_hash="$(hash_file "${BACKEND_DIR}/pyproject.toml")"

  if [[ -f "${BACKEND_DEPS_STATE_FILE}" ]]; then
    installed_hash="$(<"${BACKEND_DEPS_STATE_FILE}")"
  fi

  if [[ "${installed_hash}" != "${current_hash}" || ! -x "${VENV_DIR}/bin/uvicorn" ]]; then
    echo "Installing local transcription stack. This can take a few minutes on the first run or after dependency changes..."
    "${VENV_DIR}/bin/pip" install -e "${BACKEND_DIR}[dev]"
    printf '%s\n' "${current_hash}" > "${BACKEND_DEPS_STATE_FILE}"
  fi
}

site_packages_paths() {
  shopt -s nullglob
  printf '%s\n' "${VENV_DIR}"/lib/python*/site-packages "${VENV_DIR}"/lib64/python*/site-packages
  shopt -u nullglob
}

gpu_runtime_installed() {
  local base
  local found_cublas=0
  local found_cudnn=0

  while IFS= read -r base; do
    [[ -d "${base}/nvidia/cublas/lib" ]] && found_cublas=1
    [[ -d "${base}/nvidia/cudnn/lib" ]] && found_cudnn=1
  done < <(site_packages_paths)

  [[ ${found_cublas} -eq 1 && ${found_cudnn} -eq 1 ]]
}

ensure_gpu_runtime() {
  if [[ "${GPU_AVAILABLE}" != "1" ]]; then
    return
  fi

  if ! gpu_runtime_installed; then
    echo "NVIDIA GPU detected. Installing CUDA runtime libraries for faster-whisper..."
    "${VENV_DIR}/bin/pip" install "nvidia-cublas-cu12>=12,<13" "nvidia-cudnn-cu12>=9,<10"
  fi
}

resolve_gpu_library_path() {
  local base
  local path
  local -a paths=()

  while IFS= read -r base; do
    for path in "${base}/nvidia/cublas/lib" "${base}/nvidia/cudnn/lib"; do
      if [[ -d "${path}" ]]; then
        paths+=("${path}")
      fi
    done
  done < <(site_packages_paths)

  local IFS=:
  echo "${paths[*]}"
}

intent_classifier_enabled() {
  local env_file="${BACKEND_DIR}/.env"
  if [[ ! -f "${env_file}" ]]; then
    return 0
  fi
  grep -Eiq '^ENABLE_LOCAL_INTENT_CLASSIFIER=(1|true|yes)$' "${env_file}"
}

intent_runtime_installed() {
  local base
  local found_torch=0
  local found_transformers=0
  local found_safetensors=0
  local installed_version=""

  if [[ -f "${INTENT_RUNTIME_STATE_FILE}" ]]; then
    installed_version="$(<"${INTENT_RUNTIME_STATE_FILE}")"
  fi

  while IFS= read -r base; do
    [[ -d "${base}/torch" ]] && found_torch=1
    [[ -d "${base}/transformers" ]] && found_transformers=1
    [[ -d "${base}/safetensors" ]] && found_safetensors=1
  done < <(site_packages_paths)

  [[ ${found_torch} -eq 1 && ${found_transformers} -eq 1 && ${found_safetensors} -eq 1 && "${installed_version}" == "${INTENT_RUNTIME_VERSION}" ]]
}

ensure_intent_runtime() {
  if ! intent_classifier_enabled; then
    return
  fi

  if intent_runtime_installed; then
    return
  fi

  echo "Installing local intent-classifier runtime..."
  "${VENV_DIR}/bin/pip" install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    "torch==2.10.0" \
    "transformers>=4.48.0,<5" \
    "safetensors>=0.7.0"
  printf '%s\n' "${INTENT_RUNTIME_VERSION}" > "${INTENT_RUNTIME_STATE_FILE}"
}

port_in_use() {
  local port=$1
  python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    result = sock.connect_ex(("127.0.0.1", port))
print("1" if result == 0 else "0")
PY
}

find_available_port() {
  local preferred=$1
  local port=$preferred

  while true; do
    if [[ "$port" == "5173" ]]; then
      port=$((port + 1))
      continue
    fi

    if [[ "$(port_in_use "$port")" == "0" ]]; then
      echo "$port"
      return
    fi

    port=$((port + 1))
  done
}

cleanup() {
  local exit_code=$?
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating Python virtual environment..."
  mkdir -p "$(dirname "${VENV_DIR}")"
  python3 -m venv "${VENV_DIR}"
fi

if [[ "${VENV_DIR}" != "${ROOT_DIR}/.venv" ]]; then
  echo "Using Linux-side virtualenv at ${VENV_DIR} for better WSL performance."
fi

if [[ ! -f "${BACKEND_DIR}/.env" ]]; then
  cp "${BACKEND_DIR}/.env.example" "${BACKEND_DIR}/.env"
  echo "Created backend/.env from template."
fi

GPU_AVAILABLE=0
if gpu_visible; then
  GPU_AVAILABLE=1
fi

ensure_backend_dependencies
ensure_intent_runtime
ensure_gpu_runtime

GPU_LIBRARY_PATH=""
if [[ "${GPU_AVAILABLE}" == "1" ]]; then
  GPU_LIBRARY_PATH="$(resolve_gpu_library_path)"
  if [[ -n "${GPU_LIBRARY_PATH}" ]]; then
    echo "Configured CUDA runtime paths for the backend."
  fi
fi

if [[ ! -d "${FRONTEND_DIR}/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  (cd "${FRONTEND_DIR}" && npm install)
fi

SELECTED_BACKEND_PORT="$(find_available_port "${BACKEND_PORT}")"
SELECTED_FRONTEND_PORT="$(find_available_port "${FRONTEND_PORT}")"

if [[ "${SELECTED_BACKEND_PORT}" != "${BACKEND_PORT}" ]]; then
  echo "Backend port ${BACKEND_PORT} is busy. Using ${SELECTED_BACKEND_PORT} instead."
fi

if [[ "${SELECTED_FRONTEND_PORT}" != "${FRONTEND_PORT}" ]]; then
  echo "Frontend port ${FRONTEND_PORT} is busy. Using ${SELECTED_FRONTEND_PORT} instead."
fi

echo "Starting backend on http://localhost:${SELECTED_BACKEND_PORT}"
(
  cd "${BACKEND_DIR}"
  if [[ -n "${GPU_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${GPU_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
  fi
  "${VENV_DIR}/bin/uvicorn" app.main:app --reload --host 0.0.0.0 --port "${SELECTED_BACKEND_PORT}"
) &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:${SELECTED_FRONTEND_PORT}"
(
  cd "${FRONTEND_DIR}"
  VITE_API_BASE_URL="http://localhost:${SELECTED_BACKEND_PORT}/api" \
  VITE_WS_BASE_URL="ws://localhost:${SELECTED_BACKEND_PORT}/ws" \
  npm run dev -- --host 0.0.0.0 --port "${SELECTED_FRONTEND_PORT}" --strictPort
) &
FRONTEND_PID=$!

echo "Services are starting:"
echo "  Backend:  http://localhost:${SELECTED_BACKEND_PORT}"
echo "  Frontend: http://localhost:${SELECTED_FRONTEND_PORT}"
echo "Press Ctrl+C to stop both."

wait -n "${BACKEND_PID}" "${FRONTEND_PID}"
