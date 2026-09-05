#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 HOST REMOTE_PYTHON [compute_probe.py options...]" >&2
  exit 2
fi

host=$1
python=$2
shift 2

case "$host" in
  hbox)
    remote_root=/tank/chreatures
    host_environment=("HSA_OVERRIDE_GFX_VERSION=10.3.0")
    ;;
  persvati)
    remote_root=/home/ember/chreatures-compute
    host_environment=()
    ;;
  *)
    echo "unsupported host: $host (expected hbox or persvati)" >&2
    exit 2
    ;;
esac

ssh "$host" mkdir -p \
  "$remote_root/cache/pytorch" \
  "$remote_root/cache/torchinductor" \
  "$remote_root/cache/triton" \
  "$remote_root/probes"
scp "$(dirname "$0")/compute_probe.py" "$host:$remote_root/probes/compute_probe.py" >/dev/null

printf -v command '%q ' env \
  "${host_environment[@]}" \
  "PYTORCH_KERNEL_CACHE_PATH=$remote_root/cache/pytorch" \
  "TORCHINDUCTOR_CACHE_DIR=$remote_root/cache/torchinductor" \
  "TRITON_CACHE_DIR=$remote_root/cache/triton" \
  "$python" "$remote_root/probes/compute_probe.py" "$@"
ssh "$host" "$command"
