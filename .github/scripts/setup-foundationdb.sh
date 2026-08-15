#!/usr/bin/env bash
set -euo pipefail

readonly FDB_VERSION="7.3.69"
readonly FDB_CLIENT_SHA256="ea59d1708519798c7bc4f514cd29af1ac8e41dccbec4371f22d86b713ea81cbf"
readonly FDB_SERVER_SHA256="1a4088133d088be93a868e26e058250040ddfa725580701170ad2fb9e3d38ede"
readonly MODE="${1:-}"

if [[ $# -ne 1 || ("$MODE" != "client" && "$MODE" != "server" && "$MODE" != "restart") ]]; then
  echo "usage: $0 client|server|restart" >&2
  exit 2
fi

wait_until_ready() {
  for _ in {1..30}; do
    if timeout 3s fdbcli --exec "status minimal"; then
      return
    fi
    sleep 1
  done

  echo "FoundationDB did not become ready" >&2
  exit 1
}

if [[ "$MODE" == "restart" ]]; then
  sudo service foundationdb restart
  wait_until_ready
  exit 0
fi

PACKAGE_DIR="$(mktemp -d)"
readonly PACKAGE_DIR
readonly CLIENT_PACKAGE="$PACKAGE_DIR/foundationdb-clients.deb"
readonly SERVER_PACKAGE="$PACKAGE_DIR/foundationdb-server.deb"
readonly RELEASE_URL="https://github.com/apple/foundationdb/releases/download/$FDB_VERSION"

curl --fail --location --retry 3 \
  --output "$CLIENT_PACKAGE" \
  "$RELEASE_URL/foundationdb-clients_${FDB_VERSION}-1_amd64.deb"
printf '%s  %s\n' "$FDB_CLIENT_SHA256" "$CLIENT_PACKAGE" | sha256sum --check --strict

packages=("$CLIENT_PACKAGE")
if [[ "$MODE" == "server" ]]; then
  curl --fail --location --retry 3 \
    --output "$SERVER_PACKAGE" \
    "$RELEASE_URL/foundationdb-server_${FDB_VERSION}-1_amd64.deb"
  printf '%s  %s\n' "$FDB_SERVER_SHA256" "$SERVER_PACKAGE" | sha256sum --check --strict
  packages+=("$SERVER_PACKAGE")
fi

sudo env DEBIAN_FRONTEND=noninteractive apt-get install --yes "${packages[@]}"

if [[ "$MODE" == "client" ]]; then
  exit 0
fi

sudo service foundationdb restart
wait_until_ready
