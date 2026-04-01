#!/bin/bash
set -e

if [ -n "${SSH_PUBLIC_KEY}" ]; then
    mkdir -p /root/.ssh
    echo "${SSH_PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi

ssh-keygen -A
/usr/sbin/sshd
echo "SSH server started on port 22"

exec "$@"
