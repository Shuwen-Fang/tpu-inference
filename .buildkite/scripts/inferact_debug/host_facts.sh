#!/bin/bash
# Host facts that can differ between a Cloud TPU VM host (cicd) and a GCE
# tpu7x-standard-4t host (inferact). Runs on the host, outside docker.
set -u

md() {
  curl -sf -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/$1" || echo "<missing>"
}
section() { echo; echo "=== $1"; }

section identity
hostname
echo "machine-type: $(md machine-type)"
echo "image: $(md image)"
echo "accelerator-type: $(md attributes/accelerator-type)"
echo "RESULT host.machine_type=$(md machine-type | sed 's|.*/||')"

section tpu-env
md attributes/tpu-env
echo
md attributes/tpu-env | sed -nE "s/^([A-Z_]+): '?([^']*)'?$/RESULT tpu_env.\1=\2/p"

section kernel
uname -r
grep PRETTY_NAME /etc/os-release
cat /proc/cmdline
echo "RESULT host.kernel=$(uname -r)"

section cpu
lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket|NUMA|CPU max MHz|L3)'
echo "RESULT host.nproc=$(nproc)"
echo "RESULT host.cpu_model=$(lscpu | sed -nE 's/^Model name: +//p' | tr ' ' '_')"
echo "RESULT host.numa_nodes=$(lscpu | sed -nE 's/^NUMA node\(s\): +//p')"
gov=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo "RESULT host.cpu_governor=$( [ -r $gov ] && cat $gov || echo none)"
echo "RESULT host.smt=$(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo unknown)"

section memory
free -g
grep -E 'MemTotal|HugePages_Total|Hugepagesize' /proc/meminfo
for f in enabled defrag; do
  echo "RESULT host.thp_$f=$(sed -nE 's/.*\[(.*)\].*/\1/p' /sys/kernel/mm/transparent_hugepage/$f)"
done
echo "RESULT host.mem_total_gb=$(free -g | awk '/^Mem:/ {print $2}')"
echo "RESULT host.hugepages=$(awk '/HugePages_Total/ {print $2}' /proc/meminfo)"

section sysctl
for k in kernel.numa_balancing vm.overcommit_memory vm.max_map_count vm.swappiness \
         net.ipv4.ip_local_port_range net.ipv4.ip_local_reserved_ports; do
  echo "RESULT sysctl.$k=$(sysctl -n "$k" 2>/dev/null | tr '\t' ' ')"
done

section tpu-devices
ls -l /dev/vfio /dev/accel* 2>&1
for d in /sys/bus/pci/devices/*; do
  [ "$(cat "$d/vendor")" = 0x1ae0 ] || continue
  drv=$(basename "$(readlink "$d/driver" 2>/dev/null)" 2>/dev/null)
  echo "$(basename "$d") device=$(cat "$d/device") numa=$(cat "$d/numa_node") driver=$drv" \
       "link=$(cat "$d/current_link_speed" 2>/dev/null)x$(cat "$d/current_link_width" 2>/dev/null)"
done | tee /tmp/tpu_pci.txt
echo "RESULT host.tpu_pci=$(awk '{print $2, $3, $4, $5}' /tmp/tpu_pci.txt | sort | uniq -c | tr -s ' ' | tr ' ' '_' | paste -sd, -)"
echo "RESULT host.iommu_groups=$(ls /sys/kernel/iommu_groups 2>/dev/null | wc -l)"

section modules
lsmod | grep -iE 'vfio|tpu|accel|gasket|gve' || true
for m in vfio vfio_pci vfio_iommu_type1 gve; do
  v=$(modinfo -F version "$m" 2>/dev/null)
  echo "RESULT module.$m=${v:-$(modinfo -F vermagic "$m" 2>/dev/null | cut -d' ' -f1)}"
done

section packages
dpkg-query -W -f '${Package} ${Version}\n' 2>/dev/null \
  | grep -iE 'tpu|vfio|google|gve|accel|docker|containerd|ops-agent' | sed 's/^/pkg /'

section docker
docker info --format 'server={{.ServerVersion}} cgroup={{.CgroupDriver}} v{{.CgroupVersion}} storage={{.Driver}}'
cat /etc/docker/daemon.json 2>/dev/null
echo "RESULT docker.version=$(docker version --format '{{.Server.Version}}')"

section limits
ulimit -a | grep -E 'locked|open files|processes'
echo "RESULT host.memlock=$(ulimit -l)"

section disks
df -h / /mnt/disks/persist /var/lib/docker 2>/dev/null
findmnt -no SOURCE,FSTYPE,OPTIONS --target /mnt/disks/persist

section load
uptime
top -bn1 -o %CPU | sed -n '7,20p'
systemctl list-units --type=service --state=running --no-pager --no-legend \
  | awk '{print $1}' | paste -sd' ' -
