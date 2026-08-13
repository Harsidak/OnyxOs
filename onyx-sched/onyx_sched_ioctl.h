/*
 * onyx_sched_ioctl.h — ioctl command definitions for /dev/onyx_sched
 */

#ifndef _ONYX_SCHED_IOCTL_H
#define _ONYX_SCHED_IOCTL_H

#include <linux/ioctl.h>
#include "onyx_sched.h"

#define ONYX_IOC_MAGIC  'O'

/* Register a new agent with the scheduler */
#define ONYX_REGISTER_AGENT     _IOW(ONYX_IOC_MAGIC, 1, struct onyx_agent_register)

/* Update agent state (running/suspended/evicted) + scheduling params */
#define ONYX_SET_AGENT_STATE    _IOW(ONYX_IOC_MAGIC, 2, struct onyx_agent_state)

/* Set resource limits (cgroup controls) */
#define ONYX_SET_RESOURCE_LIMITS _IOW(ONYX_IOC_MAGIC, 3, struct onyx_resource_limits)

/* Read per-agent metrics */
#define ONYX_GET_METRICS        _IOWR(ONYX_IOC_MAGIC, 4, struct onyx_metrics)

/* Read all thermal zones in one call */
#define ONYX_GET_THERMALS       _IOR(ONYX_IOC_MAGIC, 5, struct onyx_thermal_data)

/* Emergency: immediately reduce power (all BG agents evicted) */
#define ONYX_EMERGENCY_THROTTLE _IO(ONYX_IOC_MAGIC, 6)

#endif /* _ONYX_SCHED_IOCTL_H */
