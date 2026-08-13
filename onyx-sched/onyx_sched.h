/*
 * onyx_sched.h — Shared structures between kernel module and userspace
 */

#ifndef _ONYX_SCHED_H
#define _ONYX_SCHED_H

/* Agent priority classes (mirrors onyx_kernel.config.AgentClass) */
#define ONYX_CLASS_RT_HARD  0
#define ONYX_CLASS_RT_SOFT  1
#define ONYX_CLASS_BG       2

/* Agent states (mirrors onyx_kernel.config.AgentState) */
#define ONYX_STATE_REGISTERED 0
#define ONYX_STATE_READY      1
#define ONYX_STATE_RUNNING    2
#define ONYX_STATE_DEGRADED   3
#define ONYX_STATE_SUSPENDED  4
#define ONYX_STATE_EVICTED    5

/* Structures for ioctl communication */

struct onyx_agent {
    pid_t       pid;
    int         agent_class;    /* ONYX_CLASS_* */
    int         state;          /* ONYX_STATE_* */
    unsigned int cpu_shares;    /* CFS cpu.weight (1-10000) */
    unsigned int mem_limit_mb;  /* memory.max in MB (0 = unlimited) */
    int         nice_value;     /* -20 to 19 */
    unsigned int cpu_affinity;  /* CPU mask */
};

struct onyx_agent_register {
    pid_t   pid;
    int     agent_class;
};

struct onyx_agent_state {
    pid_t       pid;
    int         state;
    unsigned int cpu_shares;
    unsigned int mem_limit_mb;
    int         nice_value;
};

struct onyx_resource_limits {
    pid_t       pid;
    unsigned int cpu_shares;
    unsigned int mem_limit_mb;
    unsigned int cpu_affinity;
};

struct onyx_metrics {
    pid_t       pid;
    int         state;
    unsigned int cpu_shares;
    unsigned int mem_limit_mb;
    int         nice_value;
    /* TODO: Add runtime metrics from /proc */
    unsigned long cpu_time_us;
    unsigned long mem_usage_kb;
    unsigned long context_switches;
};

struct onyx_thermal_data {
    int cpu_temp_mc;    /* millidegrees Celsius */
    int gpu_temp_mc;
    int soc_temp_mc;
    int power_mw;       /* milliwatts */
};

#endif /* _ONYX_SCHED_H */
