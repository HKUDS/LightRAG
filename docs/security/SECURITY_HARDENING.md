# Docker Security Hardening for LightRAG Production

**Last Updated**: Current
**Audience**: DevOps Engineers, Security Teams, Production Operations
**Related Documents**: [Complete Production Guide](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md) | [Authentication Plans](AUTHENTICATION_IMPROVEMENT_PLAN.md) | [Documentation Index](../DOCUMENTATION_INDEX.md)

This document outlines the security improvements implemented in the production Docker setup.

> **💡 For Complete Production Setup**: See the [Complete Production Deployment Guide](../production/PRODUCTION_DEPLOYMENT_COMPLETE.md) which includes all security configurations.

## Security Principles Applied

### 1. Principle of Least Privilege
- All containers now run as non-root users
- Minimal capabilities granted to containers
- `no-new-privileges` security option enabled

### 2. Defense in Depth
- Multiple security layers implemented
- Read-only root filesystem where possible
- Capability dropping and selective granting

## Container Security Configuration

### ✅ LightRAG Application
- **User**: `lightrag:1001` (defined in Dockerfile.production)
- **Security**: no-new-privileges, minimal capabilities
- **Status**: Fully secured with custom non-root user

### ✅ PostgreSQL
- **User**: Container handles user switching internally (postgres user)
- **Security**: no-new-privileges enabled
- **Note**: PostgreSQL containers require root initially for setup, then switch to postgres user

### ✅ Redis
- **User**: `999:999` (redis user)
- **Security**: no-new-privileges, minimal capabilities (SETGID, SETUID)
- **Status**: Secured with standard redis user

### ✅ Nginx
- **User**: `101:101` (nginx user)
- **Security**: no-new-privileges, web server capabilities only
- **Capabilities**: CHOWN, DAC_OVERRIDE, SETGID, SETUID, NET_BIND_SERVICE
- **Status**: Secured with standard nginx user

### ✅ Prometheus
- **User**: `65534:65534` (nobody user)
- **Security**: no-new-privileges, read-only root filesystem
- **Additional**: tmpfs for temporary files
- **Status**: Maximum security with read-only filesystem

### ✅ Grafana
- **User**: `472:472` (grafana user)
- **Security**: no-new-privileges, minimal capabilities
- **Status**: Secured with standard grafana user

### ✅ Jaeger
- **User**: `10001:10001` (custom non-root user)
- **Security**: no-new-privileges, no capabilities
- **Status**: Secured with custom user

### ✅ Loki
- **User**: `10001:10001` (custom non-root user)
- **Security**: no-new-privileges, no capabilities
- **Storage**: Dedicated volume with proper permissions
- **Status**: Secured with custom user and volume

### ✅ Backup Service
- **User**: `1000:1000` (backup user, defined in Dockerfile)
- **Security**: no-new-privileges, minimal capabilities (DAC_OVERRIDE)
- **Status**: Custom Dockerfile with non-root user

## Security Features Implemented

### Container Security Options
```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
cap_add:
  - [minimal required capabilities only]
```

### User Specifications
- All services run with specific UIDs/GIDs
- No containers run as root (UID 0)
- Standard service users where available

### Read-Only Filesystems
- Prometheus: Full read-only root filesystem with tmpfs
- Other services: Considered but may impact functionality

### Network Security
- All ports bound to localhost only (127.0.0.1)
- Internal container communication via Docker networks
- No unnecessary port exposure

## Impact Assessment

### Before (Security Risks)
- ❌ All containers running as root
- ❌ Full privileges available to all processes
- ❌ Potential privilege escalation attacks
- ❌ Container breakout risks

### After (Security Improvements)
- ✅ Zero containers running as root
- ✅ Minimal privileges per container
- ✅ Prevention of privilege escalation
- ✅ Reduced attack surface
- ✅ Container isolation maintained

## Monitoring & Validation

### Testing Non-Root Execution
```bash
# Verify container users
docker-compose -f docker-compose.production.yml ps
docker exec <container_name> id

# Check running processes
docker exec <container_name> ps aux
```

### Security Scanning
- Container images should be scanned with tools like Trivy
- Runtime security monitoring recommended
- Regular security updates for base images

## Best Practices Maintained

1. **Least Privilege**: Minimal permissions granted
2. **Defense in Depth**: Multiple security layers
3. **Fail Secure**: Secure defaults throughout
4. **Complete Mediation**: All access controlled
5. **Psychological Acceptability**: Security doesn't impede usability

## Notes

- PostgreSQL containers typically require root for initialization but drop privileges to postgres user during runtime
- Some containers may need specific capabilities based on their function (e.g., Nginx needs NET_BIND_SERVICE for port 80)
- Read-only root filesystem is ideal but may require additional tmpfs mounts for applications that write temporary files

This security hardening significantly reduces the attack surface and follows Docker security best practices.
