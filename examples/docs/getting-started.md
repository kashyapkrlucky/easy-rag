# Getting started with Orbit

Orbit is the internal service that schedules and sends customer notifications
(email, SMS and push). This wiki is a sample corpus for EasyRAG: Orbit doesn't exist.

## Requirements

- Python 3.12 or newer
- PostgreSQL 16 for the job queue
- A Redis instance for rate limiting (optional in development)

## Running locally

```bash
git clone https://git.example.com/platform/orbit.git
cd orbit
make dev        # starts PostgreSQL and Redis in Docker, then the API on port 8300
```

The development API listens on `http://localhost:8300`. Sign in to the admin console with
your SSO account; local accounts are disabled.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ORBIT_QUEUE_URL` | (required) | PostgreSQL URL of the job queue |
| `ORBIT_MAX_RETRIES` | `5` | Delivery attempts before a message is marked failed |
| `ORBIT_SMS_PROVIDER` | `twilio` | `twilio` or `vonage` |
| `ORBIT_QUIET_HOURS` | `22:00-07:00` | No SMS or push in the recipient's local time |
