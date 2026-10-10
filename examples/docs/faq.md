# Orbit FAQ

## Why was my SMS not delivered?

Most often because of quiet hours: SMS and push notifications are held until
`ORBIT_QUIET_HOURS` ends in the recipient's time zone. Email is never held.

## How many times does Orbit retry a failed delivery?

Five times by default (`ORBIT_MAX_RETRIES`), with exponential backoff starting at
30 seconds. After the last attempt the message is marked `failed` and an alert is sent to
the on-call engineer if more than 2% of messages failed in the last 10 minutes.

## Who owns Orbit?

The Platform Messaging team. Questions go to `#orbit-help`; incidents page the
`orbit-oncall` rotation.
