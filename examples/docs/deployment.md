# Deploying Orbit

## Environments

Orbit runs in two environments: **staging** (deployed on every merge to `main`) and
**production** (deployed manually, Tuesdays and Thursdays only).

## Production deploys

1. Check that the staging deploy of the same commit is green for at least one hour.
2. Announce the deploy in `#orbit-releases`.
3. Run `make deploy ENV=production` from the release branch.
4. Watch the delivery-rate dashboard for 15 minutes.

Deploys are frozen from December 15 to January 5.

## Rollback

Run `make rollback ENV=production`. It redeploys the previous release in about two
minutes. Database migrations are never rolled back automatically: write a new forward
migration instead.
