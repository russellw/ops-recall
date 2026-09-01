# Incident response

## Severity definitions

- **sev1** -- customer-visible loss of a core flow (checkout, payments, login).
  Page immediately, open a war room, post updates every 15 minutes.
- **sev2** -- degraded experience or a delayed background flow. Page the owning
  team, no war room required.
- **sev3** -- internal impact only, handle in business hours.

## During an incident

1. Mitigate before you diagnose. Roll back, pause the offending job, or shed
   load first; understanding can wait until customers are served again.
2. One person commands, one person communicates, one person investigates. The
   commander does not type into a terminal.
3. Write timestamps into the channel as you go. The post-mortem is assembled
   from that thread, and reconstructing it from memory a day later loses the
   detail that matters.

## After an incident

Every sev1 and sev2 gets a post-mortem within five working days, with a root
cause, a numbered remediation list tagged `[diagnose]`, `[mitigate]`, `[fix]`,
and the signals that were true at the time -- each bound to a probe so the next
responder can re-test the same condition automatically.
