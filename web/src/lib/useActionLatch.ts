import { useEffect, useState } from "react";

/**
 * Keeps a stage-action button in its "in progress" state from the moment the
 * user clicks until the server-reported `busy` flag has gone true (polling
 * caught up) and then back to false.
 *
 * Why this is needed: stage actions now run DETACHED on the server — the POST
 * returns in milliseconds while the work continues in the background. A local
 * "running" flag cleared in a `finally` therefore un-greys the button up to
 * ~1.5s before polling reports `busy=true`, so the button flickers back to
 * enabled and the user can double-submit (→ HTTP 409). This latch bridges that
 * gap: `inProgress` stays true on click, and once polling confirms `busy`, the
 * server's flag carries the disabled state until the action actually finishes.
 *
 * Usage:
 *   const { inProgress, begin, fail } = useActionLatch(run.busy);
 *   async function handle() {
 *     begin(); setError(null);
 *     try { await api.doThing(); onUpdate(); }   // do NOT clear on success
 *     catch (e) { setError(String(e)); fail(); } // clear only on failure
 *   }
 *   // <Button disabled={inProgress}>{inProgress ? <Spinner/> : "Go"}</Button>
 *
 * Note: intended for long actions (generate / implement / iterate / fix-loops)
 * that are reliably `busy` for longer than one ~1.5s poll. Fast synchronous
 * actions (e.g. select) don't need it.
 */
export function useActionLatch(busy: boolean) {
  const [submitting, setSubmitting] = useState(false);

  // Once the server reports the action running, hand the disabled state off to
  // `busy` so we stop relying on the local optimistic flag.
  useEffect(() => {
    if (busy) setSubmitting(false);
  }, [busy]);

  return {
    inProgress: submitting || busy,
    begin: () => setSubmitting(true),
    fail: () => setSubmitting(false),
  };
}
