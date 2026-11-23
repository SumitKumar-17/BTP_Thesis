
  Within one target cache set, you have many congruent addresses.
  Most show small/no increase, some show medium spikes, and a few show very large spikes.

  That means:

  - contention impact is uneven across addresses in the same set,
  - rare worst-case conflicts create the very high peaks,
  - those high peaks are the strongest attack signal.



› what the eveictionset order in the x axis means what is that explain more on that things in the whole code we used allso


• x-axis (Eviction-Set Address Order) means:

  - We first collect all addresses that map to the same target LLC set.
  - That list is stored as idx_host[0], idx_host[1], ....
  - X value is just the position in this list (order), not time and not cache-set number.

  So:

  - order = 0 -> first congruent address found
  - order = 1 -> second congruent address found
  - ...
  - each point corresponds to one specific memory line in that same set.

  In your code (evset_address_scan.cu):

  1. collect_congruent_indices(...) scans memory and picks indices with matching physical set.
  2. These indices are saved in array order.
  3. CSV writes:

  - order = list position
  - index = actual array index (real address proxy)
  - baseline/attack metrics per address.

  Why this axis is useful:

  - It lets you compare “which specific congruent addresses” are more sensitive.
  - Spikes at some orders mean those specific lines saw stronger rare contention.

  Important:

  - X-axis does not mean chronology.
  - It does not mean different sets.
  - All points are from the same target set, just different addresses within that set.




  • Yes, up-spike means a big latency event due to contention/thrashing.

  Down peak means relative calm, not a “negative event”:

  - It means that address did not see a strong rare miss in that window.
  - For ratio plots, going down toward 1 means attack behavior there looked closer to baseline.
  - If it goes below nearby points, it just means less interference at that address/time, not improvement in attack.

  So:

  - Up = strong interference.
  - Down = weak/no interference for that point.




• Top subplot (Median per address):

  - If line stays low/flat: normal typical latency.
  - If attack and baseline overlap: most accesses unaffected.
  - If a local bump appears: that address got slightly slower typically.

  Bottom subplot (P99 per address):

  - P99 means “near worst-case” latency for that address.
  - Spike up: rare heavy slowdown happened for that address (likely eviction/conflict event).
  - Drop down: fewer/slighter tail events at that address in that run.
  - Flat near baseline: no strong rare contention there.

  Simple interpretation:

  - Up spike = GPU contention hit that address strongly (rare but big delay).
  - Down/flat = little or no rare interference for that address.
  - So spikes mark “hot contention points” inside the same eviction set.