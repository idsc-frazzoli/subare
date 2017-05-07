// code by jph
package ch.ethz.idsc.subare.core;

/** all feasible start states and start actions must have probability > 0
 * in order to satisfy the exploring starts condition */
public interface EpisodeSupplier {
  /** @return */
  EpisodeInterface kickoff(PolicyInterface policyInterface);
}
