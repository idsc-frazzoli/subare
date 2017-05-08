// code by jph
package ch.ethz.idsc.subare.core;

/** all feasible start states and start actions must have probability > 0
 * in order to satisfy the exploring starts condition
 * 
 * an epsilon greedy policy will ensure that all actions have probability > 0 */
public interface EpisodeSupplier {
  /** @return */
  EpisodeInterface kickoff(PolicyInterface policyInterface);
}
