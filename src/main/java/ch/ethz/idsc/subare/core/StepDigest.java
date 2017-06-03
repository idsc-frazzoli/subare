// code by jph
package ch.ethz.idsc.subare.core;

/** interface is implemented by temporal difference algorithms */
public interface StepDigest {
  /** update value estimation based on single step of episode
   * 
   * @param stepInterface */
  void digest(StepInterface stepInterface);
}
