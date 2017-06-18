// code by jph
package ch.ethz.idsc.subare.ch05.wireloop;

import ch.ethz.idsc.subare.core.RewardInterface;
import ch.ethz.idsc.tensor.RealScalar;

/** collection of different cost functions */
interface WireloopReward extends RewardInterface {
  /** steps don't cost anything
   * 
   * @return constant zero */
  static WireloopReward freeSteps() {
    return (s, a, n) -> RealScalar.ZERO;
  }

  /** steps are more expensive than reward along x
   * 
   * @return constant zero */
  static WireloopReward constantCost() {
    return (s, a, n) -> RealScalar.of(-1.4); // -1.2
  }
}
