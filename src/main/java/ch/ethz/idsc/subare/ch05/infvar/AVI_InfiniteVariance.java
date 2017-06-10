// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.alg.ActionValueIteration;
import ch.ethz.idsc.tensor.RealScalar;

class AVI_InfiniteVariance {
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    ActionValueIteration avi = new ActionValueIteration(infiniteVariance);
    avi.untilBelow(RealScalar.of(.00001));
    avi.qsa().print();
  }
}
