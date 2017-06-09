// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;

class MCES_InfiniteVariance {
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(infiniteVariance);
    PolicyInterface policyInterface = new EquiprobablePolicy(infiniteVariance);
    ExploringStartsBatch.apply(infiniteVariance, policyInterface, mces);
    DiscreteQsa discreteQsa = mces.qsa();
    discreteQsa.print();
  }
}
