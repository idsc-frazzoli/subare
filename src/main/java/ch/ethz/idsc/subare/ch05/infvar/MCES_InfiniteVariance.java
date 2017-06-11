// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;

class MCES_InfiniteVariance {
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(infiniteVariance);
    PolicyInterface policyInterface = new EquiprobablePolicy(infiniteVariance);
    for (int c = 0; c < 100; ++c)
      ExploringStarts.batch(infiniteVariance, policyInterface, mces);
    DiscreteQsa qsa = mces.qsa();
    qsa.print();
  }
}
