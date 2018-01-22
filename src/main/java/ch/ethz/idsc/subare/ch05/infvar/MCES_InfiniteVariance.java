// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;

enum MCES_InfiniteVariance {
  ;
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(infiniteVariance);
    Policy policy = EquiprobablePolicy.create(infiniteVariance);
    for (int c = 0; c < 100; ++c)
      ExploringStarts.batch(infiniteVariance, policy, mces);
    DiscreteQsa qsa = mces.qsa();
    DiscreteUtils.print(qsa);
  }
}
