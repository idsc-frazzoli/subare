// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.tensor.RealScalar;

class MCES_InfiniteVariance {
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    PolicyInterface policyInterface = new EquiprobablePolicy(infiniteVariance);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts( //
        infiniteVariance, policyInterface, //
        infiniteVariance, RealScalar.ONE, RealScalar.of(.5));
    mces.simulate(1);
    DiscreteQsa discreteQsa = mces.qsa();
    discreteQsa.print();
  }
}
