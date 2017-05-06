// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.alg.IterativePolicyEvaluation;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;

class IPE_InfiniteVariance {
  public static void main(String[] args) {
    StandardModel standardModel = new InfiniteVariance();
    PolicyInterface policyInterface = new ConstantPolicy(RationalScalar.of(9, 10));
    IterativePolicyEvaluation a = new IterativePolicyEvaluation( //
        standardModel, policyInterface);
    Tensor result = a.until(RealScalar.ONE, RealScalar.of(.001));
    System.out.println(a.iterations());
    Index statesIndex = Index.build(standardModel.states());
    for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
      Tensor state = statesIndex.get(stateI);
      System.out.println("state=" + state + "  value=" + result.get(stateI));
    }
  }
}
