// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.tensor.RationalScalar;

class FVMC_InfiniteVariance {
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    PolicyInterface policyInterface = new ConstantPolicy(RationalScalar.of(9, 10));
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        infiniteVariance, policyInterface, infiniteVariance);
    fvpe.simulate(123);
    // IterativePolicyEvaluation a = new IterativePolicyEvaluation( //
    // standardModel, policyInterface, RealScalar.ONE, RealScalar.of(.001));
    // Tensor result = a.values();
    // System.out.println(a.iterations());
    // Index statesIndex = Index.build(standardModel.states());
    // for (int stateI = 0; stateI < statesIndex.size(); ++stateI) {
    // Tensor state = statesIndex.get(stateI);
    // System.out.println("state=" + state + " value=" + result.get(stateI));
    // }
  }
}
