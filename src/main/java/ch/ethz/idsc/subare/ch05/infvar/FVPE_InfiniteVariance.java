// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.mc.FirstVisitPolicyEvaluation;
import ch.ethz.idsc.subare.core.util.ExploringStartsBatch;
import ch.ethz.idsc.tensor.RationalScalar;

class FVPE_InfiniteVariance {
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    FirstVisitPolicyEvaluation fvpe = new FirstVisitPolicyEvaluation( //
        infiniteVariance, null);
    PolicyInterface policyInterface = new ConstantPolicy(RationalScalar.of(5, 10));
    for (int count = 0; count < 100; ++count)
      ExploringStartsBatch.apply(infiniteVariance, fvpe, policyInterface);
    System.out.println(fvpe.vs().values());
  }
}
